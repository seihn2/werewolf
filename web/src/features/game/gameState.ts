import type { Game, GameEvent, PlayerState } from '../../types'

export function mergeEvents(current: GameEvent[], incoming: GameEvent[]): GameEvent[] {
  const byTime = new Map<number, GameEvent>()
  for (const event of [...current, ...incoming]) byTime.set(event.logical_time, event)
  return [...byTime.values()].sort((left, right) => left.logical_time - right.logical_time)
}

export function derivePlayers(game: Game, events: GameEvent[]): Record<string, PlayerState> {
  const source = game.result?.players ?? game.players
  const players = Object.fromEntries(
    Object.entries(source).map(([id, player]) => [id, { ...player }]),
  ) as Record<string, PlayerState>
  for (const event of events) {
    const eliminated =
      event.topic === 'night_result'
        ? (event.payload.victim_id as string | null)
        : event.topic === 'vote_result'
          ? (event.payload.eliminated_id as string | null)
          : null
    if (eliminated && players[eliminated]) players[eliminated].alive = false
  }
  return players
}

export function activeVotes(events: GameEvent[]): Array<{ from: string; to: string }> {
  const currentRound = Math.max(0, ...events.map((event) => event.round_no))
  const roundEvents = events.filter((event) => event.round_no === currentRound)
  const lastResultIndex = roundEvents.findLastIndex((event) => event.topic === 'vote_result')
  const relevant = roundEvents.slice(lastResultIndex + 1)
  const votes = new Map<string, string>()
  for (const event of relevant) {
    if (event.topic !== 'vote_cast' || !event.sender || !event.payload.target_id) continue
    votes.set(event.sender, event.payload.target_id as string)
  }
  return [...votes].map(([from, to]) => ({ from, to }))
}

export function latestSpeaker(events: GameEvent[]): string | null {
  return [...events].reverse().find((event) => event.topic === 'speech')?.sender ?? null
}

export function eventTitle(event: GameEvent): string {
  const sender = event.sender?.replace('player_', '玩家 ')
  const player = (value: unknown) =>
    typeof value === 'string' ? value.replace('player_', '玩家 ') : '无人'
  switch (event.topic) {
    case 'game_started':
      return '对局开始，七位 Agent 已入场'
    case 'role_assignment':
      return `${player(event.payload.player_id)} 获得身份`
    case 'werewolf_team':
      return '狼人阵营确认队友'
    case 'werewolf_proposal':
      return `${sender} 提议袭击 ${player(event.payload.target_id)}`
    case 'seer_result':
      return `预言家完成对 ${player(event.payload.target_id)} 的查验`
    case 'doctor_choice':
      return `医生选择守护 ${player(event.payload.target_id)}`
    case 'night_result':
      return event.payload.nobody_died
        ? '平安夜，没有玩家出局'
        : `${player(event.payload.victim_id)} 倒在了夜里`
    case 'day_started':
      return `第 ${event.round_no} 天开始讨论`
    case 'speech':
      return `${sender} 发言`
    case 'vote_cast':
      return `${sender} 投向 ${player(event.payload.target_id)}`
    case 'vote_result':
      return event.payload.eliminated_id
        ? `${player(event.payload.eliminated_id)} 被放逐出局`
        : '本轮无人被放逐'
    case 'game_over':
      return `对局结束 · ${event.payload.winner === 'werewolf' ? '狼人阵营胜利' : event.payload.winner === 'village' ? '村庄阵营胜利' : '平局'}`
    default:
      return event.topic.replaceAll('_', ' ')
  }
}

export function eventDetail(event: GameEvent): string | null {
  if (event.topic === 'speech') return String(event.payload.message ?? '')
  if (event.topic === 'vote_result') {
    const tally = event.payload.tally as Record<string, number> | undefined
    if (!tally || Object.keys(tally).length === 0) return null
    return Object.entries(tally)
      .sort(([, left], [, right]) => right - left)
      .map(([id, count]) => `${id.replace('player_', '玩家 ')} ${count} 票`)
      .join(' · ')
  }
  if (event.topic === 'game_over') return String(event.payload.reason ?? '')
  return null
}

export function eventTone(event: GameEvent): 'wolf' | 'village' | 'system' | 'speech' {
  if (event.topic.startsWith('werewolf')) return 'wolf'
  if (event.topic === 'seer_result' || event.topic === 'doctor_choice') return 'village'
  if (event.topic === 'speech') return 'speech'
  return 'system'
}

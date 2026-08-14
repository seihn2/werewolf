import { Eye, Moon, PawPrint, ShieldPlus, Sparkles, UserRound, X } from 'lucide-react'

import type { Game, GameEvent, PlayerState } from '../../types'
import { formatPhase, formatRole } from '../../lib/format'
import { activeVotes, derivePlayers, latestSpeaker } from './gameState'

const roleIcons = {
  werewolf: PawPrint,
  seer: Eye,
  doctor: ShieldPlus,
  villager: UserRound,
}

function point(index: number, total: number) {
  const angle = -Math.PI / 2 + (index * Math.PI * 2) / total
  return { x: 50 + Math.cos(angle) * 39, y: 50 + Math.sin(angle) * 39 }
}

export function GameTable({
  game,
  events,
  revealRoles = false,
  playbackLabel,
}: {
  game: Game
  events: GameEvent[]
  revealRoles?: boolean
  playbackLabel?: string
}) {
  const players = derivePlayers(game, events)
  const ordered = Object.values(players).sort((left, right) => left.player_id.localeCompare(right.player_id))
  const speaker = latestSpeaker(events)
  const votes = activeVotes(events)
  const indexById = new Map(ordered.map((player, index) => [player.player_id, index]))
  const publicPhase = events.at(-1)?.phase ?? game.current_phase
  const round = events.at(-1)?.round_no ?? game.current_round

  return (
    <div className={`game-table phase-${publicPhase}`}>
      <div className="table-orbit orbit-one" />
      <div className="table-orbit orbit-two" />
      <svg className="vote-map" viewBox="0 0 100 100" aria-label="当前投票关系">
        {votes.map(({ from, to }) => {
          const fromIndex = indexById.get(from)
          const toIndex = indexById.get(to)
          if (fromIndex === undefined || toIndex === undefined) return null
          const start = point(fromIndex, ordered.length)
          const end = point(toIndex, ordered.length)
          return (
            <line
              key={`${from}-${to}`}
              x1={start.x}
              y1={start.y}
              x2={end.x}
              y2={end.y}
              markerEnd="url(#vote-arrow)"
            />
          )
        })}
        <defs>
          <marker id="vote-arrow" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
            <path d="M0,0 L5,2.5 L0,5 z" />
          </marker>
        </defs>
      </svg>

      <div className="table-center">
        <div className="moon-disc"><Moon size={30} fill="currentColor" /></div>
        <span>ROUND {String(round).padStart(2, '0')}</span>
        <strong>{formatPhase(publicPhase)}</strong>
        <small>{playbackLabel ?? (game.status === 'running' ? 'LIVE SIMULATION' : game.status.toUpperCase())}</small>
      </div>

      {ordered.map((player, index) => (
        <PlayerSeat
          key={player.player_id}
          player={player}
          index={index}
          total={ordered.length}
          speaking={speaker === player.player_id && player.alive}
          revealRole={revealRoles || game.status === 'completed'}
        />
      ))}
    </div>
  )
}

function PlayerSeat({
  player,
  index,
  total,
  speaking,
  revealRole,
}: {
  player: PlayerState
  index: number
  total: number
  speaking: boolean
  revealRole: boolean
}) {
  const role = player.role
  const RoleIcon = role ? roleIcons[role] : UserRound
  const position = point(index, total)
  return (
    <div
      className={`player-seat ${player.alive ? '' : 'eliminated'} ${speaking ? 'speaking' : ''} role-${role ?? 'hidden'}`}
      style={{ left: `${position.x}%`, top: `${position.y}%` }}
    >
      <div className="seat-avatar">
        {revealRole && role ? <RoleIcon size={22} /> : <span>{index + 1}</span>}
        {!player.alive && <X className="death-mark" size={28} />}
        {speaking && <Sparkles className="speaking-mark" size={14} />}
      </div>
      <strong>{player.name.replace('Player', '玩家')}</strong>
      <small>{revealRole && role ? formatRole(role) : player.alive ? '身份隐藏' : '已出局'}</small>
    </div>
  )
}

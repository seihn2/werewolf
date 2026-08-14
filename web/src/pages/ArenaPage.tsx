import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, Eye, Play, Radio, RotateCcw, Settings2, StopCircle, Swords } from 'lucide-react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import { api } from '../api/client'
import { useRealtime } from '../api/useRealtime'
import { ErrorState, InlineLoading } from '../components/States'
import { ConnectionBadge, PageHeader, StatusPill } from '../components/UI'
import { DecisionInspector } from '../features/game/DecisionInspector'
import { EventFeed } from '../features/game/EventFeed'
import { GameTable } from '../features/game/GameTable'
import { mergeEvents } from '../features/game/gameState'
import { formatPhase, formatWinner } from '../lib/format'
import type { Game, GameEvent } from '../types'

type GameMessage =
  | { type: 'snapshot'; game: Game; events: GameEvent[] }
  | { type: 'event'; event: GameEvent }
  | { type: 'status'; game: Game }
  | { type: 'error'; message: string }
  | { type: 'heartbeat' }

export default function ArenaPage() {
  const { gameId } = useParams()
  return gameId ? <LiveArena gameId={gameId} /> : <ArenaLobby />
}

function ArenaLobby() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const agents = useQuery({ queryKey: ['agents'], queryFn: api.agents })
  const games = useQuery({ queryKey: ['games', 'active'], queryFn: () => api.games('?limit=20') })
  const [form, setForm] = useState({ seed: '', max_rounds: 8, pace_seconds: 0.35, werewolf_agent_id: 'heuristic', village_agent_id: 'heuristic', label: '' })
  const create = useMutation({
    mutationFn: () => api.createGame({ ...form, seed: form.seed ? Number(form.seed) : null, label: form.label || null }),
    onSuccess: (game) => {
      queryClient.invalidateQueries({ queryKey: ['games'] })
      navigate(`/arena/${game.id}`)
    },
  })
  const availableAgents = agents.data?.filter((agent) => agent.enabled) ?? []
  const activeGames = games.data?.items.filter((game) => ['queued', 'running'].includes(game.status)) ?? []

  return (
    <div className="page-pad arena-lobby">
      <PageHeader eyebrow="REALTIME ARENA" title="布置今晚的七人局" description="选择阵营 Agent、对局节奏和随机种子。公开事件将实时广播，私密身份只在终局揭晓。" />
      <div className="arena-lobby-grid reveal delay-1">
        <form className="panel match-form" onSubmit={(event) => { event.preventDefault(); create.mutate() }}>
          <div className="form-heading"><Swords size={20} /><div><h2>对局配置</h2><p>Standard 7P · 2 狼 / 1 预言家 / 1 医生 / 3 村民</p></div></div>
          <label className="field full"><span>对局名称</span><input value={form.label} onChange={(event) => setForm({ ...form, label: event.target.value })} placeholder="例如：Baseline vs CFR Challenger" /></label>
          <div className="field-row">
            <label className="field"><span>随机种子</span><input inputMode="numeric" value={form.seed} onChange={(event) => setForm({ ...form, seed: event.target.value })} placeholder="自动生成" /></label>
            <label className="field"><span>最大轮数</span><input type="number" min="1" max="50" value={form.max_rounds} onChange={(event) => setForm({ ...form, max_rounds: Number(event.target.value) })} /></label>
          </div>
          <label className="field full"><span>公开事件节奏 <b>{form.pace_seconds.toFixed(2)}s</b></span><input type="range" min="0" max="1.5" step="0.05" value={form.pace_seconds} onChange={(event) => setForm({ ...form, pace_seconds: Number(event.target.value) })} /><small>演示建议 0.25–0.50 秒；批量验证可设为 0。</small></label>
          <div className="faction-picks">
            <label className="agent-pick wolf-pick"><span>狼人阵营 Agent</span><select value={form.werewolf_agent_id} onChange={(event) => setForm({ ...form, werewolf_agent_id: event.target.value })}>{availableAgents.map((agent) => <option key={agent.id} value={agent.id}>{agent.name}</option>)}</select><small>负责夜间协作、伪装与带票</small></label>
            <label className="agent-pick village-pick"><span>村庄阵营 Agent</span><select value={form.village_agent_id} onChange={(event) => setForm({ ...form, village_agent_id: event.target.value })}>{availableAgents.map((agent) => <option key={agent.id} value={agent.id}>{agent.name}</option>)}</select><small>负责查验、守护、推理与投票</small></label>
          </div>
          {create.isError && <div className="form-error">{create.error.message}</div>}
          <button className="button primary large" type="submit" disabled={create.isPending || agents.isLoading}><Play size={17} fill="currentColor" />{create.isPending ? '正在召集 Agent…' : '开始实时对局'}</button>
        </form>

        <aside className="panel active-match-list">
          <div className="panel-title"><div><Radio size={18} /><h2>运行中的竞技场</h2></div><small>{activeGames.length} ACTIVE</small></div>
          {games.isLoading && <InlineLoading />}
          {activeGames.map((game) => (
            <Link to={`/arena/${game.id}`} key={game.id} className="active-match-card">
              <div><StatusPill status={game.status} /><strong>{String(game.config.label ?? `Seed ${game.seed}`)}</strong><p>第 {game.current_round} 轮 · {formatPhase(game.current_phase)}</p></div>
              <span>{game.event_count}<small>EVENTS</small></span>
            </Link>
          ))}
          {!games.isLoading && activeGames.length === 0 && <div className="empty-stage"><div className="empty-ring" /><strong>当前没有进行中的对局</strong><p>左侧配置完成后，竞技场会立即进入实时状态。</p></div>}
          <Link className="text-link" to="/agents"><Settings2 size={14} />管理模型 Agent</Link>
        </aside>
      </div>
    </div>
  )
}

function LiveArena({ gameId }: { gameId: string }) {
  const queryClient = useQueryClient()
  const [realtimeEvents, setRealtimeEvents] = useState<GameEvent[]>([])
  const [transportError, setTransportError] = useState<string | null>(null)
  const gameQuery = useQuery({ queryKey: ['game', gameId], queryFn: () => api.game(gameId), refetchInterval: (query) => ['queued', 'running'].includes(query.state.data?.status ?? '') ? 3_000 : false })
  const eventsQuery = useQuery({ queryKey: ['game-events', gameId, 'public'], queryFn: () => api.gameEvents(gameId), refetchOnMount: true })
  const events = useMemo(
    () => mergeEvents(eventsQuery.data ?? [], realtimeEvents),
    [eventsQuery.data, realtimeEvents],
  )

  const connection = useRealtime<GameMessage>(`/ws/games/${gameId}`, (message) => {
    if (message.type === 'snapshot') {
      queryClient.setQueryData(['game', gameId], message.game)
      setRealtimeEvents((current) => mergeEvents(current, message.events))
    } else if (message.type === 'event') {
      setRealtimeEvents((current) => mergeEvents(current, [message.event]))
    } else if (message.type === 'status') {
      queryClient.setQueryData(['game', gameId], message.game)
      if (message.game.status === 'completed') queryClient.invalidateQueries({ queryKey: ['analytics'] })
    } else if (message.type === 'error') {
      setTransportError(message.message)
    }
  })
  const cancel = useMutation({ mutationFn: () => api.cancelGame(gameId), onSuccess: (game) => queryClient.setQueryData(['game', gameId], game) })
  const game = gameQuery.data
  const traces = game?.result?.decision_traces ?? []
  const title = game ? String(game.config.label ?? `Seed ${game.seed}`) : '正在进入竞技场'
  const isActive = game && ['queued', 'running'].includes(game.status)

  if (gameQuery.isError) return <div className="page-pad"><ErrorState error={gameQuery.error} retry={() => gameQuery.refetch()} /></div>
  if (!game) return <div className="page-pad"><InlineLoading label="读取对局状态" /></div>

  return (
    <div className="page-pad live-arena">
      <header className="arena-topbar reveal">
        <div className="arena-breadcrumb"><Link to="/arena"><ArrowLeft size={15} />竞技场</Link><span>/</span><strong>{title}</strong></div>
        <div className="arena-status"><ConnectionBadge state={connection} /><StatusPill status={game.status} />{isActive && <button className="button danger compact" onClick={() => cancel.mutate()} disabled={cancel.isPending}><StopCircle size={14} />停止对局</button>}</div>
      </header>
      {transportError && <div className="form-error">{transportError}</div>}
      <div className="arena-workspace reveal delay-1">
        <section className="table-stage panel">
          <div className="stage-caption"><div><span>GAME {game.id.slice(-8).toUpperCase()}</span><h1>{title}</h1></div><div><small>当前阶段</small><strong>{formatPhase(events.at(-1)?.phase ?? game.current_phase)}</strong></div></div>
          <GameTable game={game} events={events} revealRoles={game.status === 'completed'} />
          <div className="stage-footer">
            <span>Seed {game.seed}</span><span>{events.length} public events</span><span>Lamport t={events.at(-1)?.logical_time ?? 0}</span>
          </div>
        </section>
        <EventFeed events={events} />
      </div>

      {game.status === 'completed' && (
        <section className={`verdict-banner winner-${game.winner} reveal`}>
          <div><Eye size={22} /><span>FINAL VERDICT</span><h2>{formatWinner(game.winner)}获胜</h2></div>
          <p>{game.termination_reason?.replaceAll('_', ' ')} · 共 {game.rounds} 轮 · {traces.length} 条决策轨迹</p>
          <Link className="button light" to={`/replays/${game.id}`}><RotateCcw size={16} />进入全知回放</Link>
        </section>
      )}
      {game.error && <div className="error-state">{game.error}</div>}
      {traces.length > 0 && <DecisionInspector traces={traces} />}
    </div>
  )
}

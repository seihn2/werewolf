import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, ChevronLeft, ChevronRight, Pause, Play, RotateCcw, Search } from 'lucide-react'
import { Link, useParams } from 'react-router-dom'

import { api } from '../api/client'
import { EmptyState, ErrorState, InlineLoading } from '../components/States'
import { PageHeader, StatusPill } from '../components/UI'
import { DecisionInspector } from '../features/game/DecisionInspector'
import { EventFeed } from '../features/game/EventFeed'
import { GameTable } from '../features/game/GameTable'
import { formatDate, formatWinner } from '../lib/format'

export default function ReplaysPage() {
  const { gameId } = useParams()
  return gameId ? <ReplayDetail key={gameId} gameId={gameId} /> : <ReplayLibrary />
}

function ReplayLibrary() {
  const [winner, setWinner] = useState('')
  const [query, setQuery] = useState('')
  const games = useQuery({ queryKey: ['games', 'replays', winner], queryFn: () => api.games(`?limit=100&status=completed${winner ? `&winner=${winner}` : ''}`) })
  const filtered = games.data?.items.filter((game) => !query || `${game.id} ${game.seed} ${game.config.label ?? ''}`.toLowerCase().includes(query.toLowerCase())) ?? []

  return (
    <div className="page-pad replays-page">
      <PageHeader eyebrow="ARCHIVE & REPLAY" title="每一次判断，都留在时间线上" description="查看公开视角与终局全知视角，逐事件复盘发言、投票和 Agent 决策。" />
      <div className="archive-toolbar reveal delay-1">
        <label className="search-field"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索名称、Seed 或 Game ID" /></label>
        <div className="segmented">
          {([['', '全部'], ['werewolf', '狼人胜'], ['village', '村庄胜'], ['draw', '平局']] as const).map(([value, label]) => <button className={winner === value ? 'active' : ''} key={value} onClick={() => setWinner(value)}>{label}</button>)}
        </div>
      </div>
      {games.isError && <ErrorState error={games.error} retry={() => games.refetch()} />}
      {games.isLoading && <InlineLoading label="读取历史档案" />}
      <div className="replay-grid reveal delay-2">
        {filtered.map((game) => (
          <Link className={`replay-card winner-${game.winner}`} to={`/replays/${game.id}`} key={game.id}>
            <div className="replay-card-top"><span>{game.id.slice(-8).toUpperCase()}</span><StatusPill status={game.status} /></div>
            <div className="replay-verdict"><small>VERDICT</small><strong>{formatWinner(game.winner)}</strong><p>{game.termination_reason?.replaceAll('_', ' ')}</p></div>
            <div className="replay-meta"><span>{game.rounds} 轮</span><span>{game.event_count} 事件</span><span>Seed {game.seed}</span></div>
            <footer><span>{String(game.config.label ?? '未命名对局')}</span><time>{formatDate(game.completed_at)}</time></footer>
          </Link>
        ))}
      </div>
      {!games.isLoading && filtered.length === 0 && <EmptyState title="没有匹配的回放" detail="完成一局对战后，公开事件、私密行动和认知轨迹都会永久保存在这里。" action={<Link className="button primary" to="/arena">发起第一局</Link>} />}
    </div>
  )
}

function ReplayDetail({ gameId }: { gameId: string }) {
  const game = useQuery({ queryKey: ['game', gameId], queryFn: () => api.game(gameId) })
  const events = useQuery({ queryKey: ['game-events', gameId, 'omniscient'], queryFn: () => api.gameEvents(gameId, 'omniscient'), enabled: game.data?.status === 'completed' })
  const [cursor, setCursor] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const isPlaying = playing && Boolean(events.data) && cursor < (events.data?.length ?? 0)
  useEffect(() => {
    if (!isPlaying || !events.data) return
    const timer = window.setTimeout(() => setCursor((value) => Math.min(events.data!.length, value + 1)), 700 / speed)
    return () => window.clearTimeout(timer)
  }, [isPlaying, cursor, speed, events.data])
  const visible = useMemo(() => events.data?.slice(0, cursor) ?? [], [events.data, cursor])
  if (game.isError || events.isError) return <div className="page-pad"><ErrorState error={game.error ?? events.error} /></div>
  if (!game.data || !events.data) return <div className="page-pad"><InlineLoading label="装载全知时间线" /></div>

  return (
    <div className="page-pad replay-detail">
      <header className="replay-header reveal">
        <div><Link to="/replays"><ArrowLeft size={15} />回放档案</Link><span className="eyebrow">OMNISCIENT VIEW</span><h1>{String(game.data.config.label ?? `Seed ${game.data.seed}`)}</h1><p>{formatWinner(game.data.winner)} · {game.data.termination_reason?.replaceAll('_', ' ')}</p></div>
        <div className="replay-clock"><span>{String(cursor).padStart(3, '0')}</span><i>/</i><small>{String(events.data.length).padStart(3, '0')} EVENTS</small></div>
      </header>
      <div className="replay-workspace reveal delay-1">
        <section className="table-stage panel">
          <GameTable game={game.data} events={visible} revealRoles playbackLabel={isPlaying ? `${speed}× PLAYBACK` : 'PAUSED'} />
          <div className="replay-controls">
            <button aria-label="后退一步" onClick={() => setCursor((value) => Math.max(0, value - 1))}><ChevronLeft size={17} /></button>
            <button className="play-control" aria-label={isPlaying ? '暂停' : '播放'} onClick={() => { if (!isPlaying && cursor >= events.data!.length) setCursor(0); setPlaying((value) => !value) }}>{isPlaying ? <Pause size={19} fill="currentColor" /> : <Play size={19} fill="currentColor" />}</button>
            <button aria-label="前进一步" onClick={() => setCursor((value) => Math.min(events.data!.length, value + 1))}><ChevronRight size={17} /></button>
            <input aria-label="回放进度" type="range" min="0" max={events.data.length} value={cursor} onChange={(event) => setCursor(Number(event.target.value))} />
            <div className="speed-controls">{[0.5, 1, 2, 4].map((value) => <button className={speed === value ? 'active' : ''} key={value} onClick={() => setSpeed(value)}>{value}×</button>)}</div>
            <button aria-label="重新开始" onClick={() => { setCursor(0); setPlaying(false) }}><RotateCcw size={16} /></button>
          </div>
        </section>
        <EventFeed events={visible} autoFollow title="全知时间线" />
      </div>
      <div className="timeline-ticks">
        {events.data.map((event, index) => <button key={event.logical_time} className={`${index < cursor ? 'passed' : ''} topic-${event.topic}`} title={`${event.logical_time} · ${event.topic}`} onClick={() => setCursor(index + 1)} />)}
      </div>
      {game.data.result?.decision_traces && <DecisionInspector traces={game.data.result.decision_traces} />}
    </div>
  )
}

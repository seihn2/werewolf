import { useQuery } from '@tanstack/react-query'
import {
  ArrowRight,
  BrainCircuit,
  CircleDotDashed,
  FlaskConical,
  Play,
  Radio,
  ShieldCheck,
  Sparkles,
  Swords,
} from 'lucide-react'
import { Link } from 'react-router-dom'

import { api } from '../api/client'
import { ErrorState, InlineLoading } from '../components/States'
import { PageHeader, SectionHeader, StatusPill } from '../components/UI'
import { formatDate, formatPhase, formatWinner, percent } from '../lib/format'

export default function DashboardPage() {
  const analytics = useQuery({ queryKey: ['analytics'], queryFn: api.analytics })
  const games = useQuery({ queryKey: ['games', 'dashboard'], queryFn: () => api.games('?limit=8') })
  const jobs = useQuery({ queryKey: ['jobs', 'dashboard'], queryFn: () => api.jobs('?limit=5') })

  if (analytics.isError) return <div className="page-pad"><ErrorState error={analytics.error} retry={() => analytics.refetch()} /></div>
  const stats = analytics.data
  const active = games.data?.items.find((game) => game.status === 'running' || game.status === 'queued')
  const recentCompleted = games.data?.items.filter((game) => game.status === 'completed') ?? []

  return (
    <div className="page-pad dashboard-page">
      <PageHeader
        eyebrow="COMMAND OVERVIEW"
        title="今晚，谁在说谎？"
        description="从一场实时博弈出发，追踪 Agent 的判断、策略与训练演化。"
        actions={
          <Link className="button primary" to="/arena"><Play size={16} fill="currentColor" />发起新对局</Link>
        }
      />

      <section className="dashboard-hero reveal delay-1">
        <div className="hero-copy">
          <span className="hero-kicker"><Sparkles size={14} /> LIVE STRATEGIC SIMULATION</span>
          {active ? (
            <>
              <h2>{String(active.config.label ?? `对局 ${active.seed}`)}</h2>
              <p>第 {active.current_round} 轮 · {formatPhase(active.current_phase)} · 已记录 {active.event_count} 个逻辑事件</p>
              <div className="hero-actions">
                <Link className="button light" to={`/arena/${active.id}`}><Radio size={16} />进入实时竞技场</Link>
                <StatusPill status={active.status} />
              </div>
            </>
          ) : (
            <>
              <h2>竞技场正在等待下一组 Agent</h2>
              <p>启发式模式可离线运行；也可以为不同阵营接入独立的 OpenAI-compatible 模型。</p>
              <div className="hero-actions">
                <Link className="button light" to="/arena"><Swords size={16} />配置并开局</Link>
              </div>
            </>
          )}
        </div>
        <div className="hero-moon" aria-hidden="true">
          <div className="moon-surface" />
          <div className="orbit-label orbit-label-one">PLANNER</div>
          <div className="orbit-label orbit-label-two">EVALUATOR</div>
          <div className="orbit-label orbit-label-three">REFLEXION</div>
        </div>
      </section>

      <section className="metric-ribbon reveal delay-2">
        <Metric icon={CircleDotDashed} label="已完成对局" value={stats ? String(stats.completed_games) : '—'} detail={`${stats?.active_games ?? 0} 局运行中`} />
        <Metric icon={Swords} label="狼人胜局" value={String(stats?.winner_counts.werewolf ?? 0)} detail={stats?.completed_games ? percent((stats.winner_counts.werewolf ?? 0) / stats.completed_games) : '等待样本'} tone="wolf" />
        <Metric icon={ShieldCheck} label="村庄胜局" value={String(stats?.winner_counts.village ?? 0)} detail={stats?.completed_games ? percent((stats.winner_counts.village ?? 0) / stats.completed_games) : '等待样本'} tone="village" />
        <Metric icon={BrainCircuit} label="Reflexion 率" value={stats ? percent(stats.reflection_rate) : '—'} detail={`${stats?.average_rounds.toFixed(1) ?? '0.0'} 平均轮次`} />
      </section>

      <div className="dashboard-grid reveal delay-3">
        <section className="panel recent-games">
          <SectionHeader title="最近对局" detail="可继续进入实时局，或打开已完成回放" link={{ to: '/replays', label: '全部回放' }} />
          {games.isLoading && <InlineLoading />}
          <div className="record-list">
            {games.data?.items.slice(0, 6).map((game) => (
              <Link key={game.id} to={game.status === 'completed' ? `/replays/${game.id}` : `/arena/${game.id}`} className="record-row">
                <span className={`record-glyph winner-${game.winner ?? 'pending'}`}>{game.status === 'completed' ? (game.winner === 'werewolf' ? 'W' : game.winner === 'village' ? 'V' : 'D') : '•'}</span>
                <div><strong>{String(game.config.label ?? `Seed ${game.seed}`)}</strong><small>{formatDate(game.created_at)} · {game.rounds ?? game.current_round} 轮</small></div>
                <div className="record-result"><StatusPill status={game.status} /><span>{formatWinner(game.winner)}</span></div>
                <ArrowRight size={15} />
              </Link>
            ))}
            {!games.isLoading && games.data?.items.length === 0 && <p className="quiet-copy">尚无对局记录，创建第一局后这里会形成完整时间线。</p>}
          </div>
        </section>

        <section className="panel pipeline-panel">
          <SectionHeader title="策略学习流水线" detail="从语言轨迹到可采样策略网络" link={{ to: '/training', label: '打开训练工作台' }} />
          <div className="pipeline-map">
            {[
              ['01', '自博弈轨迹', '完整事件与认知候选'],
              ['02', '潜在策略聚类', 'Embedding · K-Means'],
              ['03', 'Deep CFR', 'Regret · Average Policy'],
              ['04', 'CFR-DPO', '偏好构造与策略对齐'],
            ].map(([step, title, detail], index) => (
              <div className="pipeline-step" key={step}>
                <span>{step}</span><div><strong>{title}</strong><small>{detail}</small></div>
                {index < 3 && <i />}
              </div>
            ))}
          </div>
          <div className="job-mini-list">
            <div className="mini-title"><FlaskConical size={14} />最近训练任务</div>
            {jobs.data?.items.slice(0, 3).map((job) => (
              <Link to={`/training/${job.id}`} key={job.id}><span>{job.kind.replaceAll('_', ' ')}</span><StatusPill status={job.status} /></Link>
            ))}
            {!jobs.data?.items.length && <small>暂无任务，建议从 5–20 局自博弈冒烟任务开始。</small>}
          </div>
        </section>
      </div>

      {recentCompleted.length > 0 && (
        <section className="story-strip reveal delay-4">
          <div><span>LAST VERDICT</span><strong>{formatWinner(recentCompleted[0]?.winner)}</strong></div>
          <p>{recentCompleted[0]?.termination_reason?.replaceAll('_', ' ')} · Seed {recentCompleted[0]?.seed} · {recentCompleted[0]?.rounds} 轮</p>
          <Link to={`/replays/${recentCompleted[0]?.id}`}>查看判决过程 <ArrowRight size={14} /></Link>
        </section>
      )}
    </div>
  )
}

function Metric({ icon: Icon, label, value, detail, tone = 'neutral' }: { icon: typeof Swords; label: string; value: string; detail: string; tone?: string }) {
  return (
    <div className={`metric-cell tone-${tone}`}>
      <Icon size={18} strokeWidth={1.5} />
      <div><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>
    </div>
  )
}

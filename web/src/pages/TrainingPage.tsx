import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  ArrowLeft,
  Boxes,
  BrainCircuit,
  ChevronRight,
  CircleStop,
  Database,
  FileJson2,
  FlaskConical,
  Layers3,
  Play,
  RefreshCw,
  ScrollText,
  Sparkles,
} from 'lucide-react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import { api } from '../api/client'
import { useRealtime } from '../api/useRealtime'
import { EmptyState, ErrorState, InlineLoading } from '../components/States'
import { ConnectionBadge, PageHeader, ProgressBar, StatusPill } from '../components/UI'
import { formatBytes, formatDate, formatDuration, percent } from '../lib/format'
import type { Artifact, TrainingJob, TrainingKind } from '../types'

const jobTypes: Array<{ kind: TrainingKind; label: string; detail: string; icon: typeof Database; accent: string }> = [
  { kind: 'self_play', label: '自博弈采样', detail: '生成完整对局轨迹 JSONL', icon: Database, accent: 'amber' },
  { kind: 'latent', label: '潜在策略聚类', detail: 'Embedding + 角色级 K-Means', icon: Boxes, accent: 'cyan' },
  { kind: 'deep_cfr', label: 'Deep CFR', detail: 'External Sampling + Regret 网络', icon: BrainCircuit, accent: 'red' },
  { kind: 'cfr_dpo', label: 'CFR-DPO 数据', detail: '用 Advantage 排序语言候选', icon: FileJson2, accent: 'violet' },
  { kind: 'dpo', label: 'DPO 对齐', detail: 'TRL + LoRA 策略后训练', icon: Layers3, accent: 'green' },
  { kind: 'iterative', label: '策略迭代器', detail: '采样 → 聚类 → CFR → DPO', icon: Sparkles, accent: 'gold' },
]

const defaultConfigs: Record<TrainingKind, Record<string, any>> = {
  self_play: { games: 20, concurrency: 2, seed: 2025, max_rounds: 8, backend: 'heuristic' },
  latent: { input_path: '', embedding_backend: 'hashing', hash_dimensions: 256, werewolf_clusters: 3, seer_clusters: 2, doctor_clusters: 2, villager_clusters: 2, seed: 42 },
  deep_cfr: { latent_space_path: '', iterations: 10, traversals_per_player: 4, advantage_train_steps: 100, strategy_train_steps: 200, batch_size: 128, learning_rate: 0.001, hidden_sizes: [256, 256, 256], max_traversal_depth: 64, max_rollout_steps: 512, device: 'auto', checkpoint_every: 1, seed: 42, max_rounds: 8, no_save_buffers: false },
  cfr_dpo: { input_path: '', checkpoint_path: '', embedding_backend: 'hashing', hash_dimensions: 256, device: 'cpu', winning_only: true },
  dpo: { dataset_path: '', model: '', epochs: 2, learning_rate: 0.000001, beta: 0.1, batch_size: 1, gradient_accumulation_steps: 16, max_length: 2048, use_lora: true, lora_r: 32, lora_alpha: 16 },
  iterative: { iterations: 2, games_per_iteration: 50, concurrency: 2, seed: 2025, max_rounds: 8, backend: 'heuristic', embedding_backend: 'hashing', hash_dimensions: 256, clusters_added_per_iteration: 1, cfr_iterations: 10, cfr_traversals_per_player: 4, cfr_advantage_train_steps: 100, cfr_strategy_train_steps: 200, no_resume: false, dpo_model: '' },
}

type TrainingMessage =
  | { type: 'snapshot'; job: TrainingJob; logs: string[] }
  | { type: 'log'; line: string }
  | { type: 'status'; job: TrainingJob }
  | { type: 'artifact'; path: string }
  | { type: 'error'; message: string }
  | { type: 'heartbeat' }

export default function TrainingPage() {
  const { jobId } = useParams()
  return jobId ? <TrainingDetail jobId={jobId} /> : <TrainingWorkspace />
}

function TrainingWorkspace() {
  const navigate = useNavigate()
  const client = useQueryClient()
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: () => api.jobs('?limit=50'), refetchInterval: 5_000 })
  const artifacts = useQuery({ queryKey: ['artifacts'], queryFn: api.artifacts })
  const [kind, setKind] = useState<TrainingKind>('self_play')
  const [config, setConfig] = useState<Record<string, any>>(defaultConfigs.self_play)
  const create = useMutation({
    mutationFn: () => api.createJob(kind, config),
    onSuccess: (job) => {
      client.invalidateQueries({ queryKey: ['jobs'] })
      navigate(`/training/${job.id}`)
    },
  })
  const selectKind = (next: TrainingKind) => { setKind(next); setConfig({ ...defaultConfigs[next] }) }
  const files = artifacts.data?.items ?? []

  return (
    <div className="page-pad training-page">
      <PageHeader eyebrow="TRAINING OPERATIONS" title="把一次策略实验，变成可追踪任务" description="统一编排自博弈、潜在策略、Deep CFR、CFR-DPO 与 DPO；每个任务都有状态、日志和产物。" />
      <div className="training-grid reveal delay-1">
        <section className="panel job-builder">
          <div className="panel-title"><div><FlaskConical size={18} /><h2>新建训练任务</h2></div><small>WHITELISTED RUNNER</small></div>
          <div className="job-type-grid">
            {jobTypes.map(({ kind: value, label, detail, icon: Icon, accent }) => (
              <button key={value} className={`${kind === value ? 'active' : ''} accent-${accent}`} onClick={() => selectKind(value)}>
                <Icon size={19} /><span><strong>{label}</strong><small>{detail}</small></span>
              </button>
            ))}
          </div>
          <form className="dynamic-job-form" onSubmit={(event) => { event.preventDefault(); create.mutate() }}>
            <JobFields kind={kind} config={config} setConfig={setConfig} artifacts={files} />
            {create.isError && <div className="form-error">{create.error.message}</div>}
            <div className="runner-note"><ScrollText size={15} /><span>前端参数会被类型校验并转换为固定 CLI 参数，不接受任意 shell 命令。</span></div>
            <button className="button primary large" type="submit" disabled={create.isPending}><Play size={16} fill="currentColor" />{create.isPending ? '正在加入任务队列…' : '创建并启动任务'}</button>
          </form>
        </section>

        <aside className="panel artifact-browser">
          <div className="panel-title"><div><Database size={18} /><h2>产物仓库</h2></div><small>{files.length} FILES</small></div>
          {artifacts.isLoading && <InlineLoading />}
          <div className="artifact-list">
            {files.slice(0, 18).map((artifact) => (
              <a key={artifact.path} href={`/api/artifacts/${artifact.path}/download`} className="artifact-row">
                <span className={`artifact-icon category-${artifact.category}`}><FileJson2 size={15} /></span>
                <div><strong>{artifact.name}</strong><small>{artifact.path}</small></div>
                <span>{formatBytes(artifact.size)}<small>{formatDate(artifact.modified_at)}</small></span>
              </a>
            ))}
            {!artifacts.isLoading && files.length === 0 && <p className="quiet-copy">运行一次自博弈任务后，轨迹和后续训练产物会出现在这里。</p>}
          </div>
        </aside>
      </div>

      <section className="panel job-history reveal delay-2">
        <div className="panel-title"><div><RefreshCw size={18} /><h2>任务队列与历史</h2></div><small>{jobs.data?.total ?? 0} JOBS</small></div>
        {jobs.isError && <ErrorState error={jobs.error} />}
        <div className="job-table">
          {jobs.data?.items.map((job) => (
            <Link to={`/training/${job.id}`} className="job-row" key={job.id}>
              <span className={`job-kind kind-${job.kind}`}>{jobTypes.find((item) => item.kind === job.kind)?.label ?? job.kind}</span>
              <div><strong>{job.stage}</strong><small>{formatDate(job.created_at)} · {formatDuration(job.started_at, job.completed_at)}</small></div>
              <div className="job-progress"><ProgressBar value={job.progress} /><span>{percent(job.progress)}</span></div>
              <StatusPill status={job.status} /><ChevronRight size={15} />
            </Link>
          ))}
        </div>
        {!jobs.isLoading && !jobs.data?.items.length && <EmptyState title="训练队列为空" detail="上方选择任务类型并使用冒烟参数启动第一条可追踪训练链路。" />}
      </section>
    </div>
  )
}

function TrainingDetail({ jobId }: { jobId: string }) {
  const client = useQueryClient()
  const [streamedLogs, setStreamedLogs] = useState<string[] | null>(null)
  const [streamError, setStreamError] = useState<string | null>(null)
  const job = useQuery({ queryKey: ['job', jobId], queryFn: () => api.job(jobId), refetchInterval: (query) => ['queued', 'running', 'cancelling'].includes(query.state.data?.status ?? '') ? 3_000 : false })
  const initialLogs = useQuery({ queryKey: ['job-logs', jobId], queryFn: () => api.jobLogs(jobId) })
  const logs = streamedLogs ?? initialLogs.data?.lines ?? []
  const connection = useRealtime<TrainingMessage>(`/ws/training/${jobId}`, (message) => {
    if (message.type === 'snapshot') { client.setQueryData(['job', jobId], message.job); setStreamedLogs(message.logs) }
    else if (message.type === 'log') setStreamedLogs((current) => [...(current ?? initialLogs.data?.lines ?? []), message.line].slice(-3000))
    else if (message.type === 'status') client.setQueryData(['job', jobId], message.job)
    else if (message.type === 'error') setStreamError(message.message)
  })
  const cancel = useMutation({ mutationFn: () => api.cancelJob(jobId), onSuccess: (updated) => client.setQueryData(['job', jobId], updated) })
  const current = job.data
  const metadata = current ? jobTypes.find((item) => item.kind === current.kind) : null
  if (job.isError) return <div className="page-pad"><ErrorState error={job.error} /></div>
  if (!current) return <div className="page-pad"><InlineLoading label="读取训练任务" /></div>

  return (
    <div className="page-pad training-detail">
      <header className="task-header reveal">
        <div><Link to="/training"><ArrowLeft size={15} />训练工作台</Link><span className="eyebrow">{current.id.toUpperCase()}</span><h1>{metadata?.label ?? current.kind}</h1><p>{metadata?.detail} · {current.stage}</p></div>
        <div className="task-status"><ConnectionBadge state={connection} /><StatusPill status={current.status} />{['queued', 'running', 'cancelling'].includes(current.status) && <button className="button danger compact" onClick={() => cancel.mutate()}><CircleStop size={15} />取消任务</button>}</div>
      </header>
      <section className="task-summary panel reveal delay-1">
        <div className="task-progress-ring" style={{ '--progress': `${current.progress * 360}deg` } as React.CSSProperties}><span>{Math.round(current.progress * 100)}<small>%</small></span></div>
        <div className="task-summary-copy"><span>ACTIVE STAGE</span><h2>{current.stage}</h2><ProgressBar value={current.progress} /><p>{current.status === 'running' ? '训练进程正在运行，日志会通过 WebSocket 实时追加。' : current.status === 'completed' ? '任务已完成，产物和指标已持久化。' : current.error ?? '任务正在队列中等待资源。'}</p></div>
        <dl><div><dt>创建时间</dt><dd>{formatDate(current.created_at)}</dd></div><div><dt>持续时间</dt><dd>{formatDuration(current.started_at, current.completed_at)}</dd></div><div><dt>进程 PID</dt><dd>{current.pid ?? '—'}</dd></div><div><dt>Exit Code</dt><dd>{current.exit_code ?? '—'}</dd></div></dl>
      </section>
      {streamError && <div className="form-error">{streamError}</div>}
      <div className="task-detail-grid reveal delay-2">
        <section className="terminal-panel panel">
          <div className="terminal-title"><div><i className="terminal-dot red" /><i className="terminal-dot amber" /><i className="terminal-dot green" /><strong>job.log</strong></div><span>{logs.length} lines</span></div>
          <pre>{logs.length ? logs.join('\n') : '等待训练进程输出…'}</pre>
        </section>
        <aside className="task-side">
          <section className="panel config-panel"><div className="panel-title"><div><ScrollText size={17} /><h2>任务配置</h2></div></div><dl>{Object.entries(current.config).map(([key, value]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{Array.isArray(value) ? value.join(', ') : String(value)}</dd></div>)}</dl></section>
          {current.metrics && <section className="panel metrics-panel"><div className="panel-title"><div><Sparkles size={17} /><h2>输出指标</h2></div></div><dl>{Object.entries(current.metrics).map(([key, value]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{typeof value === 'number' ? value.toFixed(Number.isInteger(value) ? 0 : 4) : String(value)}</dd></div>)}</dl></section>}
          {current.output_path && <section className="panel output-panel"><span>PRIMARY OUTPUT</span><strong>{current.output_path.split('/').at(-1)}</strong><code>{current.output_path}</code></section>}
        </aside>
      </div>
    </div>
  )
}

function JobFields({ kind, config, setConfig, artifacts }: { kind: TrainingKind; config: Record<string, any>; setConfig: (value: Record<string, any>) => void; artifacts: Artifact[] }) {
  const set = (name: string, value: any) => setConfig({ ...config, [name]: value })
  const files = (categories: Artifact['category'][]) => artifacts.filter((item) => categories.includes(item.category))
  const artifactSelect = (name: string, label: string, categories: Artifact['category'][]) => (
    <label className="field full"><span>{label}</span><select required value={config[name] ?? ''} onChange={(event) => set(name, event.target.value)}><option value="">选择已有产物</option>{files(categories).map((item) => <option value={item.path} key={item.path}>{item.path}</option>)}</select></label>
  )
  const number = (name: string, label: string, min = 0, step = 1) => <label className="field"><span>{label}</span><input type="number" min={min} step={step} value={config[name]} onChange={(event) => set(name, Number(event.target.value))} /></label>

  if (kind === 'self_play') return <><div className="field-row">{number('games', '对局数量', 1)}{number('concurrency', '并发数', 1)}</div><div className="field-row">{number('seed', '起始 Seed', 0)}{number('max_rounds', '最大轮数', 1)}</div><label className="field full"><span>推理后端</span><select value={config.backend} onChange={(event) => set('backend', event.target.value)}><option value="heuristic">Heuristic · 无需外部模型</option><option value="openai-compatible">OpenAI Compatible · 读取 WOLFPLAY_* 环境变量</option></select></label></>
  if (kind === 'latent') return <>{artifactSelect('input_path', '自博弈轨迹', ['self_play'])}<div className="field-row">{number('hash_dimensions', 'Embedding 维度', 8)}{number('seed', '聚类 Seed', 0)}</div><div className="field-grid-four">{number('werewolf_clusters', '狼人簇', 1)}{number('seer_clusters', '预言家簇', 1)}{number('doctor_clusters', '医生簇', 1)}{number('villager_clusters', '村民簇', 1)}</div></>
  if (kind === 'deep_cfr') return <>{artifactSelect('latent_space_path', '潜在策略空间', ['latent_space'])}<div className="field-row">{number('iterations', 'CFR Iterations', 1)}{number('traversals_per_player', '每玩家遍历', 1)}</div><div className="field-row">{number('advantage_train_steps', 'Advantage Steps', 0)}{number('strategy_train_steps', 'Strategy Steps', 0)}</div><div className="field-row">{number('batch_size', 'Batch Size', 1)}{number('learning_rate', 'Learning Rate', 0, 0.0001)}</div><label className="field full"><span>设备</span><select value={config.device} onChange={(event) => set('device', event.target.value)}><option value="auto">Auto</option><option value="cpu">CPU</option><option value="mps">Apple MPS</option><option value="cuda">CUDA</option></select></label></>
  if (kind === 'cfr_dpo') return <>{artifactSelect('input_path', '语言轨迹', ['self_play'])}{artifactSelect('checkpoint_path', 'Deep CFR Checkpoint', ['checkpoint'])}<div className="field-row">{number('hash_dimensions', 'Embedding 维度', 8)}<label className="field checkbox-field"><input type="checkbox" checked={config.winning_only} onChange={(event) => set('winning_only', event.target.checked)} /><span>仅保留胜方偏好</span></label></div></>
  if (kind === 'dpo') return <>{artifactSelect('dataset_path', 'DPO 偏好数据', ['preference_data'])}<label className="field full"><span>基础模型或本地路径</span><input required value={config.model} onChange={(event) => set('model', event.target.value)} placeholder="Qwen/Qwen3-4B-Instruct" /></label><div className="field-row">{number('epochs', 'Epochs', 0.1, 0.1)}{number('learning_rate', 'Learning Rate', 0, 0.000001)}</div><div className="field-row">{number('batch_size', 'Batch Size', 1)}{number('gradient_accumulation_steps', 'Gradient Accumulation', 1)}</div><label className="field checkbox-field full"><input type="checkbox" checked={config.use_lora} onChange={(event) => set('use_lora', event.target.checked)} /><span>启用 LoRA 参数高效训练</span></label></>
  return <><div className="field-row">{number('iterations', '策略迭代轮数', 1)}{number('games_per_iteration', '每轮自博弈局数', 1)}</div><div className="field-row">{number('concurrency', '采样并发', 1)}{number('seed', '起始 Seed', 0)}</div><div className="field-row">{number('cfr_iterations', '每轮 CFR Iterations', 1)}{number('cfr_traversals_per_player', 'CFR Traversals', 1)}</div><label className="field full"><span>可选 DPO 基础模型</span><input value={config.dpo_model} onChange={(event) => set('dpo_model', event.target.value)} placeholder="留空则只生成 CFR-DPO 数据" /></label></>
}

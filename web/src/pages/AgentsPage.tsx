import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Bot, CheckCircle2, KeyRound, Plus, Power, Server, Trash2, X } from 'lucide-react'

import { api } from '../api/client'
import { EmptyState, ErrorState, InlineLoading } from '../components/States'
import { PageHeader } from '../components/UI'

const emptyForm = {
  name: '',
  kind: 'openai_compatible',
  base_url: '',
  model: '',
  env_prefix: 'WOLFPLAY_CUSTOM',
  temperature: 0.7,
  timeout_seconds: 90,
}

export default function AgentsPage() {
  const client = useQueryClient()
  const agents = useQuery({ queryKey: ['agents'], queryFn: api.agents })
  const [showForm, setShowForm] = useState(false)
  const [form, setForm] = useState(emptyForm)
  const create = useMutation({
    mutationFn: () => api.createAgent({
      ...form,
      base_url: form.base_url || null,
      model: form.model || null,
      env_prefix: form.kind === 'openai_compatible' ? form.env_prefix : null,
    }),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ['agents'] })
      setShowForm(false)
      setForm(emptyForm)
    },
  })
  const update = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) => api.updateAgent(id, { enabled }),
    onSuccess: () => client.invalidateQueries({ queryKey: ['agents'] }),
  })
  const remove = useMutation({
    mutationFn: api.deleteAgent,
    onSuccess: () => client.invalidateQueries({ queryKey: ['agents'] }),
  })

  return (
    <div className="page-pad agents-page">
      <PageHeader
        eyebrow="AGENT REGISTRY"
        title="为不同阵营装配不同大脑"
        description="管理启发式与 OpenAI-compatible Agent。密钥始终留在服务环境变量中，不写入数据库。"
        actions={<button className="button primary" onClick={() => setShowForm(true)}><Plus size={16} />添加模型 Agent</button>}
      />

      <section className="security-note reveal delay-1">
        <KeyRound size={19} />
        <div><strong>Secret-safe 配置</strong><p>配置只保存 Endpoint、模型名和环境变量前缀。运行时读取 <code>{'{PREFIX}_API_KEY'}</code>，界面不会接触明文密钥。</p></div>
      </section>

      {agents.isError && <ErrorState error={agents.error} retry={() => agents.refetch()} />}
      {agents.isLoading && <InlineLoading label="读取 Agent 注册表" />}
      <div className="agent-grid reveal delay-2">
        {agents.data?.map((agent) => (
          <article className={`agent-card ${agent.enabled ? '' : 'disabled'}`} key={agent.id}>
            <header>
              <div className={`agent-avatar kind-${agent.kind}`}><Bot size={24} /></div>
              <div><span>{agent.builtin ? 'BUILT-IN CORE' : 'MODEL ENDPOINT'}</span><h2>{agent.name}</h2></div>
              <button className={`power-toggle ${agent.enabled ? 'on' : ''}`} title={agent.enabled ? '停用' : '启用'} onClick={() => update.mutate({ id: agent.id, enabled: !agent.enabled })}><Power size={16} /></button>
            </header>
            <div className="agent-specs">
              <div><span>类型</span><strong>{agent.kind === 'heuristic' ? 'Deterministic Heuristic' : 'OpenAI Compatible'}</strong></div>
              <div><span>模型</span><strong>{agent.model ?? 'Local cognition core'}</strong></div>
              <div><span>Endpoint</span><strong>{agent.base_url ?? 'In-process'}</strong></div>
              <div><span>采样</span><strong>T={agent.temperature} · {agent.timeout_seconds}s timeout</strong></div>
            </div>
            {agent.env_prefix && <div className="env-prefix"><Server size={14} /><span>{agent.env_prefix}_API_KEY</span><CheckCircle2 size={14} /></div>}
            <footer>
              <span className={agent.enabled ? 'ready' : 'paused'}><i />{agent.enabled ? '可用于新对局' : '已停用'}</span>
              {!agent.builtin && <button className="icon-button danger" title="删除 Agent" onClick={() => { if (window.confirm(`删除 ${agent.name}？`)) remove.mutate(agent.id) }}><Trash2 size={15} /></button>}
            </footer>
          </article>
        ))}
      </div>
      {!agents.isLoading && !agents.data?.length && <EmptyState title="还没有 Agent" detail="至少保留一个启发式 Agent 才能启动离线对局。" />}

      {showForm && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setShowForm(false)}>
          <form className="modal-card agent-modal" onSubmit={(event) => { event.preventDefault(); create.mutate() }} onMouseDown={(event) => event.stopPropagation()}>
            <header><div><span className="eyebrow">REGISTER ENDPOINT</span><h2>添加模型 Agent</h2></div><button type="button" className="icon-button" onClick={() => setShowForm(false)}><X size={18} /></button></header>
            <label className="field full"><span>显示名称</span><input required value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} placeholder="例如：Qwen3-32B Strategist" /></label>
            <label className="field full"><span>Agent 类型</span><select value={form.kind} onChange={(event) => setForm({ ...form, kind: event.target.value })}><option value="openai_compatible">OpenAI Compatible</option><option value="heuristic">Heuristic</option></select></label>
            {form.kind === 'openai_compatible' && <>
              <label className="field full"><span>Base URL</span><input required type="url" value={form.base_url} onChange={(event) => setForm({ ...form, base_url: event.target.value })} placeholder="http://127.0.0.1:8001/v1" /></label>
              <label className="field full"><span>模型名称</span><input required value={form.model} onChange={(event) => setForm({ ...form, model: event.target.value })} placeholder="your-model-name" /></label>
              <label className="field full"><span>环境变量前缀</span><input required pattern="[A-Z][A-Z0-9_]*" value={form.env_prefix} onChange={(event) => setForm({ ...form, env_prefix: event.target.value.toUpperCase() })} /><small>服务启动前设置 {form.env_prefix || 'PREFIX'}_API_KEY。</small></label>
              <div className="field-row"><label className="field"><span>Temperature</span><input type="number" min="0" max="2" step="0.05" value={form.temperature} onChange={(event) => setForm({ ...form, temperature: Number(event.target.value) })} /></label><label className="field"><span>Timeout (s)</span><input type="number" min="1" max="600" value={form.timeout_seconds} onChange={(event) => setForm({ ...form, timeout_seconds: Number(event.target.value) })} /></label></div>
            </>}
            {create.isError && <div className="form-error">{create.error.message}</div>}
            <footer><button type="button" className="button ghost" onClick={() => setShowForm(false)}>取消</button><button className="button primary" disabled={create.isPending}>{create.isPending ? '正在注册…' : '保存 Agent'}</button></footer>
          </form>
        </div>
      )}
    </div>
  )
}

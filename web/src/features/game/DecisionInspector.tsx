import { BrainCircuit, Check, ChevronRight, RotateCcw, X } from 'lucide-react'
import { useMemo, useState } from 'react'

import type { DecisionTrace } from '../../types'
import { formatPhase, formatRole } from '../../lib/format'

export function DecisionInspector({ traces }: { traces: DecisionTrace[] }) {
  const [selected, setSelected] = useState(0)
  const ordered = useMemo(() => [...traces].reverse(), [traces])
  const trace = ordered[selected]
  if (!trace) return null

  return (
    <section className="decision-inspector panel">
      <div className="panel-title">
        <div><BrainCircuit size={18} /><h2>认知决策轨迹</h2></div>
        <small>{traces.length} DECISIONS</small>
      </div>
      <div className="decision-layout">
        <div className="decision-index">
          {ordered.map((item, index) => (
            <button
              key={`${item.player_id}-${item.round_no}-${item.phase}-${index}`}
              className={selected === index ? 'active' : ''}
              onClick={() => setSelected(index)}
            >
              <span>R{item.round_no}</span>
              <div><strong>{item.player_id.replace('player_', '玩家 ')}</strong><small>{formatPhase(item.phase)}</small></div>
              <ChevronRight size={14} />
            </button>
          ))}
        </div>
        <div className="decision-detail">
          <div className="decision-meta">
            <span>{formatRole(trace.role)}</span>
            <span>Selected #{trace.selected_index + 1}</span>
            <span>{trace.action.strategy || trace.action.action_type}</span>
          </div>
          <div className="candidate-grid">
            {trace.candidates.map((candidate, index) => {
              const evaluation = trace.evaluations[index]
              const chosen = index === trace.selected_index
              return (
                <article className={`candidate ${chosen ? 'chosen' : ''}`} key={`${candidate.strategy}-${index}`}>
                  <header>
                    <span>方案 {index + 1}</span>
                    <strong>{evaluation?.score.toFixed(2) ?? '—'}</strong>
                  </header>
                  <h3>{candidate.strategy || candidate.action_type}</h3>
                  <p>{candidate.message || candidate.rationale || candidate.target_id || '保持静默'}</p>
                  <footer>
                    <span className={evaluation?.legal ? 'legal' : 'illegal'}>
                      {evaluation?.legal ? <Check size={12} /> : <X size={12} />}
                      {evaluation?.legal ? '合法候选' : '被规则拒绝'}
                    </span>
                    {chosen && <span className="selected-mark">最终执行</span>}
                  </footer>
                </article>
              )
            })}
          </div>
          {trace.reflection && (
            <div className="reflection-box">
              <RotateCcw size={16} />
              <div><strong>Reflexion 修正</strong><p>{trace.reflection}</p></div>
            </div>
          )}
          <details className="prompt-drawer">
            <summary>查看 Agent 私有观察 Prompt</summary>
            <pre>{trace.observation_prompt}</pre>
          </details>
        </div>
      </div>
    </section>
  )
}

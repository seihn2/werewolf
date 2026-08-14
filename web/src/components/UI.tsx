import { ArrowUpRight, Wifi, WifiOff } from 'lucide-react'
import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'

import type { ConnectionState } from '../api/useRealtime'

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow: string
  title: string
  description: string
  actions?: ReactNode
}) {
  return (
    <header className="page-header reveal">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  )
}

export function SectionHeader({
  title,
  detail,
  link,
}: {
  title: string
  detail?: string
  link?: { to: string; label: string }
}) {
  return (
    <div className="section-header">
      <div>
        <h2>{title}</h2>
        {detail && <p>{detail}</p>}
      </div>
      {link && <Link to={link.to}>{link.label}<ArrowUpRight size={14} /></Link>}
    </div>
  )
}

export function StatusPill({ status }: { status: string }) {
  const labels: Record<string, string> = {
    queued: '排队中',
    running: '进行中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
    interrupted: '已中断',
    cancelling: '正在停止',
  }
  return <span className={`status-pill status-${status}`}><i />{labels[status] ?? status}</span>
}

export function ConnectionBadge({ state }: { state: ConnectionState }) {
  const live = state === 'live'
  return (
    <span className={`connection-badge ${live ? 'live' : ''}`}>
      {live ? <Wifi size={13} /> : <WifiOff size={13} />}
      {state === 'live' ? '实时连接' : state === 'connecting' ? '建立连接' : state === 'reconnecting' ? '正在重连' : '离线'}
    </span>
  )
}

export function ProgressBar({ value }: { value: number }) {
  const clamped = Math.max(0, Math.min(1, value))
  return (
    <div className="progress-track" aria-label={`进度 ${Math.round(clamped * 100)}%`}>
      <span style={{ width: `${clamped * 100}%` }} />
    </div>
  )
}

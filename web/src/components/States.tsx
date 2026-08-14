import { AlertTriangle, Inbox, LoaderCircle, RefreshCw } from 'lucide-react'
import type { ReactNode } from 'react'

export function LoadingScreen({ label = '正在加载' }: { label?: string }) {
  return (
    <div className="loading-screen">
      <div className="moon-loader"><LoaderCircle size={28} /></div>
      <p>{label}</p>
    </div>
  )
}

export function InlineLoading({ label = '载入中' }: { label?: string }) {
  return <span className="inline-loading"><LoaderCircle size={15} /> {label}</span>
}

export function EmptyState({
  title,
  detail,
  action,
}: {
  title: string
  detail: string
  action?: ReactNode
}) {
  return (
    <div className="empty-state">
      <Inbox size={28} strokeWidth={1.4} />
      <strong>{title}</strong>
      <p>{detail}</p>
      {action}
    </div>
  )
}

export function ErrorState({
  error,
  retry,
}: {
  error: unknown
  retry?: () => void
}) {
  return (
    <div className="error-state" role="alert">
      <AlertTriangle size={22} />
      <div>
        <strong>工作区暂时不可用</strong>
        <p>{error instanceof Error ? error.message : '发生了未知错误'}</p>
      </div>
      {retry && <button className="button ghost compact" onClick={retry}><RefreshCw size={14} />重试</button>}
    </div>
  )
}

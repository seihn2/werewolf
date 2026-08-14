import type {
  AgentProfile,
  AnalyticsOverview,
  AnalyticsPoint,
  Artifact,
  Game,
  GameEvent,
  Paginated,
  TrainingJob,
  TrainingKind,
} from '../types'

const API_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '')

export class ApiError extends Error {
  status: number
  code: string
  details: unknown

  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.details = details
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    const error = payload?.error
    throw new ApiError(
      response.status,
      error?.code ?? 'request_failed',
      error?.message ?? `Request failed with status ${response.status}`,
      error?.details,
    )
  }
  if (response.status === 204) {
    return undefined as T
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string; version: string; games: number }>('/api/health'),
  games: (params = '') => request<Paginated<Game>>(`/api/games${params}`),
  game: (id: string) => request<Game>(`/api/games/${id}`),
  gameEvents: (id: string, view: 'public' | 'omniscient' = 'public') =>
    request<GameEvent[]>(`/api/games/${id}/events?view=${view}`),
  createGame: (payload: Record<string, unknown>) =>
    request<Game>('/api/games', { method: 'POST', body: JSON.stringify(payload) }),
  cancelGame: (id: string) =>
    request<Game>(`/api/games/${id}/cancel`, { method: 'POST' }),
  agents: () => request<AgentProfile[]>('/api/agents'),
  createAgent: (payload: Record<string, unknown>) =>
    request<AgentProfile>('/api/agents', { method: 'POST', body: JSON.stringify(payload) }),
  updateAgent: (id: string, payload: Record<string, unknown>) =>
    request<AgentProfile>(`/api/agents/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  deleteAgent: (id: string) => request<void>(`/api/agents/${id}`, { method: 'DELETE' }),
  jobs: (params = '') => request<Paginated<TrainingJob>>(`/api/training/jobs${params}`),
  job: (id: string) => request<TrainingJob>(`/api/training/jobs/${id}`),
  createJob: (kind: TrainingKind, config: Record<string, unknown>) =>
    request<TrainingJob>('/api/training/jobs', {
      method: 'POST',
      body: JSON.stringify({ kind, config }),
    }),
  cancelJob: (id: string) =>
    request<TrainingJob>(`/api/training/jobs/${id}/cancel`, { method: 'POST' }),
  jobLogs: (id: string, offset = 0) =>
    request<{ lines: string[]; next_offset: number; complete: boolean }>(
      `/api/training/jobs/${id}/logs?offset=${offset}`,
    ),
  artifacts: () => request<{ items: Artifact[]; total: number }>('/api/artifacts'),
  analytics: () => request<AnalyticsOverview>('/api/analytics/overview'),
  timeseries: (days = 30) =>
    request<AnalyticsPoint[]>(`/api/analytics/timeseries?days=${days}`),
}

export function websocketUrl(path: string): string {
  const configured = import.meta.env.VITE_WS_BASE as string | undefined
  if (configured) return `${configured.replace(/\/$/, '')}${path}`
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}${path}`
}

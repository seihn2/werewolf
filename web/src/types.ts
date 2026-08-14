export type GameStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'interrupted'

export type JobStatus = GameStatus | 'cancelling'

export interface PlayerState {
  player_id: string
  name: string
  alive: boolean
  role?: 'werewolf' | 'seer' | 'doctor' | 'villager'
}

export interface CandidateAction {
  action_type: string
  strategy: string
  target_id: string | null
  message: string
  rationale: string
}

export interface CandidateEvaluation {
  score: number
  legal: boolean
  reasons: string[]
}

export interface DecisionTrace {
  player_id: string
  role: string
  round_no: number
  phase: string
  observation_prompt: string
  candidates: CandidateAction[]
  evaluations: CandidateEvaluation[]
  selected_index: number
  action: {
    actor_id: string
    action_type: string
    target_id: string | null
    message: string
    strategy: string
  }
  reflection: string
}

export interface GameResult {
  game_id: string
  seed: number
  rounds: number
  winner: string
  termination_reason: string
  players: Record<string, PlayerState>
  events: GameEvent[]
  decision_traces: DecisionTrace[]
}

export interface Game {
  id: string
  seed: number
  max_rounds: number
  status: GameStatus
  winner: string | null
  termination_reason: string | null
  rounds: number | null
  current_round: number
  current_phase: string
  event_count: number
  config: {
    label?: string
    pace_seconds?: number
    werewolf_agent_id?: string
    village_agent_id?: string
    [key: string]: unknown
  }
  players: Record<string, PlayerState>
  result?: GameResult | null
  error: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
}

export interface GameEvent {
  logical_time: number
  topic: string
  round_no: number
  phase: string
  payload: Record<string, any>
  sender: string | null
  audience: string[] | null
  is_public: boolean
  created_at?: string
}

export interface AgentProfile {
  id: string
  name: string
  kind: 'heuristic' | 'openai_compatible'
  model: string | null
  base_url: string | null
  env_prefix: string | null
  temperature: number
  timeout_seconds: number
  enabled: boolean
  builtin: boolean
  created_at: string
  updated_at: string
}

export type TrainingKind =
  | 'self_play'
  | 'latent'
  | 'deep_cfr'
  | 'cfr_dpo'
  | 'dpo'
  | 'iterative'

export interface TrainingJob {
  id: string
  kind: TrainingKind
  status: JobStatus
  stage: string
  progress: number
  config: Record<string, any>
  command: string[] | null
  metrics: Record<string, any> | null
  output_path: string | null
  log_path: string | null
  pid: number | null
  exit_code: number | null
  error: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
}

export interface Artifact {
  path: string
  name: string
  extension: string
  size: number
  modified_at: string
  category:
    | 'self_play'
    | 'preference_data'
    | 'latent_space'
    | 'checkpoint'
    | 'manifest'
    | 'log'
    | 'other'
}

export interface AnalyticsOverview {
  completed_games: number
  active_games: number
  active_jobs: number
  average_rounds: number
  winner_counts: Record<string, number>
  termination_reasons: Record<string, number>
  reflection_rate: number
  illegal_evaluation_rate: number
  role_stats: Array<{
    role: string
    appearances: number
    survivals: number
    survival_rate: number
  }>
  strategy_distribution: Array<{ strategy: string; count: number }>
}

export interface AnalyticsPoint {
  date: string
  games: number
  werewolf: number
  village: number
  draw: number
  average_rounds: number
}

export interface Paginated<T> {
  items: T[]
  total: number
  limit: number
  offset: number
}

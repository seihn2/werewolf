const phaseNames: Record<string, string> = {
  setup: '角色入场',
  night_werewolf: '狼人行动',
  night_seer: '预言家查验',
  night_doctor: '医生守护',
  night_resolution: '夜晚结算',
  day_announcement: '天亮公告',
  day_discussion: '白天发言',
  day_vote: '放逐投票',
  vote_resolution: '投票结算',
  game_over: '终局',
}

const roleNames: Record<string, string> = {
  werewolf: '狼人',
  seer: '预言家',
  doctor: '医生',
  villager: '村民',
}

const winnerNames: Record<string, string> = {
  werewolf: '狼人阵营',
  village: '村庄阵营',
  draw: '平局',
}

export const formatPhase = (phase?: string | null) =>
  (phase && phaseNames[phase]) || phase || '等待开始'

export const formatRole = (role?: string | null) => (role && roleNames[role]) || '未知身份'

export const formatWinner = (winner?: string | null) =>
  (winner && winnerNames[winner]) || '尚未决出'

export function formatDate(value?: string | null, withTime = true) {
  if (!value) return '—'
  const date = new Date(value)
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    ...(withTime ? { hour: '2-digit', minute: '2-digit', second: '2-digit' } : {}),
  }).format(date)
}

export function formatDuration(start?: string | null, end?: string | null) {
  if (!start) return '—'
  const milliseconds = (end ? new Date(end) : new Date()).getTime() - new Date(start).getTime()
  const seconds = Math.max(0, Math.floor(milliseconds / 1000))
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  return `${minutes}m ${seconds % 60}s`
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`
}

export const percent = (value: number) => `${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%`

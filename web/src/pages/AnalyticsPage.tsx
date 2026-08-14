import { useQuery } from '@tanstack/react-query'
import { Activity, BrainCircuit, CircleGauge, ShieldCheck, Swords } from 'lucide-react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { api } from '../api/client'
import { EmptyState, ErrorState, InlineLoading } from '../components/States'
import { PageHeader, SectionHeader } from '../components/UI'
import { formatRole, percent } from '../lib/format'

const COLORS = { werewolf: '#c95c5c', village: '#51b6a4', draw: '#b6aa91' }

export default function AnalyticsPage() {
  const overview = useQuery({ queryKey: ['analytics'], queryFn: api.analytics })
  const timeseries = useQuery({ queryKey: ['analytics-timeseries'], queryFn: () => api.timeseries(60) })
  if (overview.isError || timeseries.isError) return <div className="page-pad"><ErrorState error={overview.error ?? timeseries.error} /></div>
  if (!overview.data) return <div className="page-pad"><InlineLoading label="计算博弈指标" /></div>
  const data = overview.data
  const winnerData = Object.entries(data.winner_counts).map(([name, value]) => ({ name, value }))
  const roleData = data.role_stats.map((item) => ({ ...item, label: formatRole(item.role), survival: Math.round(item.survival_rate * 100) }))

  return (
    <div className="page-pad analytics-page">
      <PageHeader eyebrow="STRATEGIC INTELLIGENCE" title="不只看胜率，也看策略如何发生" description="从阵营结果下钻到角色存活、认知修正、非法候选和潜在策略使用。所有指标来自已持久化对局。" />
      <section className="analytics-kpis reveal delay-1">
        <Kpi icon={Activity} label="完成样本" value={String(data.completed_games)} detail={`${data.active_games} 局运行中`} />
        <Kpi icon={CircleGauge} label="平均轮数" value={data.average_rounds.toFixed(2)} detail="每局 daylight cycles" />
        <Kpi icon={BrainCircuit} label="Reflexion 触发" value={percent(data.reflection_rate)} detail="决策层自我修正" />
        <Kpi icon={ShieldCheck} label="非法候选率" value={percent(data.illegal_evaluation_rate)} detail="Evaluator 拦截比例" />
      </section>
      {data.completed_games === 0 ? <EmptyState title="还没有可分析样本" detail="完成至少一局后，这里会生成阵营、角色和策略维度的真实统计。" /> : (
        <>
          <div className="analytics-grid reveal delay-2">
            <section className="panel chart-panel wide-chart">
              <SectionHeader title="对局趋势" detail="按完成日期统计阵营胜局与样本量" />
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={timeseries.data ?? []} margin={{ top: 12, right: 10, left: -20, bottom: 0 }}>
                    <defs><linearGradient id="wolfArea" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={COLORS.werewolf} stopOpacity={0.5} /><stop offset="1" stopColor={COLORS.werewolf} stopOpacity={0} /></linearGradient><linearGradient id="villageArea" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={COLORS.village} stopOpacity={0.45} /><stop offset="1" stopColor={COLORS.village} stopOpacity={0} /></linearGradient></defs>
                    <CartesianGrid stroke="#24303a" strokeDasharray="3 6" vertical={false} />
                    <XAxis dataKey="date" stroke="#79848d" tickLine={false} axisLine={false} tickFormatter={(value) => value.slice(5)} />
                    <YAxis stroke="#79848d" tickLine={false} axisLine={false} allowDecimals={false} />
                    <Tooltip contentStyle={{ background: '#11171d', border: '1px solid #2b3740', borderRadius: 4 }} />
                    <Area type="monotone" dataKey="werewolf" name="狼人胜局" stroke={COLORS.werewolf} fill="url(#wolfArea)" strokeWidth={2} />
                    <Area type="monotone" dataKey="village" name="村庄胜局" stroke={COLORS.village} fill="url(#villageArea)" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </section>
            <section className="panel chart-panel verdict-chart">
              <SectionHeader title="阵营判决" detail={`${data.completed_games} 局完整样本`} />
              <div className="donut-wrap">
                <ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={winnerData} dataKey="value" nameKey="name" innerRadius="62%" outerRadius="86%" paddingAngle={4} stroke="none">{winnerData.map((item) => <Cell key={item.name} fill={COLORS[item.name as keyof typeof COLORS] ?? '#8b96a0'} />)}</Pie><Tooltip contentStyle={{ background: '#11171d', border: '1px solid #2b3740', borderRadius: 4 }} /></PieChart></ResponsiveContainer>
                <div><strong>{data.completed_games}</strong><span>GAMES</span></div>
              </div>
              <div className="legend-stack">{winnerData.map((item) => <div key={item.name}><i style={{ background: COLORS[item.name as keyof typeof COLORS] }} /><span>{item.name === 'werewolf' ? '狼人阵营' : item.name === 'village' ? '村庄阵营' : '平局'}</span><strong>{item.value}</strong></div>)}</div>
            </section>
          </div>
          <div className="analytics-grid lower reveal delay-3">
            <section className="panel chart-panel">
              <SectionHeader title="角色存活率" detail="终局仍存活的角色席位比例" />
              <div className="chart-wrap small"><ResponsiveContainer width="100%" height="100%"><BarChart data={roleData} layout="vertical" margin={{ top: 0, right: 20, left: 8, bottom: 0 }}><CartesianGrid stroke="#24303a" strokeDasharray="3 6" horizontal={false} /><XAxis type="number" domain={[0, 100]} stroke="#79848d" tickLine={false} axisLine={false} unit="%" /><YAxis type="category" dataKey="label" width={58} stroke="#aab2b9" tickLine={false} axisLine={false} /><Tooltip contentStyle={{ background: '#11171d', border: '1px solid #2b3740' }} /><Bar dataKey="survival" name="存活率" fill="#d2b46f" radius={[0, 3, 3, 0]} /></BarChart></ResponsiveContainer></div>
            </section>
            <section className="panel strategy-panel">
              <SectionHeader title="策略使用排行" detail="最终执行动作中的 strategy 标签" />
              <div className="strategy-bars">{data.strategy_distribution.map((item, index) => { const max = data.strategy_distribution[0]?.count || 1; return <div key={item.strategy}><span>{String(index + 1).padStart(2, '0')}</span><div><strong>{item.strategy}</strong><i><b style={{ width: `${(item.count / max) * 100}%` }} /></i></div><em>{item.count}</em></div> })}</div>
            </section>
            <section className="panel reason-panel">
              <SectionHeader title="终局原因" detail="规则引擎判定来源" />
              <div className="reason-list">{Object.entries(data.termination_reasons).map(([reason, count]) => <div key={reason}><span>{reason.replaceAll('_', ' ')}</span><strong>{count}</strong></div>)}</div>
              <div className="faction-summary"><div className="wolf"><Swords size={17} /><span>狼人胜率</span><strong>{percent((data.winner_counts.werewolf ?? 0) / data.completed_games)}</strong></div><div className="village"><ShieldCheck size={17} /><span>村庄胜率</span><strong>{percent((data.winner_counts.village ?? 0) / data.completed_games)}</strong></div></div>
            </section>
          </div>
        </>
      )}
    </div>
  )
}

function Kpi({ icon: Icon, label, value, detail }: { icon: typeof Activity; label: string; value: string; detail: string }) {
  return <div className="analytics-kpi"><Icon size={18} /><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>
}

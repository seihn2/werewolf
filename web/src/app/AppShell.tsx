import {
  Activity,
  Bot,
  ChartNoAxesCombined,
  FlaskConical,
  History,
  MoonStar,
  Radio,
} from 'lucide-react'
import { useEffect } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'

import { api } from '../api/client'

const navItems = [
  { to: '/', label: '总览', icon: Activity, exact: true },
  { to: '/arena', label: '竞技场', icon: Radio },
  { to: '/replays', label: '回放', icon: History },
  { to: '/agents', label: 'Agent', icon: Bot },
  { to: '/training', label: '训练', icon: FlaskConical },
  { to: '/analytics', label: '分析', icon: ChartNoAxesCombined },
]

export function AppShell() {
  const location = useLocation()
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    refetchInterval: 30_000,
  })
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' })
  }, [location.pathname])

  return (
    <div className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <aside className="sidebar">
        <NavLink className="brand" to="/" aria-label="WolfPlay Studio 首页">
          <span className="brand-mark"><MoonStar size={24} strokeWidth={1.5} /></span>
          <span>
            <strong>WolfPlay</strong>
            <small>STRATEGIC AGENT STUDIO</small>
          </span>
        </NavLink>
        <nav className="primary-nav" aria-label="主导航">
          {navItems.map(({ to, label, icon: Icon, exact }) => (
            <NavLink
              key={to}
              to={to}
              end={exact}
              className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
            >
              <Icon size={18} strokeWidth={1.6} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className={`system-state ${health ? 'online' : 'offline'}`}>
            <i />
            <span>{health ? 'Studio 在线' : '正在连接服务'}</span>
          </div>
          <small>v{health?.version ?? '—'} · Local First</small>
        </div>
      </aside>

      <main className="main-stage" key={location.pathname}>
        <Outlet />
      </main>

      <nav className="mobile-nav" aria-label="移动端导航">
        {navItems.map(({ to, label, icon: Icon, exact }) => (
          <NavLink key={to} to={to} end={exact}>
            <Icon size={18} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
    </div>
  )
}

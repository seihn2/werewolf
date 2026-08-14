import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'

import { AppShell } from './AppShell'
import { LoadingScreen } from '../components/States'

const DashboardPage = lazy(() => import('../pages/DashboardPage'))
const ArenaPage = lazy(() => import('../pages/ArenaPage'))
const ReplaysPage = lazy(() => import('../pages/ReplaysPage'))
const AgentsPage = lazy(() => import('../pages/AgentsPage'))
const TrainingPage = lazy(() => import('../pages/TrainingPage'))
const AnalyticsPage = lazy(() => import('../pages/AnalyticsPage'))
const NotFoundPage = lazy(() => import('../pages/NotFoundPage'))

export function AppRouter() {
  return (
    <Suspense fallback={<LoadingScreen label="正在点亮月夜剧场" />}>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<DashboardPage />} />
          <Route path="arena" element={<ArenaPage />} />
          <Route path="arena/:gameId" element={<ArenaPage />} />
          <Route path="replays" element={<ReplaysPage />} />
          <Route path="replays/:gameId" element={<ReplaysPage />} />
          <Route path="agents" element={<AgentsPage />} />
          <Route path="training" element={<TrainingPage />} />
          <Route path="training/:jobId" element={<TrainingPage />} />
          <Route path="analytics" element={<AnalyticsPage />} />
          <Route path="home" element={<Navigate to="/" replace />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </Suspense>
  )
}

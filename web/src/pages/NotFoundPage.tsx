import { ArrowLeft } from 'lucide-react'
import { Link } from 'react-router-dom'

export default function NotFoundPage() {
  return (
    <div className="not-found page-pad">
      <span>404</span>
      <h1>这条时间线不存在</h1>
      <p>可能是对局已经迁移，或你进入了未开放的夜晚视角。</p>
      <Link className="button primary" to="/"><ArrowLeft size={16} />返回 Studio</Link>
    </div>
  )
}

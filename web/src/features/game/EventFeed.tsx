import { Eye, MessageCircle, Moon, PawPrint, Shield, Sparkles, Vote } from 'lucide-react'
import { useEffect, useRef } from 'react'

import type { GameEvent } from '../../types'
import { eventDetail, eventTitle, eventTone } from './gameState'

const topicIcons: Record<string, typeof Moon> = {
  game_started: Sparkles,
  role_assignment: Eye,
  werewolf_team: PawPrint,
  werewolf_proposal: PawPrint,
  seer_result: Eye,
  doctor_choice: Shield,
  night_result: Moon,
  day_started: Sparkles,
  speech: MessageCircle,
  vote_cast: Vote,
  vote_result: Vote,
  game_over: Sparkles,
}

export function EventFeed({
  events,
  autoFollow = true,
  title = '实时事件流',
}: {
  events: GameEvent[]
  autoFollow?: boolean
  title?: string
}) {
  const listRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (autoFollow && listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight
  }, [events.length, autoFollow])

  return (
    <section className="event-panel panel">
      <div className="panel-title">
        <div><span className="signal-dot" /><h2>{title}</h2></div>
        <small>{events.length} EVENTS</small>
      </div>
      <div className="event-list" ref={listRef}>
        {events.length === 0 && <p className="quiet-copy">等待第一条公开事件…</p>}
        {events.map((event) => {
          const Icon = topicIcons[event.topic] ?? Sparkles
          const detail = eventDetail(event)
          return (
            <article className={`event-row tone-${eventTone(event)}`} key={event.logical_time}>
              <div className="event-time">{String(event.logical_time).padStart(3, '0')}</div>
              <div className="event-icon"><Icon size={15} /></div>
              <div className="event-copy">
                <strong>{eventTitle(event)}</strong>
                {detail && <p>{detail}</p>}
                <small>R{event.round_no} · {event.phase.replaceAll('_', ' ')}</small>
              </div>
            </article>
          )
        })}
      </div>
    </section>
  )
}

import { useEffect, useRef, useState } from 'react'

import { websocketUrl } from './client'

export type ConnectionState = 'connecting' | 'live' | 'reconnecting' | 'offline'

export function useRealtime<T>(path: string | null, onMessage: (message: T) => void) {
  const callbackRef = useRef(onMessage)
  const [connection, setConnection] = useState<{ path: string; state: ConnectionState } | null>(null)

  useEffect(() => {
    callbackRef.current = onMessage
  }, [onMessage])

  useEffect(() => {
    if (!path) return

    let socket: WebSocket | null = null
    let disposed = false
    let reconnectTimer: number | undefined
    let attempts = 0

    const connect = () => {
      if (disposed) return
      socket = new WebSocket(websocketUrl(path))
      socket.addEventListener('open', () => {
        attempts = 0
        setConnection({ path, state: 'live' })
      })
      socket.addEventListener('message', (event) => {
        try {
          callbackRef.current(JSON.parse(event.data) as T)
        } catch {
          // Ignore malformed transport frames; the REST snapshot remains authoritative.
        }
      })
      socket.addEventListener('close', () => {
        if (disposed) return
        attempts += 1
        setConnection({ path, state: 'reconnecting' })
        reconnectTimer = window.setTimeout(connect, Math.min(8_000, 500 * 2 ** attempts))
      })
      socket.addEventListener('error', () => socket?.close())
    }

    connect()
    return () => {
      disposed = true
      if (reconnectTimer) window.clearTimeout(reconnectTimer)
      socket?.close()
    }
  }, [path])

  if (!path) return 'offline'
  return connection?.path === path ? connection.state : 'connecting'
}

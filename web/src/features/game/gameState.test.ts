import { describe, expect, it } from 'vitest'

import type { Game, GameEvent } from '../../types'
import { activeVotes, derivePlayers, mergeEvents } from './gameState'

const event = (logical_time: number, topic: string, payload = {}, sender: string | null = null): GameEvent => ({
  logical_time,
  topic,
  round_no: 1,
  phase: 'day_vote',
  payload,
  sender,
  audience: null,
  is_public: true,
})

describe('game state projection', () => {
  it('deduplicates realtime events by logical time', () => {
    const merged = mergeEvents([event(1, 'game_started')], [event(1, 'game_started'), event(2, 'speech')])
    expect(merged.map((item) => item.logical_time)).toEqual([1, 2])
  })

  it('projects eliminations onto public players', () => {
    const game = {
      players: {
        player_0: { player_id: 'player_0', name: 'Player 0', alive: true },
      },
    } as unknown as Game
    const players = derivePlayers(game, [event(3, 'vote_result', { eliminated_id: 'player_0' })])
    expect(players.player_0?.alive).toBe(false)
  })

  it('keeps only votes after the latest vote result', () => {
    const votes = activeVotes([
      event(1, 'vote_cast', { target_id: 'player_1' }, 'player_0'),
      event(2, 'vote_result', { eliminated_id: 'player_1' }),
      event(3, 'vote_cast', { target_id: 'player_3' }, 'player_2'),
    ])
    expect(votes).toEqual([{ from: 'player_2', to: 'player_3' }])
  })
})

import { useCallback, useEffect, useRef, useState } from 'react'
import { advance, RACE_SECONDS } from './motion.js'

export const reducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

// Race clock. progress() advances and returns p in [0, 1]; the render loop calls it every frame (no React
// re-render per frame). The UI state (playing, progress) is synced 20x a second while playing (timing tower,
// start lights); pausing stops the clock, so every view freezes on the same race state. One speed only.
// A new result starts the race from the grid (or jumps to the finish with reduced motion).
export function usePlayback(result, seconds = RACE_SECONDS) {
  const c = useRef({ p: 0, playing: false, last: 0 })
  const [ui, setUi] = useState({ p: 0, playing: false })
  const sync = useCallback(() => setUi({ p: c.current.p, playing: c.current.playing }), [])

  useEffect(() => {
    const calm = reducedMotion()
    Object.assign(c.current, { p: result && calm ? 1 : 0, playing: !!result && !calm, last: performance.now() })
    sync()
  }, [result, sync])

  useEffect(() => {
    if (!ui.playing) return
    const id = setInterval(sync, 50)
    return () => clearInterval(id)
  }, [ui.playing, sync])

  const progress = useCallback(() => {
    const s = c.current, now = performance.now()
    if (s.playing) {
      s.p = advance(s.p, (now - s.last) / 1000, 1, seconds)
      if (s.p >= 1) { s.playing = false; sync() }
    }
    s.last = now
    return s.p
  }, [seconds, sync])

  const act = {
    toggle: () => { const s = c.current; if (s.p >= 1) s.p = 0; s.playing = !s.playing; s.last = performance.now(); sync() },
    restart: () => { Object.assign(c.current, { p: 0, playing: true, last: performance.now() }); sync() },
    skip: () => { Object.assign(c.current, { p: 1, playing: false }); sync() },   // straight to the predicted result
    seek: p => { c.current.p = Math.min(1, Math.max(0, p)); sync() },
  }
  return { ui, progress, act, isPlaying: () => c.current.playing }
}

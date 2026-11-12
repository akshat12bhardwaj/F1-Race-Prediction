import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { api } from './api.js'
import Logo from './components/Logo.jsx'
import LineupEditor, { TeamMark } from './components/LineupEditor.jsx'
import CircuitPicker from './components/CircuitPicker.jsx'
import Conditions, { stateOf } from './components/Conditions.jsx'
import RaceBoard from './components/RaceBoard.jsx'
import { MODEL_TAG, teamAbbr, teamColor } from './teams.js'
import { lightsAt, LIGHTS_OUT, orderAt } from './motion.js'
import { carKey, GRID_SIZE, gridProblems, history, payloadEntries, sameLineup } from './lineup.js'
import { usePlayback } from './useRaceMotion.js'
const Track3D = lazy(() => import('./components/Track3D.jsx'))   // three.js loads only when needed

const pct = p => `${Math.round(100 * p)}%`

export default function App() {
  const [health, setHealth] = useState(null)
  const [circuits, setCircuits] = useState([])
  const [pool, setPool] = useState({ drivers: [], cars: [], teams: [] })   // eligible drivers + one peak car per constructor (/api/driver-pool)
  const [models, setModels] = useState([])
  const [circuitId, setCircuitId] = useState('spa')
  const [hist, setHist] = useState(history.start([]))   // lineup with undo / redo; hist.now is the grid, first = pole
  const [base, setBase] = useState([])                   // default 22-driver lineup in peak cars
  const [editing, setEditing] = useState(false)
  const [condSpec, setCondSpec] = useState(null)       // /api/conditions: ranges, presets, adjustable inputs
  const [cond, setCond] = useState({ rain_mm_h: 0, air_temp_c: 21.1, humidity_pct: 60.7, wind_kmh: 10.8 })
  const [cls, setCls] = useState('random_forest')
  const [reg, setReg] = useState('ridge')
  const [result, setResult] = useState(null)
  const [resultKey, setResultKey] = useState(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const [geo, setGeo] = useState(null)           // verified circuit centrelines + elevation (public/circuits.json)
  const [selected, setSelected] = useState(null)
  const [focus, setFocus] = useState({ n: 0 })

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: 'down' }))
    Promise.all([api.circuits(), api.pool(), api.models(), api.conditions()])
      .then(([c, p, m, w]) => { setCircuits(c); setPool(p); setModels(m); setBase(p.default_grid); setHist(history.start(p.default_grid)); setCondSpec(w); setCond({ ...w.presets.dry }) })
      .catch(e => setError(e.message))
    fetch('/circuits.json').then(r => r.json()).then(j => setGeo(j.circuits)).catch(() => setGeo({}))
  }, [])

  const entries = hist.now
  const setEntries = next => setHist(h => history.push(h, next))
  const drivers = pool.drivers
  const limit = GRID_SIZE                                         // fixed format: 22 drivers, 11 constructors x 2 (D-036)
  const byId = useMemo(() => Object.fromEntries(drivers.map(d => [d.driver_id, d])), [drivers])
  const carOf = useMemo(() => Object.fromEntries(pool.teams.map(c => [carKey(c.constructor_id, c.season), c])), [pool.teams])
  const names = (id, short) => {
    const d = byId[id]
    if (!d) return id
    return short ? (d.code || d.last_name.slice(0, 3)).toUpperCase() : `${d.first_name} ${d.last_name}`
  }
  const clsModels = models.filter(m => m.family === 'classification')
  const regModels = models.filter(m => m.family === 'regression')
  // a prediction belongs to exactly one scenario: any change to circuit, weather, models, drivers, cars or grid order makes it stale
  const weather = stateOf(cond, condSpec?.wet_mm_h ?? 0.25)
  const key = JSON.stringify([circuitId, cond, cls, reg, entries.map(e => [e.driver_id, e.season, e.constructor_id, e.constructor_season])])
  const stale = !!result && key !== resultKey
  const shown = stale ? null : result
  const todo = gridProblems(entries, pool.teams)
  const problem = todo.length ? `${todo.join(' ')} Fix it in Edit Lineup.` : null

  const g = geo?.[circuitId]
  const pb = usePlayback(shown)
  const slots = useMemo(() => {
    const fin = shown && Object.fromEntries(shown.standings.map(r => [r.driver_id, r.position - 1]))
    return Object.fromEntries(entries.map((e, i) => [e.driver_id, [i, fin?.[e.driver_id] ?? i]]))
  }, [entries, shown])
  const phase = !shown ? 'idle' : pb.ui.p >= 1 ? 'done' : 'racing'
  const order = phase === 'racing' ? orderAt(slots, pb.ui.p) : null   // illustrative running order (same function as the cars)
  const lit = phase === 'racing' && pb.ui.p < LIGHTS_OUT + 0.06 ? lightsAt(pb.ui.p) : null

  async function predict() {
    setError(null); setRunning(true)
    const k = key, t0 = performance.now()
    try {
      const r = await api.predict({ circuit_id: circuitId, weather, ...cond, classification: cls, regression: reg, n_sims: 5000, seed: 42, entries: payloadEntries(entries) })
      setResult({ ...r, round_trip_ms: Math.round(performance.now() - t0) }); setResultKey(k)
    } catch (e) { setError(e.message) }
    setRunning(false)
  }
  const pick = id => { setSelected(id); setFocus(f => ({ id, n: f.n + 1 })) }   // from the timing tower / standings: follow that car
  const toggle = () => { if (shown) pb.act.toggle() }

  const row = phase === 'done' ? shown.standings.find(r => r.driver_id === selected) : null
  const sel = entries.find(e => e.driver_id === selected)
  const viewer = geo === null ? <div className="track-loading">Loading circuit…</div>
    : g
      ? <Suspense fallback={<div className="track-loading">Loading 3D circuit…</div>}>
          <Track3D geometry={g} circuitId={circuitId} conditions={{ ...cond, weather }} paused={phase === 'racing' && !pb.ui.playing} entries={entries} result={shown} names={names} selected={selected} onSelect={setSelected} onToggle={toggle}
            focus={focus} progress={pb.progress} isPlaying={pb.isPlaying} />
        </Suspense>
      : <div className="track-loading" role="alert">Circuit geometry is unavailable.</div>
  const status = running ? 'Predicting…' : phase === 'racing' ? (pb.ui.playing ? 'Race in progress' : 'Paused') : phase === 'done' ? `Predicted winner: ${names(shown.podium[0], true)}`
    : stale ? 'Prediction out of date' : health?.status === 'ok' ? 'Ready' : health ? 'Server offline' : 'Connecting…'
  const carTag = e => { const c = carOf[carKey(e.constructor_id, e.constructor_season)]; return c ? `${teamAbbr(c)}${e.constructor_season !== 2025 ? ` ’${String(e.constructor_season).slice(2)}` : ''}` : '' }

  return (
    <div className="app">
      <header className="topbar">
        <Logo />
        <CircuitPicker circuits={circuits} value={circuitId} onChange={setCircuitId} />
        <div className="top-meta">
          <span className="chip" title="Drivers and cars from any eligible F1 season, 1981-2025">F1 data 1981–2025</span>
          <span className="chip" data-count>{entries.length}/{limit} cars</span>
          <span className="chip" data-conditions>{weather} · {Math.round(cond.air_temp_c)} °C</span>
          <span className={`status ${health?.status === 'ok' ? 'ok' : 'bad'}`}><span className="dot" />{status}</span>
          <a className="credits-link" href="/credits.html" target="_blank" rel="noreferrer">Credits</a>
        </div>
      </header>

      {error && <div className="alert" role="alert"><b>Prediction not possible.</b> {error}<button className="btn ghost sm" onClick={() => setError(null)}>Dismiss</button></div>}

      <main className="deck" inert={editing ? '' : undefined}>
        <aside className="side side-left" aria-label="Race setup">
          <Conditions spec={condSpec} value={cond} onChange={setCond} models={[models.find(m => m.key === cls), models.find(m => m.key === reg)]} />

          <section className="panel" aria-labelledby="models-h">
            <h2 id="models-h" className="side-h">Models</h2>
            <label className="field"><span>Classification · podium</span>
              <select value={cls} onChange={e => setCls(e.target.value)} aria-label="Classification model" title={MODEL_TAG[cls]}>
                {clsModels.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}
              </select>
            </label>
            <label className="field"><span>Regression · finishing position</span>
              <select value={reg} onChange={e => setReg(e.target.value)} aria-label="Regression model" title={MODEL_TAG[reg]}>
                {regModels.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}
              </select>
            </label>
          </section>

          <section className="panel grow" aria-labelledby="grid-h">
            <h2 id="grid-h" className="side-h">Starting grid <span className="count">{entries.length}/{limit}</span>
              {!!base.length && !sameLineup(entries, base) && <span className="tag swap">edited</span>}</h2>
            <ol className="grid-sum" aria-label="Starting grid summary">
              {entries.map((e, i) => (
                <li key={e.driver_id} data-driver={e.driver_id} data-team={e.constructor_id} data-car-season={e.constructor_season} className={selected === e.driver_id ? 'sel' : ''} style={{ '--team': teamColor(e.constructor_id) }}>
                  <button onClick={() => setSelected(e.driver_id)} title={`P${i + 1} · ${names(e.driver_id)} (${e.season}) · ${carOf[carKey(e.constructor_id, e.constructor_season)]?.constructor_name ?? e.constructor_id} ${e.constructor_season}`}>
                    <span className="gs-pos">{i + 1}</span><i className="stripe" /><b>{names(e.driver_id, true)}</b><small>{carTag(e)}</small>
                  </button>
                </li>
              ))}
            </ol>
            <button className="btn ghost edit-lineup" onClick={() => setEditing(true)} disabled={!drivers.length}>✎ Edit Lineup</button>
          </section>

          <div className="run">
            <button className="btn lime big" onClick={predict} disabled={running || !!problem || !drivers.length || !entries.length}>
              {running ? 'Predicting…' : result ? 'Predict Again' : 'Predict Race'}
            </button>
            {problem && <p className="warn">{problem}</p>}
          </div>
        </aside>

        <section className="stage" aria-label="Race view">
          {viewer}
          {running && <div className="predicting" role="status">Models predicting…</div>}
          {lit !== null && (
            <div className="lights" role="img" aria-label={lit ? `Start lights: ${lit} of 5 on` : 'Lights out'} data-lit={lit}>
              {[0, 1, 2, 3, 4].map(i => <i key={i} className={i < lit ? 'on' : ''} />)}
            </div>
          )}
          {phase === 'racing' && !pb.ui.playing && <div className="paused-chip" aria-hidden="true">❚❚ Paused · click the track to resume</div>}
          {sel && (
            <div className="driver-card" style={{ '--team': teamColor(sel.constructor_id) }} aria-live="polite">
              <span className="stripe" />
              {carOf[carKey(sel.constructor_id, sel.constructor_season)] && <TeamMark car={carOf[carKey(sel.constructor_id, sel.constructor_season)]} />}
              <div>
                <b>{names(sel.driver_id)}{byId[sel.driver_id]?.number && byId[sel.driver_id].season === sel.season ? <span className="num25" title={`${sel.season} race number`}>{byId[sel.driver_id].number}</span> : null}</b>
                <small>{sel.season} form · {carOf[carKey(sel.constructor_id, sel.constructor_season)]?.constructor_name ?? sel.constructor_id} {sel.constructor_season} · grid P{entries.indexOf(sel) + 1}{row ? ` → predicted P${row.position}` : ''}</small>
                {row && <small className="dc-probs">Win {pct(row.win_prob)} · podium {pct(row.podium_prob)}</small>}
              </div>
              <button className="btn ghost sm" onClick={() => setSelected(null)} aria-label="Clear selection">✕</button>
            </div>
          )}
          <div className="playback" aria-label="Race playback">
            <button className="btn ghost sm" disabled={!shown} onClick={pb.act.toggle} aria-label={pb.ui.playing ? 'Pause' : phase === 'done' ? 'Replay' : 'Resume'}>{pb.ui.playing ? '❚❚' : '▶'}</button>
            <button className="btn ghost sm" disabled={!shown} onClick={pb.act.restart} aria-label="Restart">↺</button>
            <button className="btn ghost sm skip" disabled={phase !== 'racing'} onClick={pb.act.skip}>Skip to Results</button>
            <input type="range" min="0" max="1000" value={Math.round(pb.ui.p * 1000)} disabled={!shown} aria-label="Race position"
              onChange={e => pb.act.seek(e.target.value / 1000)} style={{ '--p': `${pb.ui.p * 100}%` }} />
            <span className="lap">{shown ? (phase === 'done' ? 'Finish' : pb.ui.p < LIGHTS_OUT ? 'Start' : 'Racing') : running ? 'Predicting…' : 'On the grid'}</span>
            <span className="replay-note" title="The finishing order is the prediction of the two selected models. The overtakes on the way there are an illustration, not predicted events.">
              Finishing order = model prediction · overtakes illustrative</span>
          </div>
        </section>

        <aside className="side side-right" aria-label="Prediction">
          <RaceBoard result={shown} stale={stale} running={running} phase={phase} order={order} started={pb.ui.p >= LIGHTS_OUT} selected={selected} onPick={pick} onRerun={predict} onSkip={pb.act.skip} />
        </aside>
      </main>

      {editing && <LineupEditor entries={entries} setEntries={setEntries} pool={drivers} cars={pool.teams} limit={limit}
        canUndo={!!hist.past.length} canRedo={!!hist.future.length} onUndo={() => setHist(history.undo)} onRedo={() => setHist(history.redo)}
        onReset={() => setEntries(base)} isDefault={sameLineup(entries, base)} onDone={() => setEditing(false)} />}
    </div>
  )
}

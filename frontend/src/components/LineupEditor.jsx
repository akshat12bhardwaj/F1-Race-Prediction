import { useEffect, useMemo, useState } from 'react'
import { DndContext, DragOverlay, KeyboardSensor, PointerSensor, TouchSensor, pointerWithin, rectIntersection, useDraggable, useDroppable, useSensor, useSensors } from '@dnd-kit/core'
import { teamAbbr, teamColor } from '../teams.js'
import { assignCar, byName, carKey, carLabel, gridProblems, MAX_PER_CAR, MIN_CARS, pickLogo, placeDriver, removeDriver, searchText } from '../lineup.js'
import portraits from '../portraits.json'
import logos from '../logos.json'

const full = d => `${d.first_name} ${d.last_name}`

// Driver photo (freely licensed; credits page) or a neutral monogram. Never a stand-in face.
export function Portrait({ d }) {
  const [bad, setBad] = useState(false)
  const c = portraits[d.driver_id]
  if (c && !bad) return <img className="portrait" src={c.path} alt="" loading="lazy" draggable="false" onError={() => setBad(true)} />
  return <span className="portrait mono" aria-hidden="true">{(d.first_name[0] + d.last_name[0]).toUpperCase()}</span>
}

// The logo variant whose seasons cover this car (reviewed era ranges, see logos.json / credits); none -> badge.
export const logoFor = car => pickLogo(logos[car.constructor_id], car.season)

// Small side-view single-seater in the team colour: an illustration of "a car", not that season's real car.
export function CarArt({ color }) {
  return (
    <svg viewBox="0 0 120 34" className="car-art" aria-hidden="true">
      <path d="M3 9h13v4H9v9l31-2 11-10h15l7 8 27 3 13 3 4 4v2H3z" fill={color} />
      <path d="M55 11q7-8 14 0" fill="none" stroke="#0b0c0d" strokeWidth="2.2" />
      <path d="M99 27h20v2H99z" fill="#0b0c0d" />
      {[27, 93].map(x => <g key={x}><circle cx={x} cy="25.5" r="8" fill="#0b0c0d" /><circle cx={x} cy="25.5" r="3" fill="#3a3b3f" /></g>)}
    </svg>
  )
}

// Constructor logo (freely licensed, on a light tile so dark and light marks both read) or the app's own badge.
// A logo with a period sponsor (e.g. Atlassian Williams, 2025) is shown only on cars of those seasons.
// `car.season` is the car's season (never the driver's), so a cross-era pairing shows the assigned car's era logo.
// `small`: compact rows; without a verified logo they keep the plain team-colour dot instead of a badge.
export function TeamMark({ car, small }) {
  const [bad, setBad] = useState(false)
  const l = logoFor(car)
  const src = l?.path
  useEffect(() => setBad(false), [src])
  if (l && !bad) return <span className={`logo-tile${small ? ' sm' : ''}`} data-logo={car.constructor_id} data-logo-file={l.path}><img src={l.path} alt={`${car.constructor_name} logo`} loading="lazy" draggable="false" onError={() => setBad(true)} /></span>
  if (small) return <i className="dot" />
  return <span className="badge" aria-hidden="true">{teamAbbr(car)}</span>
}

function PoolCard({ d, slot, target, blocked, onAdd }) {
  const on = slot >= 0
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({ id: `pool:${d.driver_id}`, disabled: on })
  return (
    <li className={`pool-card${on ? ' on' : ''}${isDragging ? ' lifting' : ''}`} data-driver={d.driver_id} style={{ '--team': teamColor(d.constructor_id) }}>
      <div className="drag" ref={setNodeRef} {...(on ? {} : { ...attributes, ...listeners })} aria-label={on ? undefined : `Drag ${full(d)} onto a grid slot`}>
        <Portrait d={d} />
        <span className="who2">
          <b>{full(d)}</b>
          <small>{d.season} · {d.constructor_name}{d.nationality ? ` · ${d.nationality}` : ''}</small>
        </span>
      </div>
      {on ? <span className="on-grid">P{slot + 1}</span>
        : <button className="btn ghost sm" disabled={blocked} onClick={() => onAdd(d)} aria-label={target ? `${target} with ${full(d)}` : `Add ${full(d)}`}>{target ? 'Replace' : 'Add'}</button>}
    </li>
  )
}

function CarCard({ car, used, target, onAssign }) {
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({ id: `team:${car.constructor_id}:${car.season}` })
  const color = teamColor(car.constructor_id)
  return (
    <li className={`team-card${isDragging ? ' lifting' : ''}`} data-team={car.constructor_id} data-season={car.season} style={{ '--team': color }}
      title={`${car.constructor_name} races the car of its strongest season (picked automatically)`}>
      <div className="drag" ref={setNodeRef} {...attributes} {...listeners} aria-label={`Drag the ${carLabel(car)} onto a grid driver`}>
        <TeamMark car={car} />
        <span className="who2">
          <b>{car.constructor_name}</b>
          <small className="car-line"><CarArt color={color} /><span className="seats" title={`${used.length} of ${MAX_PER_CAR} seats used`}>{used.length}/{MAX_PER_CAR}</span>
            <span className="car-name">{used.length ? used.join(' + ') : car.drivers}</span></small>
        </span>
      </div>
      <button className="btn ghost sm" disabled={!target} onClick={() => onAssign(car)}
        aria-label={target ? `Give the ${carLabel(car)} to ${target}` : `Assign ${carLabel(car)}: select a grid driver first`}
        title={target ? `Give this car to ${target}` : 'Select a grid driver first, or drag this car onto a driver'}>Assign</button>
    </li>
  )
}

function Slot({ i, e, d, car, n, sel, drag, act }) {
  const { setNodeRef: drop, isOver } = useDroppable({ id: `slot:${i}` })
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({ id: `grid:${e?.driver_id ?? `empty${i}`}`, disabled: !e })
  const ok = drag && (drag.kind !== 'team' || e)             // a car can only be dropped on a driver
  if (!e) return <li ref={drop} className={`slot empty${isOver && ok ? ' over' : ''}`} data-slot={i}><span className="slot-pos">P{i + 1}</span><span className="muted">Empty</span></li>
  const name = d ? full(d) : e.driver_id
  return (
    <li ref={drop} className={`slot${sel ? ' sel' : ''}${isOver && ok ? ' over' : ''}${isDragging ? ' lifting' : ''}`} data-slot={i} data-driver={e.driver_id}
      data-team={e.constructor_id} data-season={e.constructor_season} style={{ '--team': teamColor(e.constructor_id) }}>
      <div className="drag" ref={setNodeRef} {...attributes} {...listeners} onClick={() => act.select(sel ? null : e.driver_id)}
        onKeyDown={ev => { if (ev.key === 'Enter') { ev.preventDefault(); act.select(sel ? null : e.driver_id) } else listeners.onKeyDown?.(ev) }}
        aria-label={`${name}, starting P${i + 1}. Enter selects; Space lifts for dragging with the arrow keys.`} aria-pressed={sel}>
        <span className="slot-pos">P{i + 1}</span>
        {d && <Portrait d={d} />}
        <span className="who2">
          <b>{name} <em className="yr">{e.season}</em></b>
          <small className="car-line">{car ? <TeamMark car={car} small /> : <i className="dot" />}<span className="car-name">{car ? carLabel(car) : `${e.constructor_id} ${e.constructor_season}`}</span></small>
        </span>
      </div>
      <div className="slot-tools">
        <button className="btn ghost sm" disabled={i === 0} onClick={() => act.move(e.driver_id, i - 1)} aria-label={`Move ${name} up to P${i}`}>▲</button>
        <button className="btn ghost sm" disabled={i === n - 1} onClick={() => act.move(e.driver_id, i + 1)} aria-label={`Move ${name} down to P${i + 2}`}>▼</button>
        <button className="btn ghost sm danger" disabled={n <= MIN_CARS} onClick={() => act.remove(e.driver_id)} aria-label={`Remove ${name} from the grid`}>✕</button>
      </div>
    </li>
  )
}

function Bench({ children, drag }) {
  const { setNodeRef, isOver } = useDroppable({ id: 'bench' })
  return <div ref={setNodeRef} className={`ed-col bench${isOver && drag?.kind === 'grid' ? ' over' : ''}`}>{children}</div>
}

// Full-screen lineup workspace: drivers A–Z | 22-slot starting grid | constructors A–Z (each in its peak car).
// Edits go straight into the shared lineup (with undo history), so closing it never loses anything.
export default function LineupEditor({ entries, setEntries, pool, cars, limit, canUndo, canRedo, onUndo, onRedo, onReset, isDefault, onDone }) {
  const [q, setQ] = useState('')
  const [tq, setTq] = useState('')
  const [mode, setMode] = useState('insert')
  const [sel, setSel] = useState(null)
  const [note, setNote] = useState(null)
  const [drag, setDrag] = useState(null)
  const byId = useMemo(() => Object.fromEntries(pool.map(d => [d.driver_id, d])), [pool])
  const carOf = useMemo(() => Object.fromEntries(cars.map(c => [carKey(c.constructor_id, c.season), c])), [cars])
  const slotOf = id => entries.findIndex(e => e.driver_id === id)
  const selIdx = slotOf(sel), selName = selIdx >= 0 ? byId[sel]?.last_name : null
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(TouchSensor, { activationConstraint: { delay: 150, tolerance: 6 } }),
    useSensor(KeyboardSensor, { keyboardCodes: { start: ['Space'], cancel: ['Escape'], end: ['Space', 'Enter'] } }))

  const apply = r => { if (r.entries !== entries) setEntries(r.entries); setNote(r.note ?? null) }
  const opts = { limit, cars, byId }
  const act = {
    select: setSel,
    car: (id, car) => apply(assignCar(entries, id, car, opts)),
    move: (id, to) => apply(placeDriver(entries, byId[id], to, opts)),
    remove: id => { apply(removeDriver(entries, id)); if (sel === id) setSel(null) },
    add: d => {                                              // click alternative to dragging a driver
      const r = placeDriver(entries, d, selIdx >= 0 ? selIdx : entries.length, opts)
      apply(r); if (selIdx >= 0 && r.entries !== entries) setSel(d.driver_id)
    },
  }
  useEffect(() => {
    const key = ev => {
      if (ev.target.matches?.('input, select, textarea')) return
      const z = ev.key.toLowerCase() === 'z' && (ev.ctrlKey || ev.metaKey)
      if (z && !ev.shiftKey && canUndo) { ev.preventDefault(); onUndo() }
      else if (((z && ev.shiftKey) || (ev.key.toLowerCase() === 'y' && (ev.ctrlKey || ev.metaKey))) && canRedo) { ev.preventDefault(); onRedo() }
      else if (ev.key === 'Escape') setSel(null)
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [canUndo, canRedo, onUndo, onRedo])

  const onDragEnd = ({ active, over }) => {
    setDrag(null)
    if (!over) return
    const [kind, id, season] = String(active.id).split(':'), [zone, at] = String(over.id).split(':')
    if (zone === 'bench') { if (kind === 'grid') act.remove(id); return }
    const i = Number(at), e = entries[i]
    if (kind === 'team') { if (e) act.car(e.driver_id, carOf[carKey(id, Number(season))]); else setNote('Drop a car on a driver.'); return }
    apply(placeDriver(entries, byId[id], i, { ...opts, mode }))
  }

  const sortedDrivers = useMemo(() => [...pool].sort((a, b) => byName(full(a), full(b))), [pool])
  const sortedCars = useMemo(() => [...cars].sort((a, b) => byName(a.constructor_name, b.constructor_name)), [cars])
  const n = searchText(q.trim()), tn = searchText(tq.trim())
  const drivers = sortedDrivers.filter(d => searchText(full(d)).includes(n))            // by name, accents ignored
  const carList = sortedCars.filter(c => searchText(c.constructor_name).includes(tn))
  const todo = gridProblems(entries, cars)
  const usedBy = useMemo(() => {
    const m = {}
    for (const e of entries) (m[carKey(e.constructor_id, e.constructor_season)] ||= []).push(byId[e.driver_id]?.code || e.driver_id)
    return m
  }, [entries, byId])
  const dragged = drag && (drag.kind === 'team' ? carOf[carKey(drag.id, Number(drag.season))] : byId[drag.id])

  return (
    <div className="editor" role="dialog" aria-modal="true" aria-labelledby="ed-h">
      <header className="ed-top">
        <div>
          <h1 id="ed-h">Edit Lineup</h1>
          <p className="muted">22 drivers, 11 constructors, two drivers each. Every constructor races the car of its strongest season (1981–2025), picked automatically. Drag a driver onto a grid slot, or a constructor onto a driver.</p>
        </div>
        <div className="ed-actions">
          <button className="btn ghost" onClick={onUndo} disabled={!canUndo} title="Undo (Ctrl+Z)">↶ Undo</button>
          <button className="btn ghost" onClick={onRedo} disabled={!canRedo} title="Redo (Ctrl+Shift+Z)">↷ Redo</button>
          <button className="btn ghost" onClick={() => { onReset(); setSel(null); setNote('Default lineup restored.') }} disabled={isDefault}>Restore default lineup</button>
          <button className="btn lime" onClick={onDone}>Done — Back to Race</button>
        </div>
      </header>
      <p className={`ed-note${note ? ' on' : ''}`} role="status" aria-live="polite">{note ?? (selName ? `${selName} selected: pick a driver to take this slot, or Assign a car. Esc clears.` : 'Click a grid driver to select them, then use Add or Assign instead of dragging.')}</p>

      <DndContext sensors={sensors} collisionDetection={a => { const p = pointerWithin(a); return p.length ? p : rectIntersection(a) }}
        onDragStart={({ active }) => { const [kind, id, season] = String(active.id).split(':'); setDrag({ kind, id, season }); setNote(null) }}
        onDragCancel={() => setDrag(null)} onDragEnd={onDragEnd}>
        <div className={`ed-body${drag ? ` dragging-${drag.kind}` : ''}`}>
          <Bench drag={drag}>
            <h2 className="side-h">Drivers <span className="count">{drivers.length}</span></h2>
            <div className="ed-filter">
              <input value={q} onChange={e => setQ(e.target.value)} placeholder="Search driver name" aria-label="Search drivers" />
            </div>
            <ul className="ed-list" aria-label="Available drivers">
              {drivers.map(d => <PoolCard key={d.driver_id} d={d} slot={slotOf(d.driver_id)} onAdd={act.add}
                target={selIdx >= 0 ? `Replace ${selName}` : undefined} blocked={selIdx < 0 && entries.length >= limit} />)}
              {!drivers.length && <li className="muted none">No driver matches.</li>}
            </ul>
          </Bench>

          <div className="ed-col ed-grid">
            <h2 className="side-h">Starting grid <span className="count" data-count>{entries.length}/{limit}</span>
              <span className={`tag ${todo.length ? 'swap' : 'ok'}`} data-ready={!todo.length}>{todo.length ? 'incomplete' : 'ready'}</span>
              <span className="seg sm drop-mode" role="radiogroup" aria-label="When a grid driver is dropped on another">
                {[['insert', 'Move'], ['swap', 'Swap']].map(([k, l]) => <button key={k} role="radio" aria-checked={mode === k} className={mode === k ? 'on' : ''} onClick={() => setMode(k)}
                  title={k === 'insert' ? 'Dropping a grid driver on another slot moves them there; the others shift' : 'Dropping a grid driver on another swaps the two'}>{l}</button>)}
              </span>
            </h2>
            {!!todo.length && <ul className="grid-todo" role="status" aria-label="Before predicting">{todo.map(t => <li key={t}>{t}</li>)}</ul>}
            <ol className="slots" aria-label="Starting grid">
              {Array.from({ length: Math.max(limit, entries.length) }, (_, i) => { const e = entries[i]; return (
                <Slot key={e?.driver_id ?? `empty${i}`} i={i} e={e} d={byId[e?.driver_id]} car={e && carOf[carKey(e.constructor_id, e.constructor_season)]}
                  n={entries.length} sel={!!e && e.driver_id === sel} drag={drag} act={act} />) })}
            </ol>
          </div>

          <div className="ed-col">
            <h2 className="side-h">Constructors <span className="count">{carList.length}</span></h2>
            <div className="ed-filter">
              <input value={tq} onChange={e => setTq(e.target.value)} placeholder="Search constructor" aria-label="Search constructors" />
            </div>
            <ul className="ed-list" aria-label="Constructors">
              {carList.map(c => { const k = carKey(c.constructor_id, c.season); return (
                <CarCard key={k} car={c} used={usedBy[k] ?? []} target={selName} onAssign={car => act.car(sel, car)} />) })}
              {!carList.length && <li className="muted none">No constructor matches.</li>}
            </ul>
          </div>
        </div>
        <DragOverlay dropAnimation={null}>
          {dragged && (drag.kind === 'team'
            ? <div className="ghost-card" style={{ '--team': teamColor(dragged.constructor_id) }}><TeamMark car={dragged} /><b>{carLabel(dragged)}</b></div>
            : <div className="ghost-card" style={{ '--team': teamColor(dragged.constructor_id) }}><Portrait d={dragged} /><b>{full(dragged)}</b></div>)}
        </DragOverlay>
      </DndContext>
    </div>
  )
}

import { useEffect, useMemo, useRef, useState } from 'react'

// Header circuit selector: a button showing the race, opening a searchable list (keyboard: arrows, Enter, Escape).
export default function CircuitPicker({ circuits, value, onChange }) {
  const [open, setOpen] = useState(false)
  const [q, setQ] = useState('')
  const [hi, setHi] = useState(0)
  const box = useRef(null), input = useRef(null), list = useRef(null)
  const cur = circuits.find(c => c.circuit_id === value)
  const hits = useMemo(() => {
    const n = q.trim().toLowerCase()
    return circuits.filter(c => `${c.race_name} ${c.circuit_name} ${c.country}`.toLowerCase().includes(n))
      .sort((a, b) => (b.last_season >= 2023) - (a.last_season >= 2023))
  }, [q, circuits])
  useEffect(() => {
    if (!open) return
    setQ(''); setHi(Math.max(0, circuits.findIndex(c => c.circuit_id === value)))
    setTimeout(() => input.current?.focus(), 0)
    const away = e => { if (!box.current?.contains(e.target)) setOpen(false) }
    const esc = e => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('pointerdown', away); document.addEventListener('keydown', esc)
    return () => { document.removeEventListener('pointerdown', away); document.removeEventListener('keydown', esc) }
  }, [open])
  useEffect(() => { list.current?.querySelector('[data-hi="1"]')?.scrollIntoView({ block: 'nearest' }) }, [hi, open])
  const choose = c => { if (c) onChange(c.circuit_id); setOpen(false) }
  const key = e => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setHi(h => Math.min(hits.length - 1, h + 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setHi(h => Math.max(0, h - 1)) }
    else if (e.key === 'Enter') { e.preventDefault(); choose(hits[hi]) }
    else if (e.key === 'Escape') { e.preventDefault(); setOpen(false) }
  }
  let group = null
  return (
    <div className="cpick" ref={box}>
      <button className="cpick-btn" aria-haspopup="listbox" aria-expanded={open} aria-label={`Circuit: ${cur?.race_name ?? ''}. Change circuit`} onClick={() => setOpen(o => !o)}>
        <span className="eyebrow">Circuit</span>
        <b>{cur?.race_name ?? '—'}</b>
        <small>{cur ? `${cur.circuit_name} · ${cur.country}` : ''}</small>
        <svg className="cpick-chev" viewBox="0 0 12 12" width="12" height="12" aria-hidden="true"><path d="M3 4.5 6 7.5 9 4.5" fill="none" stroke="currentColor" strokeWidth="1.6" /></svg>
      </button>
      {open && (
        <div className="cpick-pop">
          <input ref={input} value={q} onChange={e => { setQ(e.target.value); setHi(0) }} onKeyDown={key} placeholder="Search circuit, race or country"
            aria-label="Search circuits" role="combobox" aria-expanded="true" aria-controls="cpick-list" aria-activedescendant={hits[hi] ? `cp-${hits[hi].circuit_id}` : undefined} />
          <ul id="cpick-list" role="listbox" ref={list} aria-label="Circuits">
            {hits.map((c, i) => {
              const g = c.last_season >= 2023 ? 'Current calendar' : 'Classic circuits', head = g !== group
              group = g
              return [head && <li key={g} className="cpick-group" role="presentation">{g}</li>,
                <li key={c.circuit_id} id={`cp-${c.circuit_id}`} role="option" aria-selected={c.circuit_id === value} data-hi={i === hi ? '1' : undefined} data-circuit={c.circuit_id}
                  className={`${i === hi ? 'hi' : ''}${c.circuit_id === value ? ' cur' : ''}`} onPointerEnter={() => setHi(i)} onClick={() => choose(c)}>
                  <b>{c.race_name}</b><small>{c.circuit_name} · {c.country}</small>
                </li>]
            })}
            {!hits.length && <li className="cpick-none">No circuit matches.</li>}
          </ul>
        </div>
      )}
    </div>
  )
}

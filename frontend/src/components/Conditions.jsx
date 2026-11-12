// Race conditions panel. Values are the pre-start weather the models were trained on (ranges = 1981-2025 races).
// Only controls listed by /api/conditions are shown (each is an input of at least one installed model).
const STATES = [['dry', 'Dry'], ['damp', 'Damp'], ['wet', 'Wet']]
const LABEL = {
  rain_mm_h: ['Rain', 'mm/h', 0.05], air_temp_c: ['Air temperature', '°C', 0.1],
  humidity_pct: ['Humidity', '%', 0.1], wind_kmh: ['Wind speed', 'km/h', 0.1],
}
export const stateOf = (c, wet) => (c.rain_mm_h <= 0 ? 'dry' : c.rain_mm_h >= wet ? 'wet' : 'damp')

export default function Conditions({ spec, value, onChange, models }) {
  if (!spec) return null
  const set = patch => {
    const next = { ...value, ...patch }
    if (next.rain_mm_h > 0 && next.humidity_pct < spec.min_humidity_rain) next.humidity_pct = spec.min_humidity_rain   // rain never fell in drier air
    onChange(next)
  }
  const state = stateOf(value, spec.wet_mm_h)
  const isDefault = Object.entries(spec.presets.dry).every(([k, v]) => value[k] === v)
  const reads = models.filter(Boolean).map(m => [m.label, m.weather_inputs.length > 1])
  return (
    <section className="panel conditions" aria-labelledby="cond-h">
      <div className="cond-head">
        <h2 id="cond-h" className="side-h">Conditions</h2>
        <button className="btn ghost sm" onClick={() => onChange({ ...spec.presets.dry })} disabled={isDefault}>Reset</button>
      </div>
      <div className="seg sm" role="radiogroup" aria-label="Track">
        {STATES.map(([k, l]) => <button key={k} role="radio" aria-checked={state === k} className={state === k ? 'on' : ''}
          onClick={() => set({ ...spec.presets[k], air_temp_c: value.air_temp_c, wind_kmh: value.wind_kmh, ...(k === 'dry' ? {} : { humidity_pct: Math.max(value.humidity_pct, spec.presets[k].humidity_pct) }) })}>{l}</button>)}
      </div>
      {spec.controls.map(k => {
        const [label, unit, step] = LABEL[k], [lo, hi] = spec.ranges[k]
        const min = k === 'humidity_pct' && value.rain_mm_h > 0 ? spec.min_humidity_rain : lo
        return (
          <label className="slider" key={k}>
            <span><b>{label}</b><em>{k === 'rain_mm_h' && value[k] === 0 ? 'none' : `${value[k]} ${unit}`}</em></span>
            <input type="range" min={min} max={hi} step={step} value={value[k]} aria-label={`${label} (${unit})`} data-cond={k}
              onChange={e => set({ [k]: Number(e.target.value) })} style={{ '--p': `${(100 * (value[k] - min)) / (hi - min)}%` }} />
          </label>
        )
      })}
      <p className="cond-note" data-weather-models>{reads.map(([l, w], i) => <span key={i} className={w ? 'yes' : 'no'} title={w ? 'This model was trained with these weather inputs' : 'This model uses only dry / wet'}>{w ? '☂' : '◌'} {l}</span>)}</p>
    </section>
  )
}

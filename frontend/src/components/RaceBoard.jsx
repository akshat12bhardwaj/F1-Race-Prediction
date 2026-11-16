import { teamColor } from '../teams.js'
import { logoFor, TeamMark } from './LineupEditor.jsx'

const carOfRow = r => ({ constructor_id: r.constructor_id, constructor_name: r.constructor_name, season: r.constructor_season })   // the assigned car's era

const pct = p => (p >= 0.995 ? '>99%' : p > 0 && p < 0.005 ? '<1%' : `${Math.round(100 * p)}%`)
const last = n => n.split(' ').slice(-1)[0]
const ROW = 27                                       // timing-tower row height (px)
const secs = ms => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`)

function Move({ grid, now }) {
  const d = grid - now
  if (d === 0) return <span className="move same" title={`Started P${grid}`}>–</span>
  return <span className={`move ${d > 0 ? 'up' : 'down'}`} title={`Started P${grid}`}>{d > 0 ? '▲' : '▼'}{Math.abs(d)}</span>
}

// Right sidebar. Before a prediction: a prompt. While the race plays: the running order (illustrative).
// At the flag: the predicted podium and full standings with probabilities. Rows select the driver (camera follows).
export default function RaceBoard({ result, stale, running, phase, order, started, selected, onPick, onRerun, onSkip }) {
  if (!result) {
    return (
      <div className={`board empty${stale ? ' is-stale' : ''}`} id="results">
        <h2 className="side-h">Predicted result</h2>
        {stale
          ? <><p className="muted">The race setup changed, so the last prediction no longer applies and has been cleared.</p>
              <button className="btn lime sm" onClick={onRerun} disabled={running}>{running ? 'Predicting…' : 'Predict Again'}</button></>
          : <p className="muted">{running ? 'The two models are predicting your race…' : 'Choose the models, set the lineup and press Predict Race. Then watch the cars race to the predicted finish.'}</p>}
      </div>
    )
  }
  const S = result.standings
  const by = Object.fromEntries(S.map(r => [r.driver_id, r]))
  const { classification: mc, regression: mr } = result.models
  const used = (
    <div className="models-used" aria-label="Models used">
      <span title="Classification model: who reaches the podium"><i>CLS</i>{mc.label}</span>
      <span title="Regression model: finishing position"><i>REG</i>{mr.label}</span>
    </div>
  )
  if (phase === 'racing') {
    // F1-style timing tower: each row keeps its key and slides to its running position (CSS transform transition)
    const at = Object.fromEntries(order.map((id, i) => [id, i]))
    return (
      <div className="board racing" id="results" aria-labelledby="board-h">
        <div className="board-head">
          <h2 id="board-h" className="side-h"><span className="live-dot" />{started ? 'Live order' : 'On the grid'}</h2>
          <button className="btn ghost sm" onClick={onSkip}>Skip to Results</button>
        </div>
        {used}
        <ol className="tower" aria-label="Running order (illustrative)" style={{ height: `${order.length * ROW}px` }}>
          {S.map(r => { const i = at[r.driver_id], gain = r.grid - (i + 1); return (
            <li key={r.driver_id} data-driver={r.driver_id} data-pos={i + 1} className={`${i < 3 ? 'top' : ''}${selected === r.driver_id ? ' sel' : ''}`}
              style={{ transform: `translateY(${i * ROW}px)`, '--team': teamColor(r.constructor_id) }}>
              <button onClick={() => onPick(r.driver_id)} aria-label={`Running P${i + 1} ${r.driver_name}, follow car`}>
                <span className="tw-pos">{i + 1}</span><i className="tw-bar" />
                <b className="tw-code">{r.driver_code || r.driver_name.split(' ').slice(-1)[0].slice(0, 3).toUpperCase()}</b>
                <small className="tw-team">{r.constructor_name}</small>
                <span className={`move ${gain > 0 ? 'up' : gain < 0 ? 'down' : 'same'}`}>{gain > 0 ? `▲${gain}` : gain < 0 ? `▼${-gain}` : '–'}</span>
              </button>
            </li>) })}
        </ol>
        <p className="board-foot">Running order on the way to the predicted finish (overtakes are illustrative). Click the track to pause.</p>
      </div>
    )
  }
  const winLeader = S.reduce((best, row) => row.win_prob > best.win_prob ? row : best);
  const pod = [2, 1, 3].map(p => S[p - 1]).filter(Boolean)
  return (
    <div className="board done" id="results" aria-labelledby="board-h">
      <div className="board-head"><h2 id="board-h" className="side-h">Predicted result</h2></div>
      {used}
      <p className="board-foot win-leader"><b>Highest win probability</b> · {winLeader.driver_name} {pct(winLeader.win_prob)}</p>
      <div className="podium">
        {pod.map(r => (
          <button key={r.driver_id} className={`pod p${r.position}${selected === r.driver_id ? ' sel' : ''}`} data-driver={r.driver_id}
            style={{ '--team': teamColor(r.constructor_id) }} onClick={() => onPick(r.driver_id)} aria-label={`P${r.position} ${r.driver_name}, follow car`}>
            <span className="pod-pos">P{r.position}</span>
            <b className="pod-name">{last(r.driver_name)}</b>
            {logoFor(carOfRow(r)) && <TeamMark car={carOfRow(r)} small />}
            <small>{r.constructor_name}</small>
            <span className="pod-num"><em>{pct(r.win_prob)}</em> win</span>
            <span className="pod-num"><em>{pct(r.podium_prob)}</em> podium</span>
          </button>
        ))}
      </div>
      <div className="st-head" aria-hidden="true"><span>Pos</span><span>Driver</span><span>Grid</span><span>Podium</span></div>
      <ol className="standings" aria-label="Predicted finishing order">
        {S.map(r => (
          <li key={r.driver_id} data-driver={r.driver_id} className={`${r.position <= 3 ? 'top' : ''}${selected === r.driver_id ? ' sel' : ''}`}>
            <button onClick={() => onPick(r.driver_id)} style={{ '--team': teamColor(r.constructor_id) }} aria-label={`P${r.position} ${r.driver_name}, follow car`}>
              <span className="st-pos">{r.position}</span>
              <span className="st-name"><span className="stripe" /><b>{r.driver_name}</b><small>{r.constructor_name}</small></span>
              <Move grid={r.grid} now={r.position} />
              <span className="st-bar" title={`Podium ${pct(r.podium_prob)} · win ${pct(r.win_prob)}`}><i style={{ width: `${Math.max(2, 100 * r.podium_prob)}%` }} /><em>{pct(r.podium_prob)}</em></span>
            </button>
          </li>
        ))}
      </ol>
      {!!result.diagnostics?.length && <details className="prediction-context">
        <summary>ⓘ Prediction context</summary>
        <ul>{result.diagnostics.map(d => <li key={d}>{d}</li>)}</ul>
      </details>}
      <p className="board-foot" data-inference-ms={result.inference_ms ?? ''}>
        {result.inference_ms != null ? `Models computed this in ${secs(result.inference_ms)}` : `Answer received in ${secs(result.round_trip_ms)}`} ({result.n_sims.toLocaleString()} simulated races) · {result.podium_agreement ? 'both models agree on the podium' : 'the two models differ on the podium'}</p>
    </div>
  )
}

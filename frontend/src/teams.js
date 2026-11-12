// Team identity colours (approximate; names are always shown next to them).
const TEAM = {
  mclaren: '#ff8000', mercedes: '#27c4b4', red_bull: '#3671c6', ferrari: '#e8002d', williams: '#64c4ff',
  rb: '#6692ff', aston_martin: '#229971', haas: '#b6babd', audi: '#bb0a30', alpine: '#ff87bc', cadillac: '#d4af37',
  sauber: '#52e252', alphatauri: '#5e8faa', toro_rosso: '#469bff', racing_point: '#f596c8', force_india: '#f596c8',
  renault: '#fff500', alfa: '#c92d4b', manor: '#ed1c24', lotus_f1: '#d6b84a', brawn: '#d6f04a', team_lotus: '#2e6b3a',
  tyrrell: '#3a5ba0', brabham: '#1d4f91', benetton: '#2bb24c', jordan: '#f2c200', minardi: '#d9c400', jaguar: '#0f5a3a',
  toyota: '#cc0000', honda: '#e8e8e8', bmw_sauber: '#4f7fbf', ligier: '#3366cc', march: '#aa3333', lotus: '#2e6b3a',
  arrows: '#ff8c1a', footwork: '#d43a3a', bar: '#dcdcdc', stewart: '#e6e6e6', prost: '#2a5bd7', super_aguri: '#d42e2e',
  spyker: '#ff7a00', marussia: '#c8102e', virgin: '#c8102e', hrt: '#b8a36a', caterham: '#1f9a50', lotus_racing: '#1f9a50',
  toleman: '#3c8fd6', larrousse: '#2e5fd0', zakspeed: '#d9d9d9', osella: '#3a74c4', ags: '#5a7cc7', lola: '#d23c3c',
  simtek: '#7b4bd6', pacific: '#3a7bd5', forti: '#f2d000', dallara: '#cc3a3a', rial: '#2b6fd6', eurobrun: '#e5e5e5',
  coloni: '#f2d000', onyx: '#4479d6', leyton: '#3cb4c8', fondmetal: '#7a82b8', lambo: '#d6c03a', ram: '#cc3a3a',
  spirit: '#e5e5e5', theodore: '#cc3a3a', ats: '#f2d000', fittipaldi: '#d6c03a',
}
// Short neutral badge text for a team (a placeholder mark, not a logo).
const ABBR = { mclaren: 'MCL', mercedes: 'MER', red_bull: 'RBR', ferrari: 'FER', williams: 'WIL', rb: 'RB', aston_martin: 'AMR', haas: 'HAA', sauber: 'SAU', alpine: 'ALP' }
export const teamAbbr = t => ABBR[t.constructor_id] ?? t.constructor_name.replace(/[^A-Za-z]/g, '').slice(0, 3).toUpperCase()

export function teamColor(id) {
  if (TEAM[id]) return TEAM[id]
  let h = 0
  for (const ch of id || '') h = (h * 31 + ch.charCodeAt(0)) % 360
  return `hsl(${h} 50% 58%)`
}

// Plain-language one-liners for the model pickers.
export const MODEL_TAG = {
  logistic_regression: 'Weighs each factor and draws a simple yes/no line for “podium or not”.',
  decision_tree: 'Asks a few yes/no questions (grid slot? car pace?) like a flowchart.',
  knn: 'Looks for the most similar past drivers and copies how they did.',
  svm: 'Finds the clearest boundary between podium and non-podium drivers.',
  naive_bayes: 'Combines each factor as separate evidence using probabilities.',
  random_forest: 'Hundreds of decision trees vote together.',
  bagging: 'Many trees, each trained on a reshuffled history, averaged.',
  adaboost: 'Small trees in a row, each fixing the last one’s mistakes.',
  gradient_boosting: 'Builds trees step by step, each correcting the remaining error.',
  neural_network: 'A small network of artificial neurons that learns patterns.',
  linear_regression: 'A straight-line formula that estimates the finishing position.',
  ridge: 'A straight-line formula with steadier, shrunk weights.',
  lasso: 'A straight-line formula that ignores factors that don’t help.',
  decision_tree_regression: 'Yes/no questions, then the typical finish for that group.',
  random_forest_regression: 'Hundreds of trees average their finishing-position guesses.',
  gradient_boosting_regression: 'Trees added one by one to shrink the position error.',
}

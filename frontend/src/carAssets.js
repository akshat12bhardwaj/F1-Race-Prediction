// Visual selection only; never read by prediction, eligibility or spacing logic.
import { teamColor } from './teams.js'
const historical = (id, label) => ({ id, url: `/models/historical/${id}.glb`, label, length: 5.3, width: 1.9, rotation: 0, quality: 'simplified era approximation', author: '3D Assets', license: 'CC0 1.0', source: 'https://3dassets.dev/packs/grand-prix-circuit-and-open-wheel' })
export const CAR_ASSETS = {
  turbo: historical('15176', '1980s turbo flat-bottom'),
  nineties: historical('15173', '1990s high-nose'),
  v10: historical('15177', '2000s narrow-track'),
  halo: { ...historical('18685', '2018–2021 generic halo racer'), source: 'https://3dassets.dev/assets/transport-collection-hd-open-wheel-racer-aa0e5cef' },
  groundEffect: { id: 'formula-f222', url: '/models/formula-2022.glb', label: 'Formula F222 Ultimate · 2022-era base', length: 5.6, width: 1.9, rotation: Math.PI / 2,
    quality: 'detailed', source: 'https://sketchfab.com/3d-models/formula-f222-ultimate-de780743451748d6b5567b1e190c7908', author: 'CarlosCG31', license: 'CC BY 4.0' },
}
export function carEra(season) {
  if (season < 1990) return '1980s turbo era'
  if (season < 2000) return '1990s'
  if (season < 2010) return '2000s'
  if (season < 2018) return '2010–2017 pre-halo'
  if (season < 2022) return '2018–2021 halo era'
  return '2022–2025 ground effect'
}
export function carAppearance(entry) {
  // A cross-era driver's form year does not determine the car body.
  const season = Number(entry.constructor_season), id = entry.constructor_id
  let palette = [teamColor(id), '#111820', '#f2f2ee']
  if (id === 'ferrari') palette = ['#e21b26', '#b90d20', '#fff6e9']
  if (id === 'red_bull') palette = ['#112b58', '#e41c35', '#ffd320']
  if (id === 'mercedes') palette = [season >= 2020 ? '#161a20' : '#c9d0d2', '#0bbfae', '#c4dcda']
  if (id === 'mclaren') palette = season < 1997 ? ['#f5eee8', '#d52a35', '#d52a35'] : season < 2017 ? ['#bbc0c5', '#22252b', '#e52b30'] : ['#ff8000', '#111b26', '#68c8e9']
  if (id === 'williams') palette = ['#1657c9', '#efefea', '#56b9e5']
  if (/lotus/.test(id)) palette = ['#161b1c', '#a78b36', '#d9bd6c']
  if (id === 'benetton') palette = ['#168d6c', '#174fca', '#ffe03e']
  if (id === 'renault') palette = ['#f5d72b', '#171c27', '#f5eee6']
  if (id === 'brawn') palette = ['#f5f5ef', '#24282a', '#d5f62f']
  const asset = season < 1990 ? CAR_ASSETS.turbo : season < 2000 ? CAR_ASSETS.nineties : season < 2018 ? CAR_ASSETS.v10 : season < 2022 ? CAR_ASSETS.halo : season <= 2025 ? CAR_ASSETS.groundEffect : null
  return { season, constructor: id, era: carEra(season), asset, palette, approximate: true,
    modelGap: season >= 2010 && season < 2018 ? 'Independent 2010–2017 model unavailable; 2000s base approximation' : null,
    description: 'Original approximate period-colour livery; not an exact constructor chassis.' }
}

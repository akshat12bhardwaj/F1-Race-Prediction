// Original mark: three forward-slanted speed bars stepping down like a podium (P1 lit in the accent colour).
export default function Logo() {
  return (
    <span className="logo" aria-label="Apex F1 race predictor">
      <svg viewBox="0 0 64 28" width="46" height="20" aria-hidden="true">
        <polygon points="12,1 63,1 56,8.5 5,8.5" fill="var(--lime)" />
        <polygon points="8,10.5 47,10.5 40,18 1,18" fill="currentColor" />
        <polygon points="4,20 30,20 23,27.5 -3,27.5" fill="currentColor" opacity=".55" />
      </svg>
      <span className="logo-word">APEX</span>
      <span className="logo-sub">F1 race<br />predictor</span>
    </span>
  )
}

import React, { useEffect, useMemo, useRef, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer } from 'recharts'

const F = {
  kf: { n: 'Kalman', c: '#38bdf8' }, ekf: { n: 'Extended KF', c: '#a78bfa' },
  ukf: { n: 'Unscented KF', c: '#34d399' }, pf: { n: 'Particle Filter', c: '#fbbf24' },
  upf: { n: 'Unscented PF', c: '#f472b6' },
}
const DEF = { steps: 100, dt: 1, sigmaR: 1, sigmaThetaDeg: 2, q: 0.05, outlier: 0.05, particles: 200, turn: 0.01, seed: 7, breakMath: false }
const SL = [
  ['sigmaR', 'Range noise σr', 0, 5, 0.1], ['sigmaThetaDeg', 'Bearing noise σθ (°)', 0, 10, 0.1],
  ['q', 'Process noise q', 0, 1, 0.01], ['outlier', 'Outlier probability', 0, 0.3, 0.01],
  ['turn', 'Target turn rate (rad/step)', -0.05, 0.05, 0.001], ['particles', 'Particles', 20, 500, 10],
  ['steps', 'Steps', 20, 300, 10], ['seed', 'Random seed', 1, 99, 1],
]

function Plot({ res, k, on }) {
  const ref = useRef()
  useEffect(() => {
    const cv = ref.current, dpr = window.devicePixelRatio || 1
    const W = cv.clientWidth, H = cv.clientHeight
    cv.width = W * dpr; cv.height = H * dpr
    const g = cv.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, W, H)
    if (!res) { g.fillStyle = '#5b6490'; g.font = '15px sans-serif'; g.fillText('Press “Run filters”', W / 2 - 70, H / 2); return }
    const pts = [...res.truth.map(t => [t[0], t[1]]), ...res.meas, [0, 0]]
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1])
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys)
    const pad = 30, s = Math.min((W - 2 * pad) / (x1 - x0 || 1), (H - 2 * pad) / (y1 - y0 || 1))
    const ox = (W - s * (x1 - x0)) / 2, oy = (H - s * (y1 - y0)) / 2
    const X = x => ox + (x - x0) * s, Y = y => H - oy - (y - y0) * s
    g.strokeStyle = '#ffffff0d'
    for (let v = Math.ceil(x0 / 20) * 20; v < x1; v += 20) { g.beginPath(); g.moveTo(X(v), 0); g.lineTo(X(v), H); g.stroke() }
    for (let v = Math.ceil(y0 / 20) * 20; v < y1; v += 20) { g.beginPath(); g.moveTo(0, Y(v)); g.lineTo(W, Y(v)); g.stroke() }
    // radar
    g.strokeStyle = '#22d3ee66'; for (let r = 1; r <= 3; r++) { g.beginPath(); g.arc(X(0), Y(0), r * 22, 0, 7); g.stroke() }
    g.fillStyle = '#22d3ee'; g.beginPath(); g.arc(X(0), Y(0), 5, 0, 7); g.fill(); g.fillText('radar', X(0) + 8, Y(0) - 8)
    // measurements
    g.fillStyle = '#ffffff40'; res.meas.slice(0, k + 1).forEach(m => { g.beginPath(); g.arc(X(m[0]), Y(m[1]), 2, 0, 7); g.fill() })
    // clouds
    res.filters.filter(f => on[f.id] && f.cloud).forEach(f => {
      g.fillStyle = F[f.id].c + '55'
      f.cloud[k].forEach(p => { g.beginPath(); g.arc(X(p[0]), Y(p[1]), 2.2, 0, 7); g.fill() })
    })
    // truth
    g.setLineDash([6, 5]); g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.beginPath()
    res.truth.slice(0, k + 1).forEach((t, i) => i ? g.lineTo(X(t[0]), Y(t[1])) : g.moveTo(X(t[0]), Y(t[1]))); g.stroke(); g.setLineDash([])
    // estimates
    res.filters.filter(f => on[f.id]).forEach(f => {
      g.strokeStyle = F[f.id].c; g.lineWidth = 2; g.shadowColor = F[f.id].c; g.shadowBlur = 8; g.beginPath()
      f.est.slice(0, k + 1).forEach((e, i) => i ? g.lineTo(X(e[0]), Y(e[1])) : g.moveTo(X(e[0]), Y(e[1]))); g.stroke(); g.shadowBlur = 0
      const e = f.est[k]; g.beginPath(); g.arc(X(e[0]), Y(e[1]), 5, 0, 7); g.fillStyle = F[f.id].c; g.fill()
      g.beginPath(); g.arc(X(e[0]), Y(e[1]), Math.min(f.std[k] * s, 120), 0, 7); g.strokeStyle = F[f.id].c + '88'; g.lineWidth = 1; g.stroke()
    })
    const t = res.truth[k]; g.fillStyle = '#fff'; g.beginPath(); g.arc(X(t[0]), Y(t[1]), 4, 0, 7); g.fill()
  }, [res, k, on])
  return <canvas ref={ref} />
}

export default function App() {
  const [p, setP] = useState(DEF)
  const [on, setOn] = useState({ kf: 1, ekf: 1, ukf: 1, pf: 1, upf: 1 })
  const [res, setRes] = useState(null)
  const [k, setK] = useState(0)
  const [play, setPlay] = useState(false)
  const [busy, setBusy] = useState(false)
  const [hist, setHist] = useState([])
  const loadHist = () => fetch('/api/runs').then(r => r.json()).then(setHist).catch(() => {})
  useEffect(() => { loadHist() }, [])
  useEffect(() => {
    if (!play || !res) return
    const t = setInterval(() => setK(v => { if (v >= res.params.steps) { setPlay(false); return v } return v + 1 }), 70)
    return () => clearInterval(t)
  }, [play, res])
  const run = async () => {
    setBusy(true)
    try {
      const r = await fetch('/api/run', { method: 'POST', body: JSON.stringify({ ...p, filters: Object.keys(on).filter(x => on[x]) }) })
      const d = await r.json(); setRes(d); setK(0); setPlay(true); loadHist()
    } finally { setBusy(false) }
  }
  const open = id => fetch('/api/runs/' + id).then(r => r.json()).then(d => { setRes(d); setP(d.params); setK(d.params.steps); setPlay(false) })
  const data = useMemo(() => res ? res.truth.map((_, i) => ({ k: i, ...Object.fromEntries(res.filters.map(f => [f.id, Math.min(f.errPos[i], 40)])) })) : [], [res])
  return (
    <div className="wrap">
      <div className="side">
        <div><h1>Filters Lab</h1><p className="sub">KF · EKF · UKF · PF · UPF — tracking a radar target</p></div>
        <div className="card">
          <h3>Filters</h3>
          <div className="chips">{Object.entries(F).map(([id, f]) =>
            <button key={id} className={'chip' + (on[id] ? ' on' : '')} style={on[id] ? { background: f.c } : {}} onClick={() => setOn({ ...on, [id]: on[id] ? 0 : 1 })}>{f.n}</button>)}</div>
          {SL.map(([key, name, lo, hi, st]) => <label key={key}>{name}<b>{p[key]}</b>
            <input type="range" min={lo} max={hi} step={st} value={p[key]} onChange={e => setP({ ...p, [key]: +e.target.value })} /></label>)}
          <label className="tog"><input type="checkbox" checked={p.breakMath} onChange={e => setP({ ...p, breakMath: e.target.checked })} />Break the math (R = 0, Q = 0, P₀ = 0)</label>
          <button className="btn" disabled={busy} onClick={run}>{busy ? 'Filtering…' : 'Run filters'}</button>
        </div>
        <div className="card"><h3>Saved runs (Postgres)</h3><div className="hist">
          {hist.length === 0 && <span className="sub">none yet</span>}
          {hist.map(h => <div key={h.id} onClick={() => open(h.id)}>#{h.id} · {new Date(h.at).toLocaleTimeString()} · {h.summary.map(s => s.id + ' ' + s.rmse).join(' · ')}</div>)}
        </div></div>
      </div>
      <div className="main">
        <div className="card">
          <h3>Track — white: truth · grey dots: radar returns · rings: position σ · clouds: particles</h3>
          <Plot res={res} k={k} on={on} />
          <div className="bar">
            <button className="play" onClick={() => { if (res && k >= res.params.steps) setK(0); setPlay(!play) }}>{play ? '❚❚' : '▶'}</button>
            <input type="range" min={0} max={res ? res.params.steps : 1} value={k} onChange={e => { setPlay(false); setK(+e.target.value) }} />
            <span className="sub">step {k}</span>
          </div>
        </div>
        <div className="card"><h3>Position error over time (capped at 40)</h3>
          <div style={{ height: 240 }}><ResponsiveContainer>
            <LineChart data={data}><CartesianGrid stroke="#ffffff10" /><XAxis dataKey="k" stroke="#667" /><YAxis stroke="#667" />
              <Tooltip contentStyle={{ background: '#0b0d1a', border: '1px solid #333' }} /><Legend />
              {res && res.filters.map(f => <Line key={f.id} dataKey={f.id} name={F[f.id].n} stroke={F[f.id].c} dot={false} strokeWidth={2} hide={!on[f.id]} />)}
            </LineChart></ResponsiveContainer></div></div>
        {res && <div className="card"><h3>Scoreboard</h3><table>
          <thead><tr><th>Filter</th><th>RMSE pos</th><th>RMSE vel</th><th>Time</th><th>Numerical events</th></tr></thead>
          <tbody>{res.filters.map(f => <tr key={f.id}>
            <td><span className="dot" style={{ background: F[f.id].c }} />{F[f.id].n}</td><td>{f.rmsePos}</td><td>{f.rmseVel}</td><td>{f.ms} ms</td>
            <td>{Object.entries(f.warn || {}).map(([a, b]) => <span key={a} className={'w' + (a === 'resampled' ? ' i' : '')}>{a} ×{b}</span>)}
              {f.neff && <span className="w i">Neff now {f.neff[k]}</span>}</td></tr>)}</tbody></table></div>}
      </div>
    </div>
  )
}

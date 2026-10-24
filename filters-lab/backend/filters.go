package main

import (
	"math"
	"math/rand"
)

// Model: constant-velocity motion, range/bearing radar at the origin.
type Model struct {
	F, Q   M
	SR, ST float64 // measurement std: range, bearing (rad)
	Eps    float64 // outlier probability (known to the particle filters)
}

type Filter interface {
	Step(z []float64)
	Est() ([]float64, M)
	Warn() map[string]int
}
type Cloudy interface {
	Cloud(n int) [][2]float64
	Neff() float64
}

func NewModel(dt, q, sr, st, eps float64) Model {
	F := Eye(4)
	F[0][2], F[1][3] = dt, dt
	d2, d3 := dt*dt, dt*dt*dt
	Q := M{{d3 / 3, 0, d2 / 2, 0}, {0, d3 / 3, 0, d2 / 2}, {d2 / 2, 0, dt, 0}, {0, d2 / 2, 0, dt}}.Scale(q)
	return Model{F, Q, sr, st, eps}
}

func wrap(a float64) float64 { return math.Atan2(math.Sin(a), math.Cos(a)) }
func hfun(x []float64) []float64 {
	return []float64{math.Hypot(x[0], x[1]), math.Atan2(x[1], x[0])}
}
func hJac(x []float64) M {
	r2 := x[0]*x[0] + x[1]*x[1]
	r := math.Max(math.Sqrt(r2), 1e-6)
	r2 = r * r
	return M{{x[0] / r, x[1] / r, 0, 0}, {-x[1] / r2, x[0] / r2, 0, 0}}
}
func inv(a M, w map[string]int) M {
	r, reg := SafeInv(a)
	if reg {
		w["singular_S_regularised"]++
	}
	return r
}
func chol(a M, w map[string]int) M {
	l, j := SafeChol(a)
	if j {
		w["not_PSD_jitter_added"]++
	}
	return l
}
func predict(m Model, x []float64, P M) ([]float64, M) {
	return m.F.MulV(x), m.F.Mul(P).Mul(m.F.T()).Add(m.Q)
}

// Kalman measurement update with Joseph-form covariance (stays symmetric / PSD).
func update(x []float64, P M, y []float64, H, R M, w map[string]int) ([]float64, M) {
	S := H.Mul(P).Mul(H.T()).Add(R)
	K := P.Mul(H.T()).Mul(inv(S, w))
	x = vadd(x, K.MulV(y))
	A := Eye(len(x)).Sub(K.Mul(H))
	return x, A.Mul(P).Mul(A.T()).Add(K.Mul(R).Mul(K.T())).Sym()
}

// ---------- 1. Kalman filter (linear: radar converted to Cartesian) ----------
type KF struct {
	m Model
	x []float64
	P M
	w map[string]int
}

func (k *KF) Warn() map[string]int  { return k.w }
func (k *KF) Est() ([]float64, M) { return k.x, k.P }
func (k *KF) Step(z []float64) {
	k.x, k.P = predict(k.m, k.x, k.P)
	r, c, s := z[0], math.Cos(z[1]), math.Sin(z[1])
	sr2, rt := k.m.SR*k.m.SR, z[0]*z[0]*k.m.ST*k.m.ST
	R := M{{sr2*c*c + rt*s*s, (sr2 - rt) * s * c}, {(sr2 - rt) * s * c, sr2*s*s + rt*c*c}}
	H := M{{1, 0, 0, 0}, {0, 1, 0, 0}}
	y := []float64{r*c - k.x[0], r*s - k.x[1]}
	k.x, k.P = update(k.x, k.P, y, H, R, k.w)
}

// ---------- 2. Extended Kalman filter ----------
type EKF struct{ KF }

func (e *EKF) Step(z []float64) {
	e.x, e.P = predict(e.m, e.x, e.P)
	h := hfun(e.x)
	y := []float64{z[0] - h[0], wrap(z[1] - h[1])}
	R := Diag(e.m.SR*e.m.SR, e.m.ST*e.m.ST)
	e.x, e.P = update(e.x, e.P, y, hJac(e.x), R, e.w)
}

// ---------- 3. Unscented Kalman filter ----------
const ukA, ukK, ukB = 1.0, 1.0, 2.0

func ukWeights(n int) (float64, []float64, []float64) {
	fn := float64(n)
	lam := ukA*ukA*(fn+ukK) - fn
	wm, wc := make([]float64, 2*n+1), make([]float64, 2*n+1)
	for i := range wm {
		wm[i], wc[i] = 1/(2*(fn+lam)), 1/(2*(fn+lam))
	}
	wm[0] = lam / (fn + lam)
	wc[0] = wm[0] + 1 - ukA*ukA + ukB
	return lam, wm, wc
}
func sigma(x []float64, P M, lam float64, w map[string]int) [][]float64 {
	n := len(x)
	L := chol(P.Scale(float64(n)+lam), w)
	pts := [][]float64{append([]float64{}, x...)}
	for i := 0; i < n; i++ {
		col := make([]float64, n)
		for r := 0; r < n; r++ {
			col[r] = L[r][i]
		}
		pts = append(pts, vadd(x, col))
	}
	for i := 0; i < n; i++ {
		col := make([]float64, n)
		for r := 0; r < n; r++ {
			col[r] = L[r][i]
		}
		pts = append(pts, vsub(x, col))
	}
	return pts
}
func ukfStep(m Model, x []float64, P M, z []float64, w map[string]int) ([]float64, M) {
	lam, wm, wc := ukWeights(4)
	// predict: push sigma points through f
	pts := sigma(x, P, lam, w)
	xp := make([]float64, 4)
	for i := range pts {
		pts[i] = m.F.MulV(pts[i])
		for d := 0; d < 4; d++ {
			xp[d] += wm[i] * pts[i][d]
		}
	}
	Pp := m.Q
	for i := range pts {
		d := vsub(pts[i], xp)
		Pp = Pp.Add(Outer(d, d).Scale(wc[i]))
	}
	// update: redraw sigma points, push through h
	pts = sigma(xp, Pp.Sym(), lam, w)
	Zs := make([][]float64, len(pts))
	zr, sn, cs := 0.0, 0.0, 0.0
	for i := range pts {
		Zs[i] = hfun(pts[i])
		zr += wm[i] * Zs[i][0]
		sn += wm[i] * math.Sin(Zs[i][1])
		cs += wm[i] * math.Cos(Zs[i][1])
	}
	zm := []float64{zr, math.Atan2(sn, cs)} // circular mean for bearing
	S, Pxz := Diag(m.SR*m.SR, m.ST*m.ST), Z(4, 2)
	for i := range pts {
		dz := []float64{Zs[i][0] - zm[0], wrap(Zs[i][1] - zm[1])}
		S = S.Add(Outer(dz, dz).Scale(wc[i]))
		Pxz = Pxz.Add(Outer(vsub(pts[i], xp), dz).Scale(wc[i]))
	}
	K := Pxz.Mul(inv(S, w))
	y := []float64{z[0] - zm[0], wrap(z[1] - zm[1])}
	return vadd(xp, K.MulV(y)), Pp.Sub(K.Mul(S).Mul(K.T())).Sym()
}

type UKF struct{ KF }

func (u *UKF) Step(z []float64) { u.x, u.P = ukfStep(u.m, u.x, u.P, z, u.w) }

// ---------- 4 & 5. Particle filter / Unscented particle filter ----------
type PF struct {
	m    Model
	upf  bool
	xs   [][]float64
	Ps   []M // per-particle covariance (UPF only)
	ws   []float64
	rng  *rand.Rand
	w    map[string]int
	neff float64
	Lq   M
}

func NewPF(m Model, upf bool, N int, x0 []float64, P0 M, rng *rand.Rand) *PF {
	p := &PF{m: m, upf: upf, rng: rng, w: map[string]int{}}
	L, _ := SafeChol(P0)
	for i := 0; i < N; i++ {
		p.xs = append(p.xs, vadd(x0, L.MulV(randn(rng, 4))))
		p.Ps = append(p.Ps, P0)
		p.ws = append(p.ws, 1/float64(N))
	}
	p.Lq = chol(m.Q, p.w)
	p.w = map[string]int{} // don't report the one-off setup jitter
	p.neff = float64(N)
	return p
}
func randn(r *rand.Rand, n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = r.NormFloat64()
	}
	return v
}
func (p *PF) Warn() map[string]int { return p.w }
func (p *PF) Neff() float64         { return p.neff }
func (p *PF) Cloud(n int) [][2]float64 {
	var o [][2]float64
	for i := 0; i < len(p.xs) && i < n; i++ {
		o = append(o, [2]float64{p.xs[i][0], p.xs[i][1]})
	}
	return o
}

// log-likelihood: Gaussian range/bearing mixed with a wide outlier component
func (p *PF) logLik(z, x []float64) float64 {
	h := hfun(x)
	sr, st := math.Max(p.m.SR, 1e-3), math.Max(p.m.ST, 1e-4)
	g := func(d, s float64) float64 { return math.Exp(-.5*d*d/(s*s)) / s }
	dr, dt := z[0]-h[0], wrap(z[1]-h[1])
	l := (1-p.m.Eps)*g(dr, sr)*g(dt, st) + p.m.Eps*g(dr, 8*sr)*g(dt, 8*st)
	return math.Log(l + 1e-300)
}
func (p *PF) Step(z []float64) {
	N := len(p.xs)
	lw := make([]float64, N)
	for i := range p.xs {
		prev := p.xs[i]
		var xn []float64
		fp := p.m.F.MulV(prev)
		if p.upf {
			xh, Ph := ukfStep(p.m, prev, p.Ps[i], z, p.w)
			Lh := chol(Ph, p.w)
			xn = vadd(xh, Lh.MulV(randn(p.rng, 4)))
			p.Ps[i] = Ph
			lw[i] = math.Log(p.ws[i]+1e-300) + p.logLik(z, xn) + logN(vsub(xn, fp), p.m.Q) - logN(vsub(xn, xh), Ph)
		} else {
			xn = vadd(fp, p.Lq.MulV(randn(p.rng, 4)))
			lw[i] = math.Log(p.ws[i]+1e-300) + p.logLik(z, xn)
		}
		p.xs[i] = xn
	}
	mx := math.Inf(-1)
	for _, v := range lw {
		if !math.IsNaN(v) && v > mx {
			mx = v
		}
	}
	sum := 0.0
	for i, v := range lw {
		p.ws[i] = math.Exp(v - mx)
		if math.IsNaN(p.ws[i]) || math.IsInf(p.ws[i], 0) {
			p.ws[i] = 0
		}
		sum += p.ws[i]
	}
	if sum <= 0 || math.IsNaN(sum) || math.IsInf(mx, -1) {
		p.w["weight_collapse_reset"]++
		for i := range p.ws {
			p.ws[i] = 1 / float64(N)
		}
	} else {
		for i := range p.ws {
			p.ws[i] /= sum
		}
	}
	s2 := 0.0
	for _, v := range p.ws {
		s2 += v * v
	}
	p.neff = 1 / s2
	if p.neff < float64(N)/2 { // systematic resampling
		p.w["resampled"]++
		nx, nP := make([][]float64, N), make([]M, N)
		u, c, j := p.rng.Float64()/float64(N), p.ws[0], 0
		for i := 0; i < N; i++ {
			t := u + float64(i)/float64(N)
			for t > c && j < N-1 {
				j++
				c += p.ws[j]
			}
			nx[i], nP[i] = append([]float64{}, p.xs[j]...), p.Ps[j]
		}
		p.xs, p.Ps = nx, nP
		for i := range p.ws {
			p.ws[i] = 1 / float64(N)
		}
	}
}
func (p *PF) Est() ([]float64, M) {
	x := make([]float64, 4)
	for i, v := range p.xs {
		for d := 0; d < 4; d++ {
			x[d] += p.ws[i] * v[d]
		}
	}
	P := Z(4, 4)
	for i, v := range p.xs {
		d := vsub(v, x)
		P = P.Add(Outer(d, d).Scale(p.ws[i]))
	}
	return x, P
}

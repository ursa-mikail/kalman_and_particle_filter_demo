package main

import (
	"math"
	"math/rand"
	"time"
)

type Params struct {
	Steps         int      `json:"steps"`
	Dt            float64  `json:"dt"`
	SigmaR        float64  `json:"sigmaR"`
	SigmaThetaDeg float64  `json:"sigmaThetaDeg"`
	Q             float64  `json:"q"`
	Outlier       float64  `json:"outlier"`
	Particles     int      `json:"particles"`
	Turn          float64  `json:"turn"`
	Seed          int64    `json:"seed"`
	BreakMath     bool     `json:"breakMath"`
	Filters       []string `json:"filters"`
}

type FRes struct {
	ID       string         `json:"id"`
	Est      [][]float64    `json:"est"`
	Std      []float64      `json:"std"`
	ErrPos   []float64      `json:"errPos"`
	RMSEPos  float64        `json:"rmsePos"`
	RMSEVel  float64        `json:"rmseVel"`
	Ms       float64        `json:"ms"`
	Warn     map[string]int `json:"warn"`
	Cloud    [][][2]float64 `json:"cloud,omitempty"`
	Neff     []float64      `json:"neff,omitempty"`
}
type Result struct {
	Params  Params       `json:"params"`
	RunID   int          `json:"runId"`
	Truth   [][]float64  `json:"truth"`
	Meas    [][2]float64 `json:"meas"`
	Filters []FRes       `json:"filters"`
}

func fin(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	return math.Round(v*1000) / 1000
}
func clamp(v, lo, hi float64) float64 { return math.Max(lo, math.Min(hi, v)) }

func (p *Params) normalize() {
	if p.Steps <= 0 { p.Steps = 100 }
	p.Steps = int(clamp(float64(p.Steps), 5, 300))
	if p.Dt <= 0 { p.Dt = 1 }
	p.Particles = int(clamp(float64(p.Particles), 20, 500))
	p.SigmaR = clamp(p.SigmaR, 0, 20)
	p.SigmaThetaDeg = clamp(p.SigmaThetaDeg, 0, 30)
	p.Q = clamp(p.Q, 0, 10)
	p.Outlier = clamp(p.Outlier, 0, 0.5)
	if len(p.Filters) == 0 { p.Filters = []string{"kf", "ekf", "ukf", "pf", "upf"} }
	if p.BreakMath { // degenerate model: perfect sensor, no process noise, perfect prior
		p.SigmaR, p.SigmaThetaDeg, p.Q, p.Outlier = 0, 0, 0, 0
	}
}

func Run(p Params) Result {
	p.normalize()
	rng := rand.New(rand.NewSource(p.Seed))
	st := p.SigmaThetaDeg * math.Pi / 180
	// --- ground truth: a fly-by that curves (the filters assume straight-line motion)
	x := []float64{-60, -10, 3, 0.3}
	truth := [][]float64{append([]float64{}, x...)}
	for k := 0; k < p.Steps; k++ {
		c, s := math.Cos(p.Turn), math.Sin(p.Turn)
		x = []float64{x[0] + x[2]*p.Dt, x[1] + x[3]*p.Dt, c*x[2] - s*x[3], s*x[2] + c*x[3]}
		truth = append(truth, append([]float64{}, x...))
	}
	// --- measurements (range, bearing) with occasional heavy outliers
	zs := make([][]float64, len(truth))
	res := Result{Params: p, Truth: truth}
	for k, t := range truth {
		h := hfun(t)
		sc := 1.0
		if rng.Float64() < p.Outlier { sc = 8 }
		zs[k] = []float64{h[0] + sc*p.SigmaR*rng.NormFloat64(), h[1] + sc*st*rng.NormFloat64()}
		res.Meas = append(res.Meas, [2]float64{fin(zs[k][0] * math.Cos(zs[k][1])), fin(zs[k][0] * math.Sin(zs[k][1]))})
	}
	m := NewModel(p.Dt, p.Q, p.SigmaR, st, p.Outlier)
	x0 := vadd(truth[0], []float64{3 * rng.NormFloat64(), 3 * rng.NormFloat64(), rng.NormFloat64(), rng.NormFloat64()})
	P0 := Diag(25, 25, 4, 4)
	if p.BreakMath { P0 = Z(4, 4) }

	for _, id := range p.Filters {
		var f Filter
		w := map[string]int{}
		base := KF{m, append([]float64{}, x0...), P0, w}
		switch id {
		case "kf": f = &base
		case "ekf": f = &EKF{base}
		case "ukf": f = &UKF{base}
		case "pf": f = NewPF(m, false, p.Particles, x0, P0, rand.New(rand.NewSource(p.Seed+1)))
		case "upf": f = NewPF(m, true, p.Particles, x0, P0, rand.New(rand.NewSource(p.Seed+2)))
		default: continue
		}
		fr := FRes{ID: id}
		rec := func(k int) {
			e, P := f.Est()
			fr.Est = append(fr.Est, []float64{fin(e[0]), fin(e[1]), fin(e[2]), fin(e[3])})
			fr.Std = append(fr.Std, fin(math.Sqrt(math.Max(P[0][0]+P[1][1], 0))))
			fr.ErrPos = append(fr.ErrPos, fin(math.Hypot(e[0]-truth[k][0], e[1]-truth[k][1])))
			if c, ok := f.(Cloudy); ok {
				var cl [][2]float64
				for _, q := range c.Cloud(100) { cl = append(cl, [2]float64{fin(q[0]), fin(q[1])}) }
				fr.Cloud = append(fr.Cloud, cl)
				fr.Neff = append(fr.Neff, fin(c.Neff()))
			}
		}
		rec(0)
		t0 := time.Now()
		sp, sv := 0.0, 0.0
		for k := 1; k <= p.Steps; k++ {
			f.Step(zs[k])
			rec(k)
			e, _ := f.Est()
			sp += math.Pow(fr.ErrPos[k], 2)
			sv += math.Pow(fin(e[2])-truth[k][2], 2) + math.Pow(fin(e[3])-truth[k][3], 2)
		}
		fr.Ms = fin(float64(time.Since(t0).Microseconds()) / 1000)
		fr.RMSEPos = fin(math.Sqrt(sp / float64(p.Steps)))
		fr.RMSEVel = fin(math.Sqrt(sv / float64(p.Steps)))
		fr.Warn = f.Warn()
		res.Filters = append(res.Filters, fr)
	}
	return res
}

package main

import (
	"errors"
	"math"
)

// M is a dense row-major matrix. Vectors are plain []float64.
type M [][]float64

func Z(r, c int) M {
	m := make(M, r)
	for i := range m {
		m[i] = make([]float64, c)
	}
	return m
}
func Eye(n int) M {
	m := Z(n, n)
	for i := range m {
		m[i][i] = 1
	}
	return m
}
func Diag(v ...float64) M {
	m := Z(len(v), len(v))
	for i, x := range v {
		m[i][i] = x
	}
	return m
}
func (a M) T() M {
	m := Z(len(a[0]), len(a))
	for i := range a {
		for j := range a[i] {
			m[j][i] = a[i][j]
		}
	}
	return m
}
func (a M) Mul(b M) M {
	m := Z(len(a), len(b[0]))
	for i := range a {
		for k := range b {
			if a[i][k] == 0 {
				continue
			}
			for j := range b[0] {
				m[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return m
}
func (a M) zip(b M, s float64) M {
	m := Z(len(a), len(a[0]))
	for i := range a {
		for j := range a[i] {
			m[i][j] = a[i][j] + s*b[i][j]
		}
	}
	return m
}
func (a M) Add(b M) M { return a.zip(b, 1) }
func (a M) Sub(b M) M { return a.zip(b, -1) }
func (a M) Scale(s float64) M {
	m := Z(len(a), len(a[0]))
	for i := range a {
		for j := range a[i] {
			m[i][j] = a[i][j] * s
		}
	}
	return m
}
func (a M) Sym() M { return a.Add(a.T()).Scale(.5) }
func (a M) MulV(v []float64) []float64 {
	o := make([]float64, len(a))
	for i := range a {
		for j := range v {
			o[i] += a[i][j] * v[j]
		}
	}
	return o
}
func Outer(u, v []float64) M {
	m := Z(len(u), len(v))
	for i := range u {
		for j := range v {
			m[i][j] = u[i] * v[j]
		}
	}
	return m
}
func vadd(a, b []float64) []float64 {
	o := make([]float64, len(a))
	for i := range a {
		o[i] = a[i] + b[i]
	}
	return o
}
func vsub(a, b []float64) []float64 {
	o := make([]float64, len(a))
	for i := range a {
		o[i] = a[i] - b[i]
	}
	return o
}

var ErrSingular = errors.New("matrix is singular")

// Inv: Gauss-Jordan with partial pivoting. Fails on (numerically) singular input.
func (a M) Inv() (M, error) {
	n := len(a)
	scale := 0.0
	g := Z(n, 2*n)
	for i := range a {
		for j := range a[i] {
			g[i][j] = a[i][j]
			scale = math.Max(scale, math.Abs(a[i][j]))
		}
		g[i][n+i] = 1
	}
	if scale == 0 || math.IsNaN(scale) {
		return nil, ErrSingular
	}
	for c := 0; c < n; c++ {
		p := c
		for r := c + 1; r < n; r++ {
			if math.Abs(g[r][c]) > math.Abs(g[p][c]) {
				p = r
			}
		}
		if math.Abs(g[p][c]) < 1e-12*scale {
			return nil, ErrSingular
		}
		g[c], g[p] = g[p], g[c]
		d := g[c][c]
		for j := range g[c] {
			g[c][j] /= d
		}
		for r := 0; r < n; r++ {
			if r != c && g[r][c] != 0 {
				f := g[r][c]
				for j := range g[r] {
					g[r][j] -= f * g[c][j]
				}
			}
		}
	}
	o := Z(n, n)
	for i := range o {
		copy(o[i], g[i][n:])
	}
	return o, nil
}

// SafeInv: if singular, Tikhonov-regularise (A + λI) with growing λ. Returns reg=true when it had to.
func SafeInv(a M) (M, bool) {
	if r, err := a.Inv(); err == nil {
		return r, false
	}
	lam := 1e-9
	for i := 0; i < 14; i++ {
		if r, err := a.Add(Eye(len(a)).Scale(lam)).Inv(); err == nil {
			return r, true
		}
		lam *= 10
	}
	return Z(len(a), len(a)), true
}

// Chol: lower-triangular L with L Lᵀ = A. Fails if A is not positive definite.
func (a M) Chol() (M, error) {
	n := len(a)
	l := Z(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			s := a[i][j]
			for k := 0; k < j; k++ {
				s -= l[i][k] * l[j][k]
			}
			if i == j {
				if s <= 1e-14 || math.IsNaN(s) {
					return nil, errors.New("not positive definite")
				}
				l[i][i] = math.Sqrt(s)
			} else {
				l[i][j] = s / l[j][j]
			}
		}
	}
	return l, nil
}

// SafeChol: symmetrise, then add growing diagonal jitter until PD. Returns jit=true when needed.
func SafeChol(a M) (M, bool) {
	a = a.Sym()
	if l, err := a.Chol(); err == nil {
		return l, false
	}
	tr := 0.0
	for i := range a {
		tr += math.Abs(a[i][i])
	}
	base := 1 + tr/float64(len(a))
	lam := 1e-9
	for i := 0; i < 16; i++ {
		if l, err := a.Add(Eye(len(a)).Scale(lam * base)).Chol(); err == nil {
			return l, true
		}
		lam *= 10
	}
	return Eye(len(a)), true
}

// logN: log N(d; 0, P)
func logN(d []float64, P M) float64 {
	L, _ := SafeChol(P)
	n := len(d)
	y := make([]float64, n)
	ld, q := 0.0, 0.0
	for i := 0; i < n; i++ {
		s := d[i]
		for k := 0; k < i; k++ {
			s -= L[i][k] * y[k]
		}
		y[i] = s / L[i][i]
		q += y[i] * y[i]
		ld += math.Log(L[i][i])
	}
	return -0.5*(q+float64(n)*math.Log(2*math.Pi)) - ld
}

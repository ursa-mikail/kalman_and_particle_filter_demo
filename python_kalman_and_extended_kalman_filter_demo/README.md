# Kalman Filter & Extended Kalman Filter — A Ninja-Tracks-a-Quail Tutorial

**The story:** a Ninja is tracking a flying Quail. His eyes are noisy, but
he knows roughly how quails fly (physics). The Kalman Filter is the
mathematically optimal way to combine "what physics predicts" with "what
his eyes report" into one best guess — plus an honest measure of how
confident that guess is.

We build this up in two parts:

- **Part 1 — the Kalman Filter (KF).** The Ninja can see the Quail's
  position directly (just noisily). Everything is linear.
- **Part 2 — the Extended Kalman Filter (EKF).** The Ninja swaps his eyes
  for a rangefinder + compass, so he only measures *distance* and
  *bearing angle* — both nonlinear functions of position. This is the
  standard motivation for why the EKF exists at all.

## Files

| File | What it does | Randomness? |
|---|---|---|
| `kf_quail_tracking.py` | Runs the full 1‑D KF simulation, saves plots | yes |
| `kf_stepbystep.py` | Prints the KF's matrices, 3 fixed steps, no plots | **no** |
| `ekf_radar_tracking.py` | Runs the full 2‑D EKF simulation, saves plots | yes |
| `ekf_stepbystep.py` | Prints the EKF's matrices, 2 fixed steps, no plots | **no** |

Run any of them with `python3 <file>.py`. If a number in this README ever
looks surprising, run the matching `_stepbystep.py` script — its printed
output is the ground truth this README is built from, not a hand
approximation.

---

# Part 1 — The Kalman Filter

## 1.1 Picking the state, then deriving everything else from it

### Step 0 — what goes in the state vector, and why

A "state" is the smallest set of numbers that lets you predict what
happens next, given the inputs. For a Quail flying under a commanded
acceleration, Newtonian kinematics says the *next* position depends on
the *current* position **and** the *current* velocity — position alone
isn't enough to predict motion (you also need to know which way and how
fast it's already moving). Velocity, in turn, depends only on the
current velocity and the acceleration. So two numbers are necessary —
and sufficient:

$$x_k = \begin{bmatrix} p_k \cr v_k \end{bmatrix} \qquad \text{(position, velocity)}$$

We don't put acceleration in the state because we already *know* the
commanded acceleration $u$ at every step — it's an input, not something
to be estimated. (If $u$ were unknown too, *then* you'd add it to the
state. That's a design choice, not a law of nature.)

Crucially, the Ninja's eyes will only ever report $p$. Velocity is
never measured — everything the filter ever knows about $v_k$ comes
from this state choice plus the motion model below. That's the payoff
of choosing the state this way: the filter can infer an unmeasured
quantity for free, just by knowing how it's coupled to something that
*is* measured.

### Step 1 — start from continuous physics, in words

Basic kinematics, for one axis, under acceleration $a$:

$$\dot p = v, \qquad \dot v = a$$

("position changes at rate velocity; velocity changes at rate
acceleration" — that's the whole physics content of this model.)

### Step 2 — discretize: freeze $a$ over one small timestep $dt$

We don't simulate continuously; we take one step every $dt$ seconds and
assume acceleration is roughly constant over that short interval. The
standard constant-acceleration kinematic equations for that one step
are:

$$p_{k+1} = p_k + v_k\,dt + \tfrac12 a\, dt^2 \qquad\qquad v_{k+1} = v_k + a\,dt$$

(If this looks unfamiliar: it's the same $x = x_0 + v_0 t + \tfrac12 a
t^2$ from a first physics course, just relabeled with subscript $k$ for
"current step" and $k+1$ for "next step.")

Split $a$ into the part we command, $u$, plus a small random wobble
$w_k$ the Quail adds on its own (real acceleration is never *exactly*
what you commanded):

$$a = u + w_k, \qquad w_k \sim \mathcal N(0, \sigma_a^2)$$

Substituting:

$$p_{k+1} = \underbrace{p_k + v_k\,dt}_{\text{from old state}} + \underbrace{\tfrac12 u\,dt^2}_{\text{known input}} + \underbrace{\tfrac12 w_k\,dt^2}_{\text{noise}}$$
$$v_{k+1} = \underbrace{v_k}_{\text{from old state}} + \underbrace{u\,dt}_{\text{known input}} + \underbrace{w_k\,dt}_{\text{noise}}$$

### Step 3 — read the matrices straight off those two lines

Stack $p_{k+1}, v_{k+1}$ into a vector and split each equation into
"coefficient on $p_k$", "coefficient on $v_k$", and "coefficient on
$u$":

$$\begin{bmatrix} p_{k+1} \cr v_{k+1} \end{bmatrix} = \underbrace{\begin{bmatrix} 1 & dt \cr 0 & 1 \end{bmatrix}}_{A} \begin{bmatrix} p_k \cr v_k \end{bmatrix} + \underbrace{\begin{bmatrix} dt^2/2 \cr dt \end{bmatrix}}_{B} u + \text{noise}$$

Every entry of $A$ and $B$ is just "which old term multiplies which new
term" in the two kinematic equations above — nothing more is happening.
Row 1 of $A$, $[1 \ \ dt]$, says "new position = 1×(old position) +
$dt$×(old velocity)"; row 2, $[0 \ \ 1]$, says "new velocity = 0×(old
position) + 1×(old velocity)". $B=[dt^2/2;\ dt]$ is the same read-off
for the $u$-terms. So:

$$x_{k+1} = A x_k + B u, \qquad A = \begin{bmatrix} 1 & dt \cr 0 & 1 \end{bmatrix}, \quad B = \begin{bmatrix} dt^2/2 \cr dt \end{bmatrix}$$

### Step 4 — the measurement model

The Ninja's eyes report position only — literally "take the state
vector and keep the 1st entry, drop the 2nd":

$$y_k = C x_k + \text{noise}, \qquad C = \begin{bmatrix} 1 & 0 \end{bmatrix}$$

$C$ is a **selection matrix**: $C x_k = [1\ \ 0]\begin{bmatrix}p_k\cr v_k\end{bmatrix} = p_k$. If the sensor instead measured velocity too, $C$ would be the $2\times2$ identity; if it measured a *scaled* position (say, pixels instead of meters), $C$ would hold that scale factor instead of a bare $1$. The general rule: $C$ encodes whatever known, linear arithmetic converts state into sensor reading.

### Step 5 — turn the leftover noise term into a covariance matrix, $E_x$

The noise term we dropped above, on both equations at once, is:

$$\begin{bmatrix} \tfrac12 w_k\,dt^2 \cr w_k\,dt \end{bmatrix} = \underbrace{\begin{bmatrix} dt^2/2 \cr dt \end{bmatrix}}_{=B,\ \text{call it } g}\, w_k$$

Notice this is exactly $B$ again — makes sense, since the *noise* is
just an unplanned little bit of acceleration, and it enters the state
the same way the commanded acceleration $u$ does. A single scalar random
variable $w_k$, multiplying a fixed vector $g$, has covariance (this is
the vector version of $\mathrm{Var}(cX) = c^2\mathrm{Var}(X)$):

$$\mathrm{Cov}(g\,w_k) = g\,g^\top\,\mathrm{Var}(w_k) = \sigma_a^2\, g g^\top$$

Multiply it out:

$$g g^\top = \begin{bmatrix} dt^2/2 \cr dt \end{bmatrix} \begin{bmatrix} dt^2/2 & dt \end{bmatrix} = \begin{bmatrix} dt^4/4 & dt^3/2 \cr dt^3/2 & dt^2 \end{bmatrix}$$

so

$$E_x = \sigma_a^2 \begin{bmatrix} dt^4/4 & dt^3/2 \cr dt^3/2 & dt^2 \end{bmatrix}$$

This is why the off-diagonal terms exist: they're not a separate
assumption bolted on, they fall directly out of squaring the *same*
vector $g$ that appears in $B$. One random acceleration burst nudges
position and velocity **together**, because both come from the same
$w_k$ — so their errors are correlated, not independent.

Finally, $E_z=\sigma_{vision}^2$ is simpler: the measurement noise
enters $y_k$ directly (scalar in, scalar out), so its covariance is just
its own variance, no matrix algebra required.

**Summary of what we just derived, in one place:**

| symbol | what it is | derived from |
|---|---|---|
| $x_k=[p_k,v_k]^\top$ | the state | Step 0: minimal info needed to predict the future |
| $A$ | state transition | Step 3: coefficients of $p_k,v_k$ in the kinematic equations |
| $B$ | control effect | Step 3: coefficients of $u$ in the kinematic equations |
| $C$ | measurement selection | Step 4: which state entries the sensor reads out |
| $E_x$ | process noise covariance | Step 5: $\sigma_a^2 g g^\top$, where $g$ is $B$'s vector |
| $E_z$ | measurement noise covariance | variance of the sensor's own noise |

## 1.2 Deriving the filter: predict, then update

The whole filter is two steps, repeated forever.

### Predict — roll the belief forward through physics

$$\hat{x}_k^- = A \hat{x}_{k-1} + Bu \qquad\qquad P_k^- = A P_{k-1} A^\top + E_x$$

*Why $A P A^\top$?* If $x$ has covariance $P$, any linear transform $Ax$
has covariance $APA^\top$ — the matrix version of $\mathrm{Var}(aX)=a^2\mathrm{Var}(X)$.
We add $E_x$ because the motion model itself isn't perfect — every
prediction injects a little fresh uncertainty.

At this point we have a **prior belief**: a Gaussian $\mathcal N(\hat
x_k^-, P_k^-)$ about where the Quail is, *before* looking at this step's
measurement.

### Update — fuse in the measurement

$$\tilde y_k = y_k - C\hat x_k^- \quad\text{(innovation)} \qquad S_k = CP_k^-C^\top + E_z \quad\text{(innovation covariance)}$$

$$K_k = P_k^- C^\top S_k^{-1} \quad\text{(Kalman gain)}$$

$$\hat x_k = \hat x_k^- + K_k \tilde y_k \qquad\qquad P_k = (I - K_k C) P_k^-$$

**The Kalman gain $K$ is a trust ratio.** It falls out of multiplying two
Gaussians (prior × measurement likelihood) — or equivalently, out of
choosing $K$ to minimize the posterior variance. Either derivation gives
exactly this $K$, which is why the KF is called the *optimal linear
unbiased estimator*.

- Measurement much noisier than the prediction ($E_z \gg P$) → $S$ is
  dominated by $E_z$ → $K \to 0$ → **trust the model.**
- Prediction much less certain than the measurement ($P \gg E_z$) → $K$
  grows → **trust the measurement.**

That's the entire filter. Every "Kalman Filter" you'll ever meet is this
predict/update pair, just with bigger matrices.

## 1.3 Worked example: three steps, real numbers

Setup: $dt=0.1$, $u=1.5$, $\sigma_a=0.05$, $\sigma_{vision}=10$, starting
belief $\hat x_0 = [0,0]^\top$, $P_0 = E_x$. Three fixed (hand-chosen, not
random) measurements: $y_1=12.3$, $y_2=5.8$, $y_3=9.4$.

| step | measurement $y$ | predicted $\hat x^-=(p,v)$ | prior $P^-_{pp}$ | gain $K$ (pos, vel) | posterior $\hat x=(p,v)$ |
|---|---|---|---|---|---|
| 1 | 12.3 | (0.0075, 0.150) | $6.25\times10^{-7}$ | $(6.25\times10^{-9},\ 5.00\times10^{-8})$ | (0.0075, 0.1500) |
| 2 | 5.8  | (0.0300, 0.300) | $2.19\times10^{-6}$ | $(2.19\times10^{-8},\ 1.13\times10^{-7})$ | (0.0300, 0.3000) |
| 3 | 9.4  | (0.0675, 0.450) | $5.25\times10^{-6}$ | $(5.25\times10^{-8},\ 2.00\times10^{-7})$ | (0.0675, 0.4500) |

*(Run `kf_stepbystep.py` to see every one of these numbers printed with
full precision, plus $S$ and the full $2\times2$ $P$ matrix at each step.)*

**The one thing to actually notice:** the gain $K$ is tiny — around
$10^{-8}$. Even though the measurements (12.3, 5.8, 9.4) look nothing like
the predictions (~0.01–0.07), the posterior barely moves toward them. This
is not a bug. $E_z=100$ is many orders of magnitude bigger than $E_x\sim
10^{-6}$, so the filter has correctly concluded the physics model is far
more trustworthy than this measurement. **This is a real and useful
behavior, but it also means this specific example undersells how much the
gain can vary** — see the callout below.

> **How much does the gain actually vary?** Steady-state gain (after many
> steps, once $P$ stops changing) for different noise ratios:
>
> | regime | $\sigma_a$ | $\sigma_{vision}$ | steady-state $K_{pos}$ |
> |---|---|---|---|
> | this tutorial | 0.05 | 10 | ≈ 0.005 — trust the model almost completely |
> | balanced | 2.0 | 3.0 | ≈ 0.11 — model wins, but measurement matters |
> | noisy flier, sharp eyes | 8.0 | 1.0 | ≈ 0.33 — trust the measurement noticeably more |
>
> Try it yourself: change `accel_noise_mag` / `vision_noise_mag` near the
> top of `kf_quail_tracking.py` and re-run — nothing else in the code
> changes, only the filter's whole "personality."

## 1.4 Watching it actually converge

Numbers in a table only go so far — here's what the filter does across
the full 10-second flight.

**`plots/kf_convergence.png`** is the headline plot: the true position
(red), the raw noisy measurements (gray dots), the KF estimate (green
line), and a shaded green band showing the filter's own $\pm 2\sigma$
confidence region. Watch the green line track the red one closely while
the gray dots scatter wildly around both — and watch the band's width
change as $P$ evolves.

**`plots/kf_error.png`** plots $|\text{error}|$ over time for the raw
measurement vs. the KF estimate side by side — the filtered error stays
low and stable while the raw error stays noisy throughout.

**`plots/kf_distributions.png`** zooms into four specific instants and
draws the actual bell curves being multiplied together: the predicted
(prior) belief, the measurement's likelihood, and the resulting posterior
— with the true position marked — so you can see Bayes' rule happening,
not just its outcome.

**`plots/kf_velocity.png`** shows the filter recovering velocity, a
quantity that is *never measured at all*, purely from how position and
the motion model relate.

---

# Part 2 — From KF to EKF

## 2.1 Why linear isn't always enough

Everything above relied on both models being **linear**: $x_{k+1}=Ax_k+Bu$
and $y_k = Cx_k$. Now the Ninja trades his direct-vision eyes for a
rangefinder + compass:

$$h(x,y) = \begin{bmatrix} r \cr \theta \end{bmatrix} = \begin{bmatrix} \sqrt{x^2+y^2} \cr \text{atan2}(y,x) \end{bmatrix}$$

There is no matrix $C$ with $h(\text{state}) = C \cdot \text{state}$ — this
is genuinely nonlinear. If the state is Gaussian, $\sqrt{x^2+y^2}$ is not
exactly Gaussian, so the clean KF machinery breaks.

**The EKF's fix:** linearize $h$ locally, around the current estimate,
with a first-order Taylor expansion:

$$h(x) \approx h(\hat x) + H(x - \hat x), \qquad H = \left.\frac{\partial h}{\partial x}\right|_{\hat x}$$

$H$ (the **Jacobian**) slots in exactly where $C$ used to go — but it must
be **recomputed at every timestep**, since it depends on where the filter
currently thinks it is. That recomputation is the entire idea of "extended."

## 2.2 The state, and the Jacobian, derived by hand

### Which states to focus on, again

Same logic as Step 0 in Part 1, just applied twice — once per axis. The
Quail now moves in a plane, so predicting its future position needs
both position *and* velocity **on each axis**:

$$s = \begin{bmatrix} x \cr v_x \cr y \cr v_y \end{bmatrix}$$

Nothing new here: $x,v_x$ obey exactly the same constant-acceleration
kinematics as $p,v$ did in Part 1, and so do $y,v_y$, independently
(a sideways nudge doesn't change the forward-acceleration equations,
and vice versa). So $A$ and $B$ are just two side-by-side copies of
Part 1's $A,B$, one block per axis — re-derive them by repeating
Steps 1–3 above once for $x$ and once for $y$ if you want to see it
written out; the algebra is identical.

### The only genuinely new piece: $H$

The sensor no longer reads out a state entry directly (there's no row
of 0s/1s that produces "distance" from $[x,v_x,y,v_y]$) — it reports a
*nonlinear function* of the state, $h(s) = [r,\theta]^\top$. The
Jacobian $H$ is the matrix of partial derivatives of $h$ with respect
to every state variable, evaluated at the current estimate — it's
"how much does each output change per unit change in each state
entry, right now." Let $r=\sqrt{x^2+y^2}$.

**Row 1 — range, $r=\sqrt{x^2+y^2}=(x^2+y^2)^{1/2}$.** Differentiate
with the chain rule, treating $y$ as constant for $\partial/\partial x$:

$$\frac{\partial r}{\partial x} = \frac{1}{2}(x^2+y^2)^{-1/2}\cdot 2x = \frac{x}{\sqrt{x^2+y^2}} = \frac{x}{r}$$

and by the same steps (swap $x\leftrightarrow y$), $\partial r/\partial
y = y/r$. Neither $v_x$ nor $v_y$ appears in $r$ at all — range depends
only on where the Quail *is*, not how fast it's moving — so both of
those partials are $0$:

$$\frac{\partial r}{\partial x} = \frac{x}{r}, \qquad \frac{\partial r}{\partial y} = \frac{y}{r}, \qquad \frac{\partial r}{\partial v_x} = \frac{\partial r}{\partial v_y} = 0$$

**Row 2 — bearing, $\theta=\text{atan2}(y,x)$.** For $x>0$ this
is the same function as $\theta=\arctan(y/x)$ (atan2 just extends it to
handle all four quadrants and $x=0$; the derivative formula below is the
one that's valid everywhere, quadrant issues included). Using
$\frac{d}{du}\arctan(u) = \frac{1}{1+u^2}$ with $u=y/x$, and the chain
rule for $\partial u/\partial x = -y/x^2$:

$$\frac{\partial \theta}{\partial x} = \frac{1}{1+(y/x)^2}\cdot\left(-\frac{y}{x^2}\right) = \frac{x^2}{x^2+y^2}\cdot\left(-\frac{y}{x^2}\right) = \frac{-y}{x^2+y^2} = \frac{-y}{r^2}$$

and similarly, with $\partial u/\partial y = 1/x$:

$$\frac{\partial \theta}{\partial y} = \frac{1}{1+(y/x)^2}\cdot\frac{1}{x} = \frac{x^2}{x^2+y^2}\cdot\frac{1}{x} = \frac{x}{x^2+y^2} = \frac{x}{r^2}$$

Again, velocity doesn't appear in $\theta$ (bearing only depends on
where the Quail is), so those two partials are $0$ too:

$$\frac{\partial \theta}{\partial x} = \frac{-y}{r^2}, \qquad \frac{\partial \theta}{\partial y} = \frac{x}{r^2}, \qquad \frac{\partial \theta}{\partial v_x} = \frac{\partial \theta}{\partial v_y} = 0$$

**Stack the two rows** (row = output, column = state variable, in the
order $x,v_x,y,v_y$) to get $H$:

$$H = \begin{bmatrix} x/r & 0 & y/r & 0 \cr -y/r^2 & 0 & x/r^2 & 0 \end{bmatrix}$$

Because $x,y$ (and therefore $r$) change every step, $H$ must be
re-evaluated fresh, **every step**, at wherever the filter currently
thinks it is — it is not a fixed matrix the way $A$, $B$, $C$ were in
Part 1.

## 2.3 The EKF equations

$$\underbrace{\hat s_k^- = A\hat s_{k-1}+Bu,\quad P_k^- = AP_{k-1}A^\top+E_x}_{\text{predict — identical in spirit to the KF}}$$

$$\underbrace{H_k = \left.\dfrac{\partial h}{\partial s}\right|_{\hat s_k^-}}_{\text{NEW: linearize here, fresh}} \qquad \tilde z_k = z_k - h(\hat s_k^-)\ \ \text{\small(use the exact nonlinear $h$, not $H\hat s$!)}$$

$$S_k = H_kP_k^-H_k^\top + R, \qquad K_k = P_k^-H_k^\top S_k^{-1}$$

$$\hat s_k = \hat s_k^- + K_k\tilde z_k, \qquad P_k = (I-K_kH_k)P_k^-$$

Two details that matter in the code:

1. **The innovation uses the real nonlinear $h(\hat s)$**, not $H\hat s$ —
   the Jacobian is only for propagating uncertainty and computing the
   gain.
2. **Bearing wraps around a circle.** A naive $\theta_{meas}-\theta_{pred}$
   can spuriously jump near $\pm180°$. The code wraps this innovation back
   into $(-\pi,\pi]$ before using it.

## 2.4 Worked example: two steps, real numbers

Setup: $u=(a_x{=}1.2, a_y{=}0.6)$, starting belief $\hat s_0=(x{=}5,
v_x{=}0, y{=}0, v_y{=}0)$, $R=\mathrm{diag}(2^2,(3°)^2)$. Two fixed sensor
readings: $(r{=}6.0,\ \theta{=}0.05\text{ rad})$, then $(r{=}7.2,\
\theta{=}0.09\text{ rad})$.

| step | predicted $(x,y)$ | $H$ row 1 (range) | $H$ row 2 (bearing) | innovation $(r,\theta)$ | posterior $(x,v_x,y,v_y)$ |
|---|---|---|---|---|---|
| 1 | (5.006, 0.003) | (1.000, 0.0006) | (−0.00012, 0.1998) | (0.994, 0.0494) | (5.006, 0.120, 0.0030, 0.0601) |
| 2 | (5.024, 0.012) | (1.000, 0.0024) | (−0.00048, 0.1990) | (2.176, 0.0876) | (5.024, 0.240, 0.0121, 0.1206) |

*(Full precision, plus the complete $4\times4$ $P$ and $4\times2$ $K$
matrices, print from `ekf_stepbystep.py`.)*

**Two things worth actually noticing:**

- **$H$'s range row is close to $(1, 0, \approx0, 0)$.** The Quail is
  nearly due east of the Ninja right now, so range is *almost exactly*
  the x-coordinate — the Jacobian correctly recovers the "obvious" linear
  relationship exactly where the geometry makes it valid. It will look
  completely different once the Quail is, say, due north.
- **Range mostly corrects $(x,v_x)$; bearing mostly corrects $(y,v_y)$** —
  compare the two $H$ rows above: range depends almost entirely on $x$,
  bearing almost entirely on $y$ (right now). This split isn't hardcoded
  anywhere — it falls straight out of the Jacobian, every step.
- **$v_x$ and $v_y$ get updated even though neither is ever measured** —
  exactly like the plain KF, via the off-diagonal correlations built up
  during predict.

## 2.5 Watching the EKF converge

**`plots/ekf_trajectory.png`** — true 2‑D path (red), raw range/bearing
readings converted back to $(x,y)$ just for plotting (gray dots), and the
EKF estimate (green), with the Ninja marked at the origin.

**`plots/ekf_convergence.png`** — the same "headline" style as
`kf_convergence.png`: true $x$, raw sensor $x$, EKF estimate, and its
shrinking $\pm2\sigma$ band, all over time.

**`plots/ekf_position_error.png`** — $|\text{error}|$ over time, raw
sensor vs. EKF, same idea as `kf_error.png`.

**`plots/ekf_covariance_ellipse.png`** — the final 2‑D uncertainty ellipse
around the last EKF estimate, in $(x,y)$ space.

---

# Part 3 — KF vs EKF

| | **KF** | **EKF** |
|---|---|---|
| Motion model | must be linear | can be nonlinear (linearized via Jacobian $F$) |
| Measurement model | must be linear | can be nonlinear (linearized via Jacobian $H$) |
| Optimality | **provably optimal** given Gaussian noise + linear models | only **approximately** optimal — depends how "linear-ish" the model is near the estimate |
| Cost | fixed matrices | Jacobians recomputed every step |
| Failure mode | none, if assumptions hold | can diverge if the true state strays far from where you linearized |

**Rule of thumb:**
- Sensor already linear in the state (GPS → position, odometer → distance)
  → **KF**.
- Sensor/dynamics involve angles, ranges, products of state variables, or
  trig → you need at least **EKF**.
- Severe nonlinearity, or huge initial uncertainty relative to how curved
  $f$/$h$ are → the EKF's linear approximation can be too rough. Reach for
  the **Unscented Kalman Filter** (propagates sample "sigma points" through
  the exact nonlinear function — no Jacobians, more accurate, similar
  cost) or a **particle filter** (fully general, handles non-Gaussian
  noise, but expensive). Not implemented here, but the natural next step.

**In short: the KF is a special case of the EKF** where $f$ and $h$
already happen to be linear, so their Jacobians are just the constant
matrices $A$ and $C$, and the linearization is exact rather than
approximate.

## Common pitfalls

- Using $H\hat x$ instead of the real $h(\hat x)$ for the innovation.
- Forgetting to recompute the Jacobian every step (silently turns the EKF
  back into a bad linear KF).
- Not wrapping angular innovations into $(-\pi,\pi]$.
- Confusing process noise ($E_x$/$Q$ — how wrong your *model* is) with
  measurement noise ($E_z$/$R$ — how wrong your *sensor* is).
- Initializing $P$ at zero ("I'm 100% sure of my start") — this makes the
  gain ~0 forever, so the filter ignores all future measurements.
- Expecting the EKF to be optimal — it's a cheap, good approximation, not
  a guarantee. If it diverges, try a better initial estimate, a shorter
  timestep, or a UKF/particle filter.

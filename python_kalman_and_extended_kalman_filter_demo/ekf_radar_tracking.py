"""
ekf_radar_tracking.py
----------------------
An Extended Kalman Filter (EKF) example, continuing the Ninja/Quail story.

This time the Quail flies in 2-D (x, y), and the Ninja no longer has a
magic "read off the position directly" sensor. Instead he stands at the
origin with a rangefinder + compass, so he only measures:

    r     = distance to the Quail       = sqrt(x^2 + y^2)
    theta = bearing angle to the Quail  = atan2(y, x)

Both of these are NONLINEAR functions of the state, so the plain Kalman
Filter's linear-algebra update (y = C*x) doesn't apply anymore. The EKF
fixes this by linearizing the measurement function h(x) with a first-order
Taylor expansion (its Jacobian H) at every step. See README.md, section
"From KF to EKF", for the full derivation.

State:        s = [x, vx, y, vy]              (constant-velocity motion)
Control:      u = [ax, ay]  constant acceleration in x and y
Measurement:  z = [r; theta] = h(s) + noise    (NONLINEAR)

Run:
    python3 ekf_radar_tracking.py
Outputs (saved into ./plots/):
    ekf_trajectory.png      - true path vs raw (dead-reckoned) measurement vs EKF estimate
    ekf_position_error.png  - tracking error over time, KF-linear-approx vs EKF
    ekf_covariance_ellipse.png - final uncertainty ellipse around the EKF estimate
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os

np.random.seed(3)
OUTDIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Meta-variables
# ---------------------------------------------------------------------------
duration = 10.0
dt = 0.1
t_grid = np.arange(0, duration + dt, dt)
N = len(t_grid)

# ---------------------------------------------------------------------------
# 2. Motion model (state order: [x, vx, y, vy] -> two independent 1-D blocks)
#    This part is IDENTICAL in spirit to the linear KF - motion is linear,
#    only the MEASUREMENT is nonlinear here.
# ---------------------------------------------------------------------------
Ax = np.array([[1, dt],
               [0, 1]])
A = np.block([[Ax, np.zeros((2, 2))],
              [np.zeros((2, 2)), Ax]])          # 4x4 block-diagonal

Bx = np.array([[dt**2 / 2], [dt]])
B = np.block([[Bx, np.zeros((2, 1))],
              [np.zeros((2, 1)), Bx]])           # 4x2

u = np.array([[1.2], [0.6]])   # constant acceleration (ax, ay)

# ---------------------------------------------------------------------------
# 3. Noise model
# ---------------------------------------------------------------------------
accel_noise_mag = 0.05
range_noise_mag = 2.0            # meters
bearing_noise_mag = np.deg2rad(3)  # radians (~3 degrees)

Ex1 = accel_noise_mag ** 2 * np.array([[dt**4/4, dt**3/2],
                                        [dt**3/2, dt**2]])
Ex = np.block([[Ex1, np.zeros((2, 2))],
               [np.zeros((2, 2)), Ex1]])         # 4x4 process noise covariance

R = np.diag([range_noise_mag**2, bearing_noise_mag**2])   # 2x2 measurement noise

# ---------------------------------------------------------------------------
# 4. Nonlinear measurement function and its Jacobian
# ---------------------------------------------------------------------------
def h(state):
    """True nonlinear measurement model: state -> [range; bearing]."""
    x, y = state[0, 0], state[2, 0]
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return np.array([[r], [theta]])


def jacobian_H(state):
    """Linearize h(x) about the current estimate: H = dh/dstate."""
    x, y = state[0, 0], state[2, 0]
    r2 = x**2 + y**2
    r = np.sqrt(max(r2, 1e-9))
    H = np.array([
        [x / r,      0,  y / r,      0],
        [-y / r2,    0,  x / r2,     0],
    ])
    return H


def angle_wrap(a):
    """Keep an angle innovation in (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi

# ---------------------------------------------------------------------------
# 5. Simulate the TRUE quail flight + the Ninja's noisy range/bearing reading
# ---------------------------------------------------------------------------
s_true = np.array([[0.0], [0.0], [0.0], [0.0]])   # start at the origin at rest
# nudge the quail away from the origin so range/bearing are well defined
s_true[0, 0] = 5.0

true_xy = np.zeros((N, 2))
z_meas = np.zeros((N, 2))   # [range, bearing] as actually sensed

for k in range(N):
    proc_noise = np.zeros((4, 1))
    proc_noise[0:2, 0] = accel_noise_mag * np.array([(dt**2/2)*np.random.randn(), dt*np.random.randn()])
    proc_noise[2:4, 0] = accel_noise_mag * np.array([(dt**2/2)*np.random.randn(), dt*np.random.randn()])

    s_true = A @ s_true + B @ u + proc_noise
    true_xy[k] = [s_true[0, 0], s_true[2, 0]]

    z_true = h(s_true)
    noise = np.array([[range_noise_mag * np.random.randn()],
                       [bearing_noise_mag * np.random.randn()]])
    z = z_true + noise
    z_meas[k] = [z[0, 0], z[1, 0]]

# for visualization only: convert the raw noisy (r, theta) readings back to (x, y)
raw_xy = np.stack([z_meas[:, 0] * np.cos(z_meas[:, 1]),
                    z_meas[:, 0] * np.sin(z_meas[:, 1])], axis=1)

# ---------------------------------------------------------------------------
# 6. Run the EKF
# ---------------------------------------------------------------------------
s_est = np.array([[5.0], [0.0], [0.0], [0.0]])   # initial belief (roughly correct)
P = Ex.copy() * 10   # start a bit more uncertain than the process noise alone

est_xy = np.zeros((N, 2))
P_hist = np.zeros((N, 4, 4))

for k in range(N):
    # ---- PREDICT (still linear: motion model is linear) ----
    s_est = A @ s_est + B @ u
    P = A @ P @ A.T + Ex

    # ---- UPDATE (nonlinear: linearize h() at the current predicted state) ----
    H = jacobian_H(s_est)
    z_pred = h(s_est)

    innovation = z_meas[k].reshape(2, 1) - z_pred
    innovation[1, 0] = angle_wrap(innovation[1, 0])   # handle angle wraparound

    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    s_est = s_est + K @ innovation
    P = (np.eye(4) - K @ H) @ P

    est_xy[k] = [s_est[0, 0], s_est[2, 0]]
    P_hist[k] = P

# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------

# (a) trajectory: truth vs raw (r,theta)->(x,y) vs EKF estimate
plt.figure(figsize=(7, 7))
plt.plot(true_xy[:, 0], true_xy[:, 1], 'r-', linewidth=2, label='True Quail path')
plt.plot(raw_xy[:, 0], raw_xy[:, 1], 'k.', markersize=4, alpha=0.6, label='Raw range/bearing reading')
plt.plot(est_xy[:, 0], est_xy[:, 1], 'g-', linewidth=2, label='EKF estimate')
plt.plot(0, 0, 'b^', markersize=12, label='Ninja (sensor location)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Extended Kalman Filter: tracking with a nonlinear range/bearing sensor')
plt.legend()
plt.axis('equal')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'ekf_trajectory.png'), dpi=150)
plt.close()

# (b) position error over time: raw reading vs EKF estimate
raw_err = np.linalg.norm(raw_xy - true_xy, axis=1)
ekf_err = np.linalg.norm(est_xy - true_xy, axis=1)
plt.figure(figsize=(9, 5))
plt.plot(t_grid, raw_err, 'k-', alpha=0.6, label='Raw sensor reading error')
plt.plot(t_grid, ekf_err, 'g-', linewidth=2, label='EKF estimate error')
plt.xlabel('time (s)')
plt.ylabel('position error (m)')
plt.title('EKF error stays low and stable despite a nonlinear sensor')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'ekf_position_error.png'), dpi=150)
plt.close()

# (c) final uncertainty ellipse (2-sigma) around the last EKF estimate, in x-y space
def cov_ellipse(ax, mean_xy, cov_xy, n_std=2.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov_xy)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ell = Ellipse(xy=mean_xy, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ell)

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(true_xy[-1, 0], true_xy[-1, 1], 'b*', markersize=15, label='True final position')
ax.plot(est_xy[-1, 0], est_xy[-1, 1], 'go', markersize=8, label='EKF final estimate')
P_xy = P_hist[-1][np.ix_([0, 2], [0, 2])]   # pull out the x,y block of the covariance
cov_ellipse(ax, est_xy[-1], P_xy, n_std=2.0, edgecolor='g', facecolor='none',
            linewidth=2, label='95% confidence region')
ax.legend()
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('EKF uncertainty ellipse at the final timestep')
ax.axis('equal')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'ekf_covariance_ellipse.png'), dpi=150)
plt.close()

print("Done. Plots written to:", OUTDIR)
print(f"Final EKF position error: {ekf_err[-1]:.3f} m "
      f"(raw sensor reading error would have been {raw_err[-1]:.3f} m)")

# (d) THE HEADLINE PLOT: watch the x-position estimate converge onto the
# truth, with an honest shrinking uncertainty band (±2 standard deviations).
sigma_x = np.sqrt(P_hist[:, 0, 0])
plt.figure(figsize=(9, 5))
plt.fill_between(t_grid, est_xy[:, 0] - 2*sigma_x, est_xy[:, 0] + 2*sigma_x,
                  color='green', alpha=0.15, label='EKF ±2σ confidence band (x)')
plt.plot(t_grid, raw_xy[:, 0], '.', color='gray', markersize=4, alpha=0.5, label='Raw sensor reading (x)')
plt.plot(t_grid, true_xy[:, 0], 'r-', linewidth=2.5, label='True x position')
plt.plot(t_grid, est_xy[:, 0], 'g-', linewidth=2, label='EKF estimate (x)')
plt.xlabel('time (s)')
plt.ylabel('x position (m)')
plt.title('The EKF estimate locks onto the truth while its own\nuncertainty band (shaded) honestly shrinks')
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'ekf_convergence.png'), dpi=150)
plt.close()

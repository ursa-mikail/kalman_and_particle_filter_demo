"""
kf_quail_tracking.py
---------------------
A linear Kalman Filter (KF) example: a Ninja tracking a flying Quail.

This is a direct Python translation (and slight extension, with saved plots
instead of a live animation) of the classic "StudentDave" MATLAB Kalman
filter demo. It goes with the step-by-step derivation in README.md -- read
that first if the code below looks like magic.

State we track:      x = [position; velocity]        (1-D flight)
Control input:        u = constant acceleration magnitude
Measurement:          y = noisy position only (the Ninja can't measure speed)

Run:
    python3 kf_quail_tracking.py
Outputs (saved into ./plots/):
    kf_position.png   - true path vs raw measurement vs KF estimate
    kf_velocity.png   - true velocity vs KF-estimated velocity
    kf_covariance.png - how the position-variance P shrinks over time
    kf_distributions.png - snapshots of the three Gaussians being combined
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

np.random.seed(7)  # reproducible noise
OUTDIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Meta-variables: how long and how often the Ninja looks at the sky
# ---------------------------------------------------------------------------
duration = 10.0     # seconds the Quail flies
dt = 0.1            # sampling interval
t_grid = np.arange(0, duration + dt, dt)
N = len(t_grid)

# ---------------------------------------------------------------------------
# 2. Physics model: state transition / control / measurement matrices
#    (see README.md section "Deriving A, B, C" for where these come from)
# ---------------------------------------------------------------------------
A = np.array([[1, dt],
              [0, 1]])          # constant-velocity state transition
B = np.array([[dt**2 / 2],
              [dt]])             # effect of a known acceleration input
C = np.array([[1, 0]])           # we only *measure* position, not velocity

u = 1.5  # constant commanded acceleration (m/s^2) - the Quail is speeding up

# ---------------------------------------------------------------------------
# 3. Noise model
# ---------------------------------------------------------------------------
accel_noise_mag = 0.05      # how much the Quail's real acceleration wobbles
vision_noise_mag = 10.0     # how bad the Ninja's eyesight is (measurement)

Ez = vision_noise_mag ** 2                      # measurement noise covariance (1x1 here)
Ex = accel_noise_mag ** 2 * np.array([[dt**4/4, dt**3/2],
                                       [dt**3/2, dt**2]])   # process noise covariance

# ---------------------------------------------------------------------------
# 4. Simulate the TRUE quail flight + what the Ninja actually sees
# ---------------------------------------------------------------------------
Q_true = np.zeros((2, 1))     # true [position; velocity], starts at rest at origin
Q_loc_true = np.zeros(N)
vel_true = np.zeros(N)
Q_loc_meas = np.zeros(N)

for k in range(N):
    accel_noise = accel_noise_mag * np.array([[(dt**2/2) * np.random.randn()],
                                               [dt * np.random.randn()]])
    Q_true = A @ Q_true + B * u + accel_noise
    meas_noise = vision_noise_mag * np.random.randn()
    y = (C @ Q_true).item() + meas_noise

    Q_loc_true[k] = Q_true[0, 0]
    vel_true[k] = Q_true[1, 0]
    Q_loc_meas[k] = y

# ---------------------------------------------------------------------------
# 5. Run the Kalman Filter over the measurements
# ---------------------------------------------------------------------------
Q_est = np.zeros((2, 1))          # filter's belief of [position; velocity]
P = Ex.copy()                     # initial uncertainty in that belief

Q_loc_est = np.zeros(N)
vel_est = np.zeros(N)
P_mag_est = np.zeros(N)           # trace-ish scalar summary of P for plotting
predicted_state = np.zeros(N)     # prior mean before the measurement update
predicted_var = np.zeros(N)       # prior variance before the measurement update

for k in range(N):
    # ---- PREDICT ----
    Q_est = A @ Q_est + B * u
    predicted_state[k] = Q_est[0, 0]

    P = A @ P @ A.T + Ex
    predicted_var[k] = P[0, 0]

    # ---- UPDATE ----
    S = (C @ P @ C.T).item() + Ez              # innovation covariance
    K = (P @ C.T) / S                            # Kalman gain (2x1)

    innovation = Q_loc_meas[k] - (C @ Q_est).item()
    Q_est = Q_est + K * innovation
    P = (np.eye(2) - K @ C) @ P

    Q_loc_est[k] = Q_est[0, 0]
    vel_est[k] = Q_est[1, 0]
    P_mag_est[k] = P[0, 0]

# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------

# (a) position: truth vs measurement vs KF estimate
plt.figure(figsize=(9, 5))
plt.plot(t_grid, Q_loc_true, 'r-', linewidth=2, label='True Quail position')
plt.plot(t_grid, Q_loc_meas, 'k.', markersize=4, label='Noisy Ninja measurement')
plt.plot(t_grid, Q_loc_est, 'g-', linewidth=2, label='Kalman filter estimate')
plt.xlabel('time (s)')
plt.ylabel('position (m)')
plt.title('Kalman Filter: tracking the Quail\'s position')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'kf_position.png'), dpi=150)
plt.close()

# (b) velocity: truth (never measured directly!) vs KF estimate
plt.figure(figsize=(9, 5))
plt.plot(t_grid, vel_true, 'r-', linewidth=2, label='True velocity')
plt.plot(t_grid, vel_est, 'g-', linewidth=2, label='KF estimated velocity')
plt.xlabel('time (s)')
plt.ylabel('velocity (m/s)')
plt.title('Kalman Filter recovers velocity even though it is never measured')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'kf_velocity.png'), dpi=150)
plt.close()

# (c) how uncertainty shrinks over time
plt.figure(figsize=(9, 5))
plt.plot(t_grid, predicted_var, 'm--', label='Predicted variance (before update)')
plt.plot(t_grid, P_mag_est, 'g-', label='Posterior variance (after update)')
plt.xlabel('time (s)')
plt.ylabel('position variance P[0,0]')
plt.title('Uncertainty shrinks every time a measurement arrives')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'kf_covariance.png'), dpi=150)
plt.close()

# (d) snapshot of the three Gaussians being fused, at a few chosen timesteps
snapshot_steps = [5, 30, 60, 90]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, T in zip(axes.ravel(), snapshot_steps):
    center = Q_loc_est[T]
    x = np.linspace(center - 15, center + 15, 400)

    # prior / predicted belief (before seeing measurement T)
    y_pred = norm.pdf(x, predicted_state[T], np.sqrt(max(predicted_var[T], 1e-6)))
    y_pred = y_pred / y_pred.max()
    ax.plot(x, y_pred, 'm-', label='predicted state (prior)')

    # measurement likelihood
    y_meas = norm.pdf(x, Q_loc_meas[T], vision_noise_mag)
    y_meas = y_meas / y_meas.max()
    ax.plot(x, y_meas, 'k-', label='measurement likelihood')

    # posterior estimate
    y_post = norm.pdf(x, Q_loc_est[T], np.sqrt(max(P_mag_est[T], 1e-6)))
    y_post = y_post / y_post.max()
    ax.plot(x, y_post, 'g-', linewidth=2, label='posterior estimate')

    ax.axvline(Q_loc_true[T], color='b', linewidth=2, label='true position')
    ax.set_title(f't = {t_grid[T]:.1f}s')
    ax.set_yticks([])
    if T == snapshot_steps[0]:
        ax.legend(fontsize=8, loc='upper left')

fig.suptitle('The prior (magenta) and the measurement (black) fuse into a\n'
             'sharper posterior (green) that sits closer to the truth (blue)')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'kf_distributions.png'), dpi=150)
plt.close()

print("Done. Plots written to:", OUTDIR)
print(f"Final position error: {abs(Q_loc_true[-1] - Q_loc_est[-1]):.3f} m "
      f"(raw measurement error would have been {abs(Q_loc_true[-1] - Q_loc_meas[-1]):.3f} m)")

# (e) THE HEADLINE PLOT: watch the estimate converge onto the truth, with an
# honest shrinking uncertainty band (±2 standard deviations) around it.
plt.figure(figsize=(9, 5))
sigma = np.sqrt(P_mag_est)
plt.fill_between(t_grid, Q_loc_est - 2*sigma, Q_loc_est + 2*sigma,
                  color='green', alpha=0.15, label='KF ±2σ confidence band')
plt.plot(t_grid, Q_loc_meas, '.', color='gray', markersize=4, alpha=0.5, label='Raw noisy measurement')
plt.plot(t_grid, Q_loc_true, 'r-', linewidth=2.5, label='True position')
plt.plot(t_grid, Q_loc_est, 'g-', linewidth=2, label='KF estimate')
plt.xlabel('time (s)')
plt.ylabel('position (m)')
plt.title('The KF estimate locks onto the truth while its own\nuncertainty band (shaded) honestly shrinks')
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'kf_convergence.png'), dpi=150)
plt.close()

# (f) error-over-time: raw measurement error vs KF error, side by side
plt.figure(figsize=(9, 5))
plt.plot(t_grid, np.abs(Q_loc_meas - Q_loc_true), '-', color='gray', alpha=0.6, label='Raw measurement error')
plt.plot(t_grid, np.abs(Q_loc_est - Q_loc_true), 'g-', linewidth=2, label='KF estimate error')
plt.xlabel('time (s)')
plt.ylabel('|error| (m)')
plt.title('KF error stays low and stable; raw measurement error stays noisy')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'kf_error.png'), dpi=150)
plt.close()

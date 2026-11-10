"""Fixed-step integrators for first-order ODE systems dy/dt = f(t, y)."""
import numpy as np


def euler_step(f, t, y, dt):
    return y + dt * f(t, y)


def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(f, y0, t_end, dt, step=rk4_step):
    """Integrate from t=0 to t_end. Returns (t, y) arrays."""
    n = int(round(t_end / dt))
    ys = np.empty((n + 1, np.size(y0)))
    ys[0] = y0
    y = np.asarray(y0, dtype=float)
    for i in range(n):
        y = step(f, i * dt, y, dt)
        ys[i + 1] = y
    return np.linspace(0.0, n * dt, n + 1), ys

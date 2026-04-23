"""
Runge-Kutta 4th Order (RK4) Method
====================================
RK4 is a numerical technique used to solve ordinary differential equations (ODEs).
Instead of just using the slope at the beginning of a step (like Euler's method),
RK4 estimates the slope at FOUR points within the step and takes a weighted average.

This gives much better accuracy without needing tiny step sizes.

The formula:
    y_{n+1} = y_n + (1/6) * (k1 + 2*k2 + 2*k3 + k4) * h

Where:
    k1 = f(t_n,        y_n)                  <- slope at start
    k2 = f(t_n + h/2,  y_n + h*k1/2)        <- slope at midpoint using k1
    k3 = f(t_n + h/2,  y_n + h*k2/2)        <- slope at midpoint using k2
    k4 = f(t_n + h,    y_n + h*k3)          <- slope at end using k3

h  = step size
f  = the derivative function dy/dt = f(t, y)
"""


def rk4_step(f, t, y, h):
    """
    Performs a single RK4 step.

    Parameters:
        f  : function  -> the ODE dy/dt = f(t, y)
        t  : float     -> current time
        y  : float     -> current value of y
        h  : float     -> step size

    Returns:
        y_next : float -> estimated value of y at t + h
    """
    k1 = f(t,           y)
    k2 = f(t + h / 2,   y + h * k1 / 2)
    k3 = f(t + h / 2,   y + h * k2 / 2)
    k4 = f(t + h,        y + h * k3)

    y_next = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return y_next


def rk4_solve(f, t0, y0, t_end, h):
    """
    Solves an ODE using RK4 over a time interval.

    Parameters:
        f      : function -> dy/dt = f(t, y)
        t0     : float    -> start time
        y0     : float    -> initial condition y(t0)
        t_end  : float    -> end time
        h      : float    -> step size

    Returns:
        t_values : list of floats -> time points
        y_values : list of floats -> solution at each time point
    """
    t_values = [t0]
    y_values = [y0]

    t = t0
    y = y0

    while t < t_end - 1e-10:  # small tolerance to handle float precision
        # Don't overshoot the end
        if t + h > t_end:
            h = t_end - t

        y = rk4_step(f, t, y, h)
        t += h

        t_values.append(round(t, 10))
        y_values.append(y)

    return t_values, y_values


# ─────────────────────────────────────────
# EXAMPLE: Solving dy/dt = y,  y(0) = 1
# Exact solution: y(t) = e^t
# ─────────────────────────────────────────

import math

def f(t, y):
    """Our ODE: dy/dt = y"""
    return y


if __name__ == "__main__":
    t0    = 0.0   # start time
    y0    = 1.0   # initial condition: y(0) = 1
    t_end = 2.0   # solve up to t = 2
    h     = 0.5   # step size

    t_vals, y_vals = rk4_solve(f, t0, y0, t_end, h)

    print(f"{'t':>6}  {'RK4 y':>12}  {'Exact e^t':>12}  {'Error':>12}")
    print("-" * 48)
    for t, y in zip(t_vals, y_vals):
        exact = math.exp(t)
        error = abs(y - exact)
        print(f"{t:>6.2f}  {y:>12.8f}  {exact:>12.8f}  {error:>12.2e}")

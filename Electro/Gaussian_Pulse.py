import numpy as np
import matplotlib.pyplot as plt
import random

from qutip import (
    expect, basis, sigmax, sigmay, sigmaz,
    sesolve, Options
)

seed = 31
random.seed(seed)
np.random.seed(seed)

Omega0 = 2 * np.pi * 10e3 # rad/s
sigma_us = 10.0 # microseconds
sigma = sigma_us * 1e-6 # seconds

sigma_jitter_us = sigma_us * 0.005 # 0.5% of sigma, in microseconds
sigma_jitter = sigma_jitter_us * 1e-6 # seconds

Omega0_jitter = Omega0 * 0.001 # 0.1% of Omega0, in rad/s
t0 = 5 * sigma
t_start = 0.0
t_end = 10 * sigma

Nt = 60
tlist = np.linspace(t_start, t_end, Nt)

# Number of pulses
NumPulse = 50

sx = sigmax()
sy = sigmay()
sz = sigmaz()

opts = Options(store_states=True)

def Omega(t, args):
    Omega0_ = args["Omega0"]
    sigma_  = args["sigma"]
    t0_     = args["t0"]
    return Omega0_ * np.exp(-(t - t0_)**2 / (2 * sigma_**2))

basislist = ["x", "y", "z"]

def getbasis(basislist):
    choice = random.choice(basislist)
    if choice == "x":
        return 0.5 * sx, "x"
    if choice == "y":
        return 0.5 * sy, "y"
    return 0.5 * sz, "z"

def draw_jittered_args(args_nominal):
    Omega0_nom = args_nominal["Omega0"]
    sigma_nom = args_nominal["sigma"]
    t0_nom = args_nominal["t0"]
    Omega0_draw = Omega0_nom + np.random.normal(loc=0.0, scale=Omega0_jitter)
    sigma_draw  = sigma_nom  + np.random.normal(loc=0.0, scale=sigma_jitter)
    sigma_draw = abs(sigma_draw)
    return {"Omega0": Omega0_draw, "sigma": sigma_draw, "t0": t0_nom}

def concat_data(result, times_all, expect_all):
    times_k = np.array(result.times)
    if times_all is None:
        times_all = times_k
        expect_all = [np.array(e) for e in result.expect] # list of arrays
    else:
        t_offset = times_all[-1]
        new_times = times_k[1:] + t_offset
        times_all = np.concatenate([times_all, new_times])
        for j in range(len(expect_all)):
            expect_all[j] = np.concatenate([expect_all[j], np.array(result.expect[j][1:])])
    return times_all, expect_all

def getBlochVector(state):
    return np.array([
        np.real(expect(sx, state)),
        np.real(expect(sy, state)),
        np.real(expect(sz, state))
    ], dtype=float)

def trace_distance(b_ideal, b_noisy):
    """For pure states: D = 1/2 ||b_ideal - b_noisy||."""
    return 0.5 * np.linalg.norm(b_ideal - b_noisy)

psi0 = (basis(2, 0) + basis(2, 1)).unit()
psi_ideal = psi0
psi_noisy = psi0

args_nominal = {"Omega0": Omega0, "sigma": sigma, "t0": t0}

pulseNum = []
axis_used = []

times_all_ideal = None
times_all_noisy = None

expect_all_ideal = None
expect_all_noisy = None

traceD = []

e_ops = [sz]

for k in range(NumPulse):
    pulseNum.append(k + 1)
    H_axis, axis_label = getbasis(basislist)
    axis_used.append(axis_label)
    H_ideal = [[H_axis, Omega]]
    H_noisy = [[H_axis, Omega]]
    args_jitter = draw_jittered_args(args_nominal)
    res_ideal = sesolve(H_ideal, psi_ideal, tlist, e_ops=e_ops, args=args_nominal, options=opts)
    res_noisy = sesolve(H_noisy, psi_noisy, tlist, e_ops=e_ops, args=args_jitter,  options=opts)
    times_all_ideal, expect_all_ideal = concat_data(res_ideal, times_all_ideal, expect_all_ideal)
    times_all_noisy, expect_all_noisy = concat_data(res_noisy, times_all_noisy, expect_all_noisy)
    psi_ideal = res_ideal.states[-1]
    psi_noisy = res_noisy.states[-1]
    b_i = getBlochVector(psi_ideal)
    b_n = getBlochVector(psi_noisy)
    traceD.append(trace_distance(b_i, b_n))
times_us = times_all_ideal * 1e6

plt.figure(figsize=(8,4))
plt.plot(times_us, expect_all_ideal[0], label=r'Ideal $\langle\sigma_z\rangle$')
plt.plot(times_us, expect_all_noisy[0], label=r'Noisy $\langle\sigma_z\rangle$')
plt.xlabel('time (µs)')
plt.ylabel(r'$\langle\sigma_z\rangle$')
plt.title(r'$\langle\sigma_z\rangle$ vs time (Ideal vs Noisy)')
plt.grid(True)
plt.legend()
plt.tight_layout()


plt.figure(figsize=(8,4))
plt.plot(pulseNum, traceD, marker='o')
plt.xlabel('Pulse Number')
plt.ylabel('Trace Distance (Max Measurement Error)')
plt.title('Trace Distance vs Pulse Number')
plt.grid(True)
plt.tight_layout()

plt.show()
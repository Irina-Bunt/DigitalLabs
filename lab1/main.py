import matplotlib.pylab as plt
from scipy.integrate import quad
import numpy as np


def Fourier_series(x, t, N, t0, T):
    a0 = integral(x, t0, t0 + T, args=(T));
    y = (2 / T * (a0 / 2 + second_term(t, N, T, t0)))
    return y


def integral(x, t0, t1, args=()):
    factor, error = quad(x, t0, t1, args)
    return factor


def second_term(t, N, T, t0):
    sum = 0
    for n in range(1, N + 1):
        a_n = integral(sig_x_cos, t0, t0 + T, args=(n, T));
        b_n = integral(sig_x_sin, t0, t0 + T, args=(n, T));
        sum += a_n * np.cos(n * 2 * np.pi / T * t) + b_n * np.sin(n * 2 * np.pi / T * t)
    return sum


def rec_sig(t, T):
    return np.sign(np.sin(2 * np.pi * t / T))


def sig_x_cos(t, n, T):
    return rec_sig(t, T)*np.cos(n * 2 * np.pi / T * t)


def sig_x_sin(t, n, T):
    return rec_sig(t, T)*np.sin(n * 2 * np.pi / T * t)


t0 = 0
T = 1
N = 100
frequency = 200
period = 1 / frequency
duration = 1000
time = np.linspace(0.0, duration * period, duration)
x_t = Fourier_series(rec_sig, time, N, t0, T)
rec_signal = rec_sig(time, T)

fig, ax1 = plt.subplots()
ax1.set_title('Rec_sig')
ax1.set_xlabel('Время, с')
ax1.set_ylabel('Амплитуда')
ax1.set_xlim([0, duration * period])
ax1.plot(time, rec_signal)
ax1.plot(time, x_t, alpha=0.5)
ax1.grid()

fig, ax2 = plt.subplots()
ax2.set_title('f_sig')
ax2.set_xlabel('Время, с')
ax2.set_ylabel('Ошибка')
ax2.set_xlim([0, duration * period])
ax2.plot(time, rec_signal - x_t)
ax2.grid()

plt.show()


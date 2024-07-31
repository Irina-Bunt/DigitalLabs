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


def sine_sig(t, T):
    return np.sin(2 * np.pi / T * t)


def sig_x_cos(t, n, T):
    return sine_sig(t, T)*np.cos(n * 2 * np.pi / T * t)


def sig_x_sin(t, n, T):
    return sine_sig(t, T)*np.sin(n * 2 * np.pi / T * t)


t0 = 0
T = 1 / 100
N = 20
frequency = 10000
period = 1 / frequency
duration = 10000
time = np.linspace(0.0, duration * period, duration)

func = Fourier_series(sine_sig, time, N, t0, T)
signal = sine_sig(time, T)
f_fft = np.fft.fft(func)
sig_fft = np.fft.fft(signal)

x_fft = np.linspace(0.0, 1.0 / (2 * period), int(duration / 2))

fig, ax1 = plt.subplots()
ax1.set_title('Sine_sig')
ax1.set_xlabel('Время')
ax1.set_ylabel('Амплитуда')
ax1.set_xlim([0, duration * period * T * 2])
ax1.plot(time, signal)
ax1.plot(time, func, alpha=0.5)
ax1.grid()

fig, ax2 = plt.subplots()
ax2.set_title('fft_sine')
ax2.set_xlabel('Частота, Гц')
ax2.set_ylabel('Амплитуда')
ax2.set_xlim([0, 200])
ax2.set_ylim([0, 2])
ax2.plot(x_fft, 2.0 / duration * np.abs(sig_fft[:duration // 2]))
ax2.grid()

fig, ax3 = plt.subplots()
ax3.set_title('Fourier_series_sine')
ax3.set_xlabel('Частота, Гц')
ax3.set_ylabel('Амплитуда')
ax3.set_xlim([0, 200])
ax3.set_ylim([0, 2])
ax3.plot(x_fft, 2.0 / duration * np.abs(f_fft[:duration // 2]))
ax3.grid()

plt.show()


import matplotlib.pylab as plt
import matplotlib.patches as mpatches
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

func = Fourier_series(rec_sig, time, N, t0, T)
signal = rec_sig(time, T)

func_fft = np.fft.fft(func)
sig_fft = np.fft.fft(signal)

x_fft = np.linspace(0.0, 1.0 / (2 * period), int(duration / 2))

fig, ax1 = plt.subplots()
ax1.set_title('Rec_sig')
ax1.set_xlabel('Время')
ax1.set_ylabel('Амплитуда')
ax1.set_xlim([0, duration * period])
ax1.plot(time, signal, 'green')
ax1.grid()

fig, ax2 = plt.subplots()
ax2.set_title('Fec_fft')
ax2.set_xlabel('Частота, Гц')
ax2.set_ylabel('Амплитуда')
ax2.set_xlim([0, frequency / 2])
ax2.set_ylim([0, 2])
ax2.plot(x_fft, 2.0 / duration * np.abs(sig_fft[:duration // 2]))
ax2.grid()

fig, ax3 = plt.subplots()
ax3.set_title('Fourier_series_rec_sig')
ax3.set_xlabel('Время')
ax3.set_ylabel('Амплитуда')
ax3.set_xlim([0, duration * period])
ax3.plot(time, func, 'green')
ax3.grid()

fig, ax4 = plt.subplots()
ax4.set_title('Fourier_series_rec_sig_fft')
ax4.set_xlabel('Частота, Гц')
ax4.set_ylabel('Амплитуда')
ax4.set_xlim([0, frequency / 2])
ax4.set_ylim([0, 2])
ax4.plot(x_fft, 2.0 / duration * np.abs(func_fft[:duration // 2]))
ax4.grid()

plt.show()


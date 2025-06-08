import matplotlib.pyplot as plt
from scipy.fft import dct, idct
import numpy as np

# Параметры
N = 900  # длина сигнала (30 секунд * 30 fps)
fs = 30  # частота дискретизации (30 Гц)
t = np.arange(N) / fs  # временная шкала (0 до 30 секунд)
print(t)

# Случайный сигнал (имитация среднего значения пикселей)
np.random.seed(42)  # для воспроизводимости
f_i = np.random.uniform(0, 1, N)
print(f_i)

# Применяем DCT
F_i = dct(f_i, type=2, norm='ortho')
print(F_i)

# Частотная шкала
freqs = np.arange(N) * fs / N  # частоты от 0 до fs/2 (15 Гц)
print(freqs)

freq_bands = [(0.7, 1.5), (1.5, 2.5), (2.5, 4.0)]

indices = []
for f_min, f_max in freq_bands:
    idx_min = int(np.ceil(f_min * N / fs))  # минимальный индекс
    idx_max = int(np.floor(f_max * N / fs))  # максимальный индекс
    indices.append((idx_min, idx_max))

print("Индексы частотных полос:", indices)

# Создаём отфильтрованные спектры для каждой полосы
F_i_k = []
for idx_min, idx_max in indices:
    F_i_prime = np.zeros_like(F_i)  # создаём нулевой массив
    F_i_prime[idx_min:idx_max+1] = F_i[idx_min:idx_max+1]  # копируем нужные частоты
    F_i_k.append(F_i_prime)

f_i_k = []
for F_i_prime in F_i_k:
    f_i_k_prime = idct(F_i_prime, type=2, norm='ortho')
    f_i_k.append(f_i_k_prime)

# Имитация сигналов из 4 областей лица
I = 4  # количество областей
f_k = []
for k in range(len(freq_bands)):
    f_k_prime = np.stack([f_i_k[k]] * I, axis=0)  # дублируем сигнал для 4 областей
    f_k.append(f_k_prime)

plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(t, f_i, label='Исходный сигнал')
plt.title('Исходный случайный сигнал')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')

for k in range(len(freq_bands)):
    plt.subplot(4, 1, k+2)
    plt.plot(t, f_i_k[k], label=f'Полоса {freq_bands[k]} Гц')
    plt.title(f'Отфильтрованный сигнал (полоса {freq_bands[k]} Гц)')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')

plt.tight_layout()
plt.show()
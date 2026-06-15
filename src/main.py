
# Установка библиотек

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, filtfilt, iirnotch
from tqdm import tqdm
from collections import Counter

# Путь к распакованной базе MIT-BIH
DATA_PATH = './mitbih_database/'

# Стандартный список записей MIT-BIH
RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]
FS = 360  # частота дискретизации MIT-BIH


def load_record(record_name, data_path=DATA_PATH, channel=0):
    """
    Загружает один сигнал ЭКГ и аннотации из CSV/TXT файлов MIT-BIH.

    Файлы:
      - {record_name}.csv               -- сигналы (колонки: 'sample #','MLII','V5')
      - {record_name}annotations.txt    -- аннотации (Time, Sample #, Type, ...)

    channel=0 — первое отведение (обычно MLII).
    """
    # Загрузка сигнала из CSV
    csv_path = os.path.join(data_path, f'{record_name}.csv')
    df = pd.read_csv(csv_path)

    # Убираем возможные кавычки и пробелы в названиях колонок
    df.columns = [c.strip().strip("'").strip('"') for c in df.columns]

    # Колонки сигналов — всё, кроме 'sample #'
    signal_cols = [c for c in df.columns if c.lower() != 'sample #']

    if channel >= len(signal_cols):
        raise ValueError(
            f"Запрошен channel={channel}, но в записи {record_name} "
            f"только {len(signal_cols)} отведений: {signal_cols}"
        )

    signal = df[signal_cols[channel]].to_numpy(dtype=np.float32)

    # MIT-BIH в CSV хранится в целых ADC-единицах.
    # Переводим в милливольты: gain=200 ADU/mV, base=1024 (стандарт для MIT-BIH).
    signal = (signal - 1024.0) / 200.0

    # Загрузка аннотаций из TXT ---
    ann_path = os.path.join(data_path, f'{record_name}annotations.txt')

    r_peaks = []
    labels = []

    with open(ann_path, 'r') as f:
        # Пропускаем заголовок ("    Time   Sample #  Type ...")
        header = f.readline()

        for line in f:
            if not line.strip():
                continue
            # Колонки разделены пробелами/табуляциями.
            # Формат: Time  Sample#  Type  Sub  Chan  Num  Aux
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                sample = int(parts[1])
                symbol = parts[2]
            except (ValueError, IndexError):
                continue

            r_peaks.append(sample)
            labels.append(symbol)

    r_peaks = np.array(r_peaks, dtype=np.int64)
    labels = np.array(labels)

    return signal, r_peaks, labels

# Маппинг символов аннотаций MIT-BIH в 5 классов AAMI
AAMI_MAPPING = {
    # N — Normal
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    # S — Supraventricular ectopic
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    # V — Ventricular ectopic
    'V': 'V', 'E': 'V',
    # F — Fusion
    'F': 'F',
    # Q — Unknown
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}

CLASS_TO_IDX = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}

def bandpass_filter(signal, fs=FS, lowcut=0.5, highcut=40.0, order=4):
    """Полосовой фильтр Баттерворта 0.5–40 Гц."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, fs=FS, freq=50.0, quality=30.0):
    """Режекторный фильтр для подавления сетевой наводки 50 Гц."""
    b, a = iirnotch(freq / (fs / 2), quality)
    return filtfilt(b, a, signal)

def denoise(signal, fs=FS):
    """Полная цепочка фильтрации."""
    signal = bandpass_filter(signal, fs=fs)
    signal = notch_filter(signal, fs=fs, freq=50.0)
    return signal

def zscore_normalize(signal, eps=1e-8):
    """Нормализация по среднему и стандартному отклонению."""
    return (signal - np.mean(signal)) / (np.std(signal) + eps)


WINDOW_BEFORE = 100  # отсчётов до R-пика
WINDOW_AFTER = 150  # отсчётов после R-пика
WINDOW_SIZE = WINDOW_BEFORE + WINDOW_AFTER  # = 250


def segment_beats(signal, r_peaks, labels,
                  before=WINDOW_BEFORE, after=WINDOW_AFTER):
    """
    Нарезает сигнал на отдельные сердечные циклы вокруг R-пиков.
    Возвращает массив окон и соответствующие классы AAMI.
    """
    beats, beat_labels = [], []
    n = len(signal)

    for r_peak, symbol in zip(r_peaks, labels):
        # Пропускаем символы, не относящиеся к QRS-комплексам
        if symbol not in AAMI_MAPPING:
            continue

        start = r_peak - before
        end = r_peak + after

        # Проверка границ
        if start < 0 or end > n:
            continue

        segment = signal[start:end]
        aami_class = AAMI_MAPPING[symbol]

        beats.append(segment)
        beat_labels.append(CLASS_TO_IDX[aami_class])

    return np.array(beats), np.array(beat_labels)


def process_all_records(records=RECORDS, data_path=DATA_PATH):
    """Обрабатывает все записи и возвращает X, y и id пациентов."""
    X_all, y_all, patient_ids = [], [], []

    for rec in tqdm(records, desc='Обработка записей'):
        try:
            signal, r_peaks, labels = load_record(rec, data_path)
        except Exception as e:
            print(f'Ошибка при чтении {rec}: {e}')
            continue

        # Фильтрация
        signal = denoise(signal)

        # Нормализация всего сигнала перед сегментацией
        signal = zscore_normalize(signal)

        # Сегментация
        beats, beat_labels = segment_beats(signal, r_peaks, labels)

        X_all.append(beats)
        y_all.append(beat_labels)
        patient_ids.append(np.full(len(beats), int(rec)))

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    pids = np.concatenate(patient_ids, axis=0)

    return X, y, pids



# Запуск

signal, r_peaks, labels = load_record('100')
print(f"Длина сигнала: {len(signal)} отсчётов ({len(signal)/FS:.1f} c)")
print(f"Аннотаций: {len(r_peaks)}")
print(f"Первые 5 меток: {labels[:5]}")
print(f"Первые 5 R-пиков: {r_peaks[:5]}")



X, y, pids = process_all_records()

print(f'Форма X: {X.shape}')             # (N_beats, 250)
print(f'Форма y: {y.shape}')             # (N_beats,)
print(f'Распределение классов:')
class_names = ['N', 'S', 'V', 'F', 'Q']
for idx, name in enumerate(class_names):
    count = np.sum(y == idx)
    print(f'  {name}: {count:>6} ({100*count/len(y):.2f}%)')


def plot_examples(X, y, n_per_class=3):
    """Показывает примеры сегментов из каждого класса."""
    fig, axes = plt.subplots(5, n_per_class, figsize=(12, 10))
    class_names = ['N', 'S', 'V', 'F', 'Q']

    for cls_idx, name in enumerate(class_names):
        idxs = np.where(y == cls_idx)[0]
        if len(idxs) == 0:
            continue
        chosen = np.random.choice(idxs, min(n_per_class, len(idxs)), replace=False)

        for j, beat_idx in enumerate(chosen):
            ax = axes[cls_idx, j]
            ax.plot(X[beat_idx], linewidth=1)
            ax.set_title(f'Класс {name}')
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

np.savez_compressed(
    './models/mitbih_preprocessed.npz',
    X=X.astype(np.float32),
    y=y.astype(np.int64),
    pids=pids.astype(np.int32)
)
print('Данные сохранены в mitbih_preprocessed.npz')

plot_examples(X, y)


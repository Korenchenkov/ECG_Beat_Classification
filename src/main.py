
# хУФБОПЧЛБ ВЙВМЙПФЕЛ

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from tqdm import tqdm
from collections import Counter

# рХФШ Л ТБУРБЛПЧБООПК ВБЪЕ MIT-BIH
DATA_PATH = './mitbih_database/'

# уФБОДБТФОЩК УРЙУПЛ ЪБРЙУЕК MIT-BIH
RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]
FS = 360  # ЮБУФПФБ ДЙУЛТЕФЙЪБГЙЙ MIT-BIH


def load_record(record_name, data_path=DATA_PATH, channel=0):
    """
    ъБЗТХЦБЕФ ПДЙО УЙЗОБМ ьлз Й БООПФБГЙЙ ЙЪ CSV/TXT ЖБКМПЧ MIT-BIH.

    жБКМЩ:
      - {record_name}.csv               -- УЙЗОБМЩ (ЛПМПОЛЙ: 'sample #','MLII','V5')
      - {record_name}annotations.txt    -- БООПФБГЙЙ (Time, Sample #, Type, ...)

    channel=0 ? РЕТЧПЕ ПФЧЕДЕОЙЕ (ПВЩЮОП MLII).
    """
    # ъБЗТХЪЛБ УЙЗОБМБ ЙЪ CSV
    csv_path = os.path.join(data_path, f'{record_name}.csv')
    df = pd.read_csv(csv_path)

    # хВЙТБЕН ЧПЪНПЦОЩЕ ЛБЧЩЮЛЙ Й РТПВЕМЩ Ч ОБЪЧБОЙСИ ЛПМПОПЛ
    df.columns = [c.strip().strip("'").strip('"') for c in df.columns]

    # лПМПОЛЙ УЙЗОБМПЧ ? ЧУЈ, ЛТПНЕ 'sample #'
    signal_cols = [c for c in df.columns if c.lower() != 'sample #']

    if channel >= len(signal_cols):
        raise ValueError(
            f"ъБРТПЫЕО channel={channel}, ОП Ч ЪБРЙУЙ {record_name} "
            f"ФПМШЛП {len(signal_cols)} ПФЧЕДЕОЙК: {signal_cols}"
        )

    signal = df[signal_cols[channel]].to_numpy(dtype=np.float32)

    # MIT-BIH Ч CSV ИТБОЙФУС Ч ГЕМЩИ ADC-ЕДЙОЙГБИ.
    # рЕТЕЧПДЙН Ч НЙММЙЧПМШФЩ: gain=200 ADU/mV, base=1024 (УФБОДБТФ ДМС MIT-BIH).
    signal = (signal - 1024.0) / 200.0

    # ъБЗТХЪЛБ БООПФБГЙК ЙЪ TXT ---
    ann_path = os.path.join(data_path, f'{record_name}annotations.txt')

    r_peaks = []
    labels = []

    with open(ann_path, 'r') as f:
        # рТПРХУЛБЕН ЪБЗПМПЧПЛ ("    Time   Sample #  Type ...")
        header = f.readline()

        for line in f:
            if not line.strip():
                continue
            # лПМПОЛЙ ТБЪДЕМЕОЩ РТПВЕМБНЙ/ФБВХМСГЙСНЙ.
            # жПТНБФ: Time  Sample#  Type  Sub  Chan  Num  Aux
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

# нБРРЙОЗ УЙНЧПМПЧ БООПФБГЙК MIT-BIH Ч 5 ЛМБУУПЧ AAMI
AAMI_MAPPING = {
    # N ? Normal
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    # S ? Supraventricular ectopic
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    # V ? Ventricular ectopic
    'V': 'V', 'E': 'V',
    # F ? Fusion
    'F': 'F',
    # Q ? Unknown
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}

CLASS_TO_IDX = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}

def bandpass_filter(signal, fs=FS, lowcut=0.5, highcut=40.0, order=4):
    """рПМПУПЧПК ЖЙМШФТ вБФФЕТЧПТФБ 0.5?40 зГ."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, fs=FS, freq=50.0, quality=30.0):
    """тЕЦЕЛФПТОЩК ЖЙМШФТ ДМС РПДБЧМЕОЙС УЕФЕЧПК ОБЧПДЛЙ 50 зГ."""
    b, a = iirnotch(freq / (fs / 2), quality)
    return filtfilt(b, a, signal)

def denoise(signal, fs=FS):
    """рПМОБС ГЕРПЮЛБ ЖЙМШФТБГЙЙ."""
    signal = bandpass_filter(signal, fs=fs)
    signal = notch_filter(signal, fs=fs, freq=50.0)
    return signal

def zscore_normalize(signal, eps=1e-8):
    """оПТНБМЙЪБГЙС РП УТЕДОЕНХ Й УФБОДБТФОПНХ ПФЛМПОЕОЙА."""
    return (signal - np.mean(signal)) / (np.std(signal) + eps)


WINDOW_BEFORE = 100  # ПФУЮЈФПЧ ДП R-РЙЛБ
WINDOW_AFTER = 150  # ПФУЮЈФПЧ РПУМЕ R-РЙЛБ
WINDOW_SIZE = WINDOW_BEFORE + WINDOW_AFTER  # = 250


def segment_beats(signal, r_peaks, labels,
                  before=WINDOW_BEFORE, after=WINDOW_AFTER):
    """
    оБТЕЪБЕФ УЙЗОБМ ОБ ПФДЕМШОЩЕ УЕТДЕЮОЩЕ ГЙЛМЩ ЧПЛТХЗ R-РЙЛПЧ.
    чПЪЧТБЭБЕФ НБУУЙЧ ПЛПО Й УППФЧЕФУФЧХАЭЙЕ ЛМБУУЩ AAMI.
    """
    beats, beat_labels = [], []
    n = len(signal)

    for r_peak, symbol in zip(r_peaks, labels):
        # рТПРХУЛБЕН УЙНЧПМЩ, ОЕ ПФОПУСЭЙЕУС Л QRS-ЛПНРМЕЛУБН
        if symbol not in AAMI_MAPPING:
            continue

        start = r_peak - before
        end = r_peak + after

        # рТПЧЕТЛБ ЗТБОЙГ
        if start < 0 or end > n:
            continue

        segment = signal[start:end]
        aami_class = AAMI_MAPPING[symbol]

        beats.append(segment)
        beat_labels.append(CLASS_TO_IDX[aami_class])

    return np.array(beats), np.array(beat_labels)


def process_all_records(records=RECORDS, data_path=DATA_PATH):
    """пВТБВБФЩЧБЕФ ЧУЕ ЪБРЙУЙ Й ЧПЪЧТБЭБЕФ X, y Й id РБГЙЕОФПЧ."""
    X_all, y_all, patient_ids = [], [], []

    for rec in tqdm(records, desc='пВТБВПФЛБ ЪБРЙУЕК'):
        try:
            signal, r_peaks, labels = load_record(rec, data_path)
        except Exception as e:
            print(f'пЫЙВЛБ РТЙ ЮФЕОЙЙ {rec}: {e}')
            continue

        # жЙМШФТБГЙС
        signal = denoise(signal)

        # оПТНБМЙЪБГЙС ЧУЕЗП УЙЗОБМБ РЕТЕД УЕЗНЕОФБГЙЕК
        signal = zscore_normalize(signal)

        # уЕЗНЕОФБГЙС
        beats, beat_labels = segment_beats(signal, r_peaks, labels)

        X_all.append(beats)
        y_all.append(beat_labels)
        patient_ids.append(np.full(len(beats), int(rec)))

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    pids = np.concatenate(patient_ids, axis=0)

    return X, y, pids



# ъБРХУЛ

signal, r_peaks, labels = load_record('100')
print(f"дМЙОБ УЙЗОБМБ: {len(signal)} ПФУЮЈФПЧ ({len(signal)/FS:.1f} c)")
print(f"бООПФБГЙК: {len(r_peaks)}")
print(f"рЕТЧЩЕ 5 НЕФПЛ: {labels[:5]}")
print(f"рЕТЧЩЕ 5 R-РЙЛПЧ: {r_peaks[:5]}")



X, y, pids = process_all_records()

print(f'жПТНБ X: {X.shape}')             # (N_beats, 250)
print(f'жПТНБ y: {y.shape}')             # (N_beats,)
print(f'тБУРТЕДЕМЕОЙЕ ЛМБУУПЧ:')
class_names = ['N', 'S', 'V', 'F', 'Q']
for idx, name in enumerate(class_names):
    count = np.sum(y == idx)
    print(f'  {name}: {count:>6} ({100*count/len(y):.2f}%)')


def plot_examples(X, y, n_per_class=3):
    """рПЛБЪЩЧБЕФ РТЙНЕТЩ УЕЗНЕОФПЧ ЙЪ ЛБЦДПЗП ЛМБУУБ."""
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
            ax.set_title(f'лМБУУ {name}')
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

np.savez_compressed(
    './models/mitbih_preprocessed.npz',
    X=X.astype(np.float32),
    y=y.astype(np.int64),
    pids=pids.astype(np.int32)
)
print('дБООЩЕ УПИТБОЕОЩ Ч mitbih_preprocessed.npz')

plot_examples(X, y)


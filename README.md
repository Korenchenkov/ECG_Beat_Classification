# ECG Beat Classification - MIT-BIH (AAMI 5 классов)

End-to-end проект по классификации сердечных сокращений на ЭКГ из базы **MIT-BIH Arrhythmia Database** в 5 классов по стандарту **AAMI** (N, S, V, F, Q).

Проект включает три этапа, реализованных в виде Jupyter-ноутбуков:

1. **Предобработка сигнала** - фильтрация, нормализация и сегментация ЭКГ вокруг R-пиков.
2. **Устранение дисбаланса классов** - SMOTE, аугментации сигнала, взвешенный loss / sampler.
3. **Обучение 1D-CNN с residual-блоками** - обучение, валидация, оценка на тесте и сохранение модели.

В итоге обучается нейросеть (`ECGResNet`), способная классифицировать отдельные удары сердца.

---

##  Классы AAMI

| Код | Название | Описание |
|-----|----------|----------|
| **N** | Normal | Нормальные сокращения |
| **S** | Supraventricular ectopic | Наджелудочковые экстрасистолы |
| **V** | Ventricular ectopic | Желудочковые экстрасистолы |
| **F** | Fusion | Сливные комплексы |
| **Q** | Unknown | Артефакты / неизвестные |

---

## - Структура проекта

```
ECG/
├── mitbih_database/              # CSV + аннотации MIT-BIH (нужно скачать)
│   ├── 100.csv
│   ├── 100annotations.txt
│   └── ...
├── models/                       # Промежуточные .npz файлы (создаются ноутбуками)
│   ├── mitbih_preprocessed.npz   # Шаг 1: сегментированные beats
│   └── mitbih_balanced.npz       # Шаг 2: train/test + варианты балансировки
├── src/
│   └── main.py                   # Вся логика предобработки (используется в main.ipynb)
├── main.ipynb                    # Шаг 1: предобработка и сегментация
├── EliminatingClassImbalances.ipynb  # Шаг 2: борьба с дисбалансом
├── BuildingModel.ipynb           # Шаг 3: обучение 1D ResNet
├── best_ecg_model.pth            # Лучший чекпойнт (по val macro-F1)
├── ecg_model_final.pth           # Финальная модель + метаданные и история
├── requirements.txt              # Python-зависимости
└── README.md
```

---

## - Требования и установка

### Зависимости
- Python 3.9+
- numpy, pandas, scipy, matplotlib, seaborn, tqdm
- scikit-learn, imbalanced-learn
- PyTorch (CPU или CUDA)

### Установка
```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Если `requirements.txt` отсутствует, можно установить вручную:
```bash
pip install numpy pandas scipy matplotlib seaborn tqdm scikit-learn imbalanced-learn torch
```

---

## - Подготовка данных

1. Скачайте MIT-BIH Arrhythmia Database (CSV-версию) с [Kaggle](https://www.kaggle.com/datasets/taejoongyoon/mitbit-arrhythmia-database) или сконвертируйте оригинальные `.dat/.hea/.atr` с помощью `wfdb`.
2. Поместите файлы в каталог `mitbih_database/`.
3. Для каждой записи (например, `100`) должны быть:
   - `100.csv` - сигнал (колонки: `sample #`, `MLII`, `V5`)
   - `100annotations.txt` - R-пики и метки (формат: `Time  Sample#  Type ...`)

Частота дискретизации - **360 Hz**, базовая шкала ADC - `gain=200 ADU/mV`, `base=1024`.

---

## - Запуск пайплайна

Запускайте ноутбуки **строго по порядку** - каждый следующий читает результат предыдущего.

### Шаг 1 - Предобработка: `main.ipynb`
Делает то же, что и `src/main.py`:
- Загружает каждую запись MIT-BIH, переводит ADC - мВ.
- Применяет **band-pass 0.5-40 Гц** (Butterworth, order=4) и **notch 50 Гц** для подавления дрейфа и сетевой наводки.
- Выполняет **Z-score нормализацию**.
- Сегментирует beats в окне `[R-100, R+150]` - 250 отсчётов (~694 мс).
- Мэппит исходные символы MIT-BIH в 5 классов AAMI.
- Сохраняет результат в `models/mitbih_preprocessed.npz`:

| Ключ   | Shape | Описание |
|--------|-------|----------|
| `X`    | `(N_beats, 250)` | Нормализованные ЭКГ-сегменты (float32) |
| `y`    | `(N_beats,)`     | Метки классов 0..4 (int64) |
| `pids` | `(N_beats,)`     | ID пациента (int32), нужен для группового сплита |

### Шаг 2 - Балансировка: `EliminatingClassImbalances.ipynb`
Класс **N ≈ 90%** всех beats - без балансировки модель деградирует на миноритарных классах.
Ноутбук:
- Делит данные на train/test **по пациентам** (без утечки) с помощью `pids`.
- Генерирует **три варианта** обучающей выборки:
  1. **`weighted`** - исходные несбалансированные данные + веса классов в `CrossEntropyLoss`.
  2. **`smote`** - синтетические примеры через `imblearn.SMOTE`.
  3. **`aug`** - физически осмысленные аугментации ЭКГ (гауссов шум, временной сдвиг, амплитудное масштабирование, дрейф изолинии через `CubicSpline` и т.п.). **Рекомендуется как основной вариант.**
- Сохраняет всё в `models/mitbih_balanced.npz`:

| Ключ | Описание |
|------|----------|
| `X_train`, `y_train` | Исходная несбалансированная train-выборка |
| `X_train_smote`, `y_train_smote` | После SMOTE |
| `X_train_aug`, `y_train_aug` | После аугментаций |
| `X_test`, `y_test` | Тестовая выборка (**никогда не балансируется**) |
| `class_weights` | Веса классов для loss |
| `sample_weights` | Веса примеров для `WeightedRandomSampler` |
| `class_names`, `fs` | Метаинформация |

### Шаг 3 - Обучение модели: `BuildingModel.ipynb`
Архитектура - **1D ResNet** (`ECGResNet`):
- Stem-свёртка (`Conv1D`).
- 3 residual-блока с возрастающим числом каналов: **32 - 64 - 128**.
- **Global Average Pooling** + Dropout + Linear-классификатор.

Обучение:
- Выбор стратегии: `STRATEGY = 'aug' | 'smote' | 'weighted'`.
- 15% обучающей выборки - стратифицированная валидация.
- **Loss:** `CrossEntropyLoss` (с весами для `weighted`).
- **Optimizer:** Adam.
- **Scheduler:** `ReduceLROnPlateau` по val-loss.
- **Early Stopping** по **macro-F1** на валидации (patience=7, до 30 эпох).

Оценка:
- Accuracy + **macro-F1** на тесте.
- Confusion matrix (абсолютная и нормированная).
- Per-class precision / recall / F1.
- Визуализация правильных и ошибочных предсказаний.

Артефакты:
- `best_ecg_model.pth` - лучший чекпойнт по val-F1 (сохраняется в ходе обучения).
- `ecg_model_final.pth` - финальная модель с метаданными (`model_config`, `class_names`, `strategy`, `test_metrics`, `history`).

---

## - Использование обученной модели

```python
import numpy as np
import torch

# 1. Загрузить чекпойнт
ckpt = torch.load('ecg_model_final.pth', map_location='cpu', weights_only=False)
print(ckpt['test_metrics'])      # accuracy, macro_f1, loss
print(ckpt['class_names'])       # ['N', 'S', 'V', 'F', 'Q']

# 2. Создать модель той же архитектуры (определена в BuildingModel.ipynb)
#    и подгрузить веса:
# model = ECGResNet(**ckpt['model_config'])
# model.load_state_dict(ckpt['model_state_dict'])
# model.eval()

# 3. Подать сегмент ЭКГ: shape (1, 1, 250), float32, z-нормализованный
# beat = torch.from_numpy(X[0:1]).float().unsqueeze(1)
# with torch.no_grad():
#     logits = model(beat)
#     pred = logits.argmax(dim=1).item()
# print(ckpt['class_names'][pred])
```

Загрузка предобработанных данных:
```python
import numpy as np
data = np.load('models/mitbih_preprocessed.npz')
X, y, pids = data['X'], data['y'], data['pids']
```

---

## - Замечания

- Параметры фильтров (0.5-40 Гц, notch 50 Гц) подобраны под `fs=360 Гц` MIT-BIH. При смене базы их нужно пересчитать.
- Сплит **по пациентам** (через `pids`) принципиален: иначе beats одного и того же пациента попадут и в train, и в test, и метрики будут завышены.
- Балансировку **никогда** не применять к тестовой выборке - она должна отражать реальное распределение.
- AAMI-мэппинг соответствует стандартной рекомендации для оценки алгоритмов классификации аритмий.

---

## - Ссылки

- **MIT-BIH Arrhythmia Database** - https://www.kaggle.com/datasets/taejoongyoon/mitbit-arrhythmia-database
- **AAMI EC57** - стандарт оценки алгоритмов детекции аритмий.
- **SMOTE** - Enhancing model accuracy with SMOTE oversampling techniques, 2022.
- **ResNet** - He et al. (2016).

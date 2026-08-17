# ECG Beat Classification - MIT-BIH (AAMI 5 классов)

End-to-end пайплайн для классификации сердечных сокращений на сигналах ЭКГ из **MIT-BIH Arrhythmia Database** по стандартным классам **AAMI** (N, S, V, F, Q).

Проект состоит из трёх последовательных Jupyter-ноутбуков и одного итогового:

1. **Предобработка сигналов** - фильтрация, нормализация, сегментация вокруг R-пиков и расчёт RR-признаков.
2. **Устранение дисбаланса классов** - SMOTE, аугментация сигнала, взвешенный loss / sampler.
3. **Обучение 1D-CNN с residual-блоками** - двухветвевая модель (сигнал + RR-признаки), обучение, метрики на тесте и сохранение модели.
4. **Финальный ноутбук** - рефакторинг пайплайна обучения и **инференс модели для пользователя** (проверка на своих данных через `main_test()`).

> **Примечание по классам:** фактический набор классов берётся из файла данных (`class_names`). В текущем `mitbih_balanced.npz` используются **4 класса** (`N, S, V, F`) - класс `Q` отсутствует в сбалансированном наборе.

---

## Классы AAMI

| Код | Название | Описание |
|-----|----------|----------|
| **N** | Normal | Нормальные сокращения |
| **S** | Supraventricular ectopic | Наджелудочковые экстрасистолы |
| **V** | Ventricular ectopic | Желудочковые экстрасистолы |
| **F** | Fusion | Сливные комплексы |

---

## - Структура проекта

```
ECG/
├── mitbih_database/                  # CSV + аннотации MIT-BIH (нужно скачать)
│   ├── 100.csv
│   ├── 100annotations.txt
│   └── ...
├── models/                           # Промежуточные .npz файлы (создаются ноутбуками)
│   ├── mitbih_preprocessed.npz       # Шаг 1: сегменты beats + RR-признаки
│   └── mitbih_balanced.npz           # Шаг 2: сплиты + варианты балансировки
├── main.ipynb                        # Шаг 1: предобработка и сегментация
├── EliminatingClassImbalances.ipynb  # Шаг 2: борьба с дисбалансом
├── BuildingModel.ipynb               # Шаг 3: обучение 1D ResNet
├── Model.ipynb                       # Шаг 4: финальный пайплайн + инференс
├── make_my_ecg.py                    # Генерация my_ecg.npz для проверки модели
├── my_ecg.npz                        # Пример файла для main_test() (X, rr, y_true)
├── best_ecg_model.pth                # Лучший чекпойнт последнего запуска (val macro-F1)
├── best_ecg_model_aug.pth            # Лучшие чекпойнты по каждой из 3 стратегий
├── best_ecg_model_smote.pth
├── best_ecg_model_weighted.pth
├── ecg_model_final.pth               # Финальная модель + метаданные и история
├── requirements.txt                  # Python-зависимости (зафиксированные версии)
└── README.md
```

Тяжёлые артефакты (`mitbih_database/`, `models/`, `notebooks/`, `src/`, данные и картинки) исключены из git через `.gitignore`.

---

## - Требования и установка

### Зависимости
- Python 3.9+
- numpy, pandas, scipy, matplotlib, seaborn, tqdm
- scikit-learn, imbalanced-learn, wfdb
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

В `requirements.txt` зафиксированы конкретные версии (numpy 1.26.4, torch 2.3.1 и т.д.). Если файла нет, можно установить вручную:
```bash
pip install numpy pandas scipy matplotlib seaborn tqdm scikit-learn imbalanced-learn wfdb torch
```

---

## - Подготовка данных

1. Скачайте MIT-BIH Arrhythmia Database (CSV-версию) с [Kaggle](https://www.kaggle.com/datasets/taejoongyoon/mitbit-arrhythmia-database) или сконвертируйте оригинальные `.dat/.hea/.atr` с помощью `wfdb`.
2. Поместите файлы в каталог `mitbih_database/`.
3. Для каждой записи (например, `100`) должны быть:
   - `100.csv` - сигнал (колонки: `sample #`, `MLII`, `V5`);
   - `100annotations.txt` - R-пики и метки (формат: `Time  Sample#  Type ...`).

Частота дискретизации - **360 Hz**, базовая шкала ADC - `gain=200 ADU/mV`, `base=1024`.

---

## - Запуск пайплайна

Запускайте ноутбуки **строго по порядку** - каждый следующий читает результат предыдущего.

### Шаг 1 - Предобработка: `main.ipynb`

- Загружает каждую запись MIT-BIH, переводит ADC в мВ.
- Применяет **band-pass 0.5-40 Гц** (Butterworth, order=4) и **notch 50 Гц** для подавления дрейфа и сетевой наводки.
- Выполняет **Z-score нормализацию**.
- Сегментирует beats в окне `[R-100, R+150]` - 250 отсчётов (~694 мс).
- Считает для каждого сегмента **6 RR-признаков**: предыдущий и следующий RR-интервалы, локальный RR (окно +/-10 ударов), глобальный RR по пациенту и два отношения к локальному RR.
- Мэппит исходные символы MIT-BIH в 5 классов AAMI.
- Сохраняет результат в `models/mitbih_preprocessed.npz`:

| Ключ   | Shape | Описание |
|--------|-------|----------|
| `X`    | `(N_beats, 250)` | Нормализованные сегменты (float32) |
| `y`    | `(N_beats,)`     | Метки классов (int64) |
| `X_rr` | `(N_beats, 6)`   | Сырые RR-признаки (float32) |
| `pids` | `(N_beats,)`     | ID пациентов (int32), для межпациентного сплита |

### Шаг 2 - Балансировка: `EliminatingClassImbalances.ipynb`
Класс **N составляет ~90%** всех beats - без балансировки модель вырождается в предсказание N.
Что делается:
- Разделение данных на train / val / test **по пациентам** (стандартные списки AAMI DS1/DS2, через `pids`).
- Построение **трёх вариантов** балансировки обучающей выборки:
  1. **`weighted`** - без изменения данных, за счёт весов классов в `CrossEntropyLoss` и `WeightedRandomSampler`;
  2. **`smote`** - оверсэмплинг миноритарных классов через `imblearn.SMOTE` (RR-признаки интерполируются синхронно с сигналом);
  3. **`aug`** - аугментация сигналов (сдвиг во времени, масштабирование амплитуды, шум, дрейф изолинии и т.п.). **Рекомендуемая стратегия.**
- Сохраняет в `models/mitbih_balanced.npz`:

| Ключ | Описание |
|------|----------|
| `X_train`, `y_train`, `X_train_rr` | Сырой (несбалансированный) train |
| `X_train_smote`, `y_train_smote`, `X_train_smote_rr` | Вариант после SMOTE |
| `X_train_aug`, `y_train_aug`, `X_train_aug_rr` | Вариант после аугментаций |
| `X_val`, `y_val`, `X_val_rr` | Валидация (честный межпациентный сплит, **не балансируется**) |
| `X_test`, `y_test`, `X_test_rr` | Тест (**не балансируется**) |
| `class_weights` | Веса классов для loss |
| `sample_weights` | Веса примеров для `WeightedRandomSampler` |
| `rr_mean`, `rr_scale` | Статистики нормализации RR-признаков (посчитаны по train) |
| `class_names`, `fs` | Метаинформация |

---

### Шаг 3 - Обучение модели: `BuildingModel.ipynb`
Архитектура - **двухветвевой 1D ResNet** (`ECGResNet`):
- Ветка сигнала: Stem-свёртка (`Conv1D`) + MaxPool, 3 residual-блока с возрастающим числом каналов **32 - 64 - 128**, Global Average Pooling.
- Ветка RR-признаков: Linear - BatchNorm - Dropout.
- Слияние веток и MLP-классификатор.

RR-признаки нормализуются сохранёнными `rr_mean`/`rr_scale` и клиппятся до +/-6.

Обучение:
- Выбор стратегии: `STRATEGY = 'aug' | 'smote' | 'weighted'`.
- Валидация - фиксированный межпациентный `X_val` из файла данных (заново не сплитуется).
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
- `best_ecg_model_<strategy>.pth` - лучшие чекпойнты по val-F1 для каждой из трёх стратегий; `best_ecg_model.pth` - чекпойнт последнего запуска.
- `ecg_model_final.pth` - финальная модель с метаданными: `model_state_dict`, `model_class`, `model_config`, `class_names`, `rr_mean`, `rr_std`, `rr_clip`, `strategy`, `test_metrics`, `history`.

Текущая финальная модель обучена со стратегией `weighted`; её метрики на тесте: accuracy 0.820, macro-F1 0.601 (полные значения - в ключе `test_metrics` чекпоинта).

---

### Шаг 4 - Итоговый ноутбук: `Model.ipynb`
Рефакторинг `BuildingModel.ipynb`:

- весь код разбит на чистые функции (`load_data` - `clean_data` - `preprocess_data` - `create_loaders` - `get_model` - `train_model` - `plot_history` - `test_model` - `save_final_model`);
- главная функция обучения `run_training_pipeline()`;
- главная функция для пользователя `main_test()` - в последней ячейке;
- финальная модель сохраняет `rr_mean`/`rr_std`/`rr_clip` для согласованного инференса.

#### Проверка модели на своих данных (для пользователя)
Поместите рядом с `Model.ipynb` файлы `mitbih_balanced.npz` (для обучения) и `ecg_model_final.pth` (для инференса). Затем:

1. Выполните все ячейки (Runtime - Run all или Ctrl+F9).
2. В последней ячейке запустится `main_test()`:
   - введите путь к вашему файлу (в Google Colab можно просто нажать Enter и выбрать файл через диалог загрузки);
   - поддерживаемые форматы: `.npz` (ключи `X` с формой `(N, 250)` и опционально `rr` с формой `(N, 6)`), `.npy` (массив сигналов), `.csv` (колонки 0-249 - сигнал, 250-255 - RR-признаки);
   - ноутбук выполнит предобработку, предсказание и выведет понятный текстовый результат с уверенностью модели и графиком сигнала.
3. Если в файле есть истинные метки (`y_true`), доступен количественный контроль `evaluate_labeled_file('my_ecg.npz')` - таблица 'истина/предсказание', accuracy и сетка сигналов (зелёные - верные ответы, красные - ошибки).

Готовый тестовый файл `my_ecg.npz` можно сгенерировать скриптом:
```bash
python make_my_ecg.py --per_class 3
```
Он отбирает по несколько сегментов каждого класса из `models/mitbih_preprocessed.npz` и записывает ключи `X`, `rr`, `y_true` (RR-признаки строятся в том же формате, что и при обучении).

---

## - Использование обученной модели

```python
import numpy as np
import torch

# 1. Загрузка чекпоинта
ckpt = torch.load('ecg_model_final.pth', map_location='cpu', weights_only=False)
print(ckpt['test_metrics'])   # accuracy, macro_f1, loss
print(ckpt['class_names'])    # например ['N', 'S', 'V', 'F']

# 2. Создание модели по конфигу (класс ECGResNet - из Model.ipynb)
# model = ECGResNet(**ckpt['model_config'])
# model.load_state_dict(ckpt['model_state_dict'])
# model.eval()

# 3. Предсказание для одного сегмента: сигнал (1, 1, 250), RR (1, 6), float32.
#    RR-признаки нормализуются статистиками из чекпоинта:
# rr_vec = (rr_raw - ckpt['rr_mean']) / ckpt['rr_std']
# rr_vec = np.clip(rr_vec, -ckpt['rr_clip'], ckpt['rr_clip'])
# beat = torch.from_numpy(X[0:1]).float().unsqueeze(1)
# rr   = torch.from_numpy(rr_vec[0:1]).float()
# with torch.no_grad():
#     logits = model(beat, rr)
#     pred = logits.argmax(dim=1).item()
# print(ckpt['class_names'][pred])
```

Доступ к сегментированным данным:
```python
import numpy as np
data = np.load('models/mitbih_preprocessed.npz')
X, y, rr, pids = data['X'], data['y'], data['X_rr'], data['pids']
```

---


---

## - Замечания

- Параметры фильтров (0.5-40 Гц, notch 50 Гц) подобраны под `fs=360 Гц` MIT-BIH. При смене базы их нужно пересчитать.
- Сплит **по пациентам** (через `pids`, стандартные списки AAMI DS1/DS2) принципиален: иначе beats одного и того же пациента попадут и в train, и в test, и метрики будут завышены.
- Балансировку **никогда** не применять к валидационной и тестовой выборкам - они должны отражать реальное распределение.
- RR-признаки нормализуются статистиками `rr_mean`/`rr_scale`, посчитанными только по train, и клиппятся до +/-6; эти статистики сохранены и в файле данных, и в финальной модели.
- AAMI-мэппинг соответствует стандартной рекомендации для оценки алгоритмов классификации аритмий.

---

## - Ссылки

- **MIT-BIH Arrhythmia Database** - https://www.kaggle.com/datasets/taejoongyoon/mitbit-arrhythmia-database
- **AAMI EC57** - стандарт оценки алгоритмов детекции аритмий.
- **SMOTE** - Enhancing model accuracy with SMOTE oversampling techniques, 2022.
- **ResNet** - He et al. (2016).

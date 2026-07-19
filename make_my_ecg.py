# make_my_ecg.py — формирование my_ecg.npz для проверки модели (Model.ipynb -> main_test)
import argparse
import numpy as np
import torch

LOCAL_WINDOW = 10  # ударов в каждую сторону для local RR


def get_checkpoint_info(checkpoint_path='ecg_model_final.pth'):
    """Возвращает (class_names, n_rr_features) из чекпоинта модели."""
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        class_names = list(ckpt.get('class_names', ['N', 'S', 'V', 'F']))
        cfg = ckpt.get('model_config', {})
        n_rr = cfg.get('rr_features', None)
        if n_rr is None and 'rr_mean' in ckpt:          # запасной вариант
            n_rr = int(np.asarray(ckpt['rr_mean']).shape[0])
        return class_names, (n_rr or 6)
    except Exception:
        return ['N', 'S', 'V', 'F'], 6


def build_rr_matrix(r_positions, pids, fs=360, local_window=LOCAL_WINDOW):
    """
    6 СЫРЫХ RR-признаков (секунды и относительные величины), порядок как при обучении:
      [pre_rr, post_rr, local_rr, global_rr, pre_rr/local_rr, post_rr/local_rr]
    r_positions: (N,) индексы R-пиков в исходной записи; pids: (N,) id пациентов.
    """
    n = len(pids)
    pre_rr = np.zeros(n)
    post_rr = np.zeros(n)
    for i in range(n):
        if i > 0 and pids[i] == pids[i - 1]:
            pre_rr[i] = r_positions[i] - r_positions[i - 1]
        if i < n - 1 and pids[i] == pids[i + 1]:
            post_rr[i] = r_positions[i + 1] - r_positions[i]

    pre_rr = pre_rr / fs
    post_rr = post_rr / fs

    local_rr = np.zeros(n)
    for i in range(n):
        st = max(0, i - local_window)
        en = min(n - 1, i + local_window)
        diffs = [r_positions[j] - r_positions[j - 1]
                 for j in range(st, en + 1)
                 if j > 0 and pids[j] == pids[j - 1]]
        local_rr[i] = np.mean(diffs) / fs if diffs else 0.0

    global_rr = np.zeros(n)
    for p in np.unique(pids):
        idx = np.where(pids == p)[0]
        global_rr[idx] = np.mean(np.diff(r_positions[idx])) / fs if len(idx) > 1 else 0.0

    # заполнение нулей (границы пациентов) медианой
    def fill_zero(a):
        m = a != 0
        a[~m] = np.median(a[m]) if m.any() else 0.0
        return a

    pre_rr, post_rr, local_rr, global_rr = map(fill_zero, (pre_rr, post_rr, local_rr, global_rr))

    eps = 1e-8
    ratio_pre = pre_rr / (local_rr + eps)
    ratio_post = post_rr / (local_rr + eps)

    return np.stack([pre_rr, post_rr, local_rr, global_rr, ratio_pre, ratio_post],
                    axis=1).astype(np.float32)


def compute_rr_features(r_positions, patient_ids, fs=360):
    """r_positions: (N,) индексы R-пиков. Возвращает (N, 6) сырые RR-признаки."""
    return build_rr_matrix(np.asarray(r_positions, dtype=np.float64),
                           np.asarray(patient_ids), fs)


def make_my_ecg(source_path='models/mitbih_preprocessed.npz',
                out_path='my_ecg.npz',
                per_class=3,
                seed=42,
                fs=360,
                checkpoint_path='ecg_model_final.pth'):
    """Отбирает по per_class сегментов на класс и сохраняет my_ecg.npz (X, rr, y_true)."""
    rng = np.random.default_rng(seed)
    class_names, n_rr = get_checkpoint_info(checkpoint_path)
    print(f'Чекпоинт: классы={class_names}, RR-признаков={n_rr}')

    data = np.load(source_path)
    X, y, pids = data['X'], data['y'], data['pids']
    print(f'Источник: {source_path} — {len(y)} сегментов')

    sel_idx = []
    for cls_idx, name in enumerate(class_names):
        pool = np.where(y == cls_idx)[0]
        take = min(per_class, len(pool))
        sel_idx.extend(rng.choice(pool, take, replace=False))
    sel_idx = np.array(sel_idx)

    # позиции R-пиков для выбранных сегментов неизвестны — восстанавливаем порядок:
    # считаем RR внутри каждого пациента по порядку сегментов (R-пик в фикс. индексе 100)
    r_positions = np.arange(len(X)) * 250 + 100  # монотонные псевдо-позиции
    rr_raw = compute_rr_features(r_positions[sel_idx], pids[sel_idx], fs)

    np.savez_compressed(out_path,
                        X=X[sel_idx].astype(np.float32),
                        rr=rr_raw,
                        y_true=y[sel_idx].astype(np.int64))
    print(f'Сохранено: {out_path} — {len(sel_idx)} сегментов, RR shape={rr_raw.shape}')
    print('Проверка: python -c "import numpy as np; d=np.load(\'my_ecg.npz\'); '
          'print(d[\'X\'].shape, d[\'rr\'].shape)"')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Создать my_ecg.npz для main_test()')
    ap.add_argument('--source', default='models/mitbih_preprocessed.npz')
    ap.add_argument('--out', default='my_ecg.npz')
    ap.add_argument('--per_class', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    make_my_ecg(source_path=args.source, out_path=args.out,
                per_class=args.per_class, seed=args.seed)
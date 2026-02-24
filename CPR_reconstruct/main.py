import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
from scipy.spatial.distance import cosine  # 导入余弦距离计算函数

# 字体（可选）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data(path='taxi_6m.xlsx'):
    try:
        df = pd.read_excel(path)
        if 'value' in df.columns:
            ori = df['value'].values.astype(float)
        else:
            ori = df.iloc[:, 0].values.astype(float)
        mn, mx = ori.min(), ori.max()
        ori_norm = (ori - mn) / (mx - mn + 1e-12)
        print(f"Loaded {path}, length={len(ori_norm)}")
    except Exception as e:
        print(f"Warning: cannot read {path} ({e}), using simulated data")
        t = np.linspace(0, 10 * np.pi, 1000)
        ori_norm = (np.sin(t * 2) + 1) / 2 + np.random.normal(0, 0.5, len(t))
        ori_norm = np.clip(ori_norm, 0, 1)
    return ori_norm


# -------------------------
# SW 机制
# -------------------------
def sw_noisy_samples(ori_samples, l, h, eps, rng):
    ee = np.exp(eps)
    denom = (2 * ee * (ee - 1 - eps))
    if abs(denom) < 1e-12:
        w = 0.5
    else:
        w = ((eps * ee) - ee + 1) / denom * 2
    if not np.isfinite(w) or w <= 0:
        w = 0.5
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)

    ori = np.asarray(ori_samples, dtype=float)
    s = (ori - l) / (h - l + 1e-12)  # 归一化到 [0,1]
    r = rng.uniform(0.0, 1.0, size=len(s))

    noisy = np.empty_like(s)
    bound1 = q * s
    bound2 = q * s + p * w

    idx1 = r <= bound1
    idx2 = (r > bound1) & (r <= bound2)
    idx3 = r > bound2

    if np.any(idx1):
        noisy[idx1] = r[idx1] / q - w / 2
    if np.any(idx2):
        noisy[idx2] = (r[idx2] - q * s[idx2]) / p + s[idx2] - w / 2
    if np.any(idx3):
        noisy[idx3] = (r[idx3] - q * s[idx3] - p * w) / q + s[idx3] + w / 2

    noisy = np.clip(noisy, -1.0, 2.0)
    return noisy, w, p, q


# -------------------------
# 周期检测（基于 FFT）
# -------------------------
def detect_period_fft(data, min_period=10, max_period=500):
    data = np.asarray(data)
    if len(data) < 3:
        return 1
    data = data - np.mean(data)
    n = len(data)
    fft = np.fft.fft(data)
    power = np.abs(fft) ** 2
    power[0] = 0
    idx = np.arange(1, n // 2)
    periods = n / idx
    mask = (periods >= min_period) & (periods <= max_period)
    if not np.any(mask):
        return 1
    idx_masked = idx[mask]
    main_idx = idx_masked[np.argmax(power[idx_masked])]
    period = int(round(n / main_idx))
    if period <= 0:
        return 1
    return period


# -------------------------
# 多窗口周期检测（用于 CPR）
# -------------------------
def multi_window_period_detection(signal, window_sizes, min_period=10, max_period=500):
    periods = []
    for w in window_sizes:
        if len(signal) < w or w < min_period:
            period = None
        else:
            seg = signal[:w]
            period = detect_period_fft(seg, min_period=min_period, max_period=max_period)
        periods.append(period)
    valid = [p for p in periods if p is not None and p > 0]
    if not valid:
        return 1, periods
    most_common = Counter(valid).most_common(1)[0][0]
    return most_common, periods


# -------------------------
# CPR 重构实现
# -------------------------
def detect_two_periods(signal, min_period=10, max_period=500, max_candidates=6):
    x = np.asarray(signal, dtype=float)
    n = len(x)
    if n < 3:
        return [1], [1.0]
    x = x - np.mean(x)
    fft = np.fft.fft(x)
    power = np.abs(fft) ** 2
    power[0] = 0
    idx = np.arange(1, n // 2)
    periods = n / idx
    mask = (periods >= min_period) & (periods <= max_period)
    if not np.any(mask):
        return [1], [1.0]

    idx_masked = idx[mask]
    power_masked = power[idx_masked]
    # 取若干个候选峰，按能量排序
    cand_order = np.argsort(power_masked)[::-1]
    candidates = []
    weights = []
    for ci in cand_order[:max_candidates]:
        main_idx = idx_masked[ci]
        p = int(round(n / main_idx))
        # 过滤重复或过接近的周期
        if any(abs(p - pc) <= 1 for pc in candidates):
            continue
        candidates.append(p)
        weights.append(power_masked[ci])
        if len(candidates) >= 2:
            break

    if len(candidates) == 0:
        return [1], [1.0]
    if len(candidates) == 1:
        return [candidates[0]], [float(weights[0])]

    w = np.array(weights, dtype=float)
    w = w / (np.sum(w) + 1e-12)
    return candidates, w.tolist()


def cpr_reconstruct(noisy_sw,
                    kde_xpoints=100,
                    min_period=10,
                    max_period=500,
                    max_components=2):
    signal = np.asarray(noisy_sw)
    n = len(signal)
    if n == 0:
        return signal.copy()
    periods, weights = detect_two_periods(signal, min_period=min_period, max_period=max_period)
    periods = periods[:max_components]
    weights = weights[:max_components]
    components = []
    comp_weights = []

    for i, p in enumerate(periods):
        if p <= 1 or p > n:
            continue
        A = [[] for _ in range(p)]
        for idx, val in enumerate(signal):
            A[idx % p].append(val)

        reconstructed_values = np.zeros(p)
        for phase in range(p):
            vals = A[phase]
            if len(vals) == 0:
                reconstructed_values[phase] = 0.0
                continue
            try:
                # 若所有值相同或太少，gaussian_kde 可能失败，退回到中位数
                if len(vals) < 5 or np.allclose(vals, vals[0]):
                    reconstructed_values[phase] = float(np.median(vals))
                else:
                    kde = stats.gaussian_kde(vals)
                    x = np.linspace(min(vals), max(vals), kde_xpoints)
                    dens = kde(x)
                    reconstructed_values[phase] = float(x[np.argmax(dens)])
            except Exception:
                # 回退策略
                reconstructed_values[phase] = float(np.median(vals))

        full_recon = np.tile(reconstructed_values, n // p + 1)[:n]
        components.append(full_recon)
        comp_weights.append(weights[i] if i < len(weights) else 1.0)

    if len(components) == 0:
        return signal.copy()

    if len(components) == 1:
        return components[0]

    # 多分量按权重合成（权重归一化）
    comp_weights = np.array(comp_weights, dtype=float)
    if np.sum(comp_weights) <= 0:
        comp_weights = np.ones_like(comp_weights)
    comp_weights = comp_weights / (np.sum(comp_weights) + 1e-12)

    combined = np.zeros(n, dtype=float)
    for comp, w in zip(components, comp_weights):
        combined += w * np.asarray(comp, dtype=float)

    return combined


# -------------------------
# 余弦距离计算
# -------------------------
def calculate_cosine_similarity(a, b):

    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = np.dot(a, b) / (norm_a * norm_b)
    return similarity


# -------------------------
# 主程序
# -------------------------
if __name__ == "__main__":
    ori_norm = load_data()  # 归一化数据
    epsilons = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    rng = np.random.default_rng(42)

    l = 0.0
    h = 1.0

    print("=" * 60)
    print(f"{'Epsilon':<10} {'余弦相似度':<15} {'余弦距离':<15}")
    print("-" * 60)

    for eps in epsilons:
        noisy_sw, w, p, q = sw_noisy_samples(ori_norm, l, h, eps, rng)
        cpr_result = cpr_reconstruct(noisy_sw)
        cos_sim = calculate_cosine_similarity(ori_norm, cpr_result)
        cos_dist = 1 - cos_sim
        print(f"{eps:<10.1f} {cos_sim:<15.6f} {cos_dist:<15.6f}")

    print("=" * 60)
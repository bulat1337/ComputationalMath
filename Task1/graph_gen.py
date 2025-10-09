import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os
import glob
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import math

# --- Настройки ---
DATA_DIR = 'data'
OUT_DIR = 'graphs'
os.makedirs(OUT_DIR, exist_ok=True)
DO_FIT = False   # True = строить линейную аппроксимацию, False = только точки


# Модель
def linear_func(x, a, b):
    return a * x + b

# Варианты имён колонок (можно расширять)
X_CANDIDATES = ["sqrt(f)", "sqrt(f) ", "x", "X", "sqrt_f"]
Y_CANDIDATES = ["\\alpha, {m}^{-1}", "alpha", "alpha, m^-1", "alpha (m^-1)"]
XERR_CANDIDATES = ["\\sigma_{sqrt(f)}", "sigma_sqrt(f)", "xerr", "sqrt(f)_err"]
YERR_CANDIDATES = ["\\sigma_{alpha}", "sigma_alpha", "yerr", "alpha_err"]


def find_column(df, candidates):
    """Найти имя столбца в DataFrame по списку возможных вариантов.
    Если DataFrame имеет ровно 2 колонки, вызывающая логика должна
    предпочесть позиционное соответствие (см. обработку ниже).
    """
    cols = list(df.columns)
    cols_strip = {c.strip(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.strip() in cols_strip:
            return cols_strip[cand.strip()]
    # Попытка частичного совпадения (игнор регистра и пробелов)
    low = {c.lower().replace(" ", "").replace("_", ""): c for c in cols}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in low:
            return low[key]
    return None


def try_read_csv(path):
    """Попытаться прочитать CSV с разными разделителями/десятичными символами."""
    # Пробуем наборы (sep, decimal). При желании расширить.
    for sep, dec in [(";", ","), (",", "."), (",", ","), ("\t", ".")]:
        try:
            df = pd.read_csv(path, sep=sep, decimal=dec)
            if df.shape[1] >= 1:
                return df
        except Exception:
            continue
    # как последняя попытка - позволим pandas догадаться
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise ValueError(f"Не удалось прочитать CSV: {path}  ({e})")


def autolimits(values, pad_fraction=0.07):
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if np.isfinite(vmin) and np.isfinite(vmax):
        if math.isclose(vmin, vmax):
            delta = abs(vmin) * 0.01 if vmin != 0 else 0.1
            return vmin - delta, vmax + delta
        rng = vmax - vmin
        pad = rng * pad_fraction
        return vmin - pad, vmax + pad
    return None, None


def smart_ticks(ax, axis='x', n_points=10):
    if axis == 'x':
        ax.xaxis.set_major_locator(MaxNLocator(nbins=max(3, min(10, int(n_points/2)))))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=max(3, min(10, int(n_points/2)))))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))


# --- Основная логика ---
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
if not csv_files:
    raise SystemExit("Нет CSV-файлов в папке 'data'.")

results = []

for file_path in csv_files:
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    df = try_read_csv(file_path)

    # Если в файле ровно 2 колонки — трактуем их как x и y по позиции
    if df.shape[1] == 2:
        x_col = df.columns[0]
        y_col = df.columns[1]
        xerr_col = None
        yerr_col = None
    else:
        # Иначе пытаемся найти по именам
        x_col = find_column(df, X_CANDIDATES)
        y_col = find_column(df, Y_CANDIDATES)
        xerr_col = find_column(df, XERR_CANDIDATES)
        yerr_col = find_column(df, YERR_CANDIDATES)

    if x_col is None or y_col is None:
        print(f"Пропущено {file_name}: не найдены колонки для x или y.")
        continue

    # Преобразуем в числовой массив
    x = pd.to_numeric(df[x_col], errors='coerce').to_numpy()
    y = pd.to_numeric(df[y_col], errors='coerce').to_numpy()
    x_err = pd.to_numeric(df[xerr_col], errors='coerce').to_numpy() if xerr_col else None
    y_err = pd.to_numeric(df[yerr_col], errors='coerce').to_numpy() if yerr_col else None

    # Маска валидных значений. Если ошибок нет — не требуем их
    mask = np.isfinite(x) & np.isfinite(y)
    if x_err is not None:
        mask &= np.isfinite(x_err)
    if y_err is not None:
        mask &= np.isfinite(y_err)

    x = x[mask]
    y = y[mask]
    x_err = x_err[mask] if x_err is not None else None
    y_err = y_err[mask] if y_err is not None else None

    n = len(x)
    if n < 2:
        print(f"{file_name}: недостаточно точек ({n}). Пропускаю.")
        continue

    # Подбор параметров. Если y_err присутствует и имеет положительные значения — используем его.
    try:
        if y_err is not None and np.any(np.isfinite(y_err)) and np.nanmax(y_err) > 0:
            params, cov = curve_fit(linear_func, x, y, sigma=y_err, absolute_sigma=True)
        else:
            params, cov = curve_fit(linear_func, x, y)
    except Exception as e:
        print(f"Fit failed for {file_name}: {e}")
        params = (np.nan, np.nan)
        cov = None

    slope, intercept = params
    if cov is not None and cov.shape == (2, 2):
        perr = np.sqrt(np.diag(cov))
        slope_err, intercept_err = perr[0], perr[1]
    else:
        slope_err, intercept_err = np.nan, np.nan

    # Настройка графика
    width = min(12, 5 + n / 30)
    height = max(3.5, width * 0.6)
    fig, ax = plt.subplots(figsize=(width, height))

    marker_size = float(max(3, min(8, 200.0 / math.sqrt(n))))
    line_width = max(1.0, 2.0 * (n / 50) ** -0.3)

    use_log_x = (np.all(x > 0) and (np.log10(np.max(x) / np.min(x)) >= 3))
    use_log_y = (np.all(y > 0) and (np.log10(np.max(y) / np.min(y)) >= 3))
    if use_log_x:
        ax.set_xscale('log')
    if use_log_y:
        ax.set_yscale('log')

    yerr_plot = None
    xerr_plot = None
    if y_err is not None and np.any(y_err > 0):
        yerr_plot = y_err
    if x_err is not None and np.any(x_err > 0):
        xerr_plot = x_err

    ax.errorbar(
        x, y,
        yerr=yerr_plot, xerr=xerr_plot,
        fmt='o', markersize=marker_size,
        markeredgewidth=0.6, markerfacecolor='none',
        ecolor='gray', elinewidth=0.9, capsize=2.5, label=file_name
    )

    x_fit = np.linspace(np.min(x), np.max(x), 200)
    if use_log_x:
        x_fit = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 200)
    y_fit = linear_func(x_fit, slope, intercept)
    
    if DO_FIT:
        ax.plot(x_fit, y_fit, linestyle='--', linewidth=line_width, label=f'Fit: y={slope:.3g}x + {intercept:.3g}')

    xlim = autolimits(x)
    ylim = autolimits(y)
    if xlim[0] is not None and ylim[0] is not None:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    tx = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    ty = ylim[0] + 0.95 * (ylim[1] - ylim[0])
    param_text = f"Slope = {slope:.4g} ± {slope_err:.4g}\nIntercept = {intercept:.4g} ± {intercept_err:.4g}"
    ax.text(tx, ty, param_text, fontsize=9, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

    smart_ticks(ax, 'x', n_points=n)
    smart_ticks(ax, 'y', n_points=n)

    ax.grid(which='both', linestyle='--', linewidth=0.4, alpha=0.8)
    ax.set_xlabel(f"{x_col}")
    ax.set_ylabel(f"{y_col}")
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()

    out_file = os.path.join(OUT_DIR, f"{file_name}.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

    results.append({
        "file": file_name,
        "n_points": n,
        "slope": slope, "slope_err": slope_err,
        "intercept": intercept, "intercept_err": intercept_err,
        "output": out_file
    })

    print(f"Saved plot: {out_file}  (points: {n})")

# Комбинированный график (если >1 файлов)
if len(results) > 1:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, res in enumerate(results):
        fp = os.path.join(DATA_DIR, res['file'] + '.csv')
        df = try_read_csv(fp)
        # Поддержка файлов с 2 колонками
        if df.shape[1] == 2:
            x_col = df.columns[0]; y_col = df.columns[1]
        else:
            x_col = find_column(df, X_CANDIDATES)
            y_col = find_column(df, Y_CANDIDATES)
        x = pd.to_numeric(df[x_col], errors='coerce').to_numpy()
        y = pd.to_numeric(df[y_col], errors='coerce').to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = len(x)
        marker_size = float(max(3, min(7, 150.0 / math.sqrt(max(1, n)))))
        ax.plot(x, y, marker='o', linestyle='None', markersize=marker_size,
                markeredgewidth=0.6, markerfacecolor='none', label=res['file'], color=colors[i % len(colors)])
        slope = res['slope']; intercept = res['intercept']
        x_fit = np.linspace(np.min(x), np.max(x), 200)
        y_fit = linear_func(x_fit, slope, intercept)
        ax.plot(x_fit, y_fit, linestyle='--', linewidth=1.2, color=colors[i % len(colors)])

    # Автолимиты по всем данным
    all_x = np.concatenate([pd.to_numeric(try_read_csv(os.path.join(DATA_DIR, r['file'] + '.csv'))[0 if try_read_csv(os.path.join(DATA_DIR, r['file'] + '.csv')).shape[1]==2 else find_column(try_read_csv(os.path.join(DATA_DIR, r['file'] + '.csv')), X_CANDIDATES)], errors='coerce').dropna().to_numpy() for r in results])
    all_y = np.concatenate([pd.to_numeric(try_read_csv(os.path.join(DATA_DIR, r['file'] + '.csv'))[1 if try_read_csv(os.path.join(DATA_DIR, r['file'] + '.csv')).shape[1]==2 else find_column(try_read_csv(os.path.join(DATA_DIR, r['file'] + '.csv')), Y_CANDIDATES)], errors='coerce').dropna().to_numpy() for r in results])
    ax.set_xlim(*autolimits(all_x))
    ax.set_ylim(*autolimits(all_y))

    smart_ticks(ax, 'x', n_points=sum(r['n_points'] for r in results))
    smart_ticks(ax, 'y', n_points=sum(r['n_points'] for r in results))
    ax.grid(which='both', linestyle='--', linewidth=0.4, alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    combo_out = os.path.join(OUT_DIR, "combined_plot.png")
    fig.savefig(combo_out, dpi=300)
    plt.close(fig)
    print(f"Saved combined plot: {combo_out}")

# Сводная таблица
if results:
    df_res = pd.DataFrame(results)
    csv_summary = os.path.join(OUT_DIR, "fit_summary.csv")
    df_res.to_csv(csv_summary, index=False)
    print("Summary saved to:", csv_summary)
    print(df_res[["file", "n_points", "slope", "slope_err", "intercept", "intercept_err"]].to_string(index=False))

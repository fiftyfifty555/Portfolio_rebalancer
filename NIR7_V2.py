# -*- coding: utf-8 -*-
"""
Панель онлайн-ребалансировки портфеля (UI сохранён в стиле ttkbootstrap 'superhero')
----------------------------------------------------------------------
Данные: HuggingFace dataset CSV:
    df = pd.read_csv("hf://datasets/siddharthmb/stocks-ohlcv/ohlcv.csv")

Ключевые фичи:
- Поддержка колонок датасета: date, act_symbol, open, high, low, close, volume
- Разделение 70/30 train/test по времени (хронологически)
- Кэширование по тикерам (после первого извлечения тикеров читать быстро)
- Прогресс по этапам (загрузка/индикаторы/ГА-A/ГА-B/метрики)
- Автодополнение тикеров (по локальному кэшу списка тикеров)
- Портфель НЕ long-only и НЕ full-invest (есть плечо, min/max, max активов, кэш)
- Двухуровневый пайплайн:
    A) Калибровка весов индикаторов ai (ГА) на прошлом окне
    B) Портфельная оптимизация w (многокрит. ГА, NSGA-II лайт)
"""

import os, sys, math, time, threading, queue, warnings, json
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# GUI
import tkinter as tk
from tkinter import messagebox
from ttkbootstrap import Style, ttk
from ttkbootstrap.constants import SUCCESS, INFO, WARNING

# Plot
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ------------------------ CONFIG / CONSTANTS ------------------------

HF_CSV_PATH = "hf://datasets/siddharthmb/stocks-ohlcv/ohlcv.csv"

CACHE_DIR = os.path.join(os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(), "cache_hf_ohlcv")
os.makedirs(CACHE_DIR, exist_ok=True)

TICKERS_CACHE_PATH = os.path.join(CACHE_DIR, "tickers_cache.json")

# Приведение колонок (HF -> внутренний стандарт)
COL_MAP = {
    "date": "Date",
    "act_symbol": "Symbol",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}

REQUIRED_INTERNAL = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]

# ------------------------ UTILS ------------------------

def in_notebook() -> bool:
    try:
        from IPython import get_ipython  # noqa
        ip = get_ipython()
        if ip is None:
            return False
        return True
    except Exception:
        return False

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def ensure_hf_dependency():
    """
    hf:// через pandas обычно требует huggingface_hub + fsspec.
    """
    try:
        import huggingface_hub  # noqa
        import fsspec  # noqa
        return True, ""
    except Exception as e:
        return False, (
            "Не найдены зависимости для чтения hf://.\n\n"
            "Установи:\n"
            "  pip install huggingface_hub fsspec\n\n"
            f"Техническая ошибка: {e}"
        )

def normalize_l1(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = np.sum(np.abs(vec))
    if s < eps:
        return vec.copy()
    return vec / s

def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0).replace(0, np.nan)
    z = (s - m) / sd
    return z.replace([np.inf, -np.inf], np.nan)

def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-12)
    return float(np.min(dd))  # отрицательное

def annualize_vol(daily_ret: np.ndarray, periods=252) -> float:
    return float(np.std(daily_ret, ddof=0) * np.sqrt(periods))

def sharpe_ratio(daily_ret: np.ndarray, periods=252, rf=0.0) -> float:
    mu = np.mean(daily_ret) - rf/periods
    sd = np.std(daily_ret, ddof=0)
    return float((mu * periods) / (sd * np.sqrt(periods) + 1e-12))

def cagr(equity: np.ndarray, days: int) -> float:
    if days <= 0:
        return 0.0
    years = days / 252.0
    if equity[-1] <= 0:
        return -1.0
    return float(equity[-1] ** (1/years) - 1)

def format_pct(x):
    return f"{x*100:.4f}"

# ------------------------ TOOLTIP ------------------------

class ToolTip:
    def __init__(self, widget, text, delay=450):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after_id = None
        self.tip = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)

    def _schedule(self, _=None):
        self._after_id = self.widget.after(self.delay, self._show)

    def _show(self):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        lbl = ttk.Label(self.tip, text=self.text, padding=8)
        lbl.pack()

    def _hide(self, _=None):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self.tip is not None:
            try:
                self.tip.destroy()
            except Exception:
                pass
            self.tip = None

# ------------------------ DATA LOADING + CACHING ------------------------

def _ticker_cache_file(sym: str) -> str:
    return os.path.join(CACHE_DIR, f"{sym}.parquet")

def load_tickers_cache() -> list:
    if os.path.isfile(TICKERS_CACHE_PATH):
        try:
            with open(TICKERS_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_tickers_cache(tickers: list):
    try:
        with open(TICKERS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(sorted(list(set(tickers))), f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def scan_unique_tickers_hf(progress_cb=None, log_cb=None, max_seconds=45):
    """
    Быстрый (насколько возможно) проход по датасету для извлечения уникальных тикеров.
    Делается редко: 1 раз, потом берём кэш.
    max_seconds — ограничитель, чтобы не зависать навечно.
    """
    ok, msg = ensure_hf_dependency()
    if not ok:
        raise RuntimeError(msg)

    start = time.time()
    tickers = set()
    usecols = ["act_symbol"]
    chunksize = 1_000_000

    if log_cb:
        log_cb("Сканирую список доступных тикеров (первичный проход)…", "INFO")

    # Важно: hf:// может быть медленным. Берём time-limit.
    it = pd.read_csv(HF_CSV_PATH, usecols=usecols, chunksize=chunksize)
    scanned = 0
    for chunk in it:
        chunk = chunk.dropna()
        tickers.update(chunk["act_symbol"].astype(str).unique().tolist())
        scanned += len(chunk)
        if progress_cb:
            # это не точный %, просто “живой” индикатор
            progress_cb(None, f"Сканирование тикеров… просмотрено строк: {scanned:,}".replace(",", " "))
        if time.time() - start > max_seconds:
            break

    out = sorted(list(tickers))
    if out and log_cb:
        log_cb(f"Найдено тикеров (частично/полностью): {len(out)}", "READY")
    return out

def load_data_for_pool(pool: list, use_cache: bool, progress_cb=None, log_cb=None) -> pd.DataFrame:
    """
    Возвращает DataFrame только по пулу тикеров.
    Стратегия:
      - если use_cache и для всех тикеров есть parquet -> собираем из кэша
      - иначе: читаем HF CSV чанками, фильтруем по пулу, сохраняем в кэш по тикерам
    """
    ok, msg = ensure_hf_dependency()
    if not ok:
        raise RuntimeError(msg)

    pool = [p.strip().upper() for p in pool if p.strip()]
    if not pool:
        raise RuntimeError("Пул тикеров пуст. Введите хотя бы один тикер.")

    # 1) попытка собрать из кэша
    if use_cache:
        cached = []
        missing = []
        for sym in pool:
            f = _ticker_cache_file(sym)
            if os.path.isfile(f):
                try:
                    d = pd.read_parquet(f)
                    cached.append(d)
                except Exception:
                    missing.append(sym)
            else:
                missing.append(sym)

        if cached and not missing:
            if log_cb:
                log_cb(f"Загружаю данные из кэша по тикерам: {', '.join(pool)}", "INFO")
            df = pd.concat(cached, ignore_index=True)
            return df

        if log_cb and cached:
            log_cb(f"Часть тикеров найдена в кэше, но нет: {', '.join(missing)}. Дочитаю из датасета…", "WARN")

    # 2) чтение HF чанками и фильтрация
    if log_cb:
        log_cb("Загружаю датасет HF и фильтрую по пулу тикеров (первый тяжёлый проход)…", "INFO")

    usecols = list(COL_MAP.keys())
    chunksize = 1_000_000

    kept_total = 0
    read_total = 0

    # для накопления по тикерам отдельно (чтобы сохранить кэш)
    per_sym = {sym: [] for sym in pool}

    it = pd.read_csv(HF_CSV_PATH, usecols=usecols, chunksize=chunksize)
    for chunk in it:
        read_total += len(chunk)
        # фильтр по пулу
        chunk["act_symbol"] = chunk["act_symbol"].astype(str).str.upper()
        sub = chunk[chunk["act_symbol"].isin(pool)].copy()
        kept_total += len(sub)

        if len(sub):
            # нормализация колонок
            sub = sub.rename(columns=COL_MAP)
            # приведение типов
            sub["Date"] = pd.to_datetime(sub["Date"], errors="coerce")
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")
            sub = sub.dropna(subset=["Date", "Close", "Symbol"]).sort_values(["Symbol", "Date"])
            # разбиваем по тикерам
            for sym, d_sym in sub.groupby("Symbol"):
                per_sym[sym].append(d_sym)

        if progress_cb:
            progress_cb(None, f"Прочитано строк: {read_total:,} | оставлено по выбранным тикерам: {kept_total:,}".replace(",", " "))

        # небольшая оптимизация: если уже много данных по всем тикерам, можно не останавливать.
        # но точного критерия нет, поэтому читаем до конца (иначе будут дырки).
        # Можно добавить ограничение по датам, но тут оставим корректность.

    # собрать итог
    frames = []
    for sym in pool:
        if per_sym[sym]:
            frames.append(pd.concat(per_sym[sym], ignore_index=True))
    if not frames:
        raise RuntimeError("По выбранным тикерам не найдено строк в датасете.")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Date", "Close"]).sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # сохранить кэш по тикерам
    try:
        for sym, d_sym in df.groupby("Symbol"):
            f = _ticker_cache_file(sym)
            d_sym.to_parquet(f, index=False)
    except Exception:
        # если pyarrow не установлен, parquet может не работать
        # тогда просто пропускаем кэширование
        if log_cb:
            log_cb("Не удалось записать parquet-кэш (возможно нет pyarrow). Продолжаю без кэша.", "WARN")

    # обновить кэш тикеров (хотя бы те, что встретились)
    try:
        save_tickers_cache(sorted(df["Symbol"].unique().tolist()))
    except Exception:
        pass

    return df

# ------------------------ INDICATORS (без повторения формул из 1 главы) ------------------------

def add_indicators_for_symbol(df_sym: pd.DataFrame) -> pd.DataFrame:
    """
    Здесь вычисления индикаторов. Формулы ты уже описал в аналитической части,
    поэтому код просто реализует расчёт (без пояснений в UI).
    """
    d = df_sym.copy()
    close = d["Close"]

    d["SMA20"] = close.rolling(20).mean()
    d["EMA20"] = close.ewm(span=20, adjust=False).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()

    delta = close.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    rs = up.rolling(14).mean() / (dn.rolling(14).mean() + 1e-12)
    d["RSI"] = 100 - 100 / (1 + rs)

    r20 = close.rolling(20)
    d["BB_upper"] = d["SMA20"] + 2 * r20.std(ddof=0)
    d["BB_lower"] = d["SMA20"] - 2 * r20.std(ddof=0)

    low14 = d["Low"].rolling(14).min()
    high14 = d["High"].rolling(14).max()
    d["STO_K"] = (close - low14) / (high14 - low14 + 1e-12) * 100
    d["STO_D"] = d["STO_K"].rolling(3).mean()

    hl = d["High"] - d["Low"]
    hc = (d["High"] - close.shift(1)).abs()
    lc = (d["Low"] - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()

    d["OBV"] = np.where(close > close.shift(1), d["Volume"],
                 np.where(close < close.shift(1), -d["Volume"], 0)).cumsum()

    return d

FEATURES = [
    "SMA20","EMA20","MACD","MACD_signal","RSI","BB_upper","BB_lower","STO_K","STO_D","ATR","OBV"
]

# ------------------------ GA: LEVEL A (weights for indicators per asset) ------------------------

def objective_level_A(df_sym: pd.DataFrame, weights: np.ndarray, z_window: int, horizon: int) -> float:
    """
    Цель калибровки A:
    максимизировать ранговую корреляцию Спирмена между score s_t и будущей доходностью r_{t+h}.
    Возвращаем fitness (больше = лучше).
    """
    d = df_sym.copy()
    # z-score для индикаторов
    Z = []
    for f in FEATURES:
        z = rolling_zscore(d[f], z_window)
        Z.append(z)
    Z = np.vstack([z.values for z in Z]).T  # shape [T, K]
    # score
    s = np.nansum(Z * weights.reshape(1, -1), axis=1)
    # будущая доходность
    ret_fwd = d["Close"].shift(-horizon) / d["Close"] - 1.0
    tmp = pd.DataFrame({"s": s, "r": ret_fwd.values}).dropna()
    if len(tmp) < max(50, 5*len(FEATURES)):
        return -1.0  # штраф за мало данных
    # Spearman
    corr = tmp["s"].corr(tmp["r"], method="spearman")
    if np.isnan(corr):
        return -1.0
    return float(corr)

def ga_optimize_level_A(df_sym: pd.DataFrame,
                        z_window: int,
                        horizon: int,
                        pop_size=40,
                        generations=35,
                        seed=42,
                        log_cb=None,
                        progress_cb=None,
                        stage_base=40,
                        stage_span=20) -> np.ndarray:
    """
    Простой ГА для весов индикаторов:
    - гены: веса K индикаторов
    - нормировка: L1=1
    - fitness: Spearman(score, future_return)
    """
    rng = np.random.default_rng(seed)
    K = len(FEATURES)

    def random_ind():
        w = rng.normal(0, 1, size=K)
        w = normalize_l1(w)
        return w

    def mutate(w, p=0.25, sigma=0.20):
        w2 = w.copy()
        for i in range(K):
            if rng.random() < p:
                w2[i] += rng.normal(0, sigma)
        return normalize_l1(w2)

    def crossover(a, b):
        # BLX-alpha style
        alpha = 0.35
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        span = hi - lo
        child = lo - alpha*span + rng.random(K) * (span*(1+2*alpha))
        return normalize_l1(child)

    # init
    pop = [random_ind() for _ in range(pop_size)]
    fit = [objective_level_A(df_sym, ind, z_window, horizon) for ind in pop]

    best_idx = int(np.argmax(fit))
    best = pop[best_idx].copy()
    best_fit = fit[best_idx]

    for g in range(generations):
        # прогресс (этап внутри общей шкалы)
        if progress_cb:
            p = stage_base + int(stage_span * (g+1) / generations)
            progress_cb(p, f"Калибровка весов индикаторов (ГА-A): поколение {g+1}/{generations}, best={best_fit:.4f}")

        # турнирная селекция
        new_pop = []
        while len(new_pop) < pop_size:
            def tour():
                i1, i2, i3 = rng.integers(0, pop_size, size=3)
                j = max([i1, i2, i3], key=lambda i: fit[i])
                return pop[j]
            p1, p2 = tour(), tour()
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)

        pop = new_pop
        fit = [objective_level_A(df_sym, ind, z_window, horizon) for ind in pop]
        bi = int(np.argmax(fit))
        if fit[bi] > best_fit:
            best_fit = fit[bi]
            best = pop[bi].copy()

    if log_cb:
        log_cb(f"ГА-A завершён: best Spearman={best_fit:.4f}", "READY")
    return best

# ------------------------ PORTFOLIO OBJECTIVES + GA LEVEL B ------------------------

def estimate_cov(returns: pd.DataFrame) -> np.ndarray:
    """
    returns: DataFrame shape [T, N]
    """
    cov = returns.cov(ddof=0).values
    # регуляризация
    cov = cov + np.eye(cov.shape[0]) * 1e-8
    return cov

def objectives_portfolio(w: np.ndarray,
                         mu_hat: np.ndarray,
                         cov: np.ndarray,
                         w_prev: np.ndarray,
                         w_min: float,
                         w_max: float,
                         leverage: float,
                         max_assets: int):
    """
    Возвращает вектор (f1,f2,f3,f4) для minimization:
      f1 = -ожидаемая доходность
      f2 = риск
      f3 = оборот (turnover)
      f4 = концентрация
    + возвращает флаг валидности
    """
    N = len(w)

    # bounds
    if np.any(w < w_min) or np.any(w > w_max):
        return (1e6, 1e6, 1e6, 1e6), False

    gross = np.sum(np.abs(w))
    if gross > leverage + 1e-12:
        return (1e6, 1e6, 1e6, 1e6), False

    # max assets
    active = int(np.sum(np.abs(w) > 1e-6))
    if active > max_assets:
        return (1e6, 1e6, 1e6, 1e6), False

    # objectives
    R = float(np.dot(w, mu_hat))        # maximize
    V = float(w.T @ cov @ w)            # minimize
    T = float(np.sum(np.abs(w - w_prev)))
    # concentration: по долям абсолютной экспозиции
    if gross < 1e-12:
        C = 0.0
    else:
        p = np.abs(w) / gross
        C = float(np.sum(p**2))

    return (-R, V, T, C), True

def fast_nondominated_sort(F):
    """
    F: list of objective vectors (minimization)
    return: fronts as list of lists of indices
    """
    n = len(F)
    S = [[] for _ in range(n)]
    n_dom = [0]*n
    rank = [0]*n
    fronts = [[]]

    def dominates(a, b):
        # a dominates b if all <= and at least one <
        fa, fb = F[a], F[b]
        le = all(fa[i] <= fb[i] for i in range(len(fa)))
        lt = any(fa[i] < fb[i] for i in range(len(fa)))
        return le and lt

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(p, q):
                S[p].append(q)
            elif dominates(q, p):
                n_dom[p] += 1
        if n_dom[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    rank[q] = i+1
                    nxt.append(q)
        i += 1
        fronts.append(nxt)

    fronts.pop()  # last empty
    return fronts, rank

def crowding_distance(front, F):
    """
    front: list of indices
    F: list of objective vectors
    """
    if not front:
        return {}
    m = len(F[0])
    dist = {i: 0.0 for i in front}

    for k in range(m):
        front_sorted = sorted(front, key=lambda i: F[i][k])
        fmin = F[front_sorted[0]][k]
        fmax = F[front_sorted[-1]][k]
        dist[front_sorted[0]] = dist[front_sorted[-1]] = float("inf")
        if abs(fmax - fmin) < 1e-18:
            continue
        for j in range(1, len(front_sorted)-1):
            prevv = F[front_sorted[j-1]][k]
            nextv = F[front_sorted[j+1]][k]
            dist[front_sorted[j]] += (nextv - prevv) / (fmax - fmin)
    return dist

def choose_from_pareto(front_indices, F):
    """
    Если в первом фронте несколько решений: выбираем компромисс.
    Подход: нормируем каждую цель на [0,1] внутри фронта и берём
    минимальную евклидову дистанцию до "идеальной" точки (0,0,0,0).
    """
    if len(front_indices) == 1:
        return front_indices[0]

    M = len(F[0])
    vals = np.array([F[i] for i in front_indices], dtype=float)  # [K, M]
    mins = vals.min(axis=0)
    maxs = vals.max(axis=0)
    denom = np.where((maxs - mins) < 1e-12, 1.0, (maxs - mins))
    norm = (vals - mins) / denom
    d = np.sqrt(np.sum(norm**2, axis=1))
    best_local = int(np.argmin(d))
    return front_indices[best_local]

def ga_optimize_portfolio(mu_hat: np.ndarray,
                          cov: np.ndarray,
                          w_prev: np.ndarray,
                          w_min: float,
                          w_max: float,
                          leverage: float,
                          max_assets: int,
                          pop_size=60,
                          generations=45,
                          seed=123,
                          progress_cb=None,
                          stage_base=60,
                          stage_span=30):
    """
    Лайт NSGA-II:
    - популяция векторов w
    - многокритериальный отбор по рангу + crowding
    - в конце берём 1-й фронт и выбираем компромисс методом ideal-point distance
    """
    rng = np.random.default_rng(seed)
    N = len(mu_hat)

    def random_w():
        # случайные веса в границах + контроль плеча
        w = rng.uniform(w_min, w_max, size=N)
        # разрежаем: оставляем случайно max_assets активов
        if N > max_assets:
            mask = np.zeros(N, dtype=bool)
            idx = rng.choice(N, size=max_assets, replace=False)
            mask[idx] = True
            w = np.where(mask, w, 0.0)
        # ограничение плеча
        gross = np.sum(np.abs(w))
        if gross > leverage:
            w = w * (leverage / (gross + 1e-12))
        return w

    def mutate(w, p=0.20, sigma=0.08):
        w2 = w.copy()
        for i in range(N):
            if rng.random() < p:
                w2[i] += rng.normal(0, sigma)
        # clip
        w2 = np.clip(w2, w_min, w_max)
        # sparsify again
        if np.sum(np.abs(w2) > 1e-6) > max_assets:
            idx = np.argsort(-np.abs(w2))[:max_assets]
            mask = np.zeros(N, dtype=bool); mask[idx] = True
            w2 = np.where(mask, w2, 0.0)
        # leverage
        gross = np.sum(np.abs(w2))
        if gross > leverage:
            w2 *= leverage / (gross + 1e-12)
        return w2

    def crossover(a, b):
        # blend
        lam = rng.random()
        child = lam*a + (1-lam)*b
        child = np.clip(child, w_min, w_max)
        # sparsify
        if np.sum(np.abs(child) > 1e-6) > max_assets:
            idx = np.argsort(-np.abs(child))[:max_assets]
            mask = np.zeros(N, dtype=bool); mask[idx] = True
            child = np.where(mask, child, 0.0)
        # leverage
        gross = np.sum(np.abs(child))
        if gross > leverage:
            child *= leverage / (gross + 1e-12)
        return child

    pop = [random_w() for _ in range(pop_size)]
    F = []
    valid = []
    for w in pop:
        f, ok = objectives_portfolio(w, mu_hat, cov, w_prev, w_min, w_max, leverage, max_assets)
        F.append(f); valid.append(ok)

    for g in range(generations):
        if progress_cb:
            p = stage_base + int(stage_span * (g+1) / generations)
            progress_cb(p, f"Оптимизация портфеля (ГА-B): поколение {g+1}/{generations}")

        # offspring
        offspring = []
        while len(offspring) < pop_size:
            i, j = rng.integers(0, pop_size, size=2)
            child = crossover(pop[i], pop[j])
            child = mutate(child)
            offspring.append(child)

        combined = pop + offspring
        F2 = []
        for w in combined:
            f, _ = objectives_portfolio(w, mu_hat, cov, w_prev, w_min, w_max, leverage, max_assets)
            F2.append(f)

        fronts, rank = fast_nondominated_sort(F2)

        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend([combined[i] for i in front])
            else:
                # crowding
                dist = crowding_distance(front, F2)
                front_sorted = sorted(front, key=lambda i: dist[i], reverse=True)
                need = pop_size - len(new_pop)
                new_pop.extend([combined[i] for i in front_sorted[:need]])
                break

        pop = new_pop
        F = [objectives_portfolio(w, mu_hat, cov, w_prev, w_min, w_max, leverage, max_assets)[0] for w in pop]

    # выбрать решение
    fronts, _ = fast_nondominated_sort(F)
    best_front = fronts[0]
    best_idx = choose_from_pareto(best_front, F)
    return pop[best_idx], F[best_idx], len(best_front)

# ------------------------ BACKTEST PIPELINE ------------------------

def build_mu_from_score(score: float, mu_scale: float = 0.01) -> float:
    """
    Перевод score -> ожидаемая доходность (простая интерпретируемая калибровка).
    mu_scale можно понимать как "характерный дневной масштаб доходности".
    """
    return float(np.tanh(score) * mu_scale)

def run_backtest(df: pd.DataFrame,
                 train_ratio: float,
                 rebalance_every: int,
                 calib_window_days: int,
                 cov_window_days: int,
                 z_window: int,
                 horizon: int,
                 commission_rate: float,
                 w_min: float,
                 w_max: float,
                 leverage: float,
                 max_assets: int,
                 recalibrate_each_rebalance: bool,
                 seed: int,
                 log_cb=None,
                 progress_cb=None):
    """
    df: OHLCV + indicators (по каждому символу)
    """
    # привести к wide формату цен
    symbols = sorted(df["Symbol"].unique().tolist())
    if len(symbols) < 1:
        raise RuntimeError("Нет данных по символам.")

    # общий календарь: пересечение дат по всем тикерам
    # (иначе портфельные доходности будут с NaN)
    pivot_close = df.pivot_table(index="Date", columns="Symbol", values="Close", aggfunc="last").sort_index()
    pivot_close = pivot_close.dropna(axis=0, how="any")  # пересечение
    if len(pivot_close) < 400:
        raise RuntimeError("Слишком мало общих дат (пересечение) по выбранным тикерам. Попробуй другой пул.")

    dates = pivot_close.index
    T = len(dates)
    split = int(T * train_ratio)
    split = max(50, min(split, T-50))

    train_dates = dates[:split]
    test_dates = dates[split:]

    # дневные доходности
    rets = pivot_close.pct_change().dropna()
    rets = rets.loc[pivot_close.index[1]:]  # align
    dates_r = rets.index

    # индикаторные данные тоже нужно под те же даты
    # сделаем словарь по символу
    df_ind = {}
    for sym in symbols:
        ds = df[df["Symbol"] == sym].set_index("Date").sort_index()
        # выровнять по общему календарю (pivot_close.index)
        ds = ds.reindex(pivot_close.index)
        df_ind[sym] = ds

    # калибровка A: веса индикаторов на обучении (или на каждом ребалансе)
    a_weights = {sym: None for sym in symbols}

    # Мю масштаб: возьмём среднюю абсолютную дневную доходность на обучении
    mu_scale = float(np.mean(np.abs(rets.loc[train_dates[1:]].values)))
    mu_scale = max(mu_scale, 0.003)

    # параметры портфеля
    w_prev = np.zeros(len(symbols), dtype=float)

    equity = [1.0]
    commissions = 0.0
    turnovers = []

    # расписание ребалансов в тесте
    # считаем по индексу dates (pivot_close.index)
    test_idx0 = split
    rebalance_points = list(range(test_idx0, T, rebalance_every))
    if rebalance_points[-1] != T-1:
        rebalance_points.append(T-1)

    if log_cb:
        log_cb(f"Общий период: {dates[0].date()} … {dates[-1].date()} (дней: {T})", "INFO")
        log_cb(f"Разделение по времени: train={len(train_dates)} дней, test={len(test_dates)} дней (70/30)", "INFO")

    # прогресс
    if progress_cb:
        progress_cb(0, "Старт…")

    # --- функция обновления весов A на заданном окне дат (walk-forward)
    def calibrate_A_for_date(end_date):
        # окно калибровки: [end-calib_window+1, end]
        end_loc = pivot_close.index.get_loc(end_date)
        start_loc = max(0, end_loc - calib_window_days + 1)
        window_dates = pivot_close.index[start_loc:end_loc+1]

        for si, sym in enumerate(symbols):
            d_sym = df_ind[sym].loc[window_dates].copy()
            # должны быть индикаторы
            # убираем NaN
            d_sym = d_sym.dropna(subset=FEATURES + ["Close"])
            if len(d_sym) < max(120, 5*len(FEATURES)):
                # fallback: равные веса
                a_weights[sym] = normalize_l1(np.ones(len(FEATURES)))
                continue

            a = ga_optimize_level_A(
                d_sym.reset_index(),
                z_window=z_window,
                horizon=horizon,
                pop_size=38,
                generations=30,
                seed=seed + si*7 + end_loc,
                log_cb=None,
                progress_cb=progress_cb
            )
            a_weights[sym] = a

    # initial calibration (на конце train)
    if log_cb:
        log_cb("Этап: калибровка индикаторов на обучающем периоде…", "INFO")
    if progress_cb:
        progress_cb(35, "Калибровка весов индикаторов (первичная)…")

    calibrate_A_for_date(train_dates[-1])

    # --- основной цикл теста
    if log_cb:
        log_cb("Этап: тест / онлайн-ребалансировка…", "INFO")

    # подготовим матрицу доходностей для ковариации на каждом ребалансе
    # будем брать окно cov_window_days из прошлых доходностей
    rets_all = rets.copy()

    # индексы в rets: начиная со 2-го дня календаря
    # для простоты берём ковариацию по календарным датам pivot_close.index, пересечение
    # доходности доступны на dates[1:]
    for rp_i, t_loc in enumerate(rebalance_points[:-1]):
        t_date = pivot_close.index[t_loc]
        next_loc = rebalance_points[rp_i+1]
        next_date = pivot_close.index[next_loc]

        if progress_cb:
            # 60..90 примерно
            base = 60 + int(30 * (rp_i+1) / max(1, len(rebalance_points)-1))
            progress_cb(base, f"Ребаланс {rp_i+1}/{len(rebalance_points)-1}: {t_date.date()}")

        # recalibrate A if needed
        if recalibrate_each_rebalance:
            if log_cb:
                log_cb(f"Переподбор весов индикаторов (walk-forward) на дату {t_date.date()}…", "INFO")
            calibrate_A_for_date(t_date)

        # построить score и mu_hat на дату t_date
        mu_hat = np.zeros(len(symbols), dtype=float)

        for i, sym in enumerate(symbols):
            d_sym = df_ind[sym]
            row = d_sym.loc[t_date]
            if row.isna().any():
                # если на этой дате нет индикаторов — mu=0
                mu_hat[i] = 0.0
                continue

            # z-score значений индикаторов на текущей дате
            # считаем z по окну до t_date
            loc = d_sym.index.get_loc(t_date)
            w_start = max(0, loc - z_window + 1)
            win = d_sym.iloc[w_start:loc+1]
            # z текущей точки
            z_vals = []
            for f in FEATURES:
                s = win[f]
                m = s.mean()
                sd = s.std(ddof=0)
                if sd < 1e-12:
                    z = 0.0
                else:
                    z = float((row[f] - m) / sd)
                z_vals.append(z)
            z_vals = np.array(z_vals, dtype=float)

            a = a_weights.get(sym)
            if a is None:
                a = normalize_l1(np.ones(len(FEATURES)))
            score = float(np.dot(a, z_vals))
            mu_hat[i] = build_mu_from_score(score, mu_scale=mu_scale)

        # ковариация
        # окно ковариации заканчивается в t_date (используем прошлые доходности)
        cov_end_loc = pivot_close.index.get_loc(t_date)
        cov_start_loc = max(1, cov_end_loc - cov_window_days + 1)  # доходности начинаются с 1
        cov_dates = pivot_close.index[cov_start_loc:cov_end_loc+1]
        # доходности по этим датам доступны на cov_dates[1:] (потому что pct_change)
        ret_win = rets_all.loc[cov_dates[1:]]
        cov = estimate_cov(ret_win[symbols])

        # оптимизация портфеля (ГА-B)
        w_star, f_star, pareto_size = ga_optimize_portfolio(
            mu_hat=mu_hat,
            cov=cov,
            w_prev=w_prev,
            w_min=w_min,
            w_max=w_max,
            leverage=leverage,
            max_assets=max_assets,
            pop_size=60,
            generations=45,
            seed=seed + 1000 + rp_i*17,
            progress_cb=progress_cb
        )

        if log_cb:
            log_cb(f"Ребаланс: найдено решений в 1-м Парето-фронте: {pareto_size}. Выбран компромисс.", "INFO")

        # комиссии по обороту
        turn = float(np.sum(np.abs(w_star - w_prev)))
        cost = commission_rate * turn * equity[-1]
        commissions += cost
        turnovers.append(turn)

        # применяем сделки (снятие комиссии сразу)
        eq0 = equity[-1] - cost
        eq0 = max(eq0, 1e-12)
        equity[-1] = eq0

        # период удержания: (t_loc ... next_loc)
        # доходности считаем по календарю pivot_close
        hold_dates = pivot_close.index[t_loc:next_loc+1]
        # дневные доходности портфеля: sum w * r
        # для дат hold_dates используем доходности начиная со 2-й даты
        for d_i in range(1, len(hold_dates)):
            d = hold_dates[d_i]
            prev_d = hold_dates[d_i-1]
            r_vec = (pivot_close.loc[d, symbols].values / pivot_close.loc[prev_d, symbols].values) - 1.0
            # кэш: оставшаяся доля (если sum w < 1) получает 0
            # если sum w > 1 (заём) — стоимость займа игнорируем (можно добавить позже)
            port_ret = float(np.dot(w_star, r_vec))
            equity.append(equity[-1] * (1.0 + port_ret))

        w_prev = w_star.copy()

    equity = np.array(equity, dtype=float)
    # метрики
    daily_ret = equity[1:] / equity[:-1] - 1.0
    days = len(equity) - 1

    total = equity[-1] - 1.0
    vol = annualize_vol(daily_ret)
    sh = sharpe_ratio(daily_ret)
    mdd = -max_drawdown(equity)  # положительное
    cg = cagr(equity, days)
    calmar = (cg / (mdd + 1e-12))

    out = {
        "Итоговая доходность, %": total*100,
        "CAGR, %": cg*100,
        "Волатильность, % годовых": vol*100,
        "Коэффициент Шарпа": sh,
        "Макс. просадка, %": mdd*100,
        "Коэффициент Калмара": calmar,
        "Средний оборот (turnover)": float(np.mean(turnovers)) if turnovers else 0.0,
        "Суммарные комиссии": commissions,
        "Число ребалансов": len(rebalance_points)-1,
        "Период (дней)": days,
    }

    return out, equity, pivot_close.loc[pivot_close.index[split:]], symbols

# ------------------------ GUI APP ------------------------

class Dash(tk.Tk):
    def __init__(self):
        super().__init__()
        self.style = Style(theme="superhero")
        self.title("📊 Онлайн-ребалансировка портфеля (технические индикаторы + ГА)")
        self.geometry("1700x980")
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.q = queue.Queue()
        self._is_running = False

        self._tickers_cache = load_tickers_cache()
        self._suggestions = []

        self._build()
        self.after(100, self._poll_queue)

        if in_notebook():
            self._wlog("Внимание: Tkinter в Jupyter может работать нестабильно. Лучше запускать как .py файл.", "WARN")

    def _build(self):
        side = ttk.Frame(self, padding=10)
        side.grid(row=0, column=0, sticky="ns")
        side.columnconfigure(0, weight=1)

        # ---- ПУЛ ТИКЕРОВ ----
        ttk.Label(side, text="Пул тикеров (через запятую):", font=("Segoe UI", 10, "bold")).grid(sticky="w")

        self.tickers_var = tk.StringVar(value="AAPL, MSFT, AMZN, TSLA")
        tickers_entry = ttk.Entry(side, textvariable=self.tickers_var, width=34)
        tickers_entry.grid(sticky="we", pady=(4, 2))
        ToolTip(tickers_entry, "Введите тикеры из датасета (например: AAPL, MSFT). Можно 2–20 тикеров.")

        # dropdown listbox for suggestions
        self.suggest_box = tk.Listbox(side, height=6)
        self.suggest_box.grid(sticky="we")
        self.suggest_box.grid_remove()
        self.suggest_box.bind("<<ListboxSelect>>", self._pick_suggestion)

        tickers_entry.bind("<KeyRelease>", self._on_ticker_typing)
        tickers_entry.bind("<FocusOut>", lambda e: self._hide_suggestions())

        btn_t = ttk.Frame(side)
        btn_t.grid(sticky="we", pady=(6, 8))
        ttk.Button(btn_t, text="🔎 Подсказки тикеров (скан/кэш)", command=self._load_ticker_hints).pack(side="left", fill="x", expand=True)
        ttk.Button(btn_t, text="🧹 Очистить кэш тикеров", command=self._clear_cache).pack(side="left", padx=6)

        # ---- ПАРАМЕТРЫ ----
        ttk.Separator(side).grid(sticky="we", pady=8)

        ttk.Label(side, text="Окна и параметры:", font=("Segoe UI", 10, "bold")).grid(sticky="w")

        form = ttk.Frame(side)
        form.grid(sticky="we", pady=(6, 8))
        form.columnconfigure(1, weight=1)

        # 70/30 split fixed but shown
        self.train_ratio = 0.70

        self.calib_days_var = tk.StringVar(value="252")
        self.cov_days_var = tk.StringVar(value="126")
        self.zwin_var = tk.StringVar(value="60")
        self.horizon_var = tk.StringVar(value="5")
        self.reb_var = tk.StringVar(value="21")
        self.comm_var = tk.StringVar(value="0.001")  # 0.1%

        rows = [
            ("Окно калибровки индикаторов (дней):", self.calib_days_var,
             "Сколько прошлых дней использовать, чтобы подобрать веса индикаторов (ГА уровня A)."),
            ("Окно оценки риска (дней):", self.cov_days_var,
             "Сколько прошлых дней использовать для оценки риска (ковариации доходностей)."),
            ("Окно стандартизации индикаторов (дней):", self.zwin_var,
             "Для перевода индикаторов к сопоставимому масштабу используем стандартизацию по окну."),
            ("Горизонт прогноза (дней):", self.horizon_var,
             "На сколько дней вперёд оцениваем связь score и будущей доходности при калибровке A."),
            ("Частота ребалансировки (дней):", self.reb_var,
             "Как часто пересчитывать портфель на тестовом периоде (например 21 ≈ раз в месяц)."),
            ("Комиссия за сделки (доля):", self.comm_var,
             "Пропорциональные издержки: комиссия * оборот портфеля (turnover) в момент ребаланса."),
        ]

        self._widgets_for_tooltips = []
        for r, (label, var, tip) in enumerate(rows):
            lbl = ttk.Label(form, text=label)
            lbl.grid(row=r, column=0, sticky="w", pady=2)
            ent = ttk.Entry(form, textvariable=var, width=10)
            ent.grid(row=r, column=1, sticky="we", pady=2)
            ToolTip(ent, tip)
            self._widgets_for_tooltips.append(ent)

        # info split
        split_lbl = ttk.Label(side, text="Разделение данных по времени: 70% обучение / 30% тест", bootstyle=INFO)
        split_lbl.grid(sticky="we", pady=(0, 10))

        # ---- ОГРАНИЧЕНИЯ ПОРТФЕЛЯ ----
        ttk.Label(side, text="Ограничения портфеля:", font=("Segoe UI", 10, "bold")).grid(sticky="w")
        form2 = ttk.Frame(side)
        form2.grid(sticky="we", pady=(6, 8))
        form2.columnconfigure(1, weight=1)

        self.wmin_var = tk.StringVar(value="-0.30")
        self.wmax_var = tk.StringVar(value="0.50")
        self.lev_var = tk.StringVar(value="1.20")
        self.max_assets_var = tk.StringVar(value="8")
        self.recalib_var = tk.BooleanVar(value=True)

        rows2 = [
            ("Минимальная доля на актив:", self.wmin_var, "Нижняя граница доли w_i (может быть <0, если разрешён шорт)."),
            ("Максимальная доля на актив:", self.wmax_var, "Верхняя граница доли w_i."),
            ("Лимит плеча (сумма |w|):", self.lev_var, "Ограничение суммарной абсолютной экспозиции: Σ|w_i| ≤ L."),
            ("Максимум активов в портфеле:", self.max_assets_var, "Ограничение на число ненулевых позиций в портфеле."),
        ]

        for r, (label, var, tip) in enumerate(rows2):
            lbl = ttk.Label(form2, text=label)
            lbl.grid(row=r, column=0, sticky="w", pady=2)
            ent = ttk.Entry(form2, textvariable=var, width=10)
            ent.grid(row=r, column=1, sticky="we", pady=2)
            ToolTip(ent, tip)

        chk = ttk.Checkbutton(side, text="Переподбирать веса индикаторов при каждом ребалансе", variable=self.recalib_var)
        chk.grid(sticky="w", pady=(4, 10))
        ToolTip(chk, "Если включено, веса индикаторов (ГА-A) подбираются заново на каждом ребалансе по прошлому окну.")

        # ---- КНОПКИ ----
        btns = ttk.Frame(side)
        btns.grid(sticky="we", pady=(2, 6))
        self.run_btn = ttk.Button(btns, text="▶ Запуск", bootstyle=SUCCESS, command=self._thread_run)
        self.run_btn.pack(fill="x")

        # ---- ПРОГРЕСС ----
        ttk.Label(side, text="Прогресс:").grid(sticky="w", pady=(10, 0))
        self.progress_lbl = ttk.Label(side, text="—")
        self.progress_lbl.grid(sticky="we")

        self.pb = ttk.Progressbar(side, mode="determinate", maximum=100)
        self.pb.grid(sticky="we", pady=(4, 6))

        # ---- ЛОГ ----
        ttk.Label(side, text="Журнал:").grid(sticky="w", pady=(10, 0))
        self.log = tk.Text(side, height=14, font=("Consolas", 9), state="disabled", wrap="word")
        for name, color in [("INFO", "#9aa0a6"), ("READY", "#00c853"), ("WARN", "#fb8c00"), ("ERROR", "#e53935")]:
            self.log.tag_configure(name, foreground=color)
        self.log.grid(sticky="nsew")
        side.rowconfigure(self.log.grid_info()["row"], weight=1)

        # ---- ВКЛАДКИ ----
        self.nb = ttk.Notebook(self)
        self.nb.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.tab_metrics = ttk.Frame(self.nb)
        self.tab_equity = ttk.Frame(self.nb)
        self.tab_prices = ttk.Frame(self.nb)

        self.nb.add(self.tab_metrics, text="Метрики")
        self.nb.add(self.tab_equity, text="Капитал (доходность, %)")
        self.nb.add(self.tab_prices, text="Цены (тест)")

        # metrics table
        self.metrics_tree = ttk.Treeview(self.tab_metrics, columns=("k", "v"), show="headings", height=16)
        self.metrics_tree.heading("k", text="Показатель")
        self.metrics_tree.heading("v", text="Значение")
        self.metrics_tree.column("k", width=320, anchor="w")
        self.metrics_tree.column("v", anchor="e")
        self.metrics_tree.pack(fill="both", expand=True, padx=8, pady=8)

    # ------------------ UI helpers ------------------

    def _wlog(self, msg, level="INFO"):
        ts = time.strftime("%H:%M:%S")
        self.log["state"] = "normal"
        self.log.insert("end", f"[{ts}] {msg}\n", (level,))
        self.log.see("end")
        self.log["state"] = "disabled"

    def _set_progress(self, p, text):
        if p is None:
            # без %, просто текст
            self.progress_lbl.config(text=text)
            return
        p = max(0, min(100, int(p)))
        self.pb["value"] = p
        self.progress_lbl.config(text=f"{p}% — {text}")

    def _poll_queue(self):
        try:
            while True:
                item = self.q.get_nowait()
                kind = item.get("kind")
                if kind == "log":
                    self._wlog(item["msg"], item.get("level", "INFO"))
                elif kind == "progress":
                    self._set_progress(item.get("p"), item.get("text", ""))
                elif kind == "result":
                    self._render_result(item["metrics"], item["equity"], item["prices"])
                    self._is_running = False
                    self.run_btn.config(state="normal")
                elif kind == "error":
                    self._is_running = False
                    self.run_btn.config(state="normal")
                    self._wlog(item["msg"], "ERROR")
                    messagebox.showerror("Ошибка", item["msg"])
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)

    # ------------------ ticker suggestions ------------------

    def _clear_cache(self):
        # чистим только тикер-кэш и файлы parquet
        try:
            if os.path.isfile(TICKERS_CACHE_PATH):
                os.remove(TICKERS_CACHE_PATH)
            for f in os.listdir(CACHE_DIR):
                if f.endswith(".parquet"):
                    try:
                        os.remove(os.path.join(CACHE_DIR, f))
                    except Exception:
                        pass
            self._tickers_cache = []
            self._wlog("Кэш тикеров очищен.", "READY")
        except Exception as e:
            self._wlog(f"Не удалось очистить кэш: {e}", "ERROR")

    def _load_ticker_hints(self):
        # 1) если есть кэш — просто используем
        if self._tickers_cache:
            self._wlog(f"Подсказки тикеров загружены из кэша (шт.): {len(self._tickers_cache)}", "READY")
            return

        # 2) иначе — сканируем (ограниченно)
        def worker():
            try:
                self.q.put({"kind": "log", "msg": "Сканирование тикеров может занять время. Начинаю…", "level": "INFO"})
                def prog(p, t):
                    self.q.put({"kind": "progress", "p": p, "text": t})
                tickers = scan_unique_tickers_hf(progress_cb=lambda p,t: prog(None, t), log_cb=None, max_seconds=45)
                if tickers:
                    self._tickers_cache = tickers
                    save_tickers_cache(tickers)
                    self.q.put({"kind": "log", "msg": f"Подсказки тикеров готовы (шт.): {len(tickers)}", "level": "READY"})
                else:
                    self.q.put({"kind": "log", "msg": "Не удалось извлечь тикеры (возможно медленное соединение).", "level": "WARN"})
            except Exception as e:
                self.q.put({"kind": "error", "msg": str(e)})

        threading.Thread(target=worker, daemon=True).start()

    def _on_ticker_typing(self, event):
        txt = self.tickers_var.get()
        # берём последний "кусок" после запятой
        part = txt.split(",")[-1].strip().upper()
        if len(part) < 1 or not self._tickers_cache:
            self._hide_suggestions()
            return

        matches = [t for t in self._tickers_cache if t.startswith(part)]
        matches = matches[:30]
        if not matches:
            self._hide_suggestions()
            return

        self.suggest_box.delete(0, "end")
        for m in matches:
            self.suggest_box.insert("end", m)
        self.suggest_box.grid()
        self._suggestions = matches

    def _hide_suggestions(self):
        self.suggest_box.grid_remove()

    def _pick_suggestion(self, _=None):
        if not self.suggest_box.curselection():
            return
        choice = self.suggest_box.get(self.suggest_box.curselection()[0])
        # заменяем последний токен
        parts = [p.strip() for p in self.tickers_var.get().split(",")]
        if not parts:
            parts = [choice]
        else:
            parts[-1] = choice
        # убираем дубликаты, сохраняем порядок
        out = []
        seen = set()
        for p in parts:
            up = p.upper()
            if up and up not in seen:
                out.append(up); seen.add(up)
        self.tickers_var.set(", ".join(out) + ", ")
        self._hide_suggestions()

    # ------------------ RUN ------------------

    def _thread_run(self):
        if self._is_running:
            self._wlog("Уже выполняется задача — дождитесь завершения.", "WARN")
            return

        # парс параметров (минимальная валидация)
        pool = [x.strip().upper() for x in self.tickers_var.get().split(",") if x.strip()]
        if len(pool) < 1:
            messagebox.showerror("Ошибка", "Введите хотя бы один тикер.")
            return
        if len(pool) > 25:
            messagebox.showerror("Ошибка", "Слишком много тикеров. Рекомендуется до 25.")
            return

        try:
            calib_days = int(self.calib_days_var.get())
            cov_days = int(self.cov_days_var.get())
            zwin = int(self.zwin_var.get())
            horizon = int(self.horizon_var.get())
            reb = int(self.reb_var.get())
            comm = float(self.comm_var.get())
            wmin = float(self.wmin_var.get())
            wmax = float(self.wmax_var.get())
            lev = float(self.lev_var.get())
            max_assets = int(self.max_assets_var.get())
        except Exception:
            messagebox.showerror("Ошибка", "Некорректные параметры (проверь числа).")
            return

        if calib_days < 60 or cov_days < 30 or zwin < 10 or horizon < 1 or reb < 1:
            messagebox.showerror("Ошибка", "Слишком маленькие окна/параметры. Увеличьте значения.")
            return

        if wmin >= wmax:
            messagebox.showerror("Ошибка", "Минимальная доля должна быть меньше максимальной.")
            return

        if max_assets < 1:
            messagebox.showerror("Ошибка", "Максимум активов должен быть >= 1.")
            return

        if max_assets > len(pool):
            max_assets = len(pool)

        recalib = bool(self.recalib_var.get())

        self._is_running = True
        self.run_btn.config(state="disabled")
        self.q.put({"kind": "progress", "p": 0, "text": "Запуск…"})
        self.q.put({"kind": "log", "msg": "Запуск…", "level": "INFO"})
        self.q.put({"kind": "log", "msg": "Загружаю данные по выбранным тикерам…", "level": "INFO"})

        # worker thread
        def worker():
            try:
                def log(msg, level="INFO"):
                    self.q.put({"kind": "log", "msg": msg, "level": level})

                def prog(p, text):
                    self.q.put({"kind": "progress", "p": p, "text": text})

                # 1) load
                prog(5, "Загрузка данных…")
                df = load_data_for_pool(pool, use_cache=True, progress_cb=lambda p,t: prog(None, t), log_cb=lambda m,l="INFO": log(m,l))

                # check columns
                missing = [c for c in REQUIRED_INTERNAL if c not in df.columns]
                if missing:
                    raise RuntimeError(f"Не хватает колонок после преобразования: {missing}. Колонки: {list(df.columns)}")

                # 2) indicators
                prog(30, "Расчёт индикаторов…")
                frames = []
                for i, (sym, ds) in enumerate(df.groupby("Symbol")):
                    ds = ds.sort_values("Date").reset_index(drop=True)
                    ds = add_indicators_for_symbol(ds)
                    frames.append(ds)
                    if i % 2 == 0:
                        prog(30, f"Расчёт индикаторов… готово {i+1}/{len(pool)}")

                df2 = pd.concat(frames, ignore_index=True)
                # clean
                df2 = df2.dropna(subset=FEATURES + ["Close"]).reset_index(drop=True)

                # 3) backtest
                prog(45, "Калибровка и ребалансировка…")
                metrics, equity, prices_test, symbols = run_backtest(
                    df=df2,
                    train_ratio=0.70,
                    rebalance_every=reb,
                    calib_window_days=calib_days,
                    cov_window_days=cov_days,
                    z_window=zwin,
                    horizon=horizon,
                    commission_rate=comm,
                    w_min=wmin,
                    w_max=wmax,
                    leverage=lev,
                    max_assets=max_assets,
                    recalibrate_each_rebalance=recalib,
                    seed=42,
                    log_cb=log,
                    progress_cb=prog
                )

                prog(95, "Формирование отчёта…")
                # prices_test: DataFrame Close in test; we'll plot avg or first
                self.q.put({"kind": "result", "metrics": metrics, "equity": equity, "prices": prices_test})
                prog(100, "Готово")
                self.q.put({"kind": "log", "msg": "✅ Готово", "level": "READY"})

            except Exception as e:
                self.q.put({"kind": "error", "msg": str(e)})

        threading.Thread(target=worker, daemon=True).start()

    # ------------------ render ------------------

    def _render_result(self, metrics: dict, equity: np.ndarray, prices_test: pd.DataFrame):
        # metrics table
        self.metrics_tree.delete(*self.metrics_tree.get_children())
        for k, v in metrics.items():
            if isinstance(v, (int, np.integer)):
                vv = str(int(v))
            elif "комис" in k.lower():
                vv = f"{v:,.4f}".replace(",", " ")
            elif "%" in k:
                vv = f"{v:.4f}"
            else:
                vv = f"{v:.6f}" if abs(v) < 10 else f"{v:.4f}"
            self.metrics_tree.insert("", "end", values=(k, vv))

        # equity plot (%)
        for w in self.tab_equity.winfo_children():
            w.destroy()

        eq_pct = (equity - 1.0) * 100.0
        fig = Figure(figsize=(11, 5))
        ax = fig.add_subplot(111)
        ax.plot(eq_pct, linewidth=1.6)
        ax.axhline(0, linewidth=0.8)
        ax.set_title("Капитал (доходность, %)")
        ax.set_xlabel("Дни тестового периода")
        ax.set_ylabel("%")

        canvas = FigureCanvasTkAgg(fig, master=self.tab_equity)
        toolbar = NavigationToolbar2Tk(canvas, self.tab_equity)
        toolbar.update()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # prices plot (first symbol / average)
        for w in self.tab_prices.winfo_children():
            w.destroy()

        fig2 = Figure(figsize=(11, 5))
        ax2 = fig2.add_subplot(111)
        # prices_test is close matrix; plot average normalized to 100
        if isinstance(prices_test, pd.DataFrame) and prices_test.shape[1] >= 1:
            avg = prices_test.mean(axis=1)
            y = (avg / avg.iloc[0]) * 100
            ax2.plot(y.values, linewidth=1.6)
            ax2.set_title("Средняя цена по пулу (тест), нормировано к 100")
            ax2.set_xlabel("Дни тестового периода")
            ax2.set_ylabel("Индекс (100 = старт)")
        canvas2 = FigureCanvasTkAgg(fig2, master=self.tab_prices)
        toolbar2 = NavigationToolbar2Tk(canvas2, self.tab_prices)
        toolbar2.update()
        canvas2.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    app = Dash()
    app.mainloop()

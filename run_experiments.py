"""
Запуск всех экспериментов проекта.

    python run_experiments.py            # полный запуск (~1-2 минуты)
    python run_experiments.py --quick    # быстрый запуск для проверки

Результаты: таблицы в results/*.csv, графики в figures/*.png, summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mcoptions import (  # noqa: E402
    Params,
    black_scholes_call,
    discounted_payoff_std,
    mc_call,
    simulate_paths,
    terminal_prices,
)

ROOT = Path(__file__).parent
RES = ROOT / "results"
FIG = ROOT / "figures"
SEED = 2026
BASE = Params()  # S0=100, K=100, T=1, r=5%, sigma=20%

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})
BLUE, ORANGE, GREEN, RED, GREY = "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#6b7280"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("  ->", name)


# ------------------------------------------------------------------ 0. иллюстрации модели
def fig_model(rng):
    paths = simulate_paths(BASE, 30, 252, rng)
    t = np.linspace(0, BASE.T, paths.shape[1])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, paths.T, lw=0.8, alpha=0.7)
    ax.axhline(BASE.K, color="k", ls="--", lw=1, label=f"страйк K = {BASE.K:.0f}")
    ax.set(xlabel="время, годы", ylabel="цена акции", title="30 случайных сценариев цены акции (геометрическое броуновское движение)")
    ax.legend()
    save(fig, "01_paths.png")

    z = rng.standard_normal(100_000)
    st = terminal_prices(BASE, z)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(st, bins=120, color=BLUE, alpha=0.8)
    axes[0].axvline(BASE.K, color="k", ls="--", label="страйк K")
    axes[0].set(xlabel="цена акции через год $S_T$", ylabel="число сценариев", title="Распределение цены через год")
    axes[0].legend()
    s = np.linspace(40, 200, 400)
    axes[1].plot(s, np.maximum(s - BASE.K, 0), color=ORANGE, lw=2)
    axes[1].set(xlabel="цена акции через год $S_T$", ylabel="выплата", title="Выплата колл-опциона: max(S_T − K, 0)")
    save(fig, "02_terminal_distribution_and_payoff.png")


# ------------------------------------------------------------------ 1. одна серия бросков
def fig_running_mean(rng, bs):
    n = 100_000
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    disc = math.exp(-BASE.r * BASE.T)
    for i in range(5):
        z = rng.standard_normal(n)
        pay = disc * np.maximum(terminal_prices(BASE, z) - BASE.K, 0)
        running = np.cumsum(pay) / np.arange(1, n + 1)
        ax.plot(np.arange(1, n + 1), running, lw=1, alpha=0.85)
    ax.axhline(bs, color="k", ls="--", lw=1.5, label=f"Блэк — Шоулз = {bs:.4f}")
    ax.set(xscale="log", xlabel="число симуляций N (лог. шкала)", ylabel="оценка цены опциона",
           title="Как оценка «успокаивается» с ростом N (5 независимых запусков)", ylim=(bs - 5, bs + 5))
    ax.legend()
    save(fig, "03_running_mean.png")


# ------------------------------------------------------------------ 2. основной эксперимент
def convergence_experiment(rng, bs, ns, reps):
    sigma_pay = discounted_payoff_std(BASE)
    rows, raw = [], {}
    for n in ns:
        est, se, cover, times = [], [], 0, []
        for _ in range(reps):
            t0 = time.perf_counter()
            e, s = mc_call(BASE, n, rng)
            times.append(time.perf_counter() - t0)
            est.append(e)
            se.append(s)
            if abs(e - bs) <= 1.96 * s:
                cover += 1
        est = np.array(est)
        err = est - bs
        raw[n] = est
        rows.append({
            "N": n,
            "mean_estimate": est.mean(),
            "bias": err.mean(),
            "std_of_estimates": est.std(ddof=1),
            "rmse": math.sqrt((err ** 2).mean()),
            "mean_abs_error": np.abs(err).mean(),
            "max_abs_error": np.abs(err).max(),
            "theory_rmse": sigma_pay / math.sqrt(n),
            "rmse_times_sqrtN": math.sqrt((err ** 2).mean()) * math.sqrt(n),
            "ci95_coverage": cover / reps,
            "mean_time_ms": 1000 * float(np.mean(times)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RES / "convergence.csv", index=False, float_format="%.6g")
    return df, raw, sigma_pay


def fig_convergence(df, bs):
    n = df["N"].values
    slope, intercept = np.polyfit(np.log(n), np.log(df["rmse"].values), 1)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.loglog(n, df["rmse"], "o-", color=BLUE, lw=2, label="эксперимент: RMSE")
    ax.loglog(n, df["theory_rmse"], "--", color=ORANGE, lw=2, label=r"теория: $\sigma_{выпл}/\sqrt{N}$")
    ax.loglog(n, df["mean_abs_error"], "s:", color=GREEN, lw=1.5, label="средняя |ошибка|")
    ax.set(xlabel="число симуляций N", ylabel="ошибка оценки (в денежных единицах)",
           title=f"Ошибка падает как 1/√N: наклон в лог-лог осях = {slope:.3f}")
    ax.legend()
    save(fig, "05_loglog_error.png")
    return slope


def fig_histograms(raw, bs):
    chosen = [100, 1_000, 10_000, 100_000]
    chosen = [c for c in chosen if c in raw]
    fig, axes = plt.subplots(1, len(chosen), figsize=(3.2 * len(chosen), 3.6), sharey=False)
    lo, hi = bs - 4, bs + 4
    for ax, n in zip(np.atleast_1d(axes), chosen):
        ax.hist(np.clip(raw[n], lo, hi), bins=30, range=(lo, hi), color=BLUE, alpha=0.8)
        ax.axvline(bs, color="k", ls="--", lw=1.5)
        ax.set(title=f"N = {n:,}".replace(",", " "), xlabel="оценка цены")
    np.atleast_1d(axes)[0].set_ylabel("число повторов")
    fig.suptitle("Разброс оценок вокруг точного значения (пунктир): чем больше N, тем уже «колокол»", y=1.03)
    save(fig, "04_histograms_of_estimates.png")


def fig_coverage(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(df["N"], 100 * df["ci95_coverage"], "o-", color=BLUE, lw=2)
    ax.axhline(95, color="k", ls="--", label="номинал 95%")
    ax.set(xlabel="число симуляций N", ylabel="доля интервалов, накрывших истину, %",
           title="Надёжность 95% доверительного интервала", ylim=(80, 100))
    ax.legend()
    save(fig, "06_ci_coverage.png")


def fig_time(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].loglog(df["N"], df["mean_time_ms"], "o-", color=ORANGE, lw=2)
    axes[0].set(xlabel="число симуляций N", ylabel="время одной оценки, мс", title="Время растёт ~ линейно с N")
    axes[1].loglog(df["mean_time_ms"], df["rmse"], "o-", color=BLUE, lw=2)
    axes[1].set(xlabel="время, мс", ylabel="RMSE", title="Цена точности: ошибка против времени")
    save(fig, "07_time_vs_error.png")


# ------------------------------------------------------------------ 3. антитетическая выборка
def antithetic_experiment(rng, bs, ns, reps):
    rows = []
    for n in ns:
        plain = np.array([mc_call(BASE, n, rng)[0] for _ in range(reps)]) - bs
        anti = np.array([mc_call(BASE, n, rng, antithetic=True)[0] for _ in range(reps)]) - bs
        rp, ra = math.sqrt((plain ** 2).mean()), math.sqrt((anti ** 2).mean())
        rows.append({"N": n, "rmse_plain": rp, "rmse_antithetic": ra, "variance_reduction_factor": (rp / ra) ** 2})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "antithetic.csv", index=False, float_format="%.6g")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.loglog(df["N"], df["rmse_plain"], "o-", color=BLUE, lw=2, label="обычный Монте-Карло")
    ax.loglog(df["N"], df["rmse_antithetic"], "s-", color=GREEN, lw=2, label="антитетические пары (Z, −Z)")
    ax.set(xlabel="число симуляций N", ylabel="RMSE", title="Бонус: можно снизить ошибку, не увеличивая N")
    ax.legend()
    save(fig, "08_antithetic.png")
    return df


# ------------------------------------------------------------------ 4. разные опционы
def scenarios_experiment(rng, ns, reps):
    cases = {
        "в деньгах (K=80)": Params(K=80),
        "у денег (K=100)": Params(K=100),
        "вне денег (K=130)": Params(K=130),
        "σ=10%": Params(sigma=0.10),
        "σ=40%": Params(sigma=0.40),
    }
    rows = []
    for name, p in cases.items():
        bs = black_scholes_call(p)
        for n in ns:
            err = np.array([mc_call(p, n, rng)[0] for _ in range(reps)]) - bs
            rmse = math.sqrt((err ** 2).mean())
            rows.append({"case": name, "bs_price": bs, "N": n, "rmse": rmse, "relative_rmse_percent": 100 * rmse / bs})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "scenarios.csv", index=False, float_format="%.6g")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in cases:
        sub = df[df["case"] == name]
        axes[0].loglog(sub["N"], sub["rmse"], "o-", lw=1.8, label=name)
        axes[1].loglog(sub["N"], sub["relative_rmse_percent"], "o-", lw=1.8, label=name)
    axes[0].set(xlabel="N", ylabel="RMSE", title="Абсолютная ошибка")
    axes[1].set(xlabel="N", ylabel="RMSE / цена, %", title="Относительная ошибка")
    axes[1].legend(fontsize=8)
    fig.suptitle("Наклон везде один (−1/2), но уровень ошибки зависит от опциона", y=1.02)
    save(fig, "09_scenarios.png")
    return df


# ------------------------------------------------------------------ 5. сколько N нужно
def required_n_table(sigma_pay):
    rows = []
    for target in [1.0, 0.1, 0.01, 0.001]:
        n = math.ceil((1.96 * sigma_pay / target) ** 2)
        rows.append({"target_half_width_95": target, "required_N": n})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "required_n.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    bs = black_scholes_call(BASE)
    print(f"Эталон Блэка — Шоулза: {bs:.6f}")

    if args.quick:
        ns = [10, 100, 1_000, 10_000]
        reps, reps_small = 50, 30
        ns_small = [100, 1_000, 10_000]
    else:
        ns = [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000]
        reps, reps_small = 300, 150
        ns_small = [100, 1_000, 10_000, 100_000]

    print("Графики модели...")
    fig_model(rng)
    fig_running_mean(rng, bs)
    print("Основной эксперимент...")
    df, raw, sigma_pay = convergence_experiment(rng, bs, ns, reps)
    slope = fig_convergence(df, bs)
    fig_histograms(raw, bs)
    fig_coverage(df)
    fig_time(df)
    print("Антитетическая выборка...")
    anti = antithetic_experiment(rng, bs, ns_small, reps_small)
    print("Разные опционы...")
    scen = scenarios_experiment(rng, ns_small, reps_small)
    req = required_n_table(sigma_pay)

    summary = {
        "seed": SEED, "params": BASE.__dict__, "bs_price": bs, "payoff_std_theory": sigma_pay,
        "repetitions_per_N": reps, "loglog_slope": slope,
        "mean_rmse_times_sqrtN": float(df["rmse_times_sqrtN"].mean()),
        "antithetic_mean_variance_reduction": float(anti["variance_reduction_factor"].mean()),
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))
    print(anti.round(4).to_string(index=False))
    print(req.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

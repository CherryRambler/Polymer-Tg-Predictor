"""
speed_benchmark.py

Purpose: Measure, on this machine, what each way of getting a release
curve costs and how accurate it is:
  1. PINN: train from scratch for one D with the app's settings
     (800 Adam epochs + 5 L-BFGS rounds), then read the curve.
  2. DeepONet: one-time cost (generate training data + train, deployed
     recipe), then inference per D -- single curve, and batched.
  3. Analytical Fourier series: evaluate the exact curve directly.
Accuracy is measured against the exact solution with the same corner rule
as everywhere else. From the timings it computes the break-even number of
D values at which the DeepONet's total cost (one-time + inference) drops
below re-training a PINN for every D.

For this simple slab geometry the closed-form series is faster and exact,
so neither network is needed here. The point of the PINN/DeepONet is the
method, for problems that have no closed-form solution.

Writes results/speed_benchmark.json and results/speed_benchmark.png.

Run (from the project root; takes a few minutes, keep the machine idle):
    python src/phase4_neural_operator/speed_benchmark.py
"""

import json
import math
import os
import platform
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "phase3_pinn"))
from deeponet_evaluate import corner_mask, eval_grid  # noqa: E402
from deeponet_model import load_deeponet  # noqa: E402
from generate_training_data import exact_solution  # noqa: E402
from train_deeponet import CONFIGS, DEPLOYED_CONFIG, MODEL_OUT, train_deeponet  # noqa: E402
# pyrefly: ignore [missing-import]
from pinn_train import train_pinn  # noqa: E402

OUT_JSON = "results/speed_benchmark.json"
OUT_PNG = "results/speed_benchmark.png"

D_BENCH = 0.3            # the app's default slider value
PINN_SETTINGS = dict(n_epochs=800, n_lbfgs_rounds=5)   # same as app.py
N_PINN_RUNS = 3
N_TIMED = 50             # repeats for the fast measurements (>= 20 required)
N_BATCH_D = 100          # D values in one batched DeepONet call
T_CURVE = np.linspace(0, 5.0, 200)   # the app's curve
X_CURVE = np.full_like(T_CURVE, 0.5)


def cpu_name():
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def median_time(fn, n=N_TIMED, warmup=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times


def tensors(D, x, t):
    return (torch.full((len(x), 1), float(D), dtype=torch.float32),
            torch.tensor(x, dtype=torch.float32).reshape(-1, 1),
            torch.tensor(t, dtype=torch.float32).reshape(-1, 1))


def errors_vs_exact(predict_xt, D):
    """MAE on the 51 x 51 evaluation grid (corner excluded) and on the app curve."""
    x, t = eval_grid()
    keep = corner_mask(x, t)
    grid_mae = float(np.mean(np.abs(predict_xt(x, t) - exact_solution(x, t, D))[keep]))
    curve_mae = float(np.mean(np.abs(predict_xt(X_CURVE, T_CURVE) - exact_solution(X_CURVE, T_CURVE, D))))
    return {"grid_mae": grid_mae, "app_curve_mae": curve_mae}


def bench_pinn():
    runs = []
    for seed in range(N_PINN_RUNS):
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        model = train_pinn(D=D_BENCH, model_out=None, verbose=False, **PINN_SETTINGS)
        seconds = time.perf_counter() - t0
        model.eval()

        def predict(x, t, model=model):
            with torch.no_grad():
                return model(*tensors(0, x, t)[1:]).numpy().ravel().astype(float)

        runs.append({"seed": seed, "train_seconds": seconds, **errors_vs_exact(predict, D_BENCH)})
        print(f"  PINN run {seed}: {seconds:.1f} s, grid MAE {runs[-1]['grid_mae']:.4f}")
    return {
        "D": D_BENCH, "settings": PINN_SETTINGS, "runs": runs,
        "median_train_seconds": statistics.median(r["train_seconds"] for r in runs),
        "median_grid_mae": statistics.median(r["grid_mae"] for r in runs),
        "median_app_curve_mae": statistics.median(r["app_curve_mae"] for r in runs),
    }


def bench_deeponet():
    model = load_deeponet(MODEL_OUT)
    single_inputs = tensors(D_BENCH, X_CURVE, T_CURVE)
    D_batch = np.round(np.linspace(0.01, 1.0, N_BATCH_D), 4)
    batch_inputs = (torch.tensor(np.repeat(D_batch, len(T_CURVE)), dtype=torch.float32).reshape(-1, 1),
                    torch.tensor(np.tile(X_CURVE, N_BATCH_D), dtype=torch.float32).reshape(-1, 1),
                    torch.tensor(np.tile(T_CURVE, N_BATCH_D), dtype=torch.float32).reshape(-1, 1))

    def single():
        with torch.no_grad():
            model(*single_inputs)

    def batched():
        with torch.no_grad():
            model(*batch_inputs)

    single_s, _ = median_time(single)
    batch_s, _ = median_time(batched)

    def predict(x, t):
        with torch.no_grad():
            return model(*tensors(D_BENCH, x, t)).numpy().astype(float)

    print(f"  DeepONet single curve {1e3 * single_s:.3f} ms, batch of {N_BATCH_D}: {1e3 * batch_s:.2f} ms")
    print("  DeepONet one-time cost: re-generating data and re-training the deployed recipe...")
    _, timings = train_deeponet(**CONFIGS[DEPLOYED_CONFIG], seed=0, model_out=None, verbose=False)
    print(f"  ... data {timings['data_generation_seconds']:.2f} s, training {timings['training_seconds']:.1f} s")
    return {
        "model_file": MODEL_OUT, "recipe": DEPLOYED_CONFIG,
        "single_curve_median_seconds": single_s,
        "batched": {"n_D": N_BATCH_D, "median_seconds": batch_s, "per_D_seconds": batch_s / N_BATCH_D},
        "one_time": {**timings, "total_seconds": timings["data_generation_seconds"] + timings["training_seconds"]},
        **errors_vs_exact(predict, D_BENCH),
    }


def bench_analytical():
    out = {}
    for n_terms in (50, 400):
        s, _ = median_time(lambda n=n_terms: exact_solution(X_CURVE, T_CURVE, D_BENCH, n_terms=n))
        out[f"single_curve_median_seconds_{n_terms}_terms"] = s
    print(f"  analytical curve: {1e3 * out['single_curve_median_seconds_400_terms']:.3f} ms (400 terms)")
    return out


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3, fixed order

    pinn = res["pinn"]["median_train_seconds"]
    don_once = res["deeponet"]["one_time"]["total_seconds"]
    don_inf = res["deeponet"]["single_curve_median_seconds"]
    ana = res["analytical"]["single_curve_median_seconds_400_terms"]
    n_max = max(10, int(1.5 * (res["break_even_n_D"] or 10)))
    n = np.arange(1, n_max + 1)

    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=surface)
    ax.set_facecolor(surface)
    series = [("PINN, re-trained per D", n * pinn, blue),
              ("DeepONet, train once + inference", don_once + n * don_inf, orange),
              ("Analytical series", n * ana, aqua)]
    for label, y, color in series:
        ax.plot(n, y, color=color, linewidth=2, label=label)
        ax.annotate(label.split(",")[0], (n[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=ink2, fontsize=9)
    if res["break_even_n_D"]:
        ax.axvline(res["break_even_n_D"], color=ink2, linewidth=1, linestyle=":")
        ax.annotate(f"break-even: {res['break_even_n_D']} D values", (res["break_even_n_D"], pinn),
                    xytext=(6, 8), textcoords="offset points", color=ink, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlim(1, n_max * 1.18)
    ax.set_xlabel("Number of different D values needed", color=ink2)
    ax.set_ylabel("Total wall-clock time (s, log scale)", color=ink2)
    ax.set_title("Cost of release curves for many D values", color=ink, loc="left", fontsize=12)
    ax.grid(True, color=grid, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=ink2)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=ink2)
    env = res["environment"]
    fig.text(0.01, 0.01, f"{env['cpu']} | {env['torch_threads']} torch threads | torch {env['torch_version']}",
             color=ink2, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def main():
    env = {
        "cpu": cpu_name(), "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "platform": platform.platform(), "device": "cpu",
    }
    print(f"Environment: {env}")
    res = {"environment": env, "D_benchmarked": D_BENCH, "timing_repeats_fast_paths": N_TIMED}
    print("PINN ...")
    res["pinn"] = bench_pinn()
    print("DeepONet ...")
    res["deeponet"] = bench_deeponet()
    print("Analytical ...")
    res["analytical"] = bench_analytical()

    pinn = res["pinn"]["median_train_seconds"]
    don = res["deeponet"]
    # Smallest N with  one_time + N * inference  <  N * pinn
    gap = pinn - don["single_curve_median_seconds"]
    res["break_even_n_D"] = math.floor(don["one_time"]["total_seconds"] / gap) + 1 if gap > 0 else None
    res["speedup_per_D_deeponet_inference_vs_pinn_training"] = pinn / don["single_curve_median_seconds"]
    res["speedup_per_D_analytical_vs_deeponet_inference"] = (
        don["single_curve_median_seconds"] / res["analytical"]["single_curve_median_seconds_400_terms"])
    res["note"] = ("For this slab geometry the closed-form series is exact and faster than both networks; "
                   "the networks matter for problems without a closed-form solution.")

    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=2)
    plot(res, OUT_PNG)
    print(f"\nBreak-even: {res['break_even_n_D']} D values. "
          f"DeepONet inference is {res['speedup_per_D_deeponet_inference_vs_pinn_training']:.0f}x faster per D "
          f"than training a PINN.\nSaved {OUT_JSON} and {OUT_PNG}")


if __name__ == "__main__":
    main()

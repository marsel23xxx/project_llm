"""Benchmark the forecast service's models on public series (rolling-origin evaluation).

For each dataset the last `horizon` points are held out at three different cut-offs, every model
forecasts them, and errors are averaged. Runs inside the forecast container (it has TensorFlow):

    docker compose run --rm -T --no-deps -v "$PWD/ml/data/forecast:/bench:ro" forecast \
        python - < ml/benchmark_forecast.py > ml/reports/forecast_benchmark.json
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/forecast")
from forecasting import LSTMForecaster, mae, moving_average, naive, seasonal_naive, smape  # noqa: E402

DATA = Path("/bench")
DATASETS = [
    # file, label, season, horizon
    ("airline-passengers.csv", "Penumpang pesawat bulanan (144 titik)", 12, 12),
    ("monthly-car-sales.csv", "Penjualan mobil bulanan (108 titik)", 12, 12),
    ("daily-total-female-births.csv", "Kelahiran harian (365 titik)", 7, 14),
    ("daily-min-temperatures.csv", "Suhu minimum harian (600 titik terakhir)", 7, 14),
]
ORIGINS = 3
MAX_POINTS = 600


def load(name: str) -> np.ndarray:
    with (DATA / name).open(encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    values = []
    for row in rows:
        try:
            values.append(float(row[-1].strip().lstrip("?")))
        except (ValueError, IndexError):
            continue
    return np.array(values[-MAX_POINTS:])


def main() -> None:
    report = {"method": f"rolling origin, {ORIGINS} cut-off per dataset; MAE dan sMAPE dirata-rata", "datasets": []}
    for file, label, season, horizon in DATASETS:
        series = load(file)
        errors: dict[str, list[tuple[float, float]]] = {}
        lstm_seconds = []
        for k in range(ORIGINS):
            end = len(series) - k * horizon
            train, actual = series[: end - horizon], series[end - horizon : end]
            preds = {
                "naive": naive(train, horizon),
                "moving_average": moving_average(train, horizon, season),
                "seasonal_naive": seasonal_naive(train, horizon, season),
            }
            start = time.perf_counter()
            preds["lstm"] = LSTMForecaster(season).fit(train).predict(horizon)
            lstm_seconds.append(time.perf_counter() - start)
            for name, pred in preds.items():
                errors.setdefault(name, []).append((mae(actual, pred), smape(actual, pred)))

        scores = {
            name: {"mae": round(float(np.mean([e[0] for e in errs])), 3), "smape": round(float(np.mean([e[1] for e in errs])), 2)}
            for name, errs in errors.items()
        }
        best_baseline = min((n for n in scores if n != "lstm"), key=lambda n: scores[n]["mae"])
        report["datasets"].append(
            {
                "dataset": label,
                "season": season,
                "horizon": horizon,
                "scores": scores,
                "best": min(scores, key=lambda n: scores[n]["mae"]),
                "best_baseline": best_baseline,
                "lstm_vs_best_baseline_mae_pct": round(100 * (scores["lstm"]["mae"] / scores[best_baseline]["mae"] - 1), 1),
                "lstm_fit_seconds": round(float(np.mean(lstm_seconds)), 1),
            }
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


main()

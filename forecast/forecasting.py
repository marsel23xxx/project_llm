"""Time-series forecasting: naive baselines vs a small LSTM, chosen by backtest.

Every request holds out the most recent periods, scores each candidate on them, and forecasts
with the one that had the lowest error. An LSTM is not assumed to win: on short business series
a seasonal-naive forecast is often better, and the caller is shown both scores.
"""

import math
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

MIN_POINTS = 12
MIN_POINTS_LSTM = 40
MAX_POINTS = 600
SEED = 42
Z_80 = 1.2816  # two-sided 80% interval under a normal error assumption


class ForecastError(ValueError):
    pass


# ---------- metrics ----------


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.abs(actual) + np.abs(pred)
    ratio = np.where(denom == 0, 0.0, 2 * np.abs(actual - pred) / np.where(denom == 0, 1, denom))
    return float(100 * np.mean(ratio))


# ---------- baselines ----------


def naive(train: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(train[-1], horizon)


def seasonal_naive(train: np.ndarray, horizon: int, season: int) -> np.ndarray:
    last_cycle = train[-season:]
    return np.array([last_cycle[i % season] for i in range(horizon)])


def moving_average(train: np.ndarray, horizon: int, window: int) -> np.ndarray:
    return np.repeat(train[-window:].mean(), horizon)


# ---------- LSTM ----------


class LSTMForecaster:
    """One-step LSTM on the scaled first difference of (log) values, rolled forward recursively.

    Differencing lets the network follow a trend it could not extrapolate from raw levels;
    sin/cos of the position in the season tell it where in the cycle each step is.
    """

    def __init__(self, season: int, epochs: int | None = None):
        self.season = season
        self.fixed_epochs = epochs
        self.epochs_trained = 0

    def _features(self, diffs: np.ndarray, positions: np.ndarray) -> np.ndarray:
        cols = [diffs]
        if self.season > 1:
            angle = 2 * math.pi * (positions % self.season) / self.season
            cols += [np.sin(angle), np.cos(angle)]
        return np.stack(cols, axis=-1)

    def fit(self, series: np.ndarray) -> "LSTMForecaster":
        import tensorflow as tf

        tf.keras.utils.set_random_seed(SEED)
        self.use_log = bool(series.min() >= 0)
        levels = np.log1p(series) if self.use_log else series.astype(float)
        diffs = np.diff(levels)
        self.mean, self.std = float(diffs.mean()), float(diffs.std() or 1.0)
        scaled = (diffs - self.mean) / self.std
        positions = np.arange(1, len(series))  # position of the value each diff leads to

        self.window = int(min(max(self.season, 6), 24, len(scaled) // 3))
        feats = self._features(scaled, positions)
        x = np.stack([feats[i : i + self.window] for i in range(len(scaled) - self.window)])
        y = scaled[self.window :]

        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(self.window, feats.shape[-1])),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(5e-3), loss=tf.keras.losses.Huber())

        if self.fixed_epochs:
            model.fit(x, y, epochs=self.fixed_epochs, batch_size=16, verbose=0, shuffle=True)
            self.epochs_trained = self.fixed_epochs
        else:
            n_val = max(4, len(x) // 7)
            stop = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
            history = model.fit(
                x[:-n_val], y[:-n_val], validation_data=(x[-n_val:], y[-n_val:]),
                epochs=200, batch_size=16, verbose=0, shuffle=True, callbacks=[stop],
            )  # fmt: skip
            self.epochs_trained = max(1, int(np.argmin(history.history["val_loss"])) + 1)

        self.model = model
        self._scaled, self._last_level, self._n = scaled, float(levels[-1]), len(series)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        scaled = list(self._scaled)
        level, out = self._last_level, []
        for step in range(horizon):
            positions = np.arange(self._n + step - self.window, self._n + step)
            window = self._features(np.array(scaled[-self.window :]), positions)
            next_scaled = float(self.model(window[None, ...], training=False).numpy()[0, 0])
            scaled.append(next_scaled)
            level += next_scaled * self.std + self.mean
            out.append(level)
        values = np.array(out)
        return np.expm1(values) if self.use_log else values


# ---------- orchestration ----------


def run_forecast(values: list[float], horizon: int, season: int, use_lstm: bool = True) -> dict:
    series = np.asarray(values, dtype=float)[-MAX_POINTS:]
    n = len(series)
    if n < MIN_POINTS:
        raise ForecastError(f"Butuh minimal {MIN_POINTS} periode data, baru ada {n}.")
    if not np.isfinite(series).all():
        raise ForecastError("Deret berisi nilai kosong atau tidak valid.")
    horizon = int(max(1, min(horizon, n // 2)))
    if n < 2 * season:
        season = 1  # not enough history to see a full repeating cycle twice

    # At least ~6 held-out points where the history allows: fewer is too thin to choose a model on.
    holdout = int(min(max(horizon, min(6, n // 4)), n // 3))
    train, actual = series[:-holdout], series[-holdout:]

    def baselines(history: np.ndarray, h: int) -> dict[str, np.ndarray]:
        out = {"naive": naive(history, h), "moving_average": moving_average(history, h, min(len(history), max(3, season)))}
        if season > 1:
            out["seasonal_naive"] = seasonal_naive(history, h, season)
        return out

    backtest = baselines(train, holdout)
    lstm_epochs = None
    notes = []
    if use_lstm and len(train) >= MIN_POINTS_LSTM:
        lstm = LSTMForecaster(season).fit(train)
        backtest["lstm"] = lstm.predict(holdout)
        lstm_epochs = lstm.epochs_trained
    elif use_lstm:
        notes.append(f"LSTM dilewati: butuh minimal {MIN_POINTS_LSTM + holdout} periode, baru ada {n}.")

    scores = {name: {"mae": round(mae(actual, pred), 4), "smape": round(smape(actual, pred), 2)} for name, pred in backtest.items()}
    winner = min(scores, key=lambda name: scores[name]["mae"])

    if winner == "lstm":
        # Refit on all data for the same number of epochs the backtest fit found best.
        forecast = LSTMForecaster(season, epochs=lstm_epochs).fit(series).predict(horizon)
    else:
        forecast = baselines(series, horizon)[winner]

    # Interval from the winner's backtest errors: honest about scale, but only a rough guide.
    sigma = float(np.sqrt(np.mean((actual - backtest[winner]) ** 2)))
    lower, upper = forecast - Z_80 * sigma, forecast + Z_80 * sigma
    if series.min() >= 0:
        forecast, lower = np.clip(forecast, 0, None), np.clip(lower, 0, None)

    return {
        "model": winner,
        "scores": scores,
        "backtest_points": holdout,
        "season": season,
        "points_used": n,
        "forecast": [round(float(v), 4) for v in forecast],
        "lower": [round(float(v), 4) for v in lower],
        "upper": [round(float(v), 4) for v in upper],
        "notes": notes,
    }

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.data_factory import ensure_interactions_dataset


def _encode_column(series: pd.Series) -> tuple[pd.Series, dict[str, int], dict[int, str]]:
    labels = sorted(series.unique())
    forward = {label: idx for idx, label in enumerate(labels)}
    reverse = {idx: label for label, idx in forward.items()}
    encoded = series.map(forward)
    return encoded, forward, reverse


def _fallback_matrix_factorization(train_df: pd.DataFrame, test_df: pd.DataFrame, n_users: int, n_items: int) -> tuple[np.ndarray, float]:
    rating_matrix = np.full((n_users, n_items), np.nan, dtype="float32")
    for _, row in train_df.iterrows():
        rating_matrix[int(row["user_index"]), int(row["item_index"])] = float(row["rating"])

    user_means = np.nanmean(rating_matrix, axis=1)
    global_mean = np.nanmean(rating_matrix)
    user_means = np.where(np.isnan(user_means), global_mean, user_means)

    predictions = []
    for _, row in test_df.iterrows():
        user_idx = int(row["user_index"])
        predictions.append(float(user_means[user_idx]))
    rmse = float(np.sqrt(mean_squared_error(test_df["rating"], predictions)))
    return np.array(predictions, dtype="float32"), rmse


def run_pipeline(base_dir: str | Path) -> dict:
    base_path = Path(base_dir)
    dataset_path = ensure_interactions_dataset(base_path)
    dataframe = pd.read_csv(dataset_path)

    user_index, user_map, reverse_user_map = _encode_column(dataframe["user_id"])
    item_index, item_map, reverse_item_map = _encode_column(dataframe["item_id"])
    dataframe["user_index"] = user_index
    dataframe["item_index"] = item_index

    train_df, test_df = train_test_split(dataframe, test_size=0.25, random_state=42)
    n_users = len(user_map)
    n_items = len(item_map)

    artifacts_dir = base_path / "artifacts"
    processed_dir = base_path / "data" / "processed"
    log_dir = base_path / "logs" / "fit" / datetime.now().strftime("%Y%m%d-%H%M%S")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    history_artifact = log_dir / "history.json"
    recommendation_artifact = processed_dir / "recommendations.csv"
    report_artifact = processed_dir / "recommendation_model_report.json"
    model_artifact = artifacts_dir / "best_model.joblib"

    runtime_mode = "tensorflow_keras_embeddings"

    try:
        import tensorflow as tf

        user_input = tf.keras.layers.Input(shape=(1,), name="user")
        item_input = tf.keras.layers.Input(shape=(1,), name="item")
        user_embedding = tf.keras.layers.Embedding(n_users, 8, name="user_embedding")(user_input)
        item_embedding = tf.keras.layers.Embedding(n_items, 8, name="item_embedding")(item_input)
        user_vec = tf.keras.layers.Flatten()(user_embedding)
        item_vec = tf.keras.layers.Flatten()(item_embedding)
        dot_score = tf.keras.layers.Dot(axes=1)([user_vec, item_vec])
        bias = tf.keras.layers.Dense(1, activation="linear")(tf.keras.layers.Concatenate()([user_vec, item_vec]))
        output = tf.keras.layers.Add()([dot_score, bias])

        model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir), histogram_freq=1, write_graph=True
        )
        history = model.fit(
            [train_df["user_index"], train_df["item_index"]],
            train_df["rating"],
            validation_split=0.2,
            epochs=60,
            batch_size=4,
            verbose=0,
            callbacks=[tensorboard_callback],
        )
        predictions = model.predict([test_df["user_index"], test_df["item_index"]], verbose=0).reshape(-1)
        rmse = float(np.sqrt(mean_squared_error(test_df["rating"], predictions)))
        history_payload = {
            "loss": [round(float(v), 4) for v in history.history.get("loss", [])],
            "val_loss": [round(float(v), 4) for v in history.history.get("val_loss", [])],
            "mae": [round(float(v), 4) for v in history.history.get("mae", [])],
            "val_mae": [round(float(v), 4) for v in history.history.get("val_mae", [])],
        }
        history_artifact.write_text(json.dumps(history_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        model.save(artifacts_dir / "keras_recommender.keras")
        model_artifact = artifacts_dir / "keras_recommender.keras"
    except Exception:
        runtime_mode = "fallback_without_tensorflow"
        predictions, rmse = _fallback_matrix_factorization(train_df, test_df, n_users, n_items)
        history_artifact.write_text(
            json.dumps(
                {
                    "runtime_mode": runtime_mode,
                    "note": "TensorFlow was not available in the local environment; fallback recommender was used.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        joblib.dump({"user_map": user_map, "item_map": item_map}, model_artifact)

    test_with_predictions = test_df.copy()
    test_with_predictions["predicted_rating"] = np.round(predictions, 4)
    test_with_predictions["user_id"] = test_with_predictions["user_index"].map(reverse_user_map)
    test_with_predictions["item_id"] = test_with_predictions["item_index"].map(reverse_item_map)
    test_with_predictions[
        ["user_id", "item_id", "rating", "predicted_rating"]
    ].to_csv(recommendation_artifact, index=False)

    summary = {
        "runtime_mode": runtime_mode,
        "interaction_count": int(len(dataframe)),
        "user_count": int(n_users),
        "item_count": int(n_items),
        "train_interaction_count": int(len(train_df)),
        "test_interaction_count": int(len(test_df)),
        "rmse": round(float(rmse), 4),
        "dataset_artifact": str(dataset_path),
        "recommendation_artifact": str(recommendation_artifact),
        "model_artifact": str(model_artifact),
        "tensorboard_log_dir": str(log_dir),
        "history_artifact": str(history_artifact),
        "report_artifact": str(report_artifact),
    }
    report_artifact.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

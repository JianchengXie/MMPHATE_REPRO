"""
HAR (Human Activity Recognition) full pipeline.

Two entry points:
  run_har_full()    -- train LSTM from scratch, trace, embed, cache
  import_har_run()  -- convert an existing legacy run directory into project cache format
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mmphate_repro.embeddings.runners import center_columns, embed_mmphate, embed_pca
from mmphate_repro.utils.paths import cache_dir


@dataclass(frozen=True)
class HARRunConfig:
    seed: int = 42
    epochs: int = 1000
    batch_size: int = 32
    lr: float = 2e-5
    n_units: int = 30
    n_classes: int = 6

    # One sample per class -> S = n_classes = 6 (matches paper run)
    per_class_samples: int = 1

    # Epoch subsampling: same first-then-stride scheme as area2bump
    epoch_first: int = 29
    epoch_stride: int = 5

    # Embedding settings
    n_jobs_mphate: int = 1


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _set_seeds(seed: int, tf):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    tf.keras.utils.set_random_seed(seed)


def _select_trace_indices(Y_onehot_2d: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """
    Y_onehot_2d: (N, n_classes) – per-sequence label (use Y[:, 0, :] for seq-to-seq data).
    Returns indices of per_class samples from each class.
    """
    rng = np.random.default_rng(seed)
    n_classes = Y_onehot_2d.shape[1]
    idxs = []
    for c in range(n_classes):
        cand = np.where(Y_onehot_2d[:, c] == 1)[0]
        k = min(per_class, len(cand))
        idxs.append(rng.choice(cand, size=k, replace=False))
    return np.concatenate(idxs)


def _epoch_subsample(n_epochs: int, first: int, stride: int) -> np.ndarray:
    first_part = np.arange(min(first, n_epochs), dtype=int)
    rest = np.arange(first, n_epochs, stride, dtype=int) if n_epochs > first else np.array([], dtype=int)
    return np.unique(np.concatenate([first_part, rest]))


def _activity_label(trace_et_h_s: np.ndarray, sample_classes: np.ndarray) -> np.ndarray:
    """
    For each (et_idx, h) point, the activity is the class of the sample with
    the highest activation value.

    trace_et_h_s : (ET, H, S)
    sample_classes: (S,) integer class index for each sample
    Returns: (ET * H,) integer activity label
    """
    ET, H, S = trace_et_h_s.shape
    # argmax over S axis → (ET, H) index into sample_classes
    dominant_sample = trace_et_h_s.argmax(axis=2)       # (ET, H)
    activity = sample_classes[dominant_sample]           # (ET, H)
    return activity.reshape(ET * H)                      # (ET*H,)


# ---------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------

def run_har_full(data_dir: Path, run_id: str, cfg: HARRunConfig) -> Path:
    """
    Train HAR LSTM, extract traces, embed (MM-PHATE / PCA / t-SNE), cache results.
    Writes to results_cache/har/<run_id>/ and returns that path.
    """
    import tensorflow as tf
    import m_phate
    import m_phate.train
    from mmphate_repro.io.har import load_har_from_dir
    from mmphate_repro.models.har_lstm import build_har_lstm, HARLSTMConfig
    from mmphate_repro.pipelines.trace_standardize import standardize_trace_tensor

    _set_seeds(cfg.seed, tf)

    out_dir = cache_dir() / "har" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_har_from_dir(data_dir)
    trainX, trainY = data.trainX, data.trainY
    testX, testY = data.testX, data.testY
    T, F = trainX.shape[1], trainX.shape[2]

    # Trace sample selection (per_class_samples per class)
    # trainY[:, 0, :] gives (N, 6) since label is constant per sequence
    trace_idx = _select_trace_indices(trainY[:, 0, :], cfg.per_class_samples, cfg.seed)
    x_trace = trainX[trace_idx]        # (S, T, F)
    S = x_trace.shape[0]
    sample_classes = trainY[trace_idx, 0, :].argmax(axis=1)  # (S,) ground-truth class

    # Build model
    mcfg = HARLSTMConfig(n_units=cfg.n_units, n_classes=cfg.n_classes, lr=cfg.lr, seed=cfg.seed)
    model, trace_model = build_har_lstm((T, F), mcfg)

    # TraceHistory records trace_model(x_trace) after each epoch
    trace_cb = m_phate.train.TraceHistory(x_trace, trace_model)

    hist = model.fit(
        trainX, trainY,
        validation_data=(testX, testY),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=2,
        callbacks=[trace_cb],
    )

    np.save(out_dir / "train_accuracy.npy", np.array(hist.history.get("accuracy", [])))
    np.save(out_dir / "train_loss.npy",     np.array(hist.history.get("loss", [])))
    np.save(out_dir / "val_accuracy.npy",   np.array(hist.history.get("val_accuracy", [])))
    np.save(out_dir / "val_loss.npy",       np.array(hist.history.get("val_loss", [])))

    # Standardize trace tensor → (E, T, H, S)
    raw = np.array(trace_cb.trace)
    std = standardize_trace_tensor(raw, n_samples=S, n_timesteps=T, n_units=cfg.n_units)

    # Subsample epochs
    epoch_samples = _epoch_subsample(cfg.epochs, cfg.epoch_first, cfg.epoch_stride)
    std_sub = std[epoch_samples]                   # (E_s, T, H, S)
    E_s = std_sub.shape[0]
    ET = E_s * T

    trace_et_h_s = std_sub.reshape(ET, cfg.n_units, S)
    np.save(out_dir / "trace_et_h_s.npy", trace_et_h_s)
    np.save(out_dir / "epoch_samples.npy", epoch_samples)

    # Labels (length ET*H)
    epoch_per_ET    = np.repeat(epoch_samples, T)
    timestep_per_ET = np.tile(np.arange(T, dtype=int), E_s)
    H = cfg.n_units

    epoch_label    = np.repeat(epoch_per_ET, H)
    timestep_label = np.repeat(timestep_per_ET, H)
    unit_label     = np.tile(np.arange(H, dtype=int), ET)
    act_label      = _activity_label(trace_et_h_s, sample_classes)

    labdir = out_dir / "labels"
    labdir.mkdir(parents=True, exist_ok=True)
    np.save(labdir / "epoch_label.npy",    epoch_label)
    np.save(labdir / "timestep_label.npy", timestep_label)
    np.save(labdir / "unit_label.npy",     unit_label)
    np.save(labdir / "activity_label.npy", act_label)

    meta = dict(
        run_id=run_id, seed=cfg.seed, epochs=cfg.epochs,
        batch_size=cfg.batch_size, lr=cfg.lr,
        n_units=cfg.n_units, n_classes=cfg.n_classes,
        per_class_samples=cfg.per_class_samples,
        epoch_samples=epoch_samples.tolist(),
        ET=int(ET), H=int(H), S=int(S),
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Embeddings
    X  = trace_et_h_s.reshape(ET * H, S)
    Xc = center_columns(X)

    embed_mmphate(trace_et_h_s, n_components=3, n_jobs=cfg.n_jobs_mphate,
                  cache_path=out_dir / "embedding_mmphate_3D.npy")
    embed_pca(Xc, n_components=3, random_state=24,
              cache_path=out_dir / "embedding_pca_3D.npy")

    return out_dir


# ---------------------------------------------------------------
# Import from legacy run directory
# ---------------------------------------------------------------

def import_har_run(legacy_dir: Path, run_id: str, n_jobs_mphate: int = 1) -> Path:
    """
    Convert an existing legacy HAR run directory into the project cache format.

    Expected files in legacy_dir:
      "1 layer_30 units_LSTM__accuracy 0.931_digit activity.npy"  (E_s, S, T, H)
      "1 layer_30 units_LSTM__accuracy 0.931_m-phate 3D.npy"      (ET*H, 3)
      "1 layer_30 units_LSTM__accuracy 0.931_m-phate_epoch_label.npy"
      "1 layer_30 units_LSTM__accuracy 0.931_m-phate_intrinsic_step.npy"
      "1 layer_30 units_LSTM__accuracy 0.931_m-phate_hidden_unit.npy"
      "1 layer_30 units_LSTM__accuracy 0.931_m-phate_most_active_output.npy"
      "epoch_samples.npy"
      "accuracy.npy", "loss.npy", "val_loss.npy", "test_accuracy.npy"

    PCA and t-SNE are recomputed from the digit-activity trace.
    """
    prefix = "1 layer_30 units_LSTM__accuracy 0.931_"

    out_dir = cache_dir() / "har" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- trace: (E_s, S, T, H) → (E_s*T, H, S) ----
    digit_activity = np.load(legacy_dir / f"{prefix}digit activity.npy")
    E_s, S, T, H = digit_activity.shape
    # Transpose (E_s, S, T, H) → (E_s, T, H, S), then reshape to (ET, H, S)
    trace_et_h_s = digit_activity.transpose(0, 2, 3, 1).reshape(E_s * T, H, S)
    np.save(out_dir / "trace_et_h_s.npy", trace_et_h_s)

    # ---- copy epoch_samples ----
    epoch_samples = np.load(legacy_dir / "epoch_samples.npy")
    np.save(out_dir / "epoch_samples.npy", epoch_samples)

    # ---- training curves ----
    for src_name, dst_name in [
        ("accuracy.npy",      "train_accuracy.npy"),
        ("loss.npy",          "train_loss.npy"),
        ("val_loss.npy",      "val_loss.npy"),
        ("test_accuracy.npy", "test_accuracy.npy"),
    ]:
        p = legacy_dir / src_name
        if p.exists():
            np.save(out_dir / dst_name, np.load(p))

    # ---- labels ----
    labdir = out_dir / "labels"
    labdir.mkdir(parents=True, exist_ok=True)

    np.save(labdir / "epoch_label.npy",
            np.load(legacy_dir / f"{prefix}m-phate_epoch_label.npy"))
    np.save(labdir / "timestep_label.npy",
            np.load(legacy_dir / f"{prefix}m-phate_intrinsic_step.npy"))
    np.save(labdir / "unit_label.npy",
            np.load(legacy_dir / f"{prefix}m-phate_hidden_unit.npy"))
    np.save(labdir / "activity_label.npy",
            np.load(legacy_dir / f"{prefix}m-phate_most_active_output.npy").astype(int))

    # ---- copy MM-PHATE embedding (already computed) ----
    np.save(out_dir / "embedding_mmphate_3D.npy",
            np.load(legacy_dir / f"{prefix}m-phate 3D.npy"))

    # ---- compute PCA and t-SNE from trace ----
    ET = E_s * T
    X  = trace_et_h_s.reshape(ET * H, S)
    Xc = center_columns(X)

    embed_pca(Xc, n_components=3, random_state=24,
              cache_path=out_dir / "embedding_pca_3D.npy")

    meta = dict(
        run_id=run_id, source="legacy_import",
        legacy_dir=str(legacy_dir),
        E_s=int(E_s), T=int(T), H=int(H), S=int(S),
        ET=int(ET), epoch_samples=epoch_samples.tolist(),
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return out_dir

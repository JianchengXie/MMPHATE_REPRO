from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import numpy as np

# TF env flags should be set before importing tensorflow in some setups.
# Here we assume tf is already imported elsewhere; users can still set OS env vars if needed.

from mmphate_repro.embeddings.runners import center_columns, embed_pca, embed_tsne, embed_mmphate, TSNEConfig
from mmphate_repro.utils.paths import cache_dir


@dataclass(frozen=True)
class Area2BumpRunConfig:
    seed: int = 42
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-4
    n_units: int = 20
    n_classes: int = 8

    # Trace sampling for embedding (p samples total)
    per_class_samples: int = 5

    # Subsampling for embedding tensor
    epoch_scheme: str = "paper"  # "paper" or "linspace"
    n_epoch_samples: int = 30
    n_timestep_samples: int = 100

    # Embedding settings
    n_jobs_mphate: int = 1
    tsne: TSNEConfig = TSNEConfig(perplexity=50, n_iter=2000, random_state=24)


def _set_seeds(seed: int, tf):
    # determinism best-effort (GPU ops may still vary)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    tf.keras.utils.set_random_seed(seed)


def _select_trace_indices(y_onehot: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_classes = y_onehot.shape[1]
    idxs = []
    for c in range(n_classes):
        cand = np.where(y_onehot[:, c] == 1)[0]
        if len(cand) == 0:
            continue
        k = min(per_class, len(cand))
        idxs.append(rng.choice(cand, size=k, replace=False))
    return np.concatenate(idxs)


def _subsample_epoch_indices(epochs: int, n_samples: int) -> np.ndarray:
    # match your paper style: evenly spaced sample of epochs
    return np.linspace(0, epochs - 1, n_samples, endpoint=True, dtype=int)


def _subsample_timestep_indices(T: int, n_samples: int) -> np.ndarray:
    return np.linspace(0, T - 1, n_samples, endpoint=True, dtype=int)


def _epoch_samples_first_then_stride(epochs: int, first: int = 29, stride: int = 5) -> np.ndarray:
    first_part = np.arange(min(first, epochs), dtype=int)
    rest = np.arange(first, epochs, stride, dtype=int) if epochs > first else np.array([], dtype=int)
    out = np.unique(np.concatenate([first_part, rest]))
    return out


def run_area2bump_full(data_dir: Path, run_id: str, cfg: Area2BumpRunConfig) -> Path:
    """
    Writes caches to results_cache/area2bump/<run_id>/ and returns that path.
    """
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    import m_phate
    import m_phate.train
    from mmphate_repro.io.area2bump import load_area2bump_from_dir
    from mmphate_repro.models.area2bump_lstm import build_lstm_classifier, LSTMConfig
    from mmphate_repro.pipelines.trace_standardize import standardize_trace_tensor

    _set_seeds(cfg.seed, tf)

    out_dir = cache_dir() / "area2bump" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_area2bump_from_dir(data_dir)

    trainX, trainy = data.trainX, to_categorical(data.trainy, num_classes=cfg.n_classes)
    testX, testy   = data.testX,  to_categorical(data.testy,  num_classes=cfg.n_classes)

    # Choose trace subset (p samples for embedding)
    trace_idx = _select_trace_indices(trainy, cfg.per_class_samples, cfg.seed)
    x_trace = trainX[trace_idx]
    n_trace_samples = x_trace.shape[0]
    n_timesteps = trainX.shape[1]
    n_features = trainX.shape[2]

    # Build model
    mcfg = LSTMConfig(n_units=cfg.n_units, n_classes=cfg.n_classes, lr=cfg.lr, seed=cfg.seed)
    model, trace_model = build_lstm_classifier((n_timesteps, n_features), mcfg)

    # TraceHistory (records trace_model(x_trace) each epoch)
    trace_cb = m_phate.train.TraceHistory(x_trace, trace_model)

    # Train
    hist = model.fit(
        trainX, trainy,
        validation_data=(testX, testy),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=2,
        callbacks=[trace_cb],
    )

    # Save training curves (small)
    np.save(out_dir / "train_accuracy.npy", np.array(hist.history.get("accuracy", [])))
    np.save(out_dir / "train_loss.npy", np.array(hist.history.get("loss", [])))
    np.save(out_dir / "val_accuracy.npy", np.array(hist.history.get("val_accuracy", [])))
    np.save(out_dir / "val_loss.npy", np.array(hist.history.get("val_loss", [])))

    # Convert trace
    raw = np.array(trace_cb.trace)  # 4D expected
    std = standardize_trace_tensor(raw, n_samples=n_trace_samples, n_timesteps=n_timesteps, n_units=cfg.n_units)  # (E,T,H,S)

    # Subsample epochs/timesteps for embedding tensor
    if getattr(cfg, "epoch_scheme", "paper") == "paper":
        epoch_samples = _epoch_samples_first_then_stride(cfg.epochs, first=29, stride=5)
    else:
        epoch_samples = _subsample_epoch_indices(cfg.epochs, cfg.n_epoch_samples)
    timestep_samples = _subsample_timestep_indices(n_timesteps, cfg.n_timestep_samples)

    std_sub = std[epoch_samples][:, timestep_samples]  # (E_s, T_s, H, S)

    E_s, T_s, H, S = std_sub.shape
    ET = E_s * T_s

    # reshape to (ET, H, S) for MM-PHATE
    trace_et_h_s = std_sub.reshape(ET, H, S)
    np.save(out_dir / "trace_et_h_s.npy", trace_et_h_s)
    np.save(out_dir / "epoch_samples.npy", epoch_samples)
    np.save(out_dir / "timestep_samples.npy", timestep_samples)

    # label vectors for plotting (length ET*H)
    epoch_per_ET = np.repeat(epoch_samples, T_s)
    timestep_per_ET = np.tile(timestep_samples, E_s)

    epoch_label = np.repeat(epoch_per_ET, H)
    timestep_label = np.repeat(timestep_per_ET, H)
    unit_label = np.tile(np.arange(H, dtype=int), ET)

    labdir = out_dir / "labels"
    labdir.mkdir(parents=True, exist_ok=True)
    np.save(labdir / "epoch_label.npy", epoch_label)
    np.save(labdir / "timestep_label.npy", timestep_label)
    np.save(labdir / "unit_label.npy", unit_label)

    meta = dict(
        run_id=run_id,
        seed=cfg.seed,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        n_units=cfg.n_units,
        n_classes=cfg.n_classes,
        per_class_samples=cfg.per_class_samples,
        epoch_samples=epoch_samples.tolist(),
        timestep_samples=timestep_samples.tolist(),
        ET=int(ET), H=int(H), S=int(S),
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # -------------------------
    # Embeddings (cached)
    # -------------------------
    # PCA/t-SNE operate on 2D matrix (ET*H, S)
    X = trace_et_h_s.reshape(ET * H, S)
    Xc = center_columns(X)

    mm = embed_mmphate(trace_et_h_s, n_components=3, n_jobs=cfg.n_jobs_mphate, cache_path=out_dir / "embedding_mmphate_3D.npy")
    pca = embed_pca(Xc, n_components=3, random_state=24, cache_path=out_dir / "embedding_pca_3D.npy")
    tsne = embed_tsne(Xc, cfg=cfg.tsne, cache_path=out_dir / "embedding_tsne_3D.npy")

    return out_dir


# ---------------------------------------------------------------
# Import from legacy run directory
# ---------------------------------------------------------------

def import_area2bump_run(legacy_dir: Path, run_id: str, n_jobs_mphate: int = 1) -> Path:
    """
    Convert an existing legacy Area2Bump run directory into the project cache format.

    Expected files in legacy_dir:
      "1 layer_20 units_LSTM__accuracy 1.000_digit activity.npy"  (E_s, n_class, T_s, H)
      "1 layer_20 units_LSTM__accuracy 1.000_m-phate 3D.npy"      (ET*H, 3)
      "1 layer_20 units_LSTM__accuracy 1.000_m-phate_epoch_label.npy"
      "1 layer_20 units_LSTM__accuracy 1.000_m-phate_intrinsic_step.npy"
      "1 layer_20 units_LSTM__accuracy 1.000_m-phate_hidden_unit.npy"
      "1 layer_20 units_LSTM__accuracy 1.000_m-phate_most_active_output.npy"
      "epoch_samples.npy"
      "accuracy.npy", "loss.npy", "val_loss.npy", "test_accuracy.npy"

    PCA and t-SNE are recomputed from the digit-activity trace.
    """
    prefix = "1 layer_20 units_LSTM__accuracy 1.000_"

    out_dir = cache_dir() / "area2bump" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- trace: (E_s, n_class, T_s, H) -> (E_s*T_s, H, n_class) ----
    digit_activity = np.load(legacy_dir / f"{prefix}digit activity.npy")
    E_s, n_class, T_s, H = digit_activity.shape
    # Transpose (E_s, n_class, T_s, H) -> (E_s, T_s, H, n_class), reshape to (ET, H, S)
    trace_et_h_s = digit_activity.transpose(0, 2, 3, 1).reshape(E_s * T_s, H, n_class)
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
    np.save(labdir / "class_label.npy",
            np.load(legacy_dir / f"{prefix}m-phate_most_active_output.npy").astype(int))

    # ---- copy MM-PHATE embedding (already computed) ----
    np.save(out_dir / "embedding_mmphate_3D.npy",
            np.load(legacy_dir / f"{prefix}m-phate 3D.npy"))

    # ---- compute PCA and t-SNE from trace ----
    ET = E_s * T_s
    S = n_class
    X = trace_et_h_s.reshape(ET * H, S)
    Xc = center_columns(X)

    embed_pca(Xc, n_components=3, random_state=24,
              cache_path=out_dir / "embedding_pca_3D.npy")
    embed_tsne(Xc, cfg=TSNEConfig(perplexity=50, n_iter=2000, random_state=24),
               cache_path=out_dir / "embedding_tsne_3D.npy")

    meta = dict(
        run_id=run_id, source="legacy_import",
        legacy_dir=str(legacy_dir),
        E_s=int(E_s), T_s=int(T_s), H=int(H), S=int(S),
        ET=int(ET), epoch_samples=epoch_samples.tolist(),
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return out_dir
"""
Bifurcation-warp suite: generate synthetic RNN-like traces from
2-D latent ODE systems (Hopf, Pitchfork) with optional diffeomorphic
state warps and monotone time warps.

MAIN (2x2 factorial: dynamics x warp)
  1) hopf_bif__no_warp
  2) hopf_bif__warp
  3) pitchfork_bif__no_warp
  4) pitchfork_bif__warp

SUPPLEMENT (warp-only stress test with no confounds)
  5) hopf_bif__clean_vs_warped   (A and B share identical latent trajectories; B is warped)

Key confound control:
- When warp is the ONLY difference between groups (scenario 5), we simulate ONE latent
  trajectory and copy it, then warp only group B.
- In all scenarios, groups have equal unit counts: 3 and 3.

Save format: rnn_traces.npz per scenario directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Any, List, Literal, Optional, Tuple

import numpy as np


# ============================================================
# 1) ODE right-hand sides
# ============================================================

def _hopf_rhs(xy: np.ndarray, *, mu: float, omega: float = 1.0, **_kw) -> np.ndarray:
    """Supercritical Hopf normal form in R^2."""
    x, y = xy[..., 0], xy[..., 1]
    r2 = x * x + y * y
    return np.stack([mu * x - omega * y - r2 * x,
                     omega * x + mu * y - r2 * y], axis=-1)


def _pitchfork_rhs(xy: np.ndarray, *, mu: float, lam: float = 2.0, **_kw) -> np.ndarray:
    """Supercritical pitchfork in x, linearly stable y."""
    x, y = xy[..., 0], xy[..., 1]
    return np.stack([mu * x - x ** 3, -lam * y], axis=-1)


_RHS_REGISTRY: Dict[str, Callable] = {
    "hopf": _hopf_rhs,
    "pitchfork": _pitchfork_rhs,
}


def _rk4_step(f: Callable, xy: np.ndarray, dt: float, **kw) -> np.ndarray:
    k1 = f(xy, **kw)
    k2 = f(xy + 0.5 * dt * k1, **kw)
    k3 = f(xy + 0.5 * dt * k2, **kw)
    k4 = f(xy + dt * k3, **kw)
    return xy + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ============================================================
# 2) Warps
# ============================================================

def warp_state_diffeo(XY: np.ndarray, kappa: float = 0.15) -> np.ndarray:
    """Smooth near-identity diffeomorphic warp in R^2.  XY: (T, 2, S)."""
    x, y = XY[:, 0, :], XY[:, 1, :]
    return np.stack([x + kappa * x * y,
                     y + kappa * x ** 2], axis=1).astype(np.float32)


def warp_time_monotone(XY: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """Monotone time reparametrization.  XY: (T, 2, S)."""
    T, _, S = XY.shape
    t = np.linspace(0.0, 1.0, T)
    a = float(np.clip(strength, 0.0, 0.95))
    idx = ((1 - a) * t + a * t ** 2) * (T - 1)
    out = np.zeros_like(XY)
    base = np.arange(T)
    for s in range(S):
        for d in range(2):
            out[:, d, s] = np.interp(idx, base, XY[:, d, s])
    return out.astype(np.float32)


def phase_resample_epoch(XY: np.ndarray, k_cycles: float) -> np.ndarray:
    """Resample to uniform phase grid (Hopf only).  XY: (T, 2, S)."""
    T, _, S = XY.shape
    out = np.zeros_like(XY)
    for s in range(S):
        x, y = XY[:, 0, s], XY[:, 1, s]
        th = np.maximum.accumulate(np.unwrap(np.arctan2(y, x)))
        th_grid = th[0] + 2 * np.pi * k_cycles * np.linspace(0, 1, T)
        out[:, 0, s] = np.interp(th_grid, th, x)
        out[:, 1, s] = np.interp(th_grid, th, y)
    return out.astype(np.float32)


# ============================================================
# 3) Readout: latents -> hidden activations
# ============================================================

def _apply_readout(
    latent_a: np.ndarray,
    latent_b: np.ndarray,
    w_shared: np.ndarray,
    cfg: "SuiteConfig",
    rng: np.random.Generator,
    gain_a: np.ndarray, bias_a: np.ndarray,
    gain_b: np.ndarray, bias_b: np.ndarray,
) -> np.ndarray:
    """Map 2-D latents for groups A and B into (T, H, S) hidden activations."""
    T, _, S = latent_a.shape
    NA, NB = cfg.units_a, cfg.units_b
    H = NA + NB

    def _base_signal(lat: np.ndarray) -> np.ndarray:
        x, y = lat[:, 0, :], lat[:, 1, :]
        r = np.sqrt(x ** 2 + y ** 2)
        mix = x * w_shared[0] + y * w_shared[1]
        sig = cfg.alpha_mixture * mix + cfg.gamma_radius * r
        return np.tanh(sig).astype(np.float32) if cfg.nonlinearity == "tanh" else sig.astype(np.float32)

    def _expand(base: np.ndarray, n: int, gain: np.ndarray, bias: np.ndarray) -> np.ndarray:
        h = np.broadcast_to(base[None], (n, T, S)).copy()
        if cfg.unit_gain_jitter:
            h *= gain[:, None, None]
        if cfg.unit_bias_jitter:
            h += bias[:, None, None]
        if cfg.unit_noise > 0:
            h += rng.normal(0, cfg.unit_noise, size=h.shape).astype(np.float32)
        return h

    ha = _expand(_base_signal(latent_a), NA, gain_a, bias_a)
    hb = _expand(_base_signal(latent_b), NB, gain_b, bias_b)

    out = np.zeros((T, H, S), dtype=np.float32)
    out[:, :NA, :] = ha.transpose(1, 0, 2)
    out[:, NA:, :] = hb.transpose(1, 0, 2)
    return out


# ============================================================
# 4) Config
# ============================================================

@dataclass
class SuiteConfig:
    # --- Group A (first units_a hidden units) ---
    system_a: str = "hopf"
    mu_a_start: float = -1.0
    mu_a_end: float = -1.0
    units_a: int = 3

    # --- Group B (remaining units_b hidden units) ---
    system_b: str = "hopf"
    mu_b_start: float = -1.0
    mu_b_end: float = 2.0
    units_b: int = 3

    # --- Time / ensemble ---
    epochs: int = 101
    timesteps: int = 80
    samples: int = 10

    # --- Shared ODE parameters ---
    omega: float = 1.0
    lam: float = 2.0
    dt: float = 0.1
    process_noise: float = 0.001

    # --- Readout ---
    nonlinearity: str = "identity"
    alpha_mixture: float = 0.2
    gamma_radius: float = 1.2
    unit_noise: float = 0.001
    unit_gain_jitter: float = 0.0
    unit_bias_jitter: float = 0.0

    # --- Epoch continuity ---
    reset_each_epoch: bool = False

    # --- Hopf phase resampling ---
    phase_resample: bool = False
    phase_cycles_target: Optional[float] = 2.0

    # --- Per-group warps ---
    warp_state_a: bool = False
    warp_time_a: bool = False
    warp_state_b: bool = False
    warp_time_b: bool = False
    warp_state_kappa: float = 0.15
    warp_time_strength: float = 0.35

    # --- Misc ---
    seed: int = 24
    # Empty string means caller must supply out_dir explicitly
    out_dir: str = ""

    @property
    def hidden_units(self) -> int:
        return self.units_a + self.units_b


# ============================================================
# 5) Simulation helpers
# ============================================================

def _mu_schedule(start: float, end: float, n_epochs: int) -> np.ndarray:
    return np.linspace(start, end, n_epochs, dtype=np.float32)


def _simulate_latent(
    system: str,
    mu_sched: np.ndarray,
    xy_init: np.ndarray,
    cfg: SuiteConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Integrate one 2-D latent group across all epochs.  Returns (E, T, 2, S)."""
    E, T, S = cfg.epochs, cfg.timesteps, xy_init.shape[0]
    f = _RHS_REGISTRY[system]
    ode_kw: Dict[str, float] = dict(omega=cfg.omega, lam=cfg.lam)

    latent = np.zeros((E, T, 2, S), dtype=np.float32)
    xy = xy_init.copy()

    for e in range(E):
        mu_e = float(mu_sched[e])
        if cfg.reset_each_epoch:
            xy = rng.normal(0, 0.6, size=(S, 2)).astype(np.float32)
        for t in range(T):
            xy = _rk4_step(f, xy, cfg.dt, mu=mu_e, **ode_kw)
            if cfg.process_noise > 0:
                xy += rng.normal(0, cfg.process_noise, size=xy.shape).astype(np.float32)
            latent[e, t] = xy.T

    return latent


def _apply_epoch_warps(
    latent: np.ndarray,
    do_state: bool,
    do_time: bool,
    cfg: SuiteConfig,
) -> np.ndarray:
    for e in range(latent.shape[0]):
        if do_state:
            latent[e] = warp_state_diffeo(latent[e], kappa=cfg.warp_state_kappa)
        if do_time:
            latent[e] = warp_time_monotone(latent[e], strength=cfg.warp_time_strength)
    return latent


def _apply_hopf_phase_resample(latent: np.ndarray, cfg: SuiteConfig) -> np.ndarray:
    if not cfg.phase_resample:
        return latent
    k = (cfg.phase_cycles_target if cfg.phase_cycles_target is not None
         else (cfg.omega * cfg.timesteps * cfg.dt) / (2 * np.pi))
    for e in range(latent.shape[0]):
        latent[e] = phase_resample_epoch(latent[e], float(k))
    return latent


# ============================================================
# 6) Main generators
# ============================================================

def generate_dataset(cfg: SuiteConfig) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    E, T, S = cfg.epochs, cfg.timesteps, cfg.samples
    NA, NB = cfg.units_a, cfg.units_b
    H = cfg.hidden_units

    mu_a = _mu_schedule(cfg.mu_a_start, cfg.mu_a_end, E)
    mu_b = _mu_schedule(cfg.mu_b_start, cfg.mu_b_end, E)

    xy_a = rng.normal(0, 0.6, size=(S, 2)).astype(np.float32)
    xy_b = xy_a.copy()

    latent_a = _simulate_latent(cfg.system_a, mu_a, xy_a, cfg, rng)
    latent_b = _simulate_latent(cfg.system_b, mu_b, xy_b, cfg, rng)

    if cfg.system_a == "hopf":
        latent_a = _apply_hopf_phase_resample(latent_a, cfg)
    if cfg.system_b == "hopf":
        latent_b = _apply_hopf_phase_resample(latent_b, cfg)

    latent_a = _apply_epoch_warps(latent_a, cfg.warp_state_a, cfg.warp_time_a, cfg)
    latent_b = _apply_epoch_warps(latent_b, cfg.warp_state_b, cfg.warp_time_b, cfg)

    phi = rng.uniform(0.2 * np.pi, 0.8 * np.pi)
    w_shared = np.array([np.cos(phi), np.sin(phi)], dtype=np.float32)

    gain_a = 1.0 + cfg.unit_gain_jitter * rng.normal(0, 1, size=NA).astype(np.float32)
    bias_a = cfg.unit_bias_jitter * rng.normal(0, 1, size=NA).astype(np.float32)
    gain_b = 1.0 + cfg.unit_gain_jitter * rng.normal(0, 1, size=NB).astype(np.float32)
    bias_b = cfg.unit_bias_jitter * rng.normal(0, 1, size=NB).astype(np.float32)

    data = np.zeros((E * T, H, S), dtype=np.float32)
    for e in range(E):
        data[e * T:(e + 1) * T] = _apply_readout(
            latent_a[e], latent_b[e], w_shared, cfg, rng,
            gain_a, bias_a, gain_b, bias_b,
        )

    epoch_idx = np.repeat(np.arange(E, dtype=np.int32), T)
    time_idx = np.tile(np.arange(T, dtype=np.int32), E)

    meta = dict(
        system_a=cfg.system_a,
        system_b=cfg.system_b,
        epoch_idx=epoch_idx,
        time_idx=time_idx,
        mu_a=mu_a,
        mu_b=mu_b,
        mu_per_row_a=np.repeat(mu_a, T),
        mu_per_row_b=np.repeat(mu_b, T),
        units_a=NA,
        units_b=NB,
        w_shared=w_shared,
        warp_state_a=cfg.warp_state_a,
        warp_time_a=cfg.warp_time_a,
        warp_state_b=cfg.warp_state_b,
        warp_time_b=cfg.warp_time_b,
        warp_state_kappa=cfg.warp_state_kappa,
        warp_time_strength=cfg.warp_time_strength,
    )

    return dict(data=data, latent_a=latent_a, latent_b=latent_b, meta=meta)


def generate_dataset_warp_only_same_latents(cfg: SuiteConfig) -> Dict[str, Any]:
    """
    Warp-only stress test: A and B share identical latent trajectories; only B is warped.
    """
    rng = np.random.default_rng(cfg.seed)
    E, T, S = cfg.epochs, cfg.timesteps, cfg.samples
    NA, NB = cfg.units_a, cfg.units_b
    H = cfg.hidden_units

    assert cfg.system_a == cfg.system_b, "warp-only generator assumes same system for A and B"
    assert (cfg.mu_a_start, cfg.mu_a_end) == (cfg.mu_b_start, cfg.mu_b_end), \
        "warp-only generator assumes same mu schedule"

    mu = _mu_schedule(cfg.mu_a_start, cfg.mu_a_end, E)
    xy0 = rng.normal(0, 0.6, size=(S, 2)).astype(np.float32)
    latent = _simulate_latent(cfg.system_a, mu, xy0, cfg, rng)

    if cfg.system_a == "hopf":
        latent = _apply_hopf_phase_resample(latent, cfg)

    latent_a = latent.copy()
    latent_b = latent.copy()
    latent_b = _apply_epoch_warps(latent_b, cfg.warp_state_b, cfg.warp_time_b, cfg)

    phi = rng.uniform(0.2 * np.pi, 0.8 * np.pi)
    w_shared = np.array([np.cos(phi), np.sin(phi)], dtype=np.float32)

    gain_a = 1.0 + cfg.unit_gain_jitter * rng.normal(0, 1, size=NA).astype(np.float32)
    bias_a = cfg.unit_bias_jitter * rng.normal(0, 1, size=NA).astype(np.float32)
    gain_b = 1.0 + cfg.unit_gain_jitter * rng.normal(0, 1, size=NB).astype(np.float32)
    bias_b = cfg.unit_bias_jitter * rng.normal(0, 1, size=NB).astype(np.float32)

    data = np.zeros((E * T, H, S), dtype=np.float32)
    for e in range(E):
        data[e * T:(e + 1) * T] = _apply_readout(
            latent_a[e], latent_b[e], w_shared, cfg, rng,
            gain_a, bias_a, gain_b, bias_b,
        )

    epoch_idx = np.repeat(np.arange(E, dtype=np.int32), T)
    time_idx = np.tile(np.arange(T, dtype=np.int32), E)

    meta = dict(
        system_a=cfg.system_a,
        system_b=cfg.system_b,
        epoch_idx=epoch_idx,
        time_idx=time_idx,
        mu_a=mu,
        mu_b=mu,
        mu_per_row_a=np.repeat(mu, T),
        mu_per_row_b=np.repeat(mu, T),
        units_a=NA,
        units_b=NB,
        w_shared=w_shared,
        warp_state_a=False,
        warp_time_a=False,
        warp_state_b=cfg.warp_state_b,
        warp_time_b=cfg.warp_time_b,
        warp_state_kappa=cfg.warp_state_kappa,
        warp_time_strength=cfg.warp_time_strength,
        warp_only_same_latents=True,
    )

    assert np.array_equal(latent_a, latent), \
        "latent_a should exactly equal the shared latent pre-warp."

    return dict(data=data, latent_a=latent_a, latent_b=latent_b, meta=meta)


# ============================================================
# 7) I/O
# ============================================================

def _make_tag(cfg: SuiteConfig, label: str) -> str:
    return f"{label}__seed{cfg.seed}"


def save_npz(cfg: SuiteConfig, payload: Dict[str, Any], label: str) -> Path:
    if not cfg.out_dir:
        raise ValueError("SuiteConfig.out_dir must be set before calling save_npz")
    tag = _make_tag(cfg, label)
    out_dir = Path(cfg.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "rnn_traces.npz"

    save_kw = {**asdict(cfg), **payload["meta"]}
    np.savez_compressed(
        path,
        data=payload["data"],
        latent_a=payload["latent_a"],
        latent_b=payload["latent_b"],
        **save_kw,
    )
    return path


# ============================================================
# 8) Scenario definitions & suite runner
# ============================================================

_NO_WARP = dict(warp_state_a=False, warp_time_a=False,
                warp_state_b=False, warp_time_b=False)

_BOTH_WARP = dict(warp_state_a=True, warp_time_a=False,
                  warp_state_b=True, warp_time_b=False)

_WARP_ONLY_B = dict(warp_state_a=False, warp_time_a=False,
                    warp_state_b=True,  warp_time_b=False)

SCENARIOS: List[Tuple[str, Dict[str, Any]]] = [
    ("hopf_bif__no_warp", dict(
        system_a="hopf",      mu_a_start=-1.0, mu_a_end=2.0,
        system_b="hopf",      mu_b_start=-1.0, mu_b_end=3.0,
        **_NO_WARP
    )),
    ("hopf_bif__warp", dict(
        system_a="hopf",      mu_a_start=-1.0, mu_a_end=2.0,
        system_b="hopf",      mu_b_start=-1.0, mu_b_end=3.0,
        **_BOTH_WARP
    )),
    ("pitchfork_bif__no_warp", dict(
        system_a="pitchfork", mu_a_start=-1.0, mu_a_end=2.0,
        system_b="pitchfork", mu_b_start=-1.0, mu_b_end=3.0,
        **_NO_WARP
    )),
    ("pitchfork_bif__warp", dict(
        system_a="pitchfork", mu_a_start=-1.0, mu_a_end=2.0,
        system_b="pitchfork", mu_b_start=-1.0, mu_b_end=3.0,
        **_BOTH_WARP
    )),
    ("hopf_bif__pitchfork_bif__no_warp", dict(
        system_a="hopf",      mu_a_start=-1.0, mu_a_end=2.0,
        system_b="pitchfork", mu_b_start=-1.0, mu_b_end=2.0,
        **_NO_WARP
    )),
    ("hopf_bif__pitchfork_bif__warp", dict(
        system_a="hopf",      mu_a_start=-1.0, mu_a_end=2.0,
        system_b="pitchfork", mu_b_start=-1.0, mu_b_end=2.0,
        **_BOTH_WARP
    )),
    ("hopf_bif__clean_vs_warped", dict(
        system_a="hopf", mu_a_start=-1.0, mu_a_end=2.0,
        system_b="hopf", mu_b_start=-1.0, mu_b_end=2.0,
        **_WARP_ONLY_B
    )),
    ("hopf_bif__clean_x2", dict(
        system_a="hopf", mu_a_start=-1.0, mu_a_end=2.0,
        system_b="hopf", mu_b_start=-1.0, mu_b_end=2.0,
        **_NO_WARP
    )),
]

SCENARIO_NAMES: List[str] = [label for label, _ in SCENARIOS]


def run_generate_suite(
    out_dir: Path,
    base_cfg: Optional[SuiteConfig] = None,
    scenarios: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
) -> List[Path]:
    """
    Generate rnn_traces.npz for each scenario and save under out_dir/<label>__seed<N>/.

    Args:
        out_dir:    Root directory for output (e.g. data_dir() / "synthetic").
        base_cfg:   Base SuiteConfig; out_dir field will be overridden.
        scenarios:  Subset of SCENARIOS to run; defaults to all.

    Returns:
        List of paths to saved .npz files.
    """
    if base_cfg is None:
        base_cfg = SuiteConfig()
    if scenarios is None:
        scenarios = SCENARIOS

    saved: List[Path] = []
    for label, overrides in scenarios:
        cfg = SuiteConfig(**{**asdict(base_cfg), **overrides, "out_dir": str(out_dir)})

        if label == "hopf_bif__clean_vs_warped":
            payload = generate_dataset_warp_only_same_latents(cfg)
        else:
            payload = generate_dataset(cfg)

        path = save_npz(cfg, payload, label=label)
        print(f"  [{label}] -> {path}")
        saved.append(path)

    return saved

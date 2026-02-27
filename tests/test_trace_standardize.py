import numpy as np
from mmphate_repro.pipelines.trace_standardize import standardize_trace_tensor

def test_standardize_trace_tensor_axis_inference():
    E, S, T, H = 5, 7, 11, 3
    # raw in (E, S, T, H)
    raw = np.zeros((E, S, T, H))
    out = standardize_trace_tensor(raw, n_samples=S, n_timesteps=T, n_units=H)
    assert out.shape == (E, T, H, S)
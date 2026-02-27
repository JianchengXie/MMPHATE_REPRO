import numpy as np
from mmphate_repro.metrics.neighborhood import zscore_across_samples, neighborhood_preservation


def test_zscore_across_samples_basic_properties():
    ET, H, S = 3, 2, 7
    rng = np.random.default_rng(0)
    T = rng.normal(size=(ET, H, S))

    Tz = zscore_across_samples(T)

    # For each (t,h), mean across samples ~0, std across samples ~1
    means = Tz.mean(axis=2)
    stds = Tz.std(axis=2, ddof=1)

    assert np.allclose(means, 0.0, atol=1e-7)
    assert np.allclose(stds, 1.0, atol=1e-7)


def test_neighborhood_preservation_perfect_overlap_without_ties():
    """
    Construct T so that kNN structure is deterministic (no distance ties),
    and make emb match the same geometry => preservation should be 1.0.
    """
    ET, H, S = 6, 4, 5

    # Use quadratic spacing to avoid equal-distance ties:
    # h^2 gives 0,1,4,9; t^2 gives 0,1,4,9,16,25
    T = np.zeros((ET, H, S), dtype=float)
    emb = np.zeros((ET, H, 3), dtype=float)

    for t in range(ET):
        for h in range(H):
            base = 100.0 * (t ** 2) + 10.0 * (h ** 2)
            T[t, h, :] = base + 0.001 * np.arange(S)  # small within-vector structure
            emb[t, h, :] = np.array([base, 0.0, 0.0])

    intra, inter = neighborhood_preservation(T, emb, k=2)
    assert np.isclose(intra, 1.0)
    assert np.isclose(inter, 1.0)
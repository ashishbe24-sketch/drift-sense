"""Confirm the AR(1) roughness generator realises the requested sigma and
correlation length, and time it."""
import sys, pathlib, time
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from driftsense.layout import make_edge_roughness

t0 = time.perf_counter()
r = make_edge_roughness(np.random.default_rng(1), sigma_nm=1.0, xi_nm=10.0)
build_ms = 1000 * (time.perf_counter() - t0)

x = r.lo - r.lo.mean(axis=1, keepdims=True)
ac = np.array([np.correlate(row, row, "full")[len(row) - 1:len(row) + 40]
               for row in x[:64]])
ac /= ac[:, :1]
mean_ac = ac.mean(0)
xi_hat = float(np.argmax(mean_ac < 1 / np.e)) * r.ds_nm

print(f"build time        : {build_ms:.0f} ms   (Python-loop version was ~230 ms)")
print(f"realised sigma    : {r.lo.std():.3f} nm   (target 1.000)")
print(f"realised xi       : {xi_hat:.0f} nm       (target 10)")
print(f"two edges independent: correlation {np.corrcoef(r.lo.ravel()[:200000], r.hi.ravel()[:200000])[0,1]:+.4f}")
print(f"table shape       : {r.lo.shape}, {2 * r.lo.nbytes / 1e6:.1f} MB for both edges")

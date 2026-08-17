"""Solve for the SE edge-response gains that make the *rendered* profile match
the one measured on real IC SEM imagery, then verify the full chain."""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from driftsense import physics as P

REF = P.CaptureParams(px_nm=1.0, probe_sigma_nm=P.PROBE_SIGMA_NM)

go, gu = P.calibrate_edge_gains(REF)
print("targets from real SEM : overshoot %+.3f   halo %+.3f"
      % (P.EDGE_OVERSHOOT, -P.EDGE_UNDERSHOOT))
print("calibrated gains      : g_over %.4f      g_under %.4f" % (go, gu))
print("(currently compiled in: g_over %.4f      g_under %.4f)"
      % (P.EDGE_G_OVER, P.EDGE_G_UNDER))

# verify through the deterministic chain
n = 200
step = np.zeros((8, 2 * n), np.float32)
step[:, n:] = 1.0
for tag, g1, g2 in (("compiled", P.EDGE_G_OVER, P.EDGE_G_UNDER),
                    ("recalibrated", go, gu)):
    from dataclasses import replace
    q = replace(REF, edge_overshoot=g1, edge_undershoot=g2)
    prof = P.apply_optics(P.edge_response(step, q), q).mean(0)
    plateau = prof[-n // 3:].mean()
    over = prof[n:].max() - plateau
    halo = prof[:n].min()
    x = np.arange(len(prof)) - n + 0.5
    rise_lo = np.interp(0.1, prof[n - 20:n + 20], x[n - 20:n + 20])
    rise_hi = np.interp(0.9, prof[n - 20:n + 20], x[n - 20:n + 20])
    print("  %-13s overshoot %+.3f  halo %+.3f  peak at %+.1f nm  "
          "10-90%% rise %.2f nm"
          % (tag, over, halo, x[prof.argmax()], rise_hi - rise_lo))

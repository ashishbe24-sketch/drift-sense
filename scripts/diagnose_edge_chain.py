"""Push a clean step through each stage of the image-formation chain and
report what happens to the edge profile, to find where the measured overshoot
is going."""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from driftsense import physics as P

rng = np.random.default_rng(7)
H, W = 400, 400
step = np.zeros((H, W), np.float32)
step[:, 200:] = 1.0
step = step * (146.0 - 59.0) + 59.0          # real SEM grey levels


def stats(prof, px_nm, lo=59.0, hi=146.0):
    n = (prof - lo) / (hi - lo)
    x = np.arange(len(n)) - 200 + 0.5
    i = np.argmax(n[200:260]) + 200
    return dict(peak=n.max(), peak_at=x[n.argmax()] * px_nm,
                plateau=n[280:320].mean(),
                over=n[190:260].max() - n[280:320].mean(),
                halo=n[140:200].min())


def show(tag, img, px_nm):
    s = stats(img.mean(0), px_nm)
    print(f"  {tag:28s} overshoot {s['over']:+.3f}  halo {s['halo']:+.3f}  "
          f"peak at {s['peak_at']:+.1f} nm  plateau {s['plateau']:.3f}")


for px_nm, prm in ((1.0, P.CaptureParams(px_nm=1.0, probe_sigma_nm=1.0,
                                         drift_nm=2.0, vibration_nm=0.8,
                                         dose_e_per_grey=40.0, read_sigma=1.0)),
                   (10.0, P.CaptureParams(px_nm=10.0, probe_sigma_nm=1.4,
                                          defocus_sigma_nm=17.0, drift_nm=6.0,
                                          vibration_nm=3.0, charging=1.8,
                                          dose_e_per_grey=4.0, read_sigma=2.2))):
    print(f"\n=== {px_nm:.0f} nm/px ===")
    x = step.copy()
    show("raw step", x, px_nm)
    x = P.edge_response(x, prm);        show("+ edge response", x, px_nm)
    x = P.apply_charging(x, prm, rng);  show("+ charging", x, px_nm)
    x = P.apply_optics(x, prm);         show("+ optics (PSF+defocus)", x, px_nm)
    x = P.apply_scan_artefacts(x, prm, rng); show("+ drift/vibration", x, px_nm)
    x = P.add_shot_noise(x, prm, rng);  show("+ shot noise", x, px_nm)
    x = P.add_read_noise(x, prm, rng);  show("+ read noise", x, px_nm)

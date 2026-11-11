#!/usr/bin/env python
"""
surrogates.py - surrogate-data control for any per-epoch EEG measure (MFDFA, LZC, future ones).

Idea
    A phase-randomised surrogate has exactly the same power spectrum as the real epoch (so the same
    linear second-order structure, oscillatory peaks included) but no phase coupling, bursts or
    amplitude modulation. Running a measure on K surrogates gives the value the SPECTRUM ALONE predicts.
        residual = real - mean(surrogates)
    is the part not explained by the spectrum. Analyse the residual (or use the surrogate value as a
    covariate) exactly like any other per-epoch measure; the raw measure is kept alongside.

    Caveat: surrogates are stationary and roughly Gaussian, so the residual also contains
    non-stationarity within an epoch (e.g. a spindle in the middle of 30 s). It is "beyond the
    spectrum", not automatically "nonlinear dynamics".

Measure functions
    Every measure is a function  fn(rows) -> array  taking a 2-D array (n_rows, n_samples) and returning
    (n_rows,) or (n_rows, m). The real epoch and its surrogates are evaluated in ONE call, so batch
    implementations (e.g. complexeva) stay fast. Wrap a plain 1-D function with rowwise(fn).
    Ready-made: mfdfa_measure() (same settings as sof_mfdfa.py) and lzc_measure() (as lz_sof.py).

    Apply the surrogate to the signal exactly as the measure sees it: for LZC pass the BAND-PASSED
    epochs (surrogate before median binarisation); for MFDFA pass the notch-filtered epochs.

Surrogate kinds
    "phase"    phase randomisation: keeps the spectrum, destroys everything else  (default)
    "shuffle"  random permutation: keeps the amplitude distribution, destroys all temporal structure

Usage
    import surrogates as sg
    fn = sg.mfdfa_measure(sfreq=128.0, order=2)              # finds ../criticality/sof_mfdfa.py itself
    res = sg.adjust_epochs(epochs, fn, n_surr=10, subject_id="00004", epoch_idx=idx)
    cols = sg.to_columns(res, fn.columns)                    # h_q+2, h_q+2_surr, h_q+2_resid, ...
    # multifractal width is linear in h, so Delta h surrogate mean = h_q-4_surr - h_q+4_surr

    python surrogates.py --selftest                          # quick checks, no data needed

Reproducibility: the random generator is seeded from (subject_id, epoch index, kind), so reruns and
array-job shards give identical surrogates.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KINDS = ("phase", "shuffle")

# Where sof_mfdfa.py is looked for (first existing wins). This file lives in <code>/helpers/, so the
# sibling folder <code>/criticality/ comes first; the second is the cluster code folder.
SOF_MFDFA_CANDIDATES = (
    Path(__file__).resolve().parent.parent / "criticality" / "sof_mfdfa.py",
    Path("~/pl1358-01/project_data/code/criticality/sof_mfdfa.py").expanduser(),
)


# ------------------------------------------------------------------ random generator
def make_rng(*keys) -> np.random.Generator:
    """Generator seeded from any mix of ints and strings (digit strings are used as ints)."""
    ints = []
    for k in keys:
        if isinstance(k, (int, np.integer)):
            ints.append(int(k))
        else:
            s = str(k)
            ints.append(int(s) if s.isdigit() else zlib.crc32(s.encode()))
    return np.random.default_rng(np.random.SeedSequence(ints))


# ------------------------------------------------------------------ surrogate generators
def phase_randomize(x, rng, n=1):
    """n phase-randomised copies of 1-D x, shape (n, len(x)). |FFT| is preserved exactly.

    DC and (for even length) Nyquist terms are kept as in the original, so the mean is unchanged.
    """
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[-1]
    X = np.fft.rfft(x)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(n, X.size))
    Xs = np.abs(X) * np.exp(1j * phases)
    Xs[:, 0] = X[0]
    if N % 2 == 0:
        Xs[:, -1] = X[-1]
    return np.fft.irfft(Xs, n=N, axis=-1)


def shuffle(x, rng, n=1):
    """n random permutations of 1-D x, shape (n, len(x))."""
    x = np.asarray(x, dtype=np.float64)
    return np.stack([rng.permutation(x) for _ in range(n)])


def make_surrogates(x, n, kind, rng):
    if kind == "phase":
        return phase_randomize(x, rng, n)
    if kind == "shuffle":
        return shuffle(x, rng, n)
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


# ------------------------------------------------------------------ core
@dataclass
class SurrogateResult:
    real: np.ndarray        # measure on the real epoch
    surr_mean: np.ndarray   # mean over surrogates (the spectrum-predicted value)
    surr_sd: np.ndarray     # SD over surrogates (ddof=1)

    @property
    def resid(self):
        return self.real - self.surr_mean


def surrogate_adjust(x, measure_fn, n_surr=10, kind="phase", seed_keys=()):
    """Measure of one epoch x (1-D) and of n_surr surrogates, evaluated in a single call."""
    x = np.asarray(x, dtype=np.float64)
    rng = make_rng(*seed_keys, kind)
    rows = np.vstack([x[None, :], make_surrogates(x, n_surr, kind, rng)])
    out = np.asarray(measure_fn(rows), dtype=np.float64)
    real, surr = out[0], out[1:]
    with warnings.catch_warnings():  # all-NaN columns -> NaN, silently
        warnings.simplefilter("ignore")
        mean = np.nanmean(surr, axis=0)
        sd = np.nanstd(surr, axis=0, ddof=1) if len(surr) > 1 else np.full_like(mean, np.nan)
    return SurrogateResult(real, mean, sd)


def adjust_epochs(epochs, measure_fn, n_surr=10, kind="phase", subject_id="", epoch_idx=None):
    """Run surrogate_adjust on every row of epochs (n_epochs, n_samples).

    epoch_idx: epoch numbers used for seeding (default 0..n-1). Pass the real epoch indices when only
    a subset (e.g. the epochs inside the early/late windows) is analysed, so surrogates do not depend on
    which subset was chosen. Rows containing non-finite values give NaN.
    Returns a dict with arrays real, surr_mean, surr_sd, resid, each (n_epochs,) or (n_epochs, m).
    """
    epochs = np.asarray(epochs, dtype=np.float64)
    if epoch_idx is None:
        epoch_idx = np.arange(len(epochs))
    reals, means, sds = [], [], []
    template = None
    for x, e in zip(epochs, epoch_idx):
        if not np.isfinite(x).all():
            reals.append(None), means.append(None), sds.append(None)
            continue
        r = surrogate_adjust(x, measure_fn, n_surr, kind, seed_keys=(subject_id, int(e)))
        template = r if template is None else template
        reals.append(r.real), means.append(r.surr_mean), sds.append(r.surr_sd)
    if template is None:
        raise ValueError("no epoch with finite samples")
    blank = np.full_like(template.real, np.nan)
    stack = lambda lst: np.stack([blank if v is None else v for v in lst])
    real, mean, sd = stack(reals), stack(means), stack(sds)
    return {"real": real, "surr_mean": mean, "surr_sd": sd, "resid": real - mean}


def to_columns(result, names):
    """Flatten adjust_epochs output into {name, name_surr, name_resid, name_surr_sd: (n_epochs,)}.

    names: one name per column of the measure (str for a scalar measure, list for a vector one).
    """
    if isinstance(names, str):
        names = [names]
    arrays = {k: v.reshape(len(v), -1) for k, v in result.items()}
    cols = {}
    for j, name in enumerate(names):
        cols[name] = arrays["real"][:, j]
        cols[f"{name}_surr"] = arrays["surr_mean"][:, j]
        cols[f"{name}_surr_sd"] = arrays["surr_sd"][:, j]
        cols[f"{name}_resid"] = arrays["resid"][:, j]
    return cols


# ------------------------------------------------------------------ measure adapters
def rowwise(fn):
    """Turn a 1-D measure (x -> scalar or vector) into a 2-D one (rows -> array)."""
    def wrapped(rows):
        return np.array([np.atleast_1d(fn(r)) for r in rows], dtype=np.float64).squeeze()
    return wrapped


def load_sof_mfdfa(path=None):
    """Import sof_mfdfa.py from a full file path (default: SOF_MFDFA_CANDIDATES); no PYTHONPATH needed."""
    if "sof_mfdfa" in sys.modules:
        return sys.modules["sof_mfdfa"]
    candidates = [Path(path).expanduser().resolve()] if path else list(SOF_MFDFA_CANDIDATES)
    found = next((p for p in candidates if p.is_file()), None)
    if found is None:
        raise ImportError("sof_mfdfa.py not found; looked in: " + ", ".join(str(p) for p in candidates)
                          + ". Pass sof_mfdfa_path=... to mfdfa_measure().")
    spec = importlib.util.spec_from_file_location("sof_mfdfa", found)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sof_mfdfa"] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # e.g. mne or MFDFA missing in this environment
        del sys.modules["sof_mfdfa"]
        raise ImportError(f"found {found} but could not import it: {exc}") from exc
    return module


def mfdfa_measure(sfreq=128.0, epoch_s=30, order=2, scale_min_s=0.125, sof_mfdfa_path=None):
    """h(q) for every q in sof_mfdfa.Q_LIST, with the settings of sof_mfdfa.py.

    sof_mfdfa.py is loaded by full path (see load_sof_mfdfa), so this works from any folder.
    Returns fn(rows) -> (n_rows, n_q); fn.columns lists the names (h_q-4 ... h_q+4).
    """
    m = load_sof_mfdfa(sof_mfdfa_path)
    scales = m.make_scales(sfreq, int(round(sfreq * epoch_s)), scale_min_s)

    def fn(rows):
        out = np.full((len(rows), len(m.Q_LIST)), np.nan)
        for i, r in enumerate(rows):
            out[i] = m.mfdfa_exponents(r, scales, order)[0]
        return out

    fn.columns = [f"h_q{q:+d}" for q in m.Q_LIST]
    return fn


def lzc_measure(normalize=True):
    """Median-binarised Lempel-Ziv complexity (complexeva), as in lz_sof.py. Pass band-passed epochs."""
    import complexeva

    def fn(rows):
        rows = np.ascontiguousarray(rows, dtype=np.float64)
        try:
            return np.asarray(complexeva.lempel_ziv(rows, method="median", normalize=normalize), float)
        except (TypeError, ValueError) as exc:  # older complexeva without batching
            if "finite" in str(exc).lower():
                raise
            return np.array([complexeva.lempel_ziv(r, method="median", normalize=normalize) for r in rows], float)

    fn.columns = ["lz_norm" if normalize else "lz_count"]
    return fn


# ------------------------------------------------------------------ self-test
def _selftest():
    rng = np.random.default_rng(1)
    n, fs = 3840, 128.0
    f = np.fft.rfftfreq(n, 1 / fs)
    spec = np.zeros(f.size, complex)
    spec[1:] = (f[1:] ** -1.0) * np.exp(1j * rng.uniform(0, 2 * np.pi, f.size - 1))  # ~1/f amplitude -> 1/f^2 power
    x = np.fft.irfft(spec, n=n)
    x = x / x.std() * 20 + 3.0  # microvolt-ish, non-zero mean
    x[1800:1900] += 60 * np.sin(2 * np.pi * 13 * np.arange(100) / fs)  # a burst: non-stationary

    s = phase_randomize(x, make_rng("t", 1), n=4)
    assert s.shape == (4, n)
    err = np.max(np.abs(np.abs(np.fft.rfft(s, axis=-1)) - np.abs(np.fft.rfft(x))) / (np.abs(np.fft.rfft(x)) + 1e-12))
    print(f"phase: max relative |FFT| difference = {err:.1e}  (should be ~1e-12)")
    assert err < 1e-8
    assert np.allclose(s.mean(axis=1), x.mean())
    assert not np.allclose(s[0], x)

    assert np.array_equal(phase_randomize(x, make_rng("a", 5)), phase_randomize(x, make_rng("a", 5)))
    assert not np.array_equal(phase_randomize(x, make_rng("a", 5)), phase_randomize(x, make_rng("a", 6)))
    print("seeding: same keys -> identical surrogates, different keys -> different")

    sh = shuffle(x, make_rng("t", 2), n=2)
    assert np.allclose(np.sort(sh[0]), np.sort(x))
    print("shuffle: amplitude distribution preserved")

    # toy measure: kurtosis (non-Gaussianity). Surrogates are ~Gaussian, the real epoch has a burst.
    def kurt(r):
        z = (r - r.mean()) / r.std()
        return (z ** 4).mean() - 3

    epochs = np.vstack([x, x[::-1], x * 1.5])
    res = adjust_epochs(epochs, rowwise(kurt), n_surr=8, subject_id="00004", epoch_idx=[10, 11, 12])
    cols = to_columns(res, "kurtosis")
    print("toy measure (excess kurtosis): real", np.round(cols["kurtosis"], 2),
          "| surrogate mean", np.round(cols["kurtosis_surr"], 2),
          "| residual", np.round(cols["kurtosis_resid"], 2))
    bad = epochs.copy(); bad[1, 5] = np.nan
    res2 = adjust_epochs(bad, rowwise(kurt), n_surr=4, subject_id="00004")
    assert np.isnan(res2["real"][1]) and np.isfinite(res2["real"][0])
    print("non-finite epoch -> NaN row, others unaffected")

    for name, maker in (("MFDFA", lambda: mfdfa_measure()), ("LZC", lambda: lzc_measure())):
        try:
            fn = maker()
            r = surrogate_adjust(x, fn, n_surr=3, seed_keys=("00004", 0))
            print(f"{name}: real {np.round(r.real, 3)} | surrogate mean {np.round(r.surr_mean, 3)}")
        except ImportError as exc:
            print(f"{name}: skipped ({exc})")
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description="Surrogate-data control for per-epoch EEG measures (library).")
    ap.add_argument("--selftest", action="store_true", help="run quick checks on synthetic data")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

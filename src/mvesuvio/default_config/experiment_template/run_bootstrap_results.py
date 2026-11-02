"""Collect fit parameters from every bootstrap sample and show their statistics.

Edit the USER SECTION below, then run this script.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mantid import plots  # noqa: F401  registers the "mantid" projection

import mvesuvio
from mvesuvio import ConfigArgInputs
from mvesuvio.util.files_manager import FilesManager

# ---------------------------- USER SECTION ----------------------------
# Folder containing the boot_(n) folders created by run_bootstrap.py; None uses FilesManager.get_bootstrap_outputs_dir()
BOOT_OUTPUTS_DIR = None
# Fit results to collect: "lm", "minuit" or "simplex"
MINIMIZER = "minuit"
# Whether to save the figure
SAVE_FIGURE = True
# Folder to save the figure in; None uses FilesManager.get_bootstrap_results_dir()
RESULTS_DIR = None
# Whether to open the plot window
SHOW_FIGURE = True
# -------------------------- END USER SECTION --------------------------

MINIMIZERS = ("lm", "minuit", "simplex")


def find_parameters_file(sample_dir: Path, minimizer: str):
    """Return the *Parameters file of the chosen minimizer for one sample (highest index if several)."""
    pattern = re.compile(rf"_{minimizer}_.*Parameters$")
    files = [f for f in (sample_dir / "fitting_outputs").rglob("*Parameters") if pattern.search(f.name)]

    def index(f: Path):
        # File names look like front_boot_1_<index>_ws_...; pick the highest index
        m = re.search(r"boot_\d+_(\d+)_", f.name)
        return int(m.group(1)) if m else -1

    return max(files, key=index) if files else None


def read_parameters(path: Path):
    """Read Name, Value, Error columns from a Mantid ascii Parameters table."""
    params = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, value, error = (c.strip() for c in line.split(",")[:3])
        # Strip function prefix (f1.) and skip the cost function, which is not a fit parameter
        name = re.sub(r"^f\d+\.", "", name)
        if name.lower().startswith("cost function"):
            continue
        params[name] = (float(value), float(error))
    return params


def collect(boot_dir: Path, minimizer: str):
    """Return parameter names and arrays of values and errors, shape (n_samples, n_params)."""
    sample_dirs = sorted(
        (d for d in boot_dir.glob("boot_*") if d.is_dir()),
        key=lambda d: int(re.search(r"boot_(\d+)", d.name).group(1)),
    )
    names, values, errors = None, [], []
    for d in sample_dirs:
        f = find_parameters_file(d, minimizer)
        if f is None:
            print(f"Warning: no {minimizer} Parameters file in {d.name}, skipping")
            continue
        params = read_parameters(f)
        if names is None:
            names = list(params)
        if list(params) != names:
            print(f"Warning: parameters in {d.name} differ from the first sample, skipping")
            continue
        values.append([params[n][0] for n in names])
        errors.append([params[n][1] for n in names])
    if not values:
        raise SystemExit(f"No {minimizer} results found under {boot_dir}")
    return names, np.array(values), np.array(errors)


def plot_grid(names, values, title):
    n = len(names)
    cov = np.atleast_2d(np.cov(values, rowvar=False))
    corr = np.atleast_2d(np.corrcoef(values, rowvar=False))
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n), squeeze=False, subplot_kw={"projection": "mantid"})
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(values[:, j], bins="auto")
            else:
                ax.scatter(values[:, j], values[:, i], s=6, alpha=0.6)
                ax.set_title(f"cov={cov[i, j]:.2g}, r={corr[i, j]:.2f}", fontsize=7)
            if i == n - 1:
                ax.set_xlabel(names[j])
            if j == 0:
                ax.set_ylabel(names[i])
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def run_bootstrap_results(configure_with_main: bool = True):
    if configure_with_main:
        mvesuvio.main(ConfigArgInputs(experiment_dir=str(Path(__file__).parent), ip_dir=""))

    minimizer = MINIMIZER.strip().lower()
    if minimizer not in MINIMIZERS:
        raise SystemExit(f"Invalid MINIMIZER '{MINIMIZER}', choose one of {MINIMIZERS}")

    boot_outputs_dir = Path(BOOT_OUTPUTS_DIR) if BOOT_OUTPUTS_DIR else FilesManager.get_bootstrap_outputs_dir()
    names, values, _ = collect(boot_outputs_dir, minimizer)
    print(f"\nCollected {len(values)} bootstrap samples ({minimizer})\n")
    print(f"{'Name':<12}{'Mean':>14}{'Std dev':>14}")
    for k, name in enumerate(names):
        std = values[:, k].std(ddof=1) if len(values) > 1 else float("nan")
        print(f"{name:<12}{values[:, k].mean():>14.6g}{std:>14.6g}")

    fig = plot_grid(names, values, f"Bootstrap parameters ({minimizer}, {len(values)} samples)")
    if SAVE_FIGURE:
        results_dir = Path(RESULTS_DIR) if RESULTS_DIR else FilesManager.get_bootstrap_results_dir()
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(results_dir / f"bootstrap_grid_{minimizer}.png", dpi=150)
    if SHOW_FIGURE:
        plt.show()


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    run_bootstrap_results()

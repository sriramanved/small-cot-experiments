from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MethodStyle:
    label: str
    color: str
    linestyle: str = "-"
    marker: str | None = None
    linewidth: float = 3.0
    markersize: float = 7.0
    alpha: float = 1.0

    @property
    def plot_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "color": self.color,
            "linestyle": self.linestyle,
            "linewidth": self.linewidth,
            "alpha": self.alpha,
        }
        if self.marker is not None:
            kwargs["marker"] = self.marker
            kwargs["markersize"] = self.markersize
        return kwargs


# Dark, colorblind-conscious method palette for paper figures.
OFFLINE_BC_COLOR = "#000000"
OPD_R_COLOR = "#006D4F"
OPD_F_COLOR = "#9A3412"
NAIL_FORWARD_COLOR = "#B91C1C"
NAIL_REVERSE_COLOR = "#6B21A8"
NAIL_REVERSE_FULL_COLOR = "#4C1D95"
EXPERT_COLOR = "#000000"
RANDOM_COLOR = "#7F7F7F"
FALLBACK_COLOR = "#4D4D4D"

METHOD_ORDER = (
    "Expert",
    "Offline BC",
    "TM OPD",
    "TM OPD, greedy rollout",
    "TM OPD, sampled rollout",
    "NAIL-forward",
    "NAIL-forward, greedy rollout",
    "NAIL-forward, sampled rollout",
    "NAIL-reverse",
    "NAIL-reverse, greedy rollout",
    "NAIL-reverse, sampled rollout",
    "NAIL-reverse MC",
    "NAIL-reverse full",
    "Random",
)

_CANONICAL_METHOD_STYLES: dict[str, MethodStyle] = {
    "Expert": MethodStyle("Expert", EXPERT_COLOR, linestyle="-.", linewidth=3.2),
    "Offline BC": MethodStyle("Offline BC", OFFLINE_BC_COLOR, linestyle=":", linewidth=3.4),
    "TM OPD": MethodStyle("OPD-R", OPD_R_COLOR, linestyle="--", linewidth=3.4),
    "TM OPD, greedy rollout": MethodStyle("OPD-R", OPD_R_COLOR, linestyle="--", linewidth=3.4),
    "TM OPD, sampled rollout": MethodStyle("OPD-R", OPD_R_COLOR, linestyle="--", linewidth=3.4),
    "NAIL-forward": MethodStyle("NAIL-F", NAIL_FORWARD_COLOR, linestyle="-", linewidth=3.4),
    "NAIL-forward, greedy rollout": MethodStyle(
        "NAIL-F",
        NAIL_FORWARD_COLOR,
        linestyle="-",
        linewidth=3.4,
    ),
    "NAIL-forward, sampled rollout": MethodStyle(
        "OPD-F",
        OPD_F_COLOR,
        linestyle="--",
        linewidth=3.4,
    ),
    "NAIL-reverse": MethodStyle("NAIL-R", NAIL_REVERSE_COLOR, linestyle="-", linewidth=3.4),
    "NAIL-reverse, greedy rollout": MethodStyle(
        "NAIL-R",
        NAIL_REVERSE_COLOR,
        linestyle="-",
        linewidth=3.4,
    ),
    "NAIL-reverse, sampled rollout": MethodStyle(
        "NAIL-R sampled",
        NAIL_REVERSE_COLOR,
        linestyle="--",
        linewidth=3.4,
    ),
    "NAIL-reverse MC": MethodStyle("NAIL-R", NAIL_REVERSE_COLOR, linestyle="-", linewidth=3.4),
    "NAIL-reverse full": MethodStyle("NAIL-R full", NAIL_REVERSE_FULL_COLOR, linestyle="-.", linewidth=3.4),
    "Random": MethodStyle("Random", RANDOM_COLOR, linestyle=":", marker=None),
}

_METHOD_ALIASES = {
    "clean": "Expert",
    "clean teacher": "Expert",
    "expert": "Expert",
    "teacher": "Expert",
    "loglossbc": "Offline BC",
    "log loss bc": "Offline BC",
    "offline bc": "Offline BC",
    "noisy bc": "Offline BC",
    "bc": "Offline BC",
    "opd": "TM OPD",
    "opd, greedy rollout": "TM OPD, greedy rollout",
    "opd, sampled rollout": "TM OPD, sampled rollout",
    "tm opd": "TM OPD",
    "tm-opd": "TM OPD",
    "tm opd, greedy rollout": "TM OPD, greedy rollout",
    "tm opd, sampled rollout": "TM OPD, sampled rollout",
    "reverse_kl_tm": "TM OPD",
    "nail-forward": "NAIL-forward",
    "nail forward": "NAIL-forward",
    "nail-forward greedy": "NAIL-forward, greedy rollout",
    "nail-forward, greedy rollout": "NAIL-forward, greedy rollout",
    "nail-forward sampled": "NAIL-forward, sampled rollout",
    "nail-forward, sampled rollout": "NAIL-forward, sampled rollout",
    "forward nail greedy": "NAIL-forward, greedy rollout",
    "forward nail sampled": "NAIL-forward, sampled rollout",
    "nail-opd": "NAIL-forward",
    "nail opd": "NAIL-forward",
    "nail-opd mc": "NAIL-forward",
    "nail opd mc": "NAIL-forward",
    "nail-reverse": "NAIL-reverse",
    "nail reverse": "NAIL-reverse",
    "nail-reverse greedy": "NAIL-reverse, greedy rollout",
    "nail-reverse, greedy rollout": "NAIL-reverse, greedy rollout",
    "nail-reverse sampled": "NAIL-reverse, sampled rollout",
    "nail-reverse, sampled rollout": "NAIL-reverse, sampled rollout",
    "nail-reverse mc": "NAIL-reverse MC",
    "nail reverse mc": "NAIL-reverse MC",
    "nail-reverse full": "NAIL-reverse full",
    "nail reverse full": "NAIL-reverse full",
    "random": "Random",
    "random baseline": "Random",
}

METHOD_COLORS = {method: style.color for method, style in _CANONICAL_METHOD_STYLES.items()}
METHOD_LINESTYLES = {method: style.linestyle for method, style in _CANONICAL_METHOD_STYLES.items()}
METHOD_LABELS = {method: style.label for method, style in _CANONICAL_METHOD_STYLES.items()}


def canonical_method_name(method_name: str) -> str:
    text = str(method_name).strip()
    normalized = " ".join(text.replace("_", " ").split()).lower()
    normalized = normalized.replace("nail forward", "nail-forward")
    normalized = normalized.replace("nail reverse", "nail-reverse")
    return _METHOD_ALIASES.get(normalized, text)


def get_method_style(method_name: str) -> MethodStyle:
    canonical = canonical_method_name(method_name)
    if canonical in _CANONICAL_METHOD_STYLES:
        return _CANONICAL_METHOD_STYLES[canonical]

    lower = canonical.lower()
    if "nail-forward" in lower or "nail forward" in lower:
        if "sample" in lower:
            return _CANONICAL_METHOD_STYLES["NAIL-forward, sampled rollout"]
        return _CANONICAL_METHOD_STYLES["NAIL-forward, greedy rollout"]
    if "nail-reverse" in lower or "nail reverse" in lower:
        if "full" in lower:
            return _CANONICAL_METHOD_STYLES["NAIL-reverse full"]
        if "sample" in lower:
            return _CANONICAL_METHOD_STYLES["NAIL-reverse, sampled rollout"]
        return _CANONICAL_METHOD_STYLES["NAIL-reverse, greedy rollout"]
    if "opd" in lower:
        if "sample" in lower:
            return _CANONICAL_METHOD_STYLES["TM OPD, sampled rollout"]
        if "greedy" in lower:
            return _CANONICAL_METHOD_STYLES["TM OPD, greedy rollout"]
        return _CANONICAL_METHOD_STYLES["TM OPD"]
    if "offline" in lower or "bc" in lower:
        return _CANONICAL_METHOD_STYLES["Offline BC"]
    if "random" in lower:
        return _CANONICAL_METHOD_STYLES["Random"]
    if "expert" in lower or "teacher" in lower:
        return _CANONICAL_METHOD_STYLES["Expert"]

    return MethodStyle(canonical, FALLBACK_COLOR, linestyle="-", marker=None)


def method_sort_key(method_name: str) -> tuple[int, str]:
    canonical = canonical_method_name(method_name)
    try:
        return (METHOD_ORDER.index(canonical), canonical)
    except ValueError:
        style = get_method_style(canonical)
        try:
            return (METHOD_ORDER.index(style.label), style.label)
        except ValueError:
            return (len(METHOD_ORDER), style.label)


def set_publication_style() -> None:
    try:
        import seaborn as sns

        sns.set_theme(context="paper", style="whitegrid")
    except ImportError:
        pass

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "lines.linewidth": 3.4,
            "lines.markersize": 7,
            "axes.linewidth": 1.5,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.3,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def polish_axes(ax: object, *, remove_top_right: bool = True) -> None:
    ax.grid(which="major", alpha=0.3)
    ax.grid(which="minor", linestyle=":", alpha=0.2)
    if remove_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def format_iteration_k(value: float, _position: object | None = None) -> str:
    if not math.isfinite(value):
        return ""
    rounded = int(round(value))
    if abs(rounded) < 1000:
        return str(rounded)
    if abs(rounded) >= 1_000_000:
        millions = rounded / 1_000_000
        if float(millions).is_integer():
            return f"{int(millions)}M"
        return f"{millions:.1f}".rstrip("0").rstrip(".") + "M"
    thousands = rounded / 1000
    if float(thousands).is_integer():
        return f"{int(thousands)}K"
    return f"{thousands:.1f}".rstrip("0").rstrip(".") + "K"


def apply_iteration_axis(ax: object, *, nbins: int = 6) -> None:
    from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, integer=True, min_n_ticks=4))
    ax.xaxis.set_major_formatter(FuncFormatter(format_iteration_k))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="x", labelrotation=0)


def metric_display_label(metric: str) -> str:
    labels = {
        "val/clean_full_exact": "Reward",
        "val_clean_full_exact": "Reward",
        "clean_full_exact": "Reward",
        "final_clean_full_exact": "Reward",
        "val/clean_final_exact": "Final exact",
        "val_clean_final_exact": "Final exact",
        "clean_final_exact": "Final exact",
        "final_clean_final_exact": "Final exact",
        "val/loss": "Loss",
        "val_loss": "Loss",
    }
    if metric in labels:
        return labels[metric]
    short = metric.replace("val/", "").replace("val_", "")
    return short.replace("_", " ").strip().capitalize()


def save_publication_figure(fig: object, out_path: str | object, *, formats: tuple[str, ...] = ("png", "pdf")) -> None:
    from pathlib import Path

    path = Path(out_path)
    base = path.with_suffix("")
    for suffix in formats:
        fig.savefig(base.with_suffix(f".{suffix}"))


__doc__ = """
Shared publication plotting style.

Notebook usage:

    from scripts.plot_style import set_publication_style, get_method_style

    set_publication_style()
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for method in methods:
        style = get_method_style(method)
        ax.plot(x, y[method], label=style.label, **style.plot_kwargs)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Clean exact accuracy")
    ax.legend()
    fig.savefig("figure.pdf")
"""

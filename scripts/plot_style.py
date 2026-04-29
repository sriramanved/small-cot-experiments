from __future__ import annotations

from dataclasses import dataclass


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


# Okabe-Ito-inspired colorblind-friendly method palette.
OFFLINE_BC_COLOR = "#0072B2"
TM_OPD_COLOR = "#009E73"
NAIL_FORWARD_COLOR = "#D55E00"
NAIL_FORWARD_SAMPLED_COLOR = "#E69F00"
NAIL_REVERSE_COLOR = "#CC79A7"
NAIL_REVERSE_FULL_COLOR = "#8E5EA2"
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
    "Offline BC": MethodStyle("Offline BC", OFFLINE_BC_COLOR, linestyle="-", marker="o"),
    "TM OPD": MethodStyle("TM OPD", TM_OPD_COLOR, linestyle="-", marker="^"),
    "TM OPD, greedy rollout": MethodStyle("TM OPD, greedy rollout", TM_OPD_COLOR, linestyle="-", marker="^"),
    "TM OPD, sampled rollout": MethodStyle("TM OPD, sampled rollout", TM_OPD_COLOR, linestyle="--", marker="^"),
    "NAIL-forward": MethodStyle("NAIL-forward", NAIL_FORWARD_COLOR, linestyle="-", marker="s"),
    "NAIL-forward, greedy rollout": MethodStyle(
        "NAIL-forward, greedy rollout",
        NAIL_FORWARD_COLOR,
        linestyle="-",
        marker="s",
    ),
    "NAIL-forward, sampled rollout": MethodStyle(
        "NAIL-forward, sampled rollout",
        NAIL_FORWARD_SAMPLED_COLOR,
        linestyle="--",
        marker="s",
    ),
    "NAIL-reverse": MethodStyle("NAIL-reverse", NAIL_REVERSE_COLOR, linestyle="-", marker="D"),
    "NAIL-reverse, greedy rollout": MethodStyle(
        "NAIL-reverse, greedy rollout",
        NAIL_REVERSE_COLOR,
        linestyle="-",
        marker="D",
    ),
    "NAIL-reverse, sampled rollout": MethodStyle(
        "NAIL-reverse, sampled rollout",
        NAIL_REVERSE_COLOR,
        linestyle="--",
        marker="D",
    ),
    "NAIL-reverse MC": MethodStyle("NAIL-reverse MC", NAIL_REVERSE_COLOR, linestyle="-", marker="D"),
    "NAIL-reverse full": MethodStyle("NAIL-reverse full", NAIL_REVERSE_FULL_COLOR, linestyle="-.", marker="D"),
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
            "lines.linewidth": 3.0,
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

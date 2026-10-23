"""Paper tables and figures: one palette and style, CSV + LaTeX tables, CI dot plots and heatmaps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Palette from the dataviz reference: blue sequential ramp, blue<->red diverging with a gray midpoint
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6b67", "#e4e3df"
SEQ = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Diverging anchors, interpolated: blue 700/400, gray midpoint, reference red, a darker red end (2 steps per arm)
DIV = ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#8f1d1d"]


def set_style() -> None:
    """Matplotlib defaults for print: small type, recessive grid, no top/right spines."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK_MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "text.color": INK,
            "pdf.fonttype": 42,
        }
    )


def cmap(kind: str = "seq") -> Any:
    """Sequential (magnitude) or diverging (signed difference) colormap."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(kind, SEQ if kind == "seq" else DIV)


def save_table(df: pd.DataFrame, out_dir: Path, stem: str, caption: str, float_fmt: str = "{:.3f}") -> None:
    """Write `stem.csv` (full precision) and a booktabs `stem.tex` with the caption that names the setting."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    fmt: dict[str, Any] = {}
    for c in df.columns:
        if not pd.api.types.is_float_dtype(df[c]):
            continue
        vals = df[c].dropna()
        if c.startswith("p_"):
            fmt[c] = lambda v: "<.001" if v < 0.001 else f"{v:.3f}"
        elif len(vals) and (vals == vals.round()).all() and not c.endswith(("f1", "rate", "lo", "hi")):
            fmt[c] = "{:.0f}".format  # counts that became float because of NaN rows
        else:
            fmt[c] = float_fmt.format
    styler = df.style.hide(axis="index").format(fmt, na_rep="--")
    tex = styler.to_latex(caption=caption, label=f"tab:{stem}", hrules=True, position="t")
    (out_dir / f"{stem}.tex").write_text(tex)


def ci_dotplot(ax: Any, df: pd.DataFrame, title: str, ref: float | None = None, ref_label: str = "") -> None:
    """Point + 95% CI per model (rows in df order); hollow markers = not separable from the best after Holm."""
    y = np.arange(len(df))[::-1]
    ax.hlines(y, df["ci_lo"], df["ci_hi"], color=INK_MUTED, linewidth=1.2)
    tied, f1 = df["tied_with_best"].astype(bool).to_numpy(), df["macro_f1"].to_numpy()
    ax.scatter(f1[~tied], y[~tied], s=22, color=SEQ[6], zorder=3, label="below best (Holm)")
    ax.scatter(
        f1[tied],
        y[tied],
        s=22,
        facecolor="white",
        edgecolor=SEQ[6],
        linewidth=1.4,
        zorder=3,
        label="best or not separable",
    )
    if ref is not None:
        ax.axvline(ref, color=INK_MUTED, linestyle="--", linewidth=0.9)
        ax.text(ref, -0.9, f" {ref_label}", color=INK_MUTED, fontsize=7, va="center")
        ax.set_ylim(-1.4, len(df) - 0.5)
    ax.set_yticks(y, df["model"])
    ax.set_title(title, loc="left")
    ax.set_xlabel("macro-F1 (strict)")
    ax.grid(axis="y", visible=False)


def heatmap(
    ax: Any,
    mat: pd.DataFrame,
    title: str,
    kind: str = "seq",
    vmin: float = 0,
    vmax: float = 1,
    fmt: str = "{:.2f}",
) -> Any:
    """Annotated cell heatmap; text switches to white on dark cells so values stay legible."""
    im = ax.imshow(mat.to_numpy(dtype=float), cmap=cmap(kind), vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(mat.shape[0]), mat.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    span = vmax - vmin
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            if pd.isna(v):
                continue
            dark = (abs(v - (vmin + vmax) / 2) > 0.3 * span) if kind == "div" else (v - vmin) > 0.6 * span
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6, color="white" if dark else INK)
    ax.set_title(title, loc="left")
    return im


def tex_escape(s: str) -> str:
    """Escape the LaTeX specials that occur in model names, repos and labels."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def write_paper_table(
    path: Path,
    header: list[str],
    rows: list[list[str]],
    caption: str,
    label: str,
    colspec: str,
    group_header: str | None = None,
    midrule_before: tuple[int, ...] = (),
    star: bool = False,
    note: str | None = None,
    fit: bool = False,
) -> None:
    """Compact booktabs table for the paper; cells are LaTeX strings already formatted by the caller.

    fit=True scales the tabular down to the line width (wide model or per-task tables).
    """
    env = "table*" if star else "table"
    lines = [
        rf"\begin{{{env}}}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]
    if fit:
        lines.append(r"\resizebox{\linewidth}{!}{%")
    lines += [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    if group_header:
        lines.append(group_header)
    lines += [" & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i in midrule_before:
            lines.append(r"\midrule")
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}" + ("}" if fit else "")]
    if note:
        lines.append(rf"\par\smallskip\parbox{{\linewidth}}{{\footnotesize {note}}}")
    lines.append(rf"\end{{{env}}}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")

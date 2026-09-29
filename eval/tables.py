"""Minimal LaTeX table writer for paper/tables/*.tex (booktabs)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mhrag.config import PROJECT_ROOT

TABLE_DIR = PROJECT_ROOT / "paper" / "tables"


def esc(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def fmt(x, nd: int = 3) -> str:
    if x is None:
        return "--"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return esc(x)


def fmt_ci(mean, lo, hi, nd: int = 3) -> str:
    if mean is None:
        return "--"
    return f"{mean:.{nd}f} {{\\scriptsize [{lo:.{nd}f}, {hi:.{nd}f}]}}"


def write_table(name: str, header: Sequence[str], rows: Sequence[Sequence[str]], caption: str, label: str,
                note: str | None = None, align: str | None = None, source: str | None = None) -> Path:
    """Rows must already be formatted strings (use fmt / fmt_ci). `source` names the results file."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    align = align or ("l" + "r" * (len(header) - 1))
    lines = [
        f"% Auto-generated from {source or 'results/'} -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{adjustbox}{max width=\\linewidth}",  # shrink only tables wider than the text block
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(esc(h) if "\\" not in h and "$" not in h else h for h in header) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if note:
        lines.append(f"\\par\\smallskip\\footnotesize {note}")
    lines.append("\\end{table}")
    path = TABLE_DIR / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n")
    return path

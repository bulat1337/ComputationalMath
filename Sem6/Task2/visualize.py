from __future__ import annotations

import argparse
import csv
import math
from html import escape
from pathlib import Path
from typing import Iterable


COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def parse_float(value: str | None) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_csv_columns(path: Path) -> dict[str, list[float]]:
    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")

        data = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                data[name].append(parse_float(row.get(name)))

    return data


def require_columns(path: Path, data: dict[str, list[float]], columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in data]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{path} does not contain required columns: {missing_text}")


def finite(value: float) -> bool:
    return math.isfinite(value)


def linspace(left: float, right: float, count: int) -> list[float]:
    if count <= 1:
        return [left]
    step = (right - left) / (count - 1)
    return [left + i * step for i in range(count)]


def padded_range(values: list[float], padding: float = 0.06) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot build a plot from empty data")

    left = min(values)
    right = max(values)

    if left == right:
        delta = max(abs(left) * 0.1, 1.0)
        return left - delta, right + delta

    delta = (right - left) * padding
    return left - delta, right + delta


def tick_label(value: float) -> str:
    if not finite(value):
        return ""
    if value == 0.0:
        return "0"

    abs_value = abs(value)
    if abs_value >= 10000 or abs_value < 0.001:
        return f"{value:.2e}"
    if abs_value >= 100:
        return f"{value:.1f}"
    return f"{value:.4g}"


def split_line_segments(
    x_values: list[float],
    y_values: list[float],
    x_to_px,
    y_to_px,
    log_y: bool,
) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    for x_value, y_value in zip(x_values, y_values):
        valid = finite(x_value) and finite(y_value) and (not log_y or y_value > 0.0)

        if valid:
            current.append((x_to_px(x_value), y_to_px(y_value)))
        elif current:
            segments.append(current)
            current = []

    if current:
        segments.append(current)

    return segments


def plot_svg(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    x_values: list[float],
    series: list[tuple[str, list[float]]],
    log_y: bool = False,
) -> None:
    width = 960
    height = 560
    left = 92
    right = 34
    top = 58
    bottom = 76
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_data = [x for x in x_values if finite(x)]
    y_data: list[float] = []
    for _, values in series:
        for value in values:
            if finite(value) and (not log_y or value > 0.0):
                y_data.append(math.log10(value) if log_y else value)

    if not x_data:
        raise ValueError(f"no finite x values for {path}")
    if not y_data:
        raise ValueError(f"no finite y values for {path}")

    x_min, x_max = padded_range(x_data, padding=0.0)
    y_min, y_max = padded_range(y_data)

    def y_transform(value: float) -> float:
        return math.log10(value) if log_y else value

    def x_to_px(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_to_px(value: float) -> float:
        transformed = y_transform(value)
        return top + plot_height - (transformed - y_min) / (y_max - y_min) * plot_height

    def y_tick_to_px(transformed: float) -> float:
        return top + plot_height - (transformed - y_min) / (y_max - y_min) * plot_height

    x_ticks = linspace(x_min, x_max, 6)
    y_ticks = linspace(y_min, y_max, 6)

    elements: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #222; }",
        ".title { font-size: 20px; font-weight: 700; }",
        ".axis-label { font-size: 14px; }",
        ".tick { font-size: 12px; fill: #555; }",
        ".grid { stroke: #e3e3e3; stroke-width: 1; }",
        ".axis { stroke: #333; stroke-width: 1.4; }",
        ".legend { font-size: 13px; }",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="{width / 2:.1f}" y="30" text-anchor="middle">{escape(title)}</text>',
    ]

    for tick in x_ticks:
        x_px = x_to_px(tick)
        elements.append(
            f'<line class="grid" x1="{x_px:.2f}" y1="{top}" '
            f'x2="{x_px:.2f}" y2="{top + plot_height}"/>'
        )
        elements.append(
            f'<text class="tick" x="{x_px:.2f}" y="{top + plot_height + 22}" '
            f'text-anchor="middle">{escape(tick_label(tick))}</text>'
        )

    for tick in y_ticks:
        y_px = y_tick_to_px(tick)
        label_value = 10.0**tick if log_y else tick
        elements.append(
            f'<line class="grid" x1="{left}" y1="{y_px:.2f}" '
            f'x2="{left + plot_width}" y2="{y_px:.2f}"/>'
        )
        elements.append(
            f'<text class="tick" x="{left - 10}" y="{y_px + 4:.2f}" '
            f'text-anchor="end">{escape(tick_label(label_value))}</text>'
        )

    elements.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<text class="axis-label" x="{left + plot_width / 2:.1f}" '
            f'y="{height - 24}" text-anchor="middle">{escape(x_label)}</text>',
            f'<text class="axis-label" transform="translate(24,{top + plot_height / 2:.1f}) '
            f'rotate(-90)" text-anchor="middle">{escape(y_label)}</text>',
        ]
    )

    for index, (label, values) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        segments = split_line_segments(x_values, values, x_to_px, y_to_px, log_y)
        for segment in segments:
            points = " ".join(f"{x:.2f},{y:.2f}" for x, y in segment)
            if len(segment) == 1:
                x_px, y_px = segment[0]
                elements.append(f'<circle cx="{x_px:.2f}" cy="{y_px:.2f}" r="2.8" fill="{color}"/>')
            else:
                elements.append(
                    f'<polyline points="{points}" fill="none" stroke="{color}" '
                    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
                )

    legend_x = left + plot_width - 150
    legend_y = top + 18
    for index, (label, _) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        y_pos = legend_y + index * 22
        elements.append(
            f'<line x1="{legend_x}" y1="{y_pos - 4}" x2="{legend_x + 24}" y2="{y_pos - 4}" '
            f'stroke="{color}" stroke-width="2.6"/>'
        )
        elements.append(
            f'<text class="legend" x="{legend_x + 32}" y="{y_pos}" '
            f'text-anchor="start">{escape(label)}</text>'
        )

    elements.append("</svg>\n")
    path.write_text("\n".join(elements), encoding="utf-8")


def make_residual_norm(phi1: list[float], phi2: list[float]) -> list[float]:
    residuals: list[float] = []
    for left, right in zip(phi1, phi2):
        if finite(left) and finite(right):
            value = math.hypot(left, right)
            residuals.append(value if value > 0.0 else math.nan)
        else:
            residuals.append(math.nan)
    return residuals


def write_index(out_dir: Path, images: list[Path]) -> Path:
    path = out_dir / "index.html"
    body = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>main.cpp visualization</title>",
        "<style>",
        "body { font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #222; }",
        "img { display: block; max-width: 100%; margin: 24px 0; border: 1px solid #ddd; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>main.cpp visualization</h1>",
    ]

    for image in images:
        rel_path = image.relative_to(out_dir)
        body.append(f'<h2>{escape(image.stem)}</h2>')
        body.append(f'<img src="{escape(str(rel_path))}" alt="{escape(image.stem)}">')

    body.extend(["</body>", "</html>", ""])
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize CSV files generated by main.cpp.")
    parser.add_argument("--solution", default="solution_single_gamma.csv", help="profile CSV from main.cpp")
    parser.add_argument("--branch", default="branch.csv", help="gamma branch CSV from main.cpp")
    parser.add_argument("--out-dir", default="plots", help="directory for SVG output")
    args = parser.parse_args()

    solution_path = Path(args.solution)
    branch_path = Path(args.branch)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []

    if solution_path.exists():
        solution = read_csv_columns(solution_path)
        require_columns(solution_path, solution, ["x", "u", "up", "v", "vp"])

        path = out_dir / "solution_uv.svg"
        plot_svg(
            path=path,
            title="Solution profile for fixed gamma",
            x_label="x",
            y_label="u(x), v(x)",
            x_values=solution["x"],
            series=[("u(x)", solution["u"]), ("v(x)", solution["v"])],
        )
        created.append(path)

        path = out_dir / "solution_derivatives.svg"
        plot_svg(
            path=path,
            title="Derivative profile for fixed gamma",
            x_label="x",
            y_label="derivative",
            x_values=solution["x"],
            series=[("u'(x)", solution["up"]), ("v'(x)", solution["vp"])],
        )
        created.append(path)
    else:
        print(f"skip: {solution_path} not found")

    if branch_path.exists():
        branch = read_csv_columns(branch_path)
        require_columns(branch_path, branch, ["gamma", "u0", "v0", "u1", "v1", "phi1", "phi2"])

        path = out_dir / "branch_u.svg"
        plot_svg(
            path=path,
            title="u values along gamma branch",
            x_label="gamma",
            y_label="u",
            x_values=branch["gamma"],
            series=[("u(0)", branch["u0"]), ("u(1)", branch["u1"])],
        )
        created.append(path)

        path = out_dir / "branch_v.svg"
        plot_svg(
            path=path,
            title="v values along gamma branch",
            x_label="gamma",
            y_label="v",
            x_values=branch["gamma"],
            series=[("v(0)", branch["v0"]), ("v(1)", branch["v1"])],
        )
        created.append(path)

        path = out_dir / "branch_residual.svg"
        plot_svg(
            path=path,
            title="Boundary residual along gamma branch",
            x_label="gamma",
            y_label="residual norm",
            x_values=branch["gamma"],
            series=[("sqrt(phi1^2 + phi2^2)", make_residual_norm(branch["phi1"], branch["phi2"]))],
            log_y=True,
        )
        created.append(path)
    else:
        print(f"skip: {branch_path} not found")

    if not created:
        raise SystemExit("no plots created: run main.cpp first or pass CSV paths explicitly")

    index = write_index(out_dir, created)

    print("created plots:")
    for path in created:
        print(f"  {path}")
    print(f"index: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

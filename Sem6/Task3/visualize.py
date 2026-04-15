from __future__ import annotations

import argparse
import csv
import math
from html import escape
from pathlib import Path
from typing import Any, Iterable


COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]


def parse_float(value: str | None) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_csv_columns(path: Path, raw_columns: Iterable[str] = ()) -> dict[str, list[Any]]:
    raw = set(raw_columns)
    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")

        data = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                value = row.get(name)
                if name in raw:
                    data[name].append("" if value is None else value)
                else:
                    data[name].append(parse_float(value))
    return data


def require_columns(path: Path, data: dict[str, list[Any]], columns: Iterable[str]) -> None:
    missing = [name for name in columns if name not in data]
    if missing:
        raise ValueError(f"{path} does not contain required columns: {', '.join(missing)}")


def finite(value: float) -> bool:
    return math.isfinite(value)


def linspace(left: float, right: float, count: int) -> list[float]:
    if count <= 1:
        return [left]
    step = (right - left) / (count - 1)
    return [left + step * i for i in range(count)]


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
    if abs_value >= 10000.0 or abs_value < 0.001:
        return f"{value:.2e}"
    if abs_value >= 100.0:
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
        label_value = 10.0 ** tick if log_y else tick
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
            f'<line class="axis" x1="{left}" y1="{top}" '
            f'x2="{left}" y2="{top + plot_height}"/>',
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
                elements.append(
                    f'<circle cx="{x_px:.2f}" cy="{y_px:.2f}" r="2.8" fill="{color}"/>'
                )
            else:
                elements.append(
                    f'<polyline points="{points}" fill="none" stroke="{color}" '
                    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
                )

    legend_x = left + plot_width - 190
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


def absolute_error(values: list[float], exact: list[float]) -> list[float]:
    result: list[float] = []
    for left, right in zip(values, exact):
        if finite(left) and finite(right):
            diff = abs(left - right)
            result.append(diff if diff > 0.0 else math.nan)
        else:
            result.append(math.nan)
    return result


def series_by_scheme(
    data: dict[str, list[Any]],
    x_name: str,
    y_name: str,
) -> dict[str, tuple[list[float], list[float]]]:
    require_columns(Path("<in-memory>"), data, [x_name, y_name, "scheme"])
    grouped: dict[str, tuple[list[float], list[float]]] = {}
    for scheme, x_value, y_value in zip(data["scheme"], data[x_name], data[y_name]):
        scheme_name = str(scheme)
        if scheme_name not in grouped:
            grouped[scheme_name] = ([], [])
        grouped[scheme_name][0].append(float(x_value))
        grouped[scheme_name][1].append(float(y_value))
    return grouped


def plot_profiles(profile_data: dict[str, list[float]], out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    x_values = profile_data["x"]

    for index in range(1, 4):
        lax = profile_data[f"lax_u{index}"]
        box = profile_data[f"box_u{index}"]
        exact = profile_data[f"exact_u{index}"]

        profile_path = out_dir / f"profile_u{index}.svg"
        plot_svg(
            profile_path,
            title=f"u{index}(x, T): Lax vs Box vs exact",
            x_label="x",
            y_label=f"u{index}",
            x_values=x_values,
            series=[
                ("Lax", lax),
                ("Box", box),
                ("Exact", exact),
            ],
            log_y=False,
        )
        paths.append(profile_path)

        error_path = out_dir / f"error_u{index}.svg"
        plot_svg(
            error_path,
            title=f"|error| for component u{index} at T = 1",
            x_label="x",
            y_label=f"|u{index} - u{index} exact|",
            x_values=x_values,
            series=[
                ("Lax error", absolute_error(lax, exact)),
                ("Box error", absolute_error(box, exact)),
            ],
            log_y=True,
        )
        paths.append(error_path)

    return paths


def plot_convergence(convergence_data: dict[str, list[float]], out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    grouped_error = series_by_scheme(convergence_data, "M", "error_max")
    grouped_order = series_by_scheme(convergence_data, "M", "order_from_exact")

    error_path = out_dir / "convergence_error.svg"
    plot_svg(
        error_path,
        title="Maximum error vs grid size",
        x_label="M",
        y_label="max error",
        x_values=grouped_error["lax"][0],
        series=[
            ("Lax", grouped_error["lax"][1]),
            ("Box", grouped_error["box"][1]),
        ],
        log_y=True,
    )
    paths.append(error_path)

    order_path = out_dir / "convergence_order.svg"
    plot_svg(
        order_path,
        title="Observed order from exact error",
        x_label="M",
        y_label="order",
        x_values=grouped_order["lax"][0],
        series=[
            ("Lax", grouped_order["lax"][1]),
            ("Box", grouped_order["box"][1]),
        ],
        log_y=False,
    )
    paths.append(order_path)

    return paths


def write_index(path: Path, images: list[Path]) -> None:
    body = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Task 3 visualization</title>",
        "<style>",
        "body { font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #222; }",
        "h1 { font-size: 28px; margin-bottom: 8px; }",
        "p { color: #444; max-width: 900px; }",
        ".plot { margin: 24px 0 36px; }",
        ".plot h2 { font-size: 18px; margin: 0 0 12px; }",
        "img { max-width: 100%; border: 1px solid #ddd; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Sem6/Task3 visualization</h1>",
        "<p>Plots built from results/convergence.csv and results/final_profile_M400.csv.</p>",
    ]

    for image in images:
        body.extend(
            [
                '<div class="plot">',
                f"<h2>{escape(image.stem)}</h2>",
                f'<img src="{escape(image.name)}" alt="{escape(image.stem)}">',
                "</div>",
            ]
        )

    body.extend(["</body>", "</html>"])
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize Sem6/Task3 numerical results without external dependencies."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory containing convergence.csv and final_profile_M400.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots",
        help="Directory where SVG plots and index.html will be written",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    out_dir = args.out_dir.resolve()

    convergence_path = results_dir / "convergence.csv"
    profile_path = results_dir / "final_profile_M400.csv"

    convergence_data = read_csv_columns(convergence_path, raw_columns={"scheme"})
    profile_data = read_csv_columns(profile_path)

    require_columns(
        convergence_path,
        convergence_data,
        ["scheme", "M", "error_max", "order_from_exact"],
    )
    require_columns(
        profile_path,
        profile_data,
        [
            "x",
            "lax_u1", "lax_u2", "lax_u3",
            "box_u1", "box_u2", "box_u3",
            "exact_u1", "exact_u2", "exact_u3",
        ],
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    images: list[Path] = []
    images.extend(plot_profiles(profile_data, out_dir))
    images.extend(plot_convergence(convergence_data, out_dir))
    write_index(out_dir / "index.html", images)

    print(f"Plots written to {out_dir}")
    for image in images:
        print(image.name)
    print("index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

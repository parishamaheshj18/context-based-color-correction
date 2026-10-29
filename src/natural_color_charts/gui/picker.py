"""Matplotlib window: click a pixel on the photo, then the chart patch it truly is.

Each click pair is saved as (pixel RGB, chart RGB). "Ask Claude" adds Claude's own picks
(purple); right-click one to ignore or restore it. "Correct" combines both.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

from natural_color_charts.domain import BoxXyxy, UserColorAnchor
from natural_color_charts.gui.session import MatchSession, corrected_uint8, describe
from natural_color_charts.io.writers import write_image8

YOU, CLAUDE, PENDING = "#eb6834", "#6250d6", "#ffffff"
CHART_COLUMNS = 6
# The pipeline measures a small area around each clicked pixel; one pixel is too noisy.
POINT_HALF_SIZE = 0.015  # fraction of the long image side
MIN_HALF_SIZE_PX = 12


@dataclass(frozen=True)
class ColorMatch:
    """A clicked pixel and the chart patch whose true color it has."""

    point: tuple[float, float]  # normalized x, y
    pixel_rgb: tuple[int, int, int]
    patch: int

    def anchor(self, width: int, height: int) -> UserColorAnchor:
        """A small square around the pixel, in normalized coordinates."""
        half_px = max(POINT_HALF_SIZE * max(width, height), MIN_HALF_SIZE_PX)
        half_x, half_y = half_px / width, half_px / height
        x, y = self.point
        box = (
            max(x - half_x, 0.0),
            max(y - half_y, 0.0),
            min(x + half_x, 1.0),
            min(y + half_y, 1.0),
        )
        return UserColorAnchor(box, self.patch)


class MatchPicker:
    """The matching window; event handlers call the plain methods below, which tests drive."""

    def __init__(
        self, session: MatchSession, title: str, output_dir: Path, jpeg_quality: int = 95
    ) -> None:
        self._session = session
        self._output_dir = output_dir
        self._jpeg_quality = jpeg_quality
        self.matches: list[ColorMatch] = []
        self.ignored: set[int] = set()
        self.pending: tuple[tuple[float, float], tuple[int, int, int]] | None = None
        self._artists: list[Any] = []
        height, width = session.preview.shape[:2]
        self._size = (width, height)

        self.figure = plt.figure(figsize=(15, 8.5), layout="constrained")
        if self.figure.canvas.manager is not None:
            self.figure.canvas.manager.set_window_title(title)
        grid = self.figure.add_gridspec(3, 2, width_ratios=[2.6, 1], height_ratios=[4, 3, 0.4])
        self.ax_photo = self.figure.add_subplot(grid[:, 0])
        self.ax_chart = self.figure.add_subplot(grid[0, 1])
        self.ax_info = self.figure.add_subplot(grid[1, 1])
        buttons = grid[2, 1].subgridspec(1, 3)
        actions: list[tuple[str, Callable[[], object]]] = [
            ("Undo", self.undo),
            ("Ask Claude", self.ask_claude),
            ("Correct", self.correct),
        ]
        self._buttons = []
        for i, (label, action) in enumerate(actions):
            button = Button(self.figure.add_subplot(buttons[0, i]), label)
            button.on_clicked(_ignore_event(action))
            self._buttons.append(button)

        self.ax_photo.imshow(session.preview)
        self.ax_photo.set_xticks([])
        self.ax_photo.set_yticks([])
        self.ax_photo.set_title("Click a pixel, then the chart patch that is its true color")
        self._draw_chart()
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self._message = "Click a pixel whose true color is on the chart."
        self.redraw()

    @property
    def anchors(self) -> list[UserColorAnchor]:
        """The matches as regions for the pipeline."""
        width, height = self._size
        return [m.anchor(width, height) for m in self.matches]

    # ---------- actions ----------
    def pick_pixel(self, x: float, y: float) -> None:
        """Select the pixel at normalized x, y; it waits for its chart patch."""
        width, height = self._size
        column = min(int(x * width), width - 1)
        row = min(int(y * height), height - 1)
        # Average a 5x5 neighborhood so the shown RGB is not one noisy pixel.
        window = self._session.preview[
            max(row - 2, 0) : row + 3, max(column - 2, 0) : column + 3
        ].reshape(-1, 3)
        rgb = tuple(int(v) for v in np.round(window.mean(axis=0)))
        self.pending = ((x, y), (rgb[0], rgb[1], rgb[2]))
        self._message = f"Pixel RGB {self.pending[1]}. Now click its chart patch."
        self.redraw()

    def choose_patch(self, index: int) -> None:
        """Pair the pending pixel with a chart patch."""
        if self.pending is None:
            self._message = "Click a pixel on the photo first."
        else:
            point, rgb = self.pending
            self.matches.append(ColorMatch(point, rgb, index))
            self.pending = None
            swatch = self._session.swatches[index]
            self._message = f"Saved pair: pixel {rgb} = {swatch.name} {_rgb255(swatch.srgb)}."
        self.redraw()

    def undo(self) -> None:
        """Drop the pending pixel, or else the last pair."""
        if self.pending is not None:
            self.pending = None
        elif self.matches:
            self.matches.pop()
        self.redraw()

    def toggle_claude_at(self, x: float, y: float) -> None:
        """Ignore or restore the Claude pick under a normalized point."""
        context = self._session.claude
        if context is None:
            return
        for i, candidate in enumerate(context.reference_candidates):
            if candidate.usable and any(_inside(box, x, y) for box in candidate.boxes):
                self.ignored ^= {i}
                state = "ignored" if i in self.ignored else "restored"
                self._message = f"Claude's {candidate.label.value} pick {state}."
                self.redraw()
                return

    def ask_claude(self) -> None:
        """Run scene analysis and show Claude's trusted regions."""
        self._message = "Asking Claude..."
        self.redraw()
        self.figure.canvas.flush_events()
        context = self._session.ask_claude()
        self.ignored.clear()
        if context.source != "vlm":
            self._message = f"Claude unavailable ({context.rationale}); using your matches only."
        else:
            trusted = sum(1 for c in context.reference_candidates if c.usable and c.boxes)
            self._message = f"Claude trusts {trusted} material(s). Right-click one to ignore it."
        self.redraw()

    def correct(self) -> Path:
        """Correct, save, and show the result next to the original."""
        self._message = "Correcting..."
        self.redraw()
        self.figure.canvas.flush_events()
        result = self._session.correct(self.anchors, sorted(self.ignored))
        lines = describe(result)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        saved = write_image8(
            self._output_dir / "corrected.jpg", result.rendered.encoded_srgb, self._jpeg_quality
        )
        self.save_pairs(self._output_dir / "matches.csv")
        height, width = self._session.preview.shape[:2]
        figure, (before, after) = plt.subplots(
            1, 2, figsize=(2 * 6 * width / height, 6.6), layout="constrained"
        )
        for ax, image, label in [
            (before, self._session.preview, "Original"),
            (after, corrected_uint8(result), "Corrected"),
        ]:
            ax.imshow(image)
            ax.set_title(label)
            ax.set_axis_off()
        figure.suptitle(lines[0])
        figure.savefig(self._output_dir / "comparison.png", dpi=110)
        self._message = "\n".join(
            [*(_short(line, 60) for line in lines), f"Saved to {saved.parent}"]
        )
        self.redraw()
        for line in lines:
            print(line)
        figure.show()
        return saved

    def save_pairs(self, path: Path) -> Path:
        """Write the clicked pixel / chart patch RGB pairs as CSV."""
        width, height = self._size
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "x_px",
                    "y_px",
                    "pixel_r",
                    "pixel_g",
                    "pixel_b",
                    "patch",
                    "name",
                    "chart_r",
                    "chart_g",
                    "chart_b",
                ]
            )
            for m in self.matches:
                swatch = self._session.swatches[m.patch]
                writer.writerow(
                    [
                        round(m.point[0] * width),
                        round(m.point[1] * height),
                        *m.pixel_rgb,
                        m.patch + 1,
                        swatch.name,
                        *_rgb255(swatch.srgb),
                    ]
                )
        return path

    def show(self) -> None:
        """Open the window and block until it is closed."""
        plt.show()

    # ---------- events ----------
    def _on_click(self, event: MouseEvent) -> None:
        toolbar = self.figure.canvas.toolbar
        if event.xdata is None or event.ydata is None or (toolbar and toolbar.mode):
            return  # outside the axes, or the zoom/pan tool is active
        if event.inaxes is self.ax_chart:
            column, row = int(event.xdata), int(event.ydata)
            index = row * CHART_COLUMNS + column
            if 0 <= column < CHART_COLUMNS and 0 <= index < len(self._session.swatches):
                self.choose_patch(index)
        elif event.inaxes is self.ax_photo:
            width, height = self._size
            x, y = event.xdata / width, event.ydata / height
            # On a Mac trackpad, Ctrl + click arrives as a right click.
            if event.button is MouseButton.RIGHT or event.key == "control":
                self.toggle_claude_at(x, y)
            elif event.button is MouseButton.LEFT:
                self.pick_pixel(x, y)

    # ---------- drawing ----------
    def _draw_chart(self) -> None:
        ax = self.ax_chart
        for i, swatch in enumerate(self._session.swatches):
            row, column = divmod(i, CHART_COLUMNS)
            ax.add_patch(Rectangle((column + 0.06, row + 0.06), 0.88, 0.88, color=swatch.srgb))
            ink = "black" if sum(swatch.srgb) > 1.6 else "white"
            ax.text(column + 0.5, row + 0.5, str(i + 1), ha="center", va="center", color=ink)
        rows = -(-len(self._session.swatches) // CHART_COLUMNS)
        ax.set_xlim(0, CHART_COLUMNS)
        ax.set_ylim(rows, 0)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title("Color chart (true colors under daylight)")

    def redraw(self) -> None:
        """Redraw boxes and the side panel from the current state."""
        for artist in self._artists:
            artist.remove()
        self._artists = []
        context = self._session.claude
        if context is not None:
            for i, candidate in enumerate(context.reference_candidates):
                if not candidate.usable:
                    continue
                ignored = i in self.ignored
                for box in candidate.boxes:
                    self._box(box, CLAUDE, candidate.label.value, dashed=ignored)
        for number, match in enumerate(self.matches, start=1):
            swatch = self._session.swatches[match.patch]
            self._point(match.point, YOU, f"{number}: {swatch.name}", fill=swatch.srgb)
        if self.pending is not None:
            self._point(self.pending[0], PENDING, "?")
        self._draw_info()
        self.figure.canvas.draw_idle()

    def _box(
        self,
        box: BoxXyxy,
        color: str,
        label: str,
        dashed: bool = False,
        fill: tuple[float, float, float] | None = None,
    ) -> None:
        width, height = self._size
        x0, y0, x1, y1 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
        style = "--" if dashed else "-"
        frame = Rectangle(
            (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2, linestyle=style
        )
        self._artists.append(self.ax_photo.add_patch(frame))
        ink = "black" if fill is not None and sum(fill) > 1.6 else "white"
        text = self.ax_photo.text(
            x0,
            y0 - 4,
            label,
            color=ink,
            fontsize=8,
            va="bottom",
            bbox={"facecolor": fill or color, "edgecolor": color, "pad": 1.5, "alpha": 0.9},
        )
        self._artists.append(text)

    def _point(
        self,
        point: tuple[float, float],
        color: str,
        label: str,
        fill: tuple[float, float, float] | None = None,
    ) -> None:
        width, height = self._size
        x, y = point[0] * width, point[1] * height
        (marker,) = self.ax_photo.plot(
            x,
            y,
            marker="o",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=2.2,
        )
        self._artists.append(marker)
        ink = "black" if fill is not None and sum(fill) > 1.6 else "white"
        text = self.ax_photo.text(
            x + 8,
            y - 8,
            label,
            color=ink,
            fontsize=8,
            va="bottom",
            bbox={"facecolor": fill or color, "edgecolor": color, "pad": 1.5, "alpha": 0.9},
        )
        self._artists.append(text)

    def _draw_info(self) -> None:
        ax = self.ax_info
        ax.clear()
        ax.set_axis_off()
        lines = [f"Your pairs (pixel RGB = chart RGB): {len(self.matches)}"]
        for number, m in enumerate(self.matches, start=1):
            swatch = self._session.swatches[m.patch]
            lines.append(
                f"  {number}. {m.pixel_rgb} = {_rgb255(swatch.srgb)} {_short(swatch.name, 16)}"
            )
        lines = lines[:1] + lines[1:][-6:]
        context = self._session.claude
        if context is not None and context.source == "vlm":
            lines.append("Claude:")
            for i, c in enumerate(context.reference_candidates):
                mark = "ignored" if i in self.ignored else ("trusted" if c.usable else "rejected")
                lines.append(f"  {c.label.value} ({mark}): {_short(c.description)}")
        lines += ["", self._message]
        ax.text(0, 1, "\n".join(lines), va="top", fontsize=9, family="monospace", wrap=True)


def _ignore_event(action: Callable[[], object]) -> Callable[[Any], None]:
    def handler(_event: Any) -> None:
        action()

    return handler


def _rgb255(srgb: tuple[float, float, float]) -> tuple[int, int, int]:
    r, g, b = (round(v * 255) for v in srgb)
    return r, g, b


def _inside(box: BoxXyxy, x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _short(text: str, limit: int = 38) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"

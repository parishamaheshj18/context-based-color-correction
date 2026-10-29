"""Matplotlib window: draw boxes on the photo, pick the chart patch each one truly is.

Drag on the photo to draw a box, then click a chart patch to match it. "Ask Claude" adds
Claude's own picks (purple); click one to ignore or restore it. "Correct" combines both.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RectangleSelector

from natural_color_charts.domain import BoxXyxy, UserColorAnchor
from natural_color_charts.gui.session import MatchSession, corrected_uint8, describe
from natural_color_charts.io.writers import write_image8

YOU, CLAUDE, PENDING = "#eb6834", "#6250d6", "#ffffff"
CHART_COLUMNS = 6


class MatchPicker:
    """The matching window; event handlers call the plain methods below, which tests drive."""

    def __init__(
        self, session: MatchSession, title: str, output_dir: Path, jpeg_quality: int = 95
    ) -> None:
        self._session = session
        self._output_dir = output_dir
        self._jpeg_quality = jpeg_quality
        self.anchors: list[UserColorAnchor] = []
        self.ignored: set[int] = set()
        self.pending: BoxXyxy | None = None
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
        self.ax_photo.set_title("Photo: drag a box, then pick its true color on the chart")
        self._draw_chart()
        self._selector = RectangleSelector(
            self.ax_photo,
            self._on_box,
            useblit=True,
            button=[MouseButton.LEFT],
            minspanx=6,
            minspany=6,
            spancoords="pixels",
            props={"edgecolor": PENDING, "fill": False, "linestyle": "--", "linewidth": 1.5},
        )
        self._press_xy: tuple[float, float] | None = None
        self.figure.canvas.mpl_connect("button_press_event", self._on_press)
        self.figure.canvas.mpl_connect("button_release_event", self._on_click)
        self._message = "Draw a box on something whose true color is on the chart."
        self.redraw()

    # ---------- actions ----------
    def add_box(self, box: BoxXyxy) -> None:
        """A drawn box waits for its chart patch."""
        self.pending = box
        self._message = "Now click the chart patch this region truly is."
        self.redraw()

    def choose_patch(self, index: int) -> None:
        """Match the pending box to a chart patch."""
        if self.pending is None:
            self._message = "Draw a box on the photo first."
        else:
            self.anchors.append(UserColorAnchor(self.pending, index))
            self.pending = None
            self._message = f"Matched to {self._session.swatches[index].name}."
        self.redraw()

    def undo(self) -> None:
        """Drop the pending box, or else the last match."""
        if self.pending is not None:
            self.pending = None
        elif self.anchors:
            self.anchors.pop()
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
            self._message = f"Claude trusts {trusted} material(s). Click one to ignore it."
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
        self._message = f"Saved {saved}"
        self.redraw()
        for line in lines:
            print(line)
        figure.show()
        return saved

    def show(self) -> None:
        """Open the window and block until it is closed."""
        plt.show()

    # ---------- events ----------
    def _on_box(self, press: MouseEvent, release: MouseEvent) -> None:
        width, height = self._size
        xs = sorted([press.xdata or 0.0, release.xdata or 0.0])
        ys = sorted([press.ydata or 0.0, release.ydata or 0.0])
        self.add_box((xs[0] / width, ys[0] / height, xs[1] / width, ys[1] / height))

    def _on_press(self, event: MouseEvent) -> None:
        self._press_xy = (event.x, event.y)

    def _on_click(self, event: MouseEvent) -> None:
        if event.xdata is None or event.ydata is None:
            return
        if event.inaxes is self.ax_chart:
            column, row = int(event.xdata), int(event.ydata)
            index = row * CHART_COLUMNS + column
            if 0 <= column < CHART_COLUMNS and 0 <= index < len(self._session.swatches):
                self.choose_patch(index)
        elif event.inaxes is self.ax_photo and self._was_click(event):
            # A plain click (too small to be a box) toggles a Claude pick.
            width, height = self._size
            self.toggle_claude_at(event.xdata / width, event.ydata / height)

    def _was_click(self, release: MouseEvent) -> bool:
        if self._press_xy is None:
            return False
        return abs(self._press_xy[0] - release.x) < 6 and abs(self._press_xy[1] - release.y) < 6

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
        for anchor in self.anchors:
            swatch = self._session.swatches[anchor.chart_patch_index]
            self._box(anchor.box_xyxy, YOU, swatch.name, fill=swatch.srgb)
        if self.pending is not None:
            self._box(self.pending, PENDING, "?", dashed=True)
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

    def _draw_info(self) -> None:
        ax = self.ax_info
        ax.clear()
        ax.set_axis_off()
        lines = [f"Your matches: {len(self.anchors)}"]
        lines += [
            f"  {self._session.swatches[a.chart_patch_index].name}" for a in self.anchors[-6:]
        ]
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


def _inside(box: BoxXyxy, x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _short(text: str, limit: int = 38) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"

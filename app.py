"""
ClaimGuard - desktop application (PySide6).

Pure presentation layer. All model interaction goes through ModelService.

Run:  python app.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from string import Template
from typing import Callable, Optional

import pandas as pd
from PySide6.QtCore import (
    QAbstractTableModel, QModelIndex, QObject, QPointF, QRectF, QRegularExpression, QSettings, QSize, Qt,
    QThread, QVariantAnimation, Signal,
)
from PySide6.QtGui import (
    QColor, QFont, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap,
    QPolygonF, QRegularExpressionValidator,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QApplication, QButtonGroup, QComboBox,
    QDoubleSpinBox, QFileDialog, QFrame, QGraphicsDropShadowEffect, QLineEdit, QListView,
    QGridLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QStackedWidget,
    QTableView, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from model_service import (
    Assessment, ClassificationSummary, FieldSpec, ModelService, ServiceError,
)
from ai_assistant import generate_investigation_brief

# ===========================================================================
# Theme: orange / neon green / hot pink, in light and dark
#   orange = primary action, green = low risk / positive, pink = high risk
# ===========================================================================
PALETTES = {
    "light": dict(
        bg="#F5F3F0", sidebar="#FFFFFF", surface="#FFFFFF", surface2="#F4F2EF",
        border="#ECE8E2", border2="#D8D2CA", text="#1C1917", muted="#8A837B",
        track="#EFEBE6",
        orange="#FF6A13", orange_hover="#E85D0A", orange_text="#E85D0A",
        on_orange="#FFFFFF",
        green="#22E06B", green_text="#0B9B45",
        pink="#FF2E93", pink_text="#E0157A",
        tint=0.12,
    ),
    "dark": dict(
        bg="#0E0D10", sidebar="#131216", surface="#19181C", surface2="#232127",
        border="#2A282F", border2="#3A3741", text="#F6F3EF", muted="#8F8A96",
        track="#2A282F",
        orange="#FF7A2B", orange_hover="#FF8F4D", orange_text="#FF8F4D",
        on_orange="#1A0E05",
        green="#3DFF8F", green_text="#3DFF8F",
        pink="#FF4DA6", pink_text="#FF6FB8",
        tint=0.16,
    ),
}
LEVEL_TONE = {"high": "pink", "review": "orange", "low": "green"}
TONES = ["orange", "green", "pink"]


def rgba(hex_: str, alpha: float) -> str:
    c = QColor(hex_)
    return f"rgba({c.red()},{c.green()},{c.blue()},{alpha})"


class ThemeManager(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        saved = QSettings("ClaimGuard", "ClaimGuard").value("theme", "light")
        self.name = saved if saved in PALETTES else "light"

    @property
    def p(self) -> dict:
        return PALETTES[self.name]

    def color(self, key: str) -> QColor:
        return QColor(self.p[key])

    def tone(self, tone: str) -> QColor:
        return QColor(self.p[tone])

    def set_name(self, name: str):
        if name != self.name:
            self.name = name
            QSettings("ClaimGuard", "ClaimGuard").setValue("theme", name)
            self.changed.emit()


THEME = ThemeManager()

QSS = Template("""
QMainWindow, QWidget#root { background: $bg; }
QWidget { color: $text; font-size: 14px; }
QLabel { background: transparent; }
QWidget#canvas { background: transparent; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }

QToolTip { background: $surface; color: $text; border: 1px solid $border; padding: 6px; }
QDialog, QMessageBox, QInputDialog { background: $surface; }

QWidget#sidebar { background: $sidebar; border-right: 1px solid $border; }
QLabel#logo {
    background: $orange; color: $on_orange; border-radius: 11px;
    font-size: 18px; font-weight: 800;
}
QLabel#brand { font-size: 17px; font-weight: 700; }
QLabel#navSection { color: $muted; font-size: 11px; font-weight: 600; padding-left: 14px; }
QLabel#sideNote { color: $muted; font-size: 12px; }
QPushButton#nav {
    text-align: left; padding: 12px 14px; border: none; border-radius: 12px;
    color: $muted; background: transparent; font-weight: 600;
}
QPushButton#nav:hover { background: $surface2; color: $text; }
QPushButton#nav:checked { background: $orange_tint; color: $orange_text; }

QLabel#pageTitle { font-size: 24px; font-weight: 700; }
QLabel#muted { color: $muted; font-size: 12px; }
QLabel#cardHeading { font-size: 15px; font-weight: 650; }
QLabel#statValue { font-size: 28px; font-weight: 700; }
QLabel#tileValue { font-size: 26px; font-weight: 700; }

QFrame#card { background: $surface; border: 1px solid $border; border-radius: 18px; }
QFrame#dropzone { background: transparent; border: 1.5px dashed $border2; border-radius: 18px; }
QFrame#dropzone[hover="true"] { border-color: $orange; background: $orange_tint; }
QFrame#tile { border-radius: 14px; border: none; }
QFrame#tile[tone="green"] { background: $green_tint; }
QFrame#tile[tone="orange"] { background: $orange_tint; }
QFrame#tile[tone="pink"] { background: $pink_tint; }
QLabel#tileValue[tone="green"] { color: $green_text; }
QLabel#tileValue[tone="orange"] { color: $orange_text; }
QLabel#tileValue[tone="pink"] { color: $pink_text; }

QLabel#pill { padding: 4px 10px; border-radius: 10px; font-size: 12px; font-weight: 600; }
QLabel#pill[tone="green"] { background: $green_tint; color: $green_text; }
QLabel#pill[tone="orange"] { background: $orange_tint; color: $orange_text; }
QLabel#pill[tone="pink"] { background: $pink_tint; color: $pink_text; }

QLabel#resultTitle { font-size: 30px; font-weight: 700; }
QLabel#resultTitle[tone="green"] { color: $green_text; }
QLabel#resultTitle[tone="orange"] { color: $orange_text; }
QLabel#resultTitle[tone="pink"] { color: $pink_text; }

QPushButton {
    background: $surface2; color: $text; border: 1px solid $border;
    border-radius: 10px; padding: 8px 16px;
}
QPushButton#primary {
    background: $orange; color: $on_orange; border: none; border-radius: 12px;
    padding: 11px 22px; font-weight: 700;
}
QPushButton#primary:hover { background: $orange_hover; }
QPushButton#primary:disabled { background: $surface2; color: $muted; }
QPushButton#secondary {
    background: $surface; color: $text; border: 1px solid $border2;
    border-radius: 12px; padding: 11px 20px; font-weight: 600;
}
QPushButton#secondary:hover { background: $surface2; }
QPushButton#secondary:disabled { color: $muted; border-color: $border; }

QLineEdit, QComboBox {
    background: $surface2; color: $text; border: 1.5px solid transparent;
    border-radius: 10px; padding: 9px 12px; min-height: 22px;
    selection-background-color: $orange; selection-color: $on_orange;
}
QLineEdit:focus, QComboBox:focus { border: 1.5px solid $orange; }
QComboBox::drop-down { border: none; width: 30px; }
QComboBox QAbstractItemView {
    background: $surface; border: 1px solid $border; padding: 4px;
    selection-background-color: $orange_tint; selection-color: $orange_text;
    outline: none;
}
QComboBox QAbstractItemView::item { min-height: 30px; padding-left: 8px; border-radius: 6px; }

QTableView, QTableWidget {
    background: $surface; border: none; gridline-color: transparent;
    selection-background-color: $orange_tint; selection-color: $text; outline: none;
}
QTableView::item, QTableWidget::item { border-bottom: 1px solid $border; padding: 0 12px; }
QHeaderView { background: $surface; }
QHeaderView::section {
    background: $surface; color: $muted; padding: 10px 12px; border: none;
    border-bottom: 1px solid $border; font-weight: 600; font-size: 12px;
}
QTableCornerButton::section { background: $surface; border: none; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: $border2; border-radius: 4px; min-height: 30px; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: $border2; border-radius: 4px; min-width: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
""")


def build_style() -> str:
    p = THEME.p
    t = p["tint"]
    return QSS.substitute(
        **p,
        orange_tint=rgba(p["orange"], t),
        green_tint=rgba(p["green"], t + 0.04),
        pink_tint=rgba(p["pink"], t),
    )


def apply_theme(app: QApplication):
    p = THEME.p
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(p["bg"]))
    pal.setColor(QPalette.WindowText, QColor(p["text"]))
    pal.setColor(QPalette.Base, QColor(p["surface"]))
    pal.setColor(QPalette.AlternateBase, QColor(p["surface2"]))
    pal.setColor(QPalette.Text, QColor(p["text"]))
    pal.setColor(QPalette.Button, QColor(p["surface2"]))
    pal.setColor(QPalette.ButtonText, QColor(p["text"]))
    pal.setColor(QPalette.Highlight, QColor(p["orange"]))
    pal.setColor(QPalette.HighlightedText, QColor(p["on_orange"]))
    pal.setColor(QPalette.ToolTipBase, QColor(p["surface"]))
    pal.setColor(QPalette.ToolTipText, QColor(p["text"]))
    pal.setColor(QPalette.PlaceholderText, QColor(p["muted"]))
    app.setPalette(pal)
    app.setStyleSheet(build_style())


def repolish(w: QWidget):
    w.style().unpolish(w)
    w.style().polish(w)


# ===========================================================================
# Session log (feeds the "Recent assessments" list)
# ===========================================================================
class SessionLog(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.items: list[tuple[datetime, float, str, str]] = []

    def add(self, score: float, level: str, title: str):
        self.items.insert(0, (datetime.now(), score, level, title))
        del self.items[20:]
        self.changed.emit()


LOG = SessionLog()


# ===========================================================================
# Helpers
# ===========================================================================
class Worker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable, *args):
        super().__init__()
        self._fn, self._args = fn, args

    def run(self):
        try:
            self.done.emit(self._fn(*self._args))
        except ServiceError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Something went wrong: {exc}")


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            clear_layout(item.layout())


def make_button(text: str, kind: str = "primary") -> QPushButton:
    b = QPushButton(text)
    b.setObjectName(kind)
    b.setCursor(Qt.PointingHandCursor)
    return b


def label(text: str, name: str = "", wrap: bool = False) -> QLabel:
    l = QLabel(text)
    if name:
        l.setObjectName(name)
    l.setWordWrap(wrap)
    return l


def pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.1%}"


def scrollable(inner: QWidget) -> QScrollArea:
    inner.setObjectName("canvas")
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setWidget(inner)
    return area


# ===========================================================================
# Custom-drawn widgets
# ===========================================================================
class Surface(QFrame):
    """White (or charcoal) card; soft shadow in light mode."""

    def __init__(self, shadow: bool = True):
        super().__init__()
        self.setObjectName("card")
        self._shadow = shadow
        THEME.changed.connect(self._sync)
        self._sync()

    def _sync(self):
        if self._shadow and THEME.name == "light":
            fx = QGraphicsDropShadowEffect(self)
            fx.setBlurRadius(28)
            fx.setOffset(0, 4)
            fx.setColor(QColor(60, 40, 20, 26))
            self.setGraphicsEffect(fx)
        else:
            self.setGraphicsEffect(None)


class Badge(QWidget):
    def __init__(self, tone: str = "orange", size: int = 34):
        super().__init__()
        self._tone = tone
        self.setFixedSize(size, size)
        THEME.changed.connect(self.update)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = THEME.tone(self._tone)
        tint = QColor(c)
        tint.setAlphaF(0.18)
        p.setPen(Qt.NoPen)
        p.setBrush(tint)
        p.drawRoundedRect(self.rect(), self.width() * 0.32, self.width() * 0.32)
        p.setBrush(c)
        r = self.width() * 0.16
        p.drawEllipse(QPointF(self.rect().center()), r, r)


class Stat(Surface):
    def __init__(self, title: str, value: str = "-", caption: str = "",
                 tone: str = "orange"):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(2)
        top = QHBoxLayout()
        top.addWidget(label(title, "muted"), 1)
        top.addWidget(Badge(tone))
        lay.addLayout(top)
        self._v = label(value, "statValue")
        lay.addWidget(self._v)
        if caption:
            lay.addWidget(label(caption, "muted", True))

    def set_value(self, text: str):
        self._v.setText(text)


class ThemeToggle(QWidget):
    toggled = Signal(bool)  # True = dark

    def __init__(self, dark: bool):
        super().__init__()
        self.setFixedSize(58, 30)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Switch light / dark mode")
        self._dark = dark
        self._pos = 1.0 if dark else 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(180)
        self._anim.valueChanged.connect(self._tick)
        THEME.changed.connect(self.update)

    def _tick(self, v):
        self._pos = float(v)
        self.update()

    def mouseReleaseEvent(self, _):
        self._dark = not self._dark
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if self._dark else 0.0)
        self._anim.start()
        self.toggled.emit(self._dark)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setPen(QPen(THEME.color("border2"), 1))
        p.setBrush(THEME.color("surface2"))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)

        d = 22
        x = 4 + self._pos * (self.width() - 8 - d)
        knob = QRectF(x, (self.height() - d) / 2, d, d)
        p.setPen(Qt.NoPen)
        p.setBrush(THEME.color("orange"))
        p.drawEllipse(knob)

        c = knob.center()
        glyph = THEME.color("on_orange")
        if self._pos < 0.5:  # sun
            p.setBrush(glyph)
            p.drawEllipse(c, 3.2, 3.2)
            pen = QPen(glyph, 1.4)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            for i in range(8):
                p.save()
                p.translate(c)
                p.rotate(i * 45)
                p.drawLine(QPointF(0, -5.4), QPointF(0, -7.2))
                p.restore()
        else:  # moon
            moon = QPainterPath()
            moon.addEllipse(c, 5.0, 5.0)
            cut = QPainterPath()
            cut.addEllipse(c + QPointF(2.6, -1.9), 4.4, 4.4)
            p.setBrush(glyph)
            p.drawPath(moon.subtracted(cut))


class Combo(QComboBox):
    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(THEME.color("muted"), 1.8)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        x, y = self.width() - 20, self.height() / 2
        p.drawPolyline([QPointF(x - 4, y - 2), QPointF(x, y + 2), QPointF(x + 4, y - 2)])


class NumberEdit(QLineEdit):
    """Plain typed number: digits (and a decimal point where needed) only."""

    def __init__(self, spec: FieldSpec):
        super().__init__()
        self._spec = spec
        pattern = r"\d{0,12}" if spec.decimals == 0 else r"\d{0,12}([.]\d{0,%d})?" % spec.decimals
        self.setValidator(QRegularExpressionValidator(QRegularExpression(pattern)))
        self.setInputMethodHints(Qt.ImhFormattedNumbersOnly)
        self.reset()

    def reset(self):
        d = self._spec.default
        self.setText(str(int(d)) if float(d).is_integer() else f"{d:.2f}")

    def value(self):
        text = self.text().strip()
        if not text or text == ".":
            raise ValueError(f"Please enter a value for “{self._spec.label}”.")
        num = float(text)
        lo, hi = self._spec.minimum, self._spec.maximum
        if num < lo or (hi is not None and num > hi):
            raise ValueError(f"“{self._spec.label}” must be between {lo:g} and {hi:g}."
                             if hi is not None else
                             f"“{self._spec.label}” must be {lo:g} or more.")
        return int(num) if self._spec.decimals == 0 else num


class Gauge(QWidget):
    def __init__(self):
        super().__init__()
        self._value = 0.0
        self._tone = "orange"
        self.setMinimumSize(240, 240)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(800)
        self._anim.valueChanged.connect(self._tick)
        THEME.changed.connect(self.update)

    def _tick(self, v):
        self._value = float(v)
        self.update()

    def show_score(self, score: float, tone: str):
        self._tone = tone
        self._anim.stop()
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(float(score))
        self._anim.start()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        side = min(self.width(), self.height())
        m = 20
        rect = QRectF((self.width() - side) / 2 + m, (self.height() - side) / 2 + m,
                      side - 2 * m, side - 2 * m)
        pen = QPen(THEME.color("track"), 16)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, 225 * 16, -270 * 16)
        pen.setColor(THEME.tone(self._tone))
        p.setPen(pen)
        p.drawArc(rect, 225 * 16, int(-270 * 16 * min(max(self._value, 0), 1)))

        f = QFont(self.font())
        f.setPixelSize(44)
        f.setBold(True)
        p.setFont(f)
        p.setPen(THEME.color("text"))
        p.drawText(rect.adjusted(0, -10, 0, -10), Qt.AlignCenter, f"{self._value * 100:.0f}%")
        f.setPixelSize(13)
        f.setBold(False)
        p.setFont(f)
        p.setPen(THEME.color("muted"))
        p.drawText(rect.adjusted(0, 56, 0, 0), Qt.AlignHCenter | Qt.AlignVCenter, "risk score")


class ColumnChart(QWidget):
    def __init__(self):
        super().__init__()
        self._items: list[tuple[str, float, str]] = []
        self.setMinimumHeight(240)
        THEME.changed.connect(self.update)

    def set_items(self, items):
        self._items = items
        self.update()

    def paintEvent(self, _):
        if not self._items:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        f = QFont(self.font())
        f.setPixelSize(12)
        p.setFont(f)
        w, h = self.width(), self.height()
        left, top, bottom = 44, 26, 30
        plot = QRectF(left, top, w - left - 8, h - top - bottom)

        for frac in (0, 0.5, 1):
            y = plot.bottom() - plot.height() * frac
            p.setPen(QPen(THEME.color("track"), 1))
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            p.setPen(THEME.color("muted"))
            p.drawText(QRectF(0, y - 8, left - 10, 16),
                       Qt.AlignRight | Qt.AlignVCenter, f"{int(frac * 100)}%")

        n = len(self._items)
        slot = plot.width() / n
        bw = min(46.0, slot * 0.5)
        for i, (name, val, text) in enumerate(self._items):
            cx = plot.left() + slot * (i + 0.5)
            bh = plot.height() * max(0.0, min(val, 1.0))
            bar = QRectF(cx - bw / 2, plot.bottom() - max(bh, 3), bw, max(bh, 3))
            p.setPen(Qt.NoPen)
            p.setBrush(THEME.tone(TONES[i % 3]))
            p.drawRoundedRect(bar, 9, 9)
            bold = QFont(f)
            bold.setBold(True)
            p.setFont(bold)
            p.setPen(THEME.color("text"))
            p.drawText(QRectF(cx - slot / 2, bar.top() - 22, slot, 18), Qt.AlignCenter, text)
            p.setFont(f)
            p.setPen(THEME.color("muted"))
            p.drawText(QRectF(cx - slot / 2, plot.bottom() + 8, slot, 18), Qt.AlignCenter, name)


class HBarChart(QWidget):
    ROW = 36

    def __init__(self):
        super().__init__()
        self._items: list[tuple[str, float]] = []
        THEME.changed.connect(self.update)

    def set_items(self, items):
        self._items = items
        self.setMinimumHeight(max(1, len(items)) * self.ROW + 4)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        f = QFont(self.font())
        f.setPixelSize(13)
        p.setFont(f)
        fm = p.fontMetrics()
        label_w = int(self.width() * 0.38)
        for i, (name, val) in enumerate(self._items):
            y = i * self.ROW
            p.setPen(THEME.color("text"))
            p.drawText(QRectF(0, y, label_w, self.ROW), Qt.AlignVCenter | Qt.AlignLeft,
                       fm.elidedText(name, Qt.ElideRight, label_w - 8))
            track = QRectF(label_w + 6, y + self.ROW / 2 - 5, self.width() - label_w - 6, 10)
            p.setPen(Qt.NoPen)
            p.setBrush(THEME.color("track"))
            p.drawRoundedRect(track, 5, 5)
            fill = QRectF(track.left(), track.top(), max(10.0, track.width() * val), track.height())
            p.setBrush(THEME.tone("pink" if i == 0 else "orange"))
            p.drawRoundedRect(fill, 5, 5)


class Stepper(QWidget):
    def __init__(self, labels: list[str]):
        super().__init__()
        self._labels = labels
        self._index = 0
        self.setFixedHeight(74)
        THEME.changed.connect(self.update)

    def set_index(self, i: int):
        self._index = i
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        n = len(self._labels)
        w = self.width()
        margin = 56
        xs = [margin + i * (w - 2 * margin) / max(n - 1, 1) for i in range(n)]
        r, cy = 15, 18
        f = QFont(self.font())
        f.setPixelSize(13)

        for i in range(n - 1):
            p.setPen(QPen(THEME.color("orange") if i < self._index
                          else THEME.color("border2"), 2))
            p.drawLine(QPointF(xs[i] + r + 8, cy), QPointF(xs[i + 1] - r - 8, cy))

        for i, x in enumerate(xs):
            c = QPointF(x, cy)
            if i < self._index:
                p.setPen(Qt.NoPen)
                p.setBrush(THEME.color("orange"))
                p.drawEllipse(c, r, r)
                pen = QPen(THEME.color("on_orange"), 2)
                pen.setCapStyle(Qt.RoundCap)
                pen.setJoinStyle(Qt.RoundJoin)
                p.setPen(pen)
                p.drawPolyline([QPointF(x - 5, cy), QPointF(x - 1.5, cy + 4), QPointF(x + 5.5, cy - 4)])
            elif i == self._index:
                tint = THEME.color("orange")
                tint.setAlphaF(0.16)
                p.setPen(QPen(THEME.color("orange"), 2))
                p.setBrush(tint)
                p.drawEllipse(c, r, r)
                p.setFont(f)
                p.setPen(THEME.color("orange_text"))
                p.drawText(QRectF(x - r, cy - r, 2 * r, 2 * r), Qt.AlignCenter, str(i + 1))
            else:
                p.setPen(QPen(THEME.color("border2"), 2))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(c, r, r)
                p.setFont(f)
                p.setPen(THEME.color("muted"))
                p.drawText(QRectF(x - r, cy - r, 2 * r, 2 * r), Qt.AlignCenter, str(i + 1))

            lf = QFont(f)
            lf.setBold(i == self._index)
            p.setFont(lf)
            p.setPen(THEME.color("text") if i <= self._index else THEME.color("muted"))
            p.drawText(QRectF(x - 60, cy + r + 8, 120, 20), Qt.AlignCenter, self._labels[i])


def make_icon(name: str, off: QColor, on: QColor) -> QIcon:
    icon = QIcon()
    for color, state in ((off, QIcon.Off), (on, QIcon.On)):
        pm = QPixmap(44, 44)
        pm.setDevicePixelRatio(2)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(color, 1.7)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        if name == "home":
            pts = [(4, 9.5), (11, 4), (18, 9.5), (18, 18), (13.5, 18), (13.5, 12.5),
                   (8.5, 12.5), (8.5, 18), (4, 18)]
            p.drawPolygon(QPolygonF([QPointF(*t) for t in pts]))
        elif name == "claim":
            p.drawRoundedRect(QRectF(4.5, 3, 13, 16), 3, 3)
            p.drawLine(QPointF(11, 8), QPointF(11, 14))
            p.drawLine(QPointF(8, 11), QPointF(14, 11))
        elif name == "upload":
            p.drawLine(QPointF(11, 14.5), QPointF(11, 5))
            p.drawPolyline([QPointF(7.5, 8.5), QPointF(11, 5), QPointF(14.5, 8.5)])
            p.drawPolyline([QPointF(4, 14), QPointF(4, 18), QPointF(18, 18), QPointF(18, 14)])
        elif name == "chart":
            for x, y, h in ((4, 11, 7), (9.25, 6, 12), (14.5, 3, 15)):
                p.drawRoundedRect(QRectF(x, y, 3.5, h), 1.2, 1.2)
        p.end()
        icon.addPixmap(pm, QIcon.Normal, state)
    return icon


class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, limit: int = 1000):
        super().__init__()
        self._df = df.head(limit).reset_index(drop=True)

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        value = self._df.iat[index.row(), index.column()]
        col = self._df.columns[index.column()]
        if role == Qt.DisplayRole:
            return f"{value:,.2f}" if isinstance(value, float) else str(value)
        if role == Qt.ForegroundRole and col == "Priority":
            tone = {"High priority": "pink", "Review recommended": "orange",
                    "Lower priority": "green"}.get(str(value))
            if tone:
                return THEME.color(f"{tone}_text")
        if role == Qt.FontRole and col == "Priority":
            f = QFont()
            f.setBold(True)
            return f
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return None


# ===========================================================================
# Pages
# ===========================================================================
def card_with_heading(heading: str, sub: str = "", shadow: bool = True) -> tuple[Surface, QVBoxLayout]:
    s = Surface(shadow)
    lay = QVBoxLayout(s)
    lay.setContentsMargins(22, 20, 22, 20)
    lay.setSpacing(4)
    lay.addWidget(label(heading, "cardHeading"))
    if sub:
        lay.addWidget(label(sub, "muted", True))
    lay.addSpacing(10)
    return s, lay


class HomePage(QWidget):
    title = "Overview"
    subtitle = "A quick look at how the screening is performing."

    def __init__(self, service: ModelService, go_to: Callable[[int], None]):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setContentsMargins(0, 0, 18, 24)
        root.setSpacing(20)

        metrics = service.headline_metrics()

        kpis = QHBoxLayout()
        kpis.setSpacing(18)
        spec = [("Accuracy", "Claims classified correctly", "orange"),
                ("Precision", "Flags that were real fraud", "green"),
                ("Recall", "Fraud cases that were caught", "pink"),
                ("F1 score", "Balance of precision and recall", "orange")]
        for name, cap, tone in spec:
            v = "-"
            if metrics and name in metrics:
                v = f"{metrics[name]:.3f}" if name == "F1 score" else f"{metrics[name]:.1%}"
            kpis.addWidget(Stat(name, v, cap, tone))
        root.addLayout(kpis)

        row2 = QGridLayout()
        row2.setHorizontalSpacing(18)
        row2.setColumnStretch(0, 3)
        row2.setColumnStretch(1, 2)

        quality, ql = card_with_heading(
            "Screening quality", "How the final model scored on claims it had never seen.")
        chart = ColumnChart()
        if metrics:
            chart.set_items([
                (k, v, f"{v:.3f}" if k in ("F1 score", "ROC-AUC") else f"{v:.0%}")
                for k, v in metrics.items()])
            ql.addWidget(chart, 1)
        else:
            ql.addWidget(label("Quality figures aren't available yet.", "muted"))
            ql.addStretch()
        row2.addWidget(quality, 0, 0)

        start, sl = card_with_heading(
            "Get started", "Pick up where you left off, or begin something new.")
        b1 = make_button("Assess a claim")
        b1.clicked.connect(lambda: go_to(1))
        b2 = make_button("Upload many claims", "secondary")
        b2.clicked.connect(lambda: go_to(2))
        b3 = make_button("See the performance report", "secondary")
        b3.clicked.connect(lambda: go_to(3))
        for b in (b1, b2, b3):
            sl.addWidget(b)
        sl.addStretch()
        sl.addWidget(label("ClaimGuard supports human judgement. A flag is not proof of "
                           "fraud, and a low score does not guarantee a claim is genuine.",
                           "muted", True))
        row2.addWidget(start, 0, 1)
        root.addLayout(row2)

        row3 = QGridLayout()
        row3.setHorizontalSpacing(18)
        row3.setColumnStretch(0, 3)
        row3.setColumnStretch(1, 2)

        factors_card, fl = card_with_heading(
            "What influences the scores most", "Overall, across all claims.")
        self._factors = HBarChart()
        factors = service.top_factors(7)
        if factors:
            self._factors.set_items(factors)
            fl.addWidget(self._factors)
        else:
            fl.addWidget(label("Not available yet.", "muted"))
        fl.addStretch()
        row3.addWidget(factors_card, 0, 0)

        recent_card, rl = card_with_heading("Recent assessments", "From this session.")
        self._recent = QVBoxLayout()
        self._recent.setSpacing(12)
        rl.addLayout(self._recent)
        rl.addStretch()
        row3.addWidget(recent_card, 0, 1)
        root.addLayout(row3)
        root.addStretch()

        LOG.changed.connect(self._rebuild_recent)
        self._rebuild_recent()
        outer.addWidget(scrollable(inner))

    def _rebuild_recent(self):
        clear_layout(self._recent)
        if not LOG.items:
            self._recent.addWidget(label("No claims assessed yet.", "muted"))
            return
        for ts, score, level, title in LOG.items[:5]:
            row = QHBoxLayout()
            row.setSpacing(12)
            row.addWidget(Badge(LEVEL_TONE[level], 32))
            col = QVBoxLayout()
            col.setSpacing(0)
            col.addWidget(label("Claim assessed"))
            col.addWidget(label(ts.strftime("%H:%M"), "muted"))
            row.addLayout(col, 1)
            pill = label(f"{title}  ·  {score:.0%}", "pill")
            pill.setProperty("tone", LEVEL_TONE[level])
            row.addWidget(pill)
            holder = QWidget()
            holder.setLayout(row)
            self._recent.addWidget(holder)


class NewClaimPage(QWidget):
    title = "Assess a claim"
    subtitle = "Use details known when the claim was first submitted."

    def __init__(self, service: ModelService):
        super().__init__()
        self._service = service
        self._sections = service.form_sections()
        self._inputs: dict[str, QWidget] = {}
        self._specs: dict[str, FieldSpec] = {}
        self._last: Optional[tuple[dict, Assessment]] = None
        self._last_ai_brief = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 24)
        self._stack = QStackedWidget()
        root.addWidget(self._stack, 1)
        self._stack.addWidget(self._build_form())
        self._stack.addWidget(self._build_result())

    def _build_form(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        self._stepper = Stepper([t for t, _ in self._sections])
        lay.addWidget(self._stepper)

        self._pages = QStackedWidget()
        for i, (title, specs) in enumerate(self._sections, 1):
            self._pages.addWidget(self._build_step(i, title, specs))
        lay.addWidget(scrollable(self._pages), 1)

        nav = QHBoxLayout()
        reset = make_button("Start over", "secondary")
        reset.clicked.connect(self.reset)
        self._back = make_button("Back", "secondary")
        self._next = make_button("Next")
        self._back.clicked.connect(lambda: self._go(self._pages.currentIndex() - 1))
        self._next.clicked.connect(self._on_next)
        nav.addWidget(reset)
        nav.addStretch()
        nav.addWidget(self._back)
        nav.addWidget(self._next)
        lay.addLayout(nav)
        self._go(0)
        return w

    def _build_step(self, number: int, title: str, specs: list[FieldSpec]) -> QWidget:
        panel, lay = card_with_heading(f"{title} details",
                                       f"Step {number} of {len(self._sections)}", False)
        lay.setContentsMargins(28, 24, 28, 28)
        grid = QGridLayout()
        grid.setHorizontalSpacing(28)
        grid.setVerticalSpacing(18)
        for i, spec in enumerate(specs):
            cell = QVBoxLayout()
            cell.setSpacing(6)
            cell.addWidget(label(spec.label, "muted"))
            widget = self._make_input(spec)
            self._inputs[spec.name] = widget
            self._specs[spec.name] = spec
            cell.addWidget(widget)
            grid.addLayout(cell, i // 2, i % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)
        lay.addStretch()
        wrap = QWidget()
        wl = QVBoxLayout(wrap)
        wl.setContentsMargins(0, 0, 14, 0)
        wl.addWidget(panel)
        wl.addStretch()
        return wrap

    @staticmethod
    def _make_input(spec: FieldSpec) -> QWidget:
        if spec.kind == "number":
            return NumberEdit(spec)
        cb = Combo()
        cb.setView(QListView())
        cb.setMaxVisibleItems(10)
        cb.setFocusPolicy(Qt.StrongFocus)
        for text, value in zip(spec.option_labels, spec.options):
            cb.addItem(text, value)
        NewClaimPage._select_default(cb, spec)
        return cb

    @staticmethod
    def _select_default(cb: QComboBox, spec: FieldSpec):
        idx = cb.findData(spec.default)
        cb.setCurrentIndex(idx if idx >= 0 else 0)

    def _go(self, index: int):
        index = max(0, min(index, len(self._sections) - 1))
        self._pages.setCurrentIndex(index)
        self._stepper.set_index(index)
        self._back.setEnabled(index > 0)
        self._next.setText("Assess claim" if index == len(self._sections) - 1 else "Continue")

    def _validate_step(self, index: int) -> bool:
        for spec in self._sections[index][1]:
            w = self._inputs[spec.name]
            if isinstance(w, NumberEdit):
                try:
                    w.value()
                except ValueError as exc:
                    self._go(index)
                    QMessageBox.warning(self, "Check this field", str(exc))
                    w.setFocus()
                    w.selectAll()
                    return False
        return True

    def _on_next(self):
        i = self._pages.currentIndex()
        if i < len(self._sections) - 1:
            if self._validate_step(i):
                self._go(i + 1)
            return
        for n in range(len(self._sections)):
            if not self._validate_step(n):
                return
        self._assess()

    def _values(self) -> dict:
        out = {}
        for name, w in self._inputs.items():
            out[name] = w.value() if isinstance(w, NumberEdit) else w.currentData()
        return out

    def reset(self):
        for name, w in self._inputs.items():
            if isinstance(w, NumberEdit):
                w.reset()
            else:
                self._select_default(w, self._specs[name])
        self._go(0)
        self._stack.setCurrentIndex(0)

    def _assess(self):
        values = self._values()
        self._next.setEnabled(False)
        self._next.setText("Assessing...")
        self._worker = Worker(self._service.assess, values)
        self._worker.done.connect(lambda r: self._show_result(values, r))
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_fail(self, msg: str):
        self._next.setEnabled(True)
        self._next.setText("Assess claim")
        QMessageBox.warning(self, "Couldn't assess claim", msg)

    def _build_result(self) -> QWidget:
        panel = Surface(False)
        lay = QHBoxLayout(panel)
        lay.setContentsMargins(48, 36, 48, 36)
        lay.setSpacing(48)
        self._gauge = Gauge()
        lay.addWidget(self._gauge, 0, Qt.AlignVCenter)

        right = QVBoxLayout()
        right.setSpacing(10)
        right.addStretch()
        right.addWidget(label("Assessment result", "muted"))
        self._r_title = label("", "resultTitle")
        self._r_msg = label("", wrap=True)
        right.addWidget(self._r_title)
        right.addWidget(self._r_msg)
        right.addWidget(label("This is a prompt for human review, not a finding of fraud.",
                              "muted", True))
        right.addSpacing(10)
        self._ai_button = make_button("Generate AI investigation brief", "secondary")
        self._ai_button.clicked.connect(self._generate_ai_brief)
        right.addWidget(self._ai_button)
        right.addWidget(label(
            "AI support explains what to verify. The Random Forest keeps control of the score.",
            "muted", True))
        self._ai_summary = label(
            "Advisory explanation only • The model score and claim decision remain separate.",
            "muted", True)
        right.addWidget(self._ai_summary)
        self._ai_output = QTextEdit()
        self._ai_output.setReadOnly(True)
        self._ai_output.setPlaceholderText(
            "Your AI-generated investigation brief will appear here.")
        self._ai_output.setMinimumHeight(210)
        self._ai_output.setMaximumHeight(340)
        right.addWidget(self._ai_output)
        right.addSpacing(8)
        btns = QHBoxLayout()
        again = make_button("Assess another claim")
        again.clicked.connect(self.reset)
        edit = make_button("Edit details", "secondary")
        edit.clicked.connect(lambda: self._stack.setCurrentIndex(0))
        save = make_button("Save result", "secondary")
        save.clicked.connect(self._save)
        for b in (again, edit, save):
            btns.addWidget(b)
        btns.addStretch()
        right.addLayout(btns)
        right.addStretch()
        lay.addLayout(right, 1)

        wrap = QWidget()
        wl = QVBoxLayout(wrap)
        wl.setContentsMargins(0, 8, 0, 0)
        wl.addWidget(panel)
        wl.addStretch()
        return wrap

    def _show_result(self, values: dict, result: Assessment):
        self._last = (values, result)
        tone = LEVEL_TONE[result.level]
        self._r_title.setText(result.title)
        self._r_title.setProperty("tone", tone)
        repolish(self._r_title)
        self._r_msg.setText(result.message)
        self._next.setEnabled(True)
        self._next.setText("Assess claim")
        self._stack.setCurrentIndex(1)
        self._gauge.show_score(result.score, tone)
        self._ai_output.clear()
        self._last_ai_brief = ""
        self._ai_button.setEnabled(True)
        self._ai_button.setText("Generate AI investigation brief")
        LOG.add(result.score, result.level, result.title)

    def _generate_ai_brief(self):
        if not self._last:
            return
        values, assessment = self._last
        self._ai_button.setEnabled(False)
        self._ai_button.setText("Generating brief...")
        self._ai_output.setPlainText(
            "Contacting the configured AI provider. The ML score remains unchanged.")
        self._worker = Worker(generate_investigation_brief, values, assessment)
        self._worker.done.connect(self._show_ai_brief)
        self._worker.failed.connect(self._ai_failed)
        self._worker.start()

    def _show_ai_brief(self, brief: str):
        self._last_ai_brief = brief
        # Keep the model score in the gauge; the AI panel is for plain-language
        # explanation only, not a second score or a competing decision.
        self._ai_summary.setText(
            "AI-generated explanation • Advisory only • Verify original records"
        )
        self._ai_output.setMarkdown(brief)
        self._ai_button.setEnabled(True)
        self._ai_button.setText("Regenerate AI explanation")

    def _ai_failed(self, message: str):
        self._ai_output.setPlainText(
            "AI assistance is unavailable. Your original ML assessment is still valid.\n\n"
            + message
        )
        self._ai_button.setEnabled(True)
        self._ai_button.setText("Retry AI investigation brief")

    def _save(self):
        if not self._last:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save result", "claim_assessment.csv", "CSV files (*.csv)")
        if path:
            export = self._service.export_frame(*self._last)
            export["AI investigation brief"] = self._last_ai_brief
            export.to_csv(path, index=False)


class DropZone(QFrame):
    fileDropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("dropzone")
        self.setAcceptDrops(True)

    def _hover(self, on: bool):
        self.setProperty("hover", on)
        repolish(self)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._hover(True)

    def dragLeaveEvent(self, e):
        self._hover(False)

    def dropEvent(self, e):
        self._hover(False)
        urls = e.mimeData().urls()
        if urls:
            self.fileDropped.emit(urls[0].toLocalFile())


class BatchPage(QWidget):
    title = "Batch upload"
    subtitle = "Rank many claims from highest to lowest risk."

    def __init__(self, service: ModelService):
        super().__init__()
        self._service = service
        self._data: Optional[pd.DataFrame] = None
        self._results: Optional[pd.DataFrame] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 24)
        root.setSpacing(18)

        self._drop = DropZone()
        self._drop.fileDropped.connect(self._load)
        dl = QVBoxLayout(self._drop)
        dl.setContentsMargins(24, 26, 24, 26)
        dl.setSpacing(6)
        dl.setAlignment(Qt.AlignCenter)
        h = label("Upload a claims file", "cardHeading")
        h.setAlignment(Qt.AlignCenter)
        self._file_lbl = label("Drag a CSV here, or choose one from your computer.", "muted")
        self._file_lbl.setAlignment(Qt.AlignCenter)
        dl.addWidget(h)
        dl.addWidget(self._file_lbl)
        dl.addSpacing(10)
        row = QHBoxLayout()
        row.setAlignment(Qt.AlignCenter)
        self._choose = make_button("Choose file")
        self._choose.clicked.connect(self._pick)
        tmpl = make_button("Download template", "secondary")
        tmpl.clicked.connect(self._template)
        self._run = make_button("Assess all claims", "secondary")
        self._run.setEnabled(False)
        self._run.clicked.connect(self._assess)
        for b in (self._choose, tmpl, self._run):
            row.addWidget(b)
        dl.addLayout(row)
        root.addWidget(self._drop)

        cards = QHBoxLayout()
        cards.setSpacing(18)
        self._c_total = Stat("Claims assessed", tone="orange")
        self._c_flag = Stat("Flagged for review", tone="pink")
        self._c_share = Stat("Share flagged", tone="green")
        for c in (self._c_total, self._c_flag, self._c_share):
            cards.addWidget(c)
        self._cards = QWidget()
        self._cards.setLayout(cards)
        self._cards.hide()
        root.addWidget(self._cards)

        self._table_card = Surface(False)
        tl = QVBoxLayout(self._table_card)
        tl.setContentsMargins(8, 8, 8, 8)
        self._table = QTableView()
        self._table.setShowGrid(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(44)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        tl.addWidget(self._table)
        self._table_card.hide()
        root.addWidget(self._table_card, 1)
        THEME.changed.connect(self._table.viewport().update)

        # AI guidance is generated only for the single row selected by the
        # user. This avoids sending an entire uploaded claims file to the API.
        self._batch_ai_card = Surface(False)
        ai_lay = QVBoxLayout(self._batch_ai_card)
        ai_lay.setContentsMargins(18, 16, 18, 16)
        ai_lay.setSpacing(10)
        ai_head = QHBoxLayout()
        ai_title = label("AI claim explanation", "cardHeading")
        ai_head.addWidget(ai_title)
        ai_head.addStretch()
        self._batch_ai_button = make_button(
            "Explain selected claim with AI", "secondary"
        )
        self._batch_ai_button.setEnabled(False)
        self._batch_ai_button.clicked.connect(self._explain_selected_claim)
        ai_head.addWidget(self._batch_ai_button)
        ai_lay.addLayout(ai_head)
        self._batch_ai_note = label(
            "Select one claim in the results table to generate a plain-language explanation. "
            "Only that selected claim is sent for AI guidance.",
            "muted", True
        )
        ai_lay.addWidget(self._batch_ai_note)
        self._batch_ai_output = QTextEdit()
        self._batch_ai_output.setReadOnly(True)
        self._batch_ai_output.setPlaceholderText(
            "The selected claim's AI explanation will appear here."
        )
        self._batch_ai_output.setMinimumHeight(130)
        self._batch_ai_output.setMaximumHeight(240)
        ai_lay.addWidget(self._batch_ai_output)
        self._batch_ai_card.hide()
        root.addWidget(self._batch_ai_card)

        foot = QHBoxLayout()
        self._info = label("", "muted")
        self._export = make_button("Download ranked results", "secondary")
        self._export.clicked.connect(self._save)
        self._export.hide()
        foot.addWidget(self._info, 1)
        foot.addWidget(self._export)
        root.addLayout(foot)
        root.addStretch(0)

    def _template(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save template", "claimguard_template.csv", "CSV files (*.csv)")
        if path:
            self._service.template_dataframe().to_csv(path, index=False)

    def _pick(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose claims file", "", "CSV files (*.csv)")
        if path:
            self._load(path)

    def _load(self, path: str):
        try:
            self._data = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Couldn't open file", str(exc))
            return
        name = path.replace("\\", "/").split("/")[-1]
        self._file_lbl.setText(f"{name}  ·  {len(self._data):,} claims ready")
        self._run.setEnabled(True)
        self._run.setObjectName("primary")
        repolish(self._run)
        self._results = None
        self._cards.hide()
        self._table_card.hide()
        self._export.hide()
        self._info.setText("")

    def _assess(self):
        self._run.setEnabled(False)
        self._run.setText("Assessing...")
        self._worker = Worker(self._service.assess_batch, self._data)
        self._worker.done.connect(self._show)
        self._worker.failed.connect(self._fail)
        self._worker.start()

    def _fail(self, msg: str):
        self._run.setEnabled(True)
        self._run.setText("Assess all claims")
        QMessageBox.warning(self, "Couldn't assess file", msg)

    def _show(self, results: pd.DataFrame):
        self._results = results
        self._run.setEnabled(True)
        self._run.setText("Assess all claims")
        flagged = int((results["Flagged for review"] == "Yes").sum())
        total = len(results)
        self._c_total.set_value(f"{total:,}")
        self._c_flag.set_value(f"{flagged:,}")
        self._c_share.set_value(f"{flagged / total:.1%}" if total else "0%")
        self._cards.show()
        self._table.setModel(PandasModel(results))
        self._table.resizeColumnsToContents()
        self._table_card.show()
        self._batch_ai_output.clear()
        self._batch_ai_note.setText(
            "Select one claim in the results table, then choose Explain selected claim with AI. "
            "Only the selected row is sent to the AI provider."
        )
        self._batch_ai_button.setEnabled(True)
        self._batch_ai_button.setText("Explain selected claim with AI")
        self._batch_ai_card.show()
        self._export.show()
        self._info.setText("Showing the first 1,000 rows. The download includes every row."
                           if total > 1000 else "")

    def _explain_selected_claim(self):
        if self._results is None:
            return
        selected = self._table.selectionModel().selectedRows()
        if not selected:
            QMessageBox.information(
                self, "Select a claim",
                "Select one row in the results table before requesting an AI explanation."
            )
            return

        row_index = selected[0].row()
        record = self._results.iloc[row_index]
        # Recover only the model input features; exclude result columns.
        feature_names = self._service._features
        values = {name: record[name] for name in feature_names}
        score = float(record["Risk score (%)"]) / 100.0
        level, title, message = self._service._level(score)
        assessment = Assessment(
            score=score,
            flagged=str(record["Flagged for review"]).strip().lower() == "yes",
            level=level,
            title=str(record["Priority"]),
            message=message,
        )

        self._batch_ai_button.setEnabled(False)
        self._batch_ai_button.setText("Generating explanation...")
        self._batch_ai_output.setPlainText(
            "Preparing an explanation for the selected claim. "
            "The batch model score and ranking will not be changed."
        )
        self._worker = Worker(generate_investigation_brief, values, assessment)
        self._worker.done.connect(self._show_selected_claim_ai)
        self._worker.failed.connect(self._selected_claim_ai_failed)
        self._worker.start()

    def _show_selected_claim_ai(self, brief: str):
        self._batch_ai_output.setMarkdown(brief)
        self._batch_ai_button.setEnabled(True)
        self._batch_ai_button.setText("Regenerate selected claim explanation")
        self._batch_ai_note.setText(
            "AI guidance for the selected claim only. It does not alter the model score, "
            "priority, batch ranking, or claim decision."
        )

    def _selected_claim_ai_failed(self, message: str):
        self._batch_ai_output.setPlainText(
            "AI assistance is unavailable. The original batch assessment is unchanged.\n\n"
            + message
        )
        self._batch_ai_button.setEnabled(True)
        self._batch_ai_button.setText("Retry selected claim explanation")

    def _save(self):
        if self._results is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save results", "claimguard_results.csv", "CSV files (*.csv)")
        if path:
            self._results.to_csv(path, index=False)


class PerformancePage(QWidget):
    title = "Performance report"
    subtitle = "How well ClaimGuard separates genuine and fraudulent claims."

    def __init__(self, service: ModelService):
        super().__init__()
        self._service = service
        self._auto_ran = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setContentsMargins(0, 0, 18, 24)
        root.setSpacing(18)

        bar = QHBoxLayout()
        self._run_default = make_button("Run on test claims")
        self._run_default.setVisible(service.has_default_evaluation_data())
        self._run_default.clicked.connect(self._use_default)
        pick = make_button("Choose a labelled file", "secondary")
        pick.clicked.connect(self._pick)
        self._status = label("", "muted")
        bar.addWidget(self._run_default)
        bar.addWidget(pick)
        bar.addSpacing(8)
        bar.addWidget(self._status, 1)
        root.addLayout(bar)

        self._empty = label("Run the report to see precision, recall and more.", "muted")
        self._empty.setAlignment(Qt.AlignCenter)
        self._empty.setMinimumHeight(180)
        root.addWidget(self._empty)

        self._results = QWidget()
        rl = QVBoxLayout(self._results)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(18)

        cards = QHBoxLayout()
        cards.setSpacing(18)
        spec = [("Accuracy", "orange"), ("Precision", "green"), ("Recall", "pink"),
                ("F1 score", "orange"), ("ROC-AUC", "green")]
        self._cards = {k: Stat(k, tone=t) for k, t in spec}
        for c in self._cards.values():
            cards.addWidget(c)
        rl.addLayout(cards)

        table_card, tl = card_with_heading("Classification report",
                                           "Fraud is the class that matters most.", False)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["", "Precision", "Recall", "F1-score", "Claims"])
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.setShowGrid(False)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(44)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        tl.addWidget(self._table)
        rl.addWidget(table_card)

        lower = QGridLayout()
        lower.setHorizontalSpacing(18)
        lower.setColumnStretch(0, 1)
        lower.setColumnStretch(1, 1)

        mcard, ml = card_with_heading("What the screening got right and wrong")
        self._matrix = QGridLayout()
        self._matrix.setSpacing(12)
        ml.addLayout(self._matrix)
        ml.addStretch()
        lower.addWidget(mcard, 0, 0)

        self._fcard, fl = card_with_heading("What influences the scores most")
        self._factors = HBarChart()
        fl.addWidget(self._factors)
        fl.addStretch()
        lower.addWidget(self._fcard, 0, 1)
        rl.addLayout(lower)

        self._results.hide()
        root.addWidget(self._results)
        root.addStretch()
        outer.addWidget(scrollable(inner))

    def showEvent(self, e):
        super().showEvent(e)
        if not self._auto_ran and self._service.has_default_evaluation_data():
            self._auto_ran = True
            self._use_default()

    def _use_default(self):
        try:
            data = self._service.load_default_evaluation_data()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Couldn't load test claims", str(exc))
            return
        self._run(data)

    def _pick(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a labelled file", "", "CSV files (*.csv)")
        if not path:
            return
        try:
            data = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Couldn't open file", str(exc))
            return
        self._run(data)

    def _run(self, data: pd.DataFrame):
        column = self._service.detect_label_column(data)
        if column is None:
            choices = self._service.label_column_choices(data)
            if not choices:
                QMessageBox.warning(self, "No outcome found",
                                    "This file has no column with the true outcome.")
                return
            column, ok = QInputDialog.getItem(
                self, "Which column is the real outcome?",
                "Column that marks a claim as fraud or not:", choices, 0, False)
            if not ok:
                return
        self._status.setText("Running the report...")
        self._worker = Worker(self._service.classification_report, data, column)
        self._worker.done.connect(self._show)
        self._worker.failed.connect(self._fail)
        self._worker.start()

    def _fail(self, msg: str):
        self._status.setText("")
        QMessageBox.warning(self, "Couldn't build report", msg)

    def _show(self, r: ClassificationSummary):
        self._status.setText(f"Based on {r.total:,} claims, {r.fraud_total:,} of them fraudulent.")
        self._cards["Accuracy"].set_value(pct(r.accuracy))
        self._cards["Precision"].set_value(pct(r.precision))
        self._cards["Recall"].set_value(pct(r.recall))
        self._cards["F1 score"].set_value(f"{r.f1:.3f}")
        self._cards["ROC-AUC"].set_value("-" if r.roc_auc is None else f"{r.roc_auc:.3f}")

        self._table.setRowCount(len(r.rows))
        for i, row in enumerate(r.rows):
            vals = [row.name,
                    "" if row.precision is None else f"{row.precision:.3f}",
                    "" if row.recall is None else f"{row.recall:.3f}",
                    "" if row.f1 is None else f"{row.f1:.3f}",
                    f"{row.support:,}"]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                if j == 0:
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                self._table.setItem(i, j, item)
        self._table.setFixedHeight(self._table.horizontalHeader().height()
                                   + 44 * len(r.rows) + 8)

        self._fill_matrix(r.confusion)
        factors = self._service.top_factors(7)
        self._fcard.setVisible(bool(factors))
        self._factors.set_items(factors)
        self._empty.hide()
        self._results.show()

    def _fill_matrix(self, cm):
        clear_layout(self._matrix)
        (tn, fp), (fn, tp) = cm
        cells = [("Genuine claims cleared", tn, "green"),
                 ("Genuine claims flagged", fp, "orange"),
                 ("Fraud missed", fn, "pink"),
                 ("Fraud caught", tp, "green")]
        for i, (text, n, tone) in enumerate(cells):
            tile = QFrame()
            tile.setObjectName("tile")
            tile.setProperty("tone", tone)
            bl = QVBoxLayout(tile)
            bl.setContentsMargins(18, 14, 18, 14)
            bl.setSpacing(2)
            bl.addWidget(label(text, "muted"))
            v = label(f"{n:,}", "tileValue")
            v.setProperty("tone", tone)
            bl.addWidget(v)
            self._matrix.addWidget(tile, i // 2, i % 2)


# ===========================================================================
# Shell
# ===========================================================================
class MainWindow(QMainWindow):
    NAV = [("Overview", "home"), ("Assess a claim", "claim"),
           ("Batch upload", "upload"), ("Performance", "chart")]

    def __init__(self, service: ModelService):
        super().__init__()
        self.setWindowTitle("ClaimGuard")
        self.resize(1320, 840)
        self.setMinimumSize(1080, 700)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- sidebar ----
        side = QWidget()
        side.setObjectName("sidebar")
        side.setAttribute(Qt.WA_StyledBackground, True)
        side.setFixedWidth(248)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(18, 26, 18, 22)
        sl.setSpacing(4)

        brand = QHBoxLayout()
        logo = label("C", "logo")
        logo.setFixedSize(36, 36)
        logo.setAlignment(Qt.AlignCenter)
        brand.addWidget(logo)
        brand.addSpacing(6)
        brand.addWidget(label("ClaimGuard", "brand"), 1)
        sl.addLayout(brand)
        sl.addSpacing(28)
        sl.addWidget(label("MENU", "navSection"))
        sl.addSpacing(4)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._nav_buttons: list[tuple[QPushButton, str]] = []
        for i, (name, icon) in enumerate(self.NAV):
            b = QPushButton("  " + name)
            b.setObjectName("nav")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setIconSize(QSize(20, 20))
            self._group.addButton(b, i)
            self._nav_buttons.append((b, icon))
            sl.addWidget(b)
        sl.addStretch()
        sl.addWidget(label("A flag is a prompt for human review, not a finding of fraud.",
                           "sideNote", True))
        self._group.idClicked.connect(self.go_to)

        # ---- content ----
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(36, 28, 18, 0)
        cl.setSpacing(22)

        top = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self._title = label("", "pageTitle")
        self._subtitle = label("", "muted")
        titles.addWidget(self._title)
        titles.addWidget(self._subtitle)
        top.addLayout(titles, 1)
        top.addWidget(label("Dark mode", "muted"))
        top.addSpacing(8)
        self._toggle = ThemeToggle(THEME.name == "dark")
        self._toggle.toggled.connect(lambda d: THEME.set_name("dark" if d else "light"))
        top.addWidget(self._toggle)
        top.addSpacing(18)
        cl.addLayout(top)

        self._stack = QStackedWidget()
        self._pages = [HomePage(service, self.go_to), NewClaimPage(service),
                       BatchPage(service), PerformancePage(service)]
        for pg in self._pages:
            self._stack.addWidget(pg)
        cl.addWidget(self._stack, 1)

        outer.addWidget(side)
        outer.addWidget(content, 1)

        THEME.changed.connect(self._refresh_icons)
        self._refresh_icons()
        self.go_to(0)

    def _refresh_icons(self):
        off, on = THEME.color("muted"), THEME.color("orange_text")
        for b, icon in self._nav_buttons:
            b.setIcon(make_icon(icon, off, on))

    def go_to(self, index: int):
        self._group.button(index).setChecked(True)
        self._stack.setCurrentIndex(index)
        page = self._pages[index]
        self._title.setText(page.title)
        self._subtitle.setText(page.subtitle)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    base = QFont()
    base.setFamilies(["Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue", "Arial"])
    base.setPointSize(10)
    app.setFont(base)
    apply_theme(app)
    THEME.changed.connect(lambda: apply_theme(app))

    try:
        service = ModelService()
    except Exception as exc:  # noqa: BLE001
        box = QMessageBox()
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("ClaimGuard")
        box.setText("ClaimGuard couldn't start.")
        box.setInformativeText("The screening service isn't available right now.")
        box.setDetailedText(str(exc))
        box.exec()
        sys.exit(1)

    win = MainWindow(service)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
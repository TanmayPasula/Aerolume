"""Colours, type, the global style sheet and icons for the desktop app (UI v4).

The visual language is specified in ``docs/ui_guidelines.md`` (binding). v4
("a little more like the website", 0.1.0) carries the website's language over
to the app without changing any layout, workflow or data colour: the site's
monochrome ink-on-paper palette (light) and near-black (dark) mapped onto the
app's semantic tokens (:data:`SITE_LIGHT` / :data:`SITE_DARK` record the site
values each token came from), Atkinson Hyperlegible Next / Mono at 9 pt, 6 px
radii on controls and 8 px on framed blocks, 1 px hairlines, the site's two
buttons (primary filled ink, secondary outline) and its quiet ink focus ring.
24 px controls, no shadows and no gradients, as before. Every colour in the
application comes from the two token tables below. Pages never hard-code a
colour: they set one of the style-hook properties listed here and the global
style sheet does the rest, or they call :func:`tokens` when they paint by hand
(plots, canvases, delegates). That is what lets a theme switch at runtime
restyle the whole window without any page knowing about it.

Tokens (``tokens()`` keys)
==========================

v3 is a *semantic* set (DESIGN.md section 13): each
name says what the colour is for, and :data:`CONTRAST_RULES` lists the WCAG
2.x contrast every pairing must reach in both themes (``tests/test_tokens.py``
enforces it). The v1/v2 names stay as aliases, so code written against them
keeps working unchanged.

=================  ======================  ==============================================
v3 token           v1/v2 alias             use
=================  ======================  ==============================================
``bg``             ``bg2``                 window chrome: toolbars, status bar, dock rails
``bg_subtle``                              dock panel headers, title rows
``surface``        ``input``, ``plot_bg``  inputs, tables, plot and viewport wells
``surface_2``      ``panel``               side panels, inspectors
``fg``             ``text``                primary text
``fg2``            ``text2``               secondary text: property labels, captions
``muted``                                  hints and units (still >= 4.5:1: it is text)
``faint``                                  disabled text and icons (WCAG-exempt, kept >= 2.2:1)
``line``           ``border``              decorative 1 px separators (never a control's only edge)
``line_strong``    ``border_strong``       stronger separators, table grid
``control``                                outline of inputs, check boxes, sliders (>= 3:1)
``control_hover``                          the same, hovered
``primary``        ``accent_fill``         primary button and checked-box fill
``primary_hover``  ``accent_hover``
``primary_pressed````accent_pressed``
``primary_fg``     ``on_accent``           text and icons on ``primary``
``primary_soft``   ``accent_soft``         selected rows and items (``selection``)
``accent``                                 blue marks in the data views: the selected item
                                           of a canvas, the seed rake, the running state
                                           (v3: also links, tabs and checked icons; v4
                                           draws those chrome states in ink)
``accent2``                                a step further from the surface: built-in names
                                           in the Script console
``highlight``                            amber, non-text marks only: live state, heat,
                                           emitters (``highlight_soft`` behind them)
``success``        ``good``                converged / done text and glyphs
``warn``                                   warnings
``danger``         ``bad``                 errors
``*_fg``                                   text on a solid status fill (danger buttons)
``*_soft``         ``good_soft`` ...       opaque soft background a status text sits on
``focus``                                  keyboard focus ring / border
``hover``          ``panel2``              hovered rows and buttons
``pressed``
``series``                                 six plot colours (lighter steps in dark)
``grid``                                   plot grid lines
``field_*``                                field-view chrome: ``field_bg``, ``field_grid``,
                                           ``field_text``, ``field_solid``, ``field_solid_line``
``scrim``                                  the welcome tour's dimming layer (with alpha)
=================  ======================  ==============================================

v4 moved the chrome's interactive states to ink (``primary``, ``focus``, the
checked box, the current tab and step, links, checked icons through
:data:`ACTIVE_ICON_TOKEN`), as on the website. ``accent`` stays blue because the
data views use it to mark the selected or live item against ink outlines, and an
ink mark there would be lost; the second emphasis colour is
``highlight``. In the dark theme the primary fill is the site's light ink with
dark text, so a filled button inverts with the theme as the site's does.

Style hooks
===========

Set them with ``w.setProperty(name, "value")`` (strings!) or :func:`set_prop`
(which also re-polishes a widget that is already shown).

======================================================  =====================================
``QPushButton[variant="primary"]``                      the one main action of a view (primary fill)
``QPushButton[variant="ghost"]``                        low-emphasis, borderless until hovered
``QPushButton[variant="danger"]``                       destructive (red text, red fill on hover)
``QPushButton[variant="link"]`` / ``QToolButton[..]``   text-only accent action (start page lists)
``QToolButton[variant="icon"]``                         square icon-only tool button
``QToolButton[variant="toggle"]``                       a flat filter toggle (text when on, muted when off)
``QToolButton[variant="layer"]``                        a viewport layer toggle (on = soft fill + accent line)
``QToolButton[dropdown="true"]``                        a button that opens a menu: small chevron at the right
``QToolButton[role="section"]``                         collapsible group header (PropertyGrid)
``QFrame[card="true"]``                                 flat panel: panel fill, 1 px border, 8 px radius
``QFrame[card="true"][interactive="true"]``             clickable panel (hover border)
``QFrame[panel="true"]``                                inspector container: panel fill, no radius
``QFrame[surface="true"]``                              viewport / table well: surface fill + border
``QFrame[chrome="toolbar"]``                            a page-level tool strip (panel fill, bottom line)
``QFrame[chrome="page"|"dock"]``                        the shell's page area / dock content frames
``QFrame[chrome="dock-header"]``                        a dock panel's header row (widgets.docks)
``QFrame[divider="true"|"vertical"]``                   1 px separator line
``QFrame[notice="info"|"good"|"warn"|"bad"]``           an inline notice on its soft status background
``QFrame[callout="true"]``                              the welcome tour's step box
``QAbstractItemView[flat="true"]``                      tree/list/table without its own border
``QAbstractItemView[pane="true"]``                      borderless list on the panel colour (dock lists)
``QScrollArea[scrollstop="true"]``                      a frameless scroll area kept as a Tab stop (focus ring)
``QLabel[role="title"]``                                14 pt: the product name on Start, splash, tour
``QLabel[role="h1"|"h2"|"h3"]``                         11 / 10 / 9 pt semibold titles
``QLabel[role="secondary"]``                            fg2 (property labels, captions)
``QLabel[role="muted"]`` / ``QLabel[role="small"]``     hints and units / 8 pt footnotes
``QLabel[role="mono"]``                                 monospace numbers (data tables, logs)
``*[mono="true"|"small"]``                              any widget in the mono face (9 / 8 pt): :func:`use_mono`
``QLabel[role="kpi-value"|"kpi-label"]``                a value (11 pt, normal weight) and its caption
``QLabel[role="kpi-delta-good"|"kpi-delta-bad"]``       a delta in good / bad colour
``QLabel[role="badge"|"badge-ok"|"badge-warn"|...]``    status as small plain text in a status colour
                                                        (``badge-bad``, ``badge-accent`` too)
``QLabel[role="status-good"|"status-warn"|"status-bad"]``  same, the preferred spelling
``QLabel[role="hero"|"eyebrow"]``                       legacy: rendered as h1 / secondary text
``QLabel[role="pg-label"]``                             PropertyGrid label column (fg2)
``QLineEdit[invalid="true"]`` (also spin boxes)          validation error border
``QFrame[segmented="true"]``                            widgets.common.SegmentedControl
======================================================  =====================================

Status colours (success / warn / danger) are for text and small glyphs, or
text on their own ``*_soft`` background (an inline notice); never a solid fill
behind body text. :func:`status_colour` maps a simulation status to its token.

Type
====

Atkinson Hyperlegible Next for the interface and Atkinson Hyperlegible Mono
for numbers in tables and the log (SIL OFL, bundled in ``resources/fonts`` as
TrueType: Qt cannot read the WOFF2 files they are published in, see
``resources/fonts/README.md``). The Braille Institute drew them for low-vision
readers: 0 / O, 1 / l / I and 5 / S never look alike, which matters in a tool
whose output is numbers. Greek letters, superscripts and arrows fall back to
Segoe UI glyph by glyph (the Latin subsets have none). ``font="system"`` in
:func:`apply` (Preferences, Interface font) uses Segoe UI throughout.

``changed`` is a bound Qt signal ``(name: str)`` that fires after a theme
switch, so custom-painted widgets can refresh: ``theme.changed.connect(...)``.
Icons made with :func:`icon` follow the theme by themselves.
"""
from __future__ import annotations

import os
import sys
import tempfile
from functools import cache, lru_cache
from pathlib import Path
from string import Template

from PySide6.QtCore import QByteArray, QObject, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QGuiApplication,
    QIcon,
    QIconEngine,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QLabel, QWidget

RESOURCES = Path(__file__).resolve().parent / "resources"
ICON_DIR = RESOURCES / "icons"
FONT_DIR = RESOURCES / "fonts"
APP_ICON_PATH = RESOURCES / "aerolume.svg"

THEMES = ("light", "dark")
#: what a user can choose: a theme, or "system" (follow the operating system's light / dark setting)
THEME_CHOICES = ("light", "dark", "system")
DEFAULT_THEME = "light"
#: interface font choices: the bundled Atkinson Hyperlegible faces, or the platform font
FONT_CHOICES = ("bundled", "system")

# ---------------------------------------------------------------------------
# token tables
# ---------------------------------------------------------------------------

#: plot series on the light theme (and in reports, which are always light)
SERIES = ["#2463c4", "#d9730d", "#2f855a", "#b8322b", "#6b46c1", "#5f6b7a"]
#: the same hues one step lighter: on the graphite surface the light-theme blue was 2.6:1
SERIES_DARK = ["#5e9df4", "#f0943f", "#4fb47b", "#f0716a", "#a68af2", "#a3acb9"]

#: The website's colour tokens (website/assets/site.css ``:root`` and its dark block), which the
#: app's v4 tables below are built from: ink on paper, hairlines, one filled-ink button. Kept
#: here so the mapping is explicit and ``tests/test_tokens.py`` can check that the app still
#: speaks the site's palette (same paper, ink, hairline and control-border values).
SITE_LIGHT: dict = {"bg": "#f6f6f6", "surface": "#ffffff", "code_bg": "#e9e9e9", "fg": "#0e0e0e",
                    "muted": "#474747", "faint": "#5c5c5c", "line": "#cdcdcd", "line_strong": "#858585",
                    "btn_bg": "#0e0e0e", "btn_fg": "#f6f6f6", "focus": "#0e0e0e"}
SITE_DARK: dict = {"bg": "#000000", "surface": "#0f0f0f", "code_bg": "#171717", "fg": "#f1f1f1",
                   "muted": "#b1b1b1", "faint": "#939393", "line": "#2e2e2e", "line_strong": "#6e6e6e",
                   "btn_bg": "#f1f1f1", "btn_fg": "#000000", "focus": "#f1f1f1"}

#: v4 semantic tokens, light theme (contrast measured by tests/test_tokens.py).
#: The website's paper and ink (v4, "a little more like the website"): the side panels are the
#: site's page colour, the chrome one step darker, panel headers the site's code-block grey,
#: wells (inputs, tables, plots, the 3D view) the site's white frames. Text is the site's ink
#: and its two greys; control outlines are the site's control border. The chrome is neutral grey
#: with no blue cast; the primary button, checked boxes and the focus ring are ink, as on the
#: site. ``accent`` stays a blue, but only for marks in the data views (the selected item in a
#: canvas, the seed rake, the running state), where it has to stand apart from ink outlines.
SEMANTIC_LIGHT: dict = {
    "bg": "#efefef", "bg_subtle": "#e9e9e9", "surface": "#ffffff", "surface_2": "#f6f6f6",
    "fg": "#0e0e0e", "fg2": "#474747", "muted": "#5c5c5c", "faint": "#8e8e8e",
    "line": "#d6d6d6", "line_strong": "#b8b8b8", "control": "#858585", "control_hover": "#474747",
    "primary": "#0e0e0e", "primary_hover": "#2e2e2e", "primary_pressed": "#000000", "primary_fg": "#ffffff",
    "primary_soft": "#e2e2e2", "accent": "#2463c4", "accent2": "#1b4f9e",
    "highlight": "#b87508", "highlight_soft": "#fbefd7",
    "success": "#17703c", "success_fg": "#ffffff", "success_soft": "#e3f1e8",
    "warn": "#8f5a0b", "warn_fg": "#ffffff", "warn_soft": "#fbf0db",
    "danger": "#b02d27", "danger_fg": "#ffffff", "danger_soft": "#fae4e2",
    "focus": "#0e0e0e", "hover": "#ebebeb", "pressed": "#dcdcdc", "grid": "#e6e6e6",
    "field_bg": "#ffffff", "field_grid": "#e6e6e6", "field_text": "#474747",
    "field_solid": "#8f8f8f", "field_solid_line": "#262626", "scrim": "#0e0e0e",
    "series": SERIES,
}

#: v4 semantic tokens, dark theme: the website's near-black instead of v3's blue-grey graphite.
#: Chrome nearly black, side panels a step up, wells the site's code-block grey (#171717 lifted a
#: notch so inputs read as wells); light ink, a light filled button with dark text, light focus.
SEMANTIC_DARK: dict = {
    "bg": "#0a0a0a", "bg_subtle": "#151515", "surface": "#1a1a1a", "surface_2": "#111111",
    "fg": "#f1f1f1", "fg2": "#b1b1b1", "muted": "#a3a3a3", "faint": "#6b6b6b",
    "line": "#2b2b2b", "line_strong": "#3d3d3d", "control": "#7a7a7a", "control_hover": "#b1b1b1",
    "primary": "#f1f1f1", "primary_hover": "#d4d4d4", "primary_pressed": "#bdbdbd", "primary_fg": "#0a0a0a",
    "primary_soft": "#2e2e2e", "accent": "#7fb1f7", "accent2": "#a4c7f8",
    "highlight": "#e2a640", "highlight_soft": "#2b2213",
    "success": "#6dcb93", "success_fg": "#0a0a0a", "success_soft": "#13271b",
    "warn": "#e6b765", "warn_fg": "#0a0a0a", "warn_soft": "#2b2213",
    "danger": "#f5928a", "danger_fg": "#0a0a0a", "danger_soft": "#331b1a",
    "focus": "#f1f1f1", "hover": "#222222", "pressed": "#363636", "grid": "#2a2a2a",
    "field_bg": "#1a1a1a", "field_grid": "#2a2a2a", "field_text": "#b1b1b1",
    "field_solid": "#8f8f8f", "field_solid_line": "#0a0a0a", "scrim": "#000000",
    "series": SERIES_DARK,
}

#: v1/v2 name -> v3 token: every alias :func:`tokens` adds (the old names keep working)
ALIASES: dict[str, str] = {
    "bg2": "bg", "panel": "surface_2", "input": "surface", "plot_bg": "surface", "text": "fg",
    "text2": "fg2", "border": "line", "border_strong": "line_strong", "border2": "line_strong",
    "accent_fill": "primary", "accent_hover": "primary_hover", "accent_pressed": "primary_pressed",
    "on_accent": "primary_fg", "accent_soft": "primary_soft", "selection": "primary_soft",
    "good": "success", "bad": "danger", "good_soft": "success_soft", "bad_soft": "danger_soft",
    "panel2": "hover", "bad_border": "danger",
}

#: the v3 names DESIGN.md section 13 requires (tests/test_tokens.py checks each exists in both themes)
TOKEN_KEYS = tuple(k for k in SEMANTIC_LIGHT if k != "series") + ("series",)

#: (foreground, background, minimum WCAG 2.x ratio, why) - every pairing the UI draws.
#: Text >= 4.5:1, control outlines and focus >= 3:1 (WCAG 1.4.11), disabled text is exempt
#: (WCAG 1.4.3) but stays >= 2.2:1 so a greyed label can still be read.
_TEXT_BACKS = ("bg", "bg_subtle", "surface", "surface_2")
CONTRAST_RULES: tuple[tuple[str, str, float, str], ...] = (
    *((fg, bg, 4.5, "text") for fg in ("fg", "fg2", "muted") for bg in (*_TEXT_BACKS, "primary_soft", "hover")),
    *(("accent", bg, 4.5, "link text") for bg in _TEXT_BACKS),
    # a hovered link turns accent2 (a step further from the surface): primary_hover, used up to
    # v3.0, measured 4.37:1 on the dark surface
    *(("accent2", bg, 4.5, "link text, hovered") for bg in _TEXT_BACKS),
    *((s, bg, 4.5, "status text") for s in ("success", "warn", "danger") for bg in _TEXT_BACKS),
    *((s, f"{s}_soft", 4.5, "status text on its soft background") for s in ("success", "warn", "danger")),
    *(("fg", f"{s}_soft", 4.5, "body text inside a notice") for s in ("success", "warn", "danger")),
    *((f"{s}_fg", s, 4.5, "text on a solid status fill") for s in ("success", "warn", "danger")),
    ("primary_fg", "primary", 4.5, "primary button label"),
    ("primary_fg", "primary_hover", 4.5, "primary button label, hovered"),
    ("primary_fg", "primary_pressed", 4.5, "primary button label, pressed"),
    *(("primary", bg, 3.0, "primary button and checked-box shape") for bg in ("bg", "surface", "surface_2")),
    *(("control", bg, 3.0, "control outline and scroll-bar handle") for bg in _TEXT_BACKS),
    *(("focus", bg, 3.0, "focus indicator") for bg in _TEXT_BACKS),
    # checked tool buttons and segments: the 1 px ink border is the state indicator (WCAG 1.4.11),
    # against the panel around the button and against its own soft fill
    *(("fg", bg, 3.0, "checked indicator (ink border)") for bg in (*_TEXT_BACKS, "primary_soft")),
    ("field_solid", "field_bg", 3.0, "solid body in the field view"),
    *(("highlight", bg, 3.0, "non-text mark") for bg in ("bg", "surface", "surface_2")),
    *(("faint", bg, 2.2, "disabled text (exempt; kept legible)") for bg in ("bg", "surface", "surface_2")),
)


def _legacy(sem: dict) -> dict:
    """A theme table with every v3 token plus the v1/v2 aliases."""
    out = dict(sem)
    for old, new in ALIASES.items():
        out[old] = sem[new]
    return out


LIGHT: dict = _legacy(SEMANTIC_LIGHT)
DARK: dict = _legacy(SEMANTIC_DARK)
PALETTES = {"light": LIGHT, "dark": DARK}

# Type: 9 pt UI text; titles 9-11 pt semibold; 14 pt only for the product name (Start, splash, tour).
FONT_PT = {"small": 8.0, "body": 9.0, "h3": 9.0, "h2": 10.0, "h1": 11.0, "kpi": 11.0, "title": 14.0}
# Pixel equivalents at 96 dpi, for code that sizes fonts in pixels (custom painting).
FONT_PX = {"small": 11, "body": 12, "h3": 12, "h2": 13, "h1": 15, "hero": 15, "kpi": 15, "title": 19}

RADIUS = 6          # inputs and buttons: the website's 6 px controls
RADIUS_LG = 8       # framed blocks (cards, the tour's callout, Start tiles): the site's 8 px frames
RADIUS_SM = 3       # small marks inside a control (check-box indicators, scroll-bar handles)
CONTROL_H = 24      # every button, input and combo box
#: smallest hit area of any button, icon-only ones included (WCAG 2.2 SC 2.5.8 target size, 24 x 24
#: CSS px); a 16 px glyph keeps its size, the padding around it makes up the rest
MIN_TARGET = 24
ROW_H = 22          # item-view rows
SPACING = 4         # the 4 px grid
SCRIM_ALPHA = 0.42  # the tour's dimming layer over the window

BUNDLED_UI_FAMILY = "Atkinson Hyperlegible Next"
BUNDLED_MONO_FAMILY = "Atkinson Hyperlegible Mono"
#: the bundled faces (resources/fonts); a missing file is skipped, never fatal
BUNDLED_FONT_FILES = ("AtkinsonHyperlegibleNext-Regular.ttf", "AtkinsonHyperlegibleNext-Medium.ttf",
                      "AtkinsonHyperlegibleNext-SemiBold.ttf", "AtkinsonHyperlegibleNext-Bold.ttf",
                      "AtkinsonHyperlegibleMono-Regular.ttf", "AtkinsonHyperlegibleMono-SemiBold.ttf")
_UI_FAMILIES = ("Segoe UI", "Segoe UI Variable Text", "Inter", "SF Pro Text", "Helvetica Neue",
                "Noto Sans", "DejaVu Sans")
_MONO_FAMILIES = ("Consolas", "Cascadia Mono", "JetBrains Mono", "SF Mono", "Menlo",
                  "DejaVu Sans Mono", "Courier New")

_current = DEFAULT_THEME
_follow_system = False                  # True after apply(name="system"): OS scheme changes re-apply
_scheme_hooked = False
_font_choice = "bundled"
_bundled_ok: bool | None = None          # None until ensure_fonts has run with an application


class _Signals(QObject):
    themeChanged = Signal(str)


signals = _Signals()
changed = signals.themeChanged          # bound signal: theme.changed.connect(slot)


def current() -> str:
    """Name of the active theme ("light" or "dark")."""
    return _current


def preference() -> str:
    """What the user chose: "system" while following the OS, else the active theme name."""
    return "system" if _follow_system else _current


def system_theme(app: QApplication | None = None) -> str:
    """The operating system's light / dark setting as a theme name ("light" when unknown).

    Qt reports ``Qt.ColorScheme.Unknown`` on platforms (and the offscreen
    test platform) that do not expose one; light is this app's default, so
    Unknown means light.
    """
    app = app or QApplication.instance()
    try:
        scheme = app.styleHints().colorScheme() if app is not None else None
    except AttributeError:          # pragma: no cover - Qt < 6.5 has no colorScheme
        scheme = None
    return "dark" if scheme == Qt.ColorScheme.Dark else "light"


def resolve(name: str | None) -> str:
    """A theme choice ("light" | "dark" | "system" | None) as the theme it shows now."""
    if name == "system":
        return system_theme()
    return name if name in THEMES else _current


def _on_system_scheme(*_args) -> None:
    if _follow_system:
        apply(None, "system")


def is_dark(name: str | None = None) -> bool:
    """True when the active (or named) theme is the dark one."""
    return (name or _current) == "dark"


def tokens(name: str | None = None) -> dict:
    """Colour tokens of the active (or named) theme: v3 names plus the v1/v2 aliases.

    A fresh copy, so callers may annotate it freely. Every value is an opaque
    ``#rrggbb`` string (the v2 ``*_soft`` keys were ``rgba()`` strings that
    ``QColor`` could not parse); ``series`` is a list of them.
    """
    pal = dict(PALETTES[name or _current])
    pal["series"] = list(pal["series"])
    pal["accent2_soft"] = mix(pal["accent2"], pal["surface"], 0.14)
    return pal


def semantic_tokens(name: str | None = None) -> dict:
    """Only the v3 token names (no aliases) - what the token test and the docs table list."""
    sem = SEMANTIC_LIGHT if (name or _current) == "light" else SEMANTIC_DARK
    out = dict(sem)
    out["series"] = list(sem["series"])
    return out


_STATUS_TOKEN = {"new": "muted", "queued": "text2", "running": "accent", "done": "good",
                 "failed": "bad", "cancelled": "warn"}


def status_colour(status: str) -> str:
    """Token name for a simulation status ("new", "queued", "running", "done", ...)."""
    return _STATUS_TOKEN.get(status, "muted")


def rgba(hex_colour: str, alpha: float) -> str:
    """``#rrggbb`` + alpha -> a QSS ``rgba(r, g, b, a)`` string."""
    c = QColor(hex_colour)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {round(alpha * 255)})"


def mix(fg: str, bg: str, amount: float) -> str:
    """``amount`` of ``fg`` over ``bg`` as an opaque ``#rrggbb`` (0 = bg, 1 = fg)."""
    a, b = QColor(fg), QColor(bg)
    t = max(0.0, min(1.0, float(amount)))
    r = round(a.red() * t + b.red() * (1 - t))
    g = round(a.green() * t + b.green() * (1 - t))
    bl = round(a.blue() * t + b.blue() * (1 - t))
    return f"#{r:02x}{g:02x}{bl:02x}"


def relative_luminance(hex_colour: str) -> float:
    """WCAG 2.x relative luminance of an sRGB colour (0 black .. 1 white)."""
    c = QColor(hex_colour)

    def lin(v: int) -> float:
        x = v / 255.0
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(c.red()) + 0.7152 * lin(c.green()) + 0.0722 * lin(c.blue())


def contrast_ratio(a: str, b: str) -> float:
    """WCAG 2.x contrast ratio of two colours (1 .. 21), order-independent."""
    la, lb = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def contrast_report(name: str | None = None) -> list[tuple[str, str, float, float, str]]:
    """(fg, bg, measured, required, why) for every :data:`CONTRAST_RULES` pairing of a theme."""
    t = semantic_tokens(name)
    return [(fg, bg, contrast_ratio(t[fg], t[bg]), need, why) for fg, bg, need, why in CONTRAST_RULES]


def qcolor(key_or_hex: str, alpha: float | None = None) -> QColor:
    """A QColor from a token name (``"accent"``) or a hex string, optionally translucent."""
    t = tokens()
    c = QColor(t.get(key_or_hex, key_or_hex))
    if alpha is not None:
        c.setAlphaF(alpha)
    return c


# ---------------------------------------------------------------------------
# fonts
# ---------------------------------------------------------------------------

def ensure_fonts() -> bool:
    """Register the bundled Atkinson faces (and, on a bare font database, the system ones).

    Returns whether the bundled UI family is available. The ``offscreen``
    platform (tests, headless screenshots) starts with an empty font database
    on Windows, which would draw every label as boxes, so the system's UI and
    mono files are registered there too - they are also the glyph fallback
    for Greek letters, superscripts and arrows, which the bundled Latin faces
    do not have. Safe to call repeatedly.
    """
    global _bundled_ok
    if QGuiApplication.instance() is None:
        return False
    have = set(QFontDatabase.families())
    if BUNDLED_UI_FAMILY not in have:
        for name in BUNDLED_FONT_FILES:
            path = FONT_DIR / name
            if path.exists():
                QFontDatabase.addApplicationFont(str(path))
        have = set(QFontDatabase.families())
    _bundled_ok = BUNDLED_UI_FAMILY in have and BUNDLED_MONO_FAMILY in have
    if not (any(f in have for f in _UI_FAMILIES[:2]) and any(f in have for f in _MONO_FAMILIES[:2])):
        windir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        candidates = [windir / n for n in ("segoeui.ttf", "segoeuib.ttf", "seguisb.ttf", "segoeuii.ttf",
                                           "seguisym.ttf", "CascadiaMono.ttf", "consola.ttf", "consolab.ttf")]
        if sys.platform != "win32":
            candidates += [Path("/usr/share/fonts/truetype/dejavu") / n
                           for n in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSansMono.ttf")]
        for path in candidates:
            if path.exists():
                QFontDatabase.addApplicationFont(str(path))
    _font_family_cached.cache_clear()
    _mono_family_cached.cache_clear()
    _fallback_family_cached.cache_clear()
    return bool(_bundled_ok)


def bundled_fonts_available() -> bool:
    """Whether the bundled Atkinson faces loaded (False before an application exists)."""
    if _bundled_ok is None:
        ensure_fonts()
    return bool(_bundled_ok)


def font_choice() -> str:
    """"bundled" (Atkinson Hyperlegible, the default) or "system" (Segoe UI)."""
    return _font_choice


@cache
def _fallback_family_cached() -> str:
    have = set(QFontDatabase.families())
    for fam in _UI_FAMILIES:
        if fam in have:
            return fam
    return QFont().defaultFamily()


def fallback_family() -> str:
    """The platform UI family, used for glyphs the bundled faces lack (and for ``font="system"``)."""
    if QGuiApplication.instance() is None:
        return _UI_FAMILIES[0]
    return _fallback_family_cached()


@cache
def _font_family_cached(choice: str) -> str:
    if choice == "bundled" and BUNDLED_UI_FAMILY in set(QFontDatabase.families()):
        return BUNDLED_UI_FAMILY
    return _fallback_family_cached()


def font_family() -> str:
    """The UI family in use: Atkinson Hyperlegible Next, or Segoe UI with ``font="system"``."""
    if QGuiApplication.instance() is None:      # the font database needs an application
        return BUNDLED_UI_FAMILY if _font_choice == "bundled" else _UI_FAMILIES[0]
    return _font_family_cached(_font_choice)


def mono_family() -> str:
    """The monospace family in use (Atkinson Hyperlegible Mono, else Consolas / Cascadia Mono)."""
    if QGuiApplication.instance() is None:
        return BUNDLED_MONO_FAMILY if _font_choice == "bundled" else _MONO_FAMILIES[0]
    return _mono_family_cached(_font_choice)


@cache
def _mono_family_cached(choice: str) -> str:
    have = set(QFontDatabase.families())
    if choice == "bundled" and BUNDLED_MONO_FAMILY in have:
        return BUNDLED_MONO_FAMILY
    for fam in _MONO_FAMILIES:
        if fam in have:
            return fam
    return "monospace"


def mono_fallback_family() -> str:
    """The system monospace family (fallback for glyphs the bundled mono face lacks)."""
    if QGuiApplication.instance() is None:
        return _MONO_FAMILIES[0]
    return _mono_family_cached("system")


def ui_font(px: int | None = None, weight: QFont.Weight = QFont.Weight.Normal, *,
            pt: float | None = None) -> QFont:
    """The UI font: 9 pt by default, or an explicit ``pt`` / legacy pixel size ``px``.

    The family list ends with the platform font, so a Greek letter or an arrow
    in a label is drawn from it instead of a missing-glyph box.
    """
    f = QFont()
    fams = [font_family()]
    if fams[0] != fallback_family():
        fams.append(fallback_family())
    f.setFamilies(fams)
    if px is not None and pt is None:
        f.setPixelSize(px)
    else:
        f.setPointSizeF(pt or FONT_PT["body"])
    f.setWeight(weight)
    return f


def mono_font(px: int | None = None, weight: QFont.Weight = QFont.Weight.Normal, *,
              pt: float | None = None) -> QFont:
    """The monospace font (9 pt by default) for data tables and the message log."""
    f = QFont()
    fams = [mono_family()]
    if fams[0] != mono_fallback_family():
        fams.append(mono_fallback_family())
    f.setFamilies(fams)
    if px is not None and pt is None:
        f.setPixelSize(px)
    else:
        f.setPointSizeF(pt or FONT_PT["body"])
    f.setWeight(weight)
    f.setStyleHint(QFont.StyleHint.Monospace)
    return f


def use_mono(widget: QWidget, *, small: bool = False) -> QWidget:
    """Show ``widget`` (a log, a code view, a report box) in the mono face, 9 pt or 8 pt.

    ``widget.setFont(mono_font())`` alone does nothing visible: the app style
    sheet's ``QWidget { font-family }`` rule wins over ``setFont`` (measured: the
    crash report box drew in Atkinson Hyperlegible Next). The ``mono`` style
    hook is a rule of the same sheet, so it wins instead, follows a font switch
    in Preferences and needs no widget-level style sheet. The font is set too,
    for metrics read before the widget is polished.
    """
    set_prop(widget, "mono", "small" if small else "true")
    widget.setFont(mono_font(pt=FONT_PT["small"] if small else None))
    return widget


# ---------------------------------------------------------------------------
# icons: one 16 px line set (resources/icons/*.svg, stroke = currentColor)
# ---------------------------------------------------------------------------

def icon_names() -> list[str]:
    """Names of every bundled icon (file stems in ``resources/icons``)."""
    return sorted(p.stem for p in ICON_DIR.glob("*.svg"))


@cache
def _svg_source(name: str) -> str:
    path = ICON_DIR / f"{name}.svg"
    if not path.exists():
        path = ICON_DIR / "dot.svg"
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=512)
def _renderer(name: str, colour: str) -> QSvgRenderer:
    return QSvgRenderer(QByteArray(_svg_source(name).replace("currentColor", colour).encode()))


#: the colour a checked / selected icon is drawn in, by the token its call site asked for
ACTIVE_ICON_TOKEN = {"accent": "fg"}


def render_icon(name: str, size: int, colour: str, dpr: float = 1.0) -> QPixmap:
    """A crisp pixmap of a line icon in a solid colour (``colour`` = token or hex)."""
    hexc = tokens().get(colour, colour)
    px = max(1, round(size * dpr))
    pm = QPixmap(px, px)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    _renderer(name, hexc).render(p, QRectF(0, 0, px, px))
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


class _SvgIconEngine(QIconEngine):
    """Renders a tinted SVG at exactly the requested size and device pixel ratio.

    A pixmap-backed QIcon would be resampled at 125 % / 150 % display scaling,
    which turns 1.25 px strokes to mush; drawing the vector each time does not.
    The colour is resolved at paint time, so icons follow a theme switch.
    Disabled icons use ``faint``; checked / selected ones use ``active``.
    """

    def __init__(self, name: str, colour: str | None, active: str | None) -> None:
        super().__init__()
        self._name, self._colour, self._active = name, colour, active

    def _colour_for(self, mode: QIcon.Mode, state: QIcon.State) -> str:
        t = tokens()
        if mode == QIcon.Mode.Disabled:
            return t["faint"]
        if self._active and (state == QIcon.State.On or mode == QIcon.Mode.Selected):
            # v4: checked chrome icons are ink, like everything else interactive in the chrome; the
            # blue ``accent`` is kept for marks in the data views. Call sites ask for "accent" (the
            # v3 active colour), so it is mapped here rather than at forty call sites.
            active = ACTIVE_ICON_TOKEN.get(self._active, self._active)
            return t.get(active, active)
        c = self._colour or "text2"
        return t.get(c, c)

    def paint(self, painter: QPainter, rect, mode, state) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(rect.width(), rect.height())
        r = QRectF(rect.x() + (rect.width() - side) / 2, rect.y() + (rect.height() - side) / 2, side, side)
        _renderer(self._name, self._colour_for(mode, state)).render(painter, r)
        painter.restore()

    def pixmap(self, size: QSize, mode, state) -> QPixmap:
        pm = QPixmap(size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        self.paint(p, pm.rect(), mode, state)
        p.end()
        return pm

    def scaledPixmap(self, size: QSize, mode, state, scale: float) -> QPixmap:
        pm = QPixmap(QSize(int(size.width() * scale), int(size.height() * scale)))
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        self.paint(p, pm.rect(), mode, state)
        p.end()
        pm.setDevicePixelRatio(scale)
        return pm

    def clone(self) -> QIconEngine:
        return _SvgIconEngine(self._name, self._colour, self._active)


def icon(name: str, colour: str | None = None, active: str | None = None) -> QIcon:
    """A bundled 16 px line icon as a QIcon.

    ``colour`` is a token name or hex (default ``text2``, the guideline icon
    colour); ``active`` is used for the checked / selected state (typically
    ``"accent"``). The colour is looked up at paint time, so an icon created
    before a theme switch follows the new theme automatically.
    """
    return QIcon(_SvgIconEngine(name, colour, active))


def app_icon() -> QIcon:
    """The application icon: a paper airfoil on an ink square, the website's mark (taskbar, window)."""
    ic = QIcon()
    ren = QSvgRenderer(str(APP_ICON_PATH))
    for s in (16, 20, 24, 32, 40, 48, 64, 128, 256):
        pm = QPixmap(s, s)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        ren.render(p, QRectF(0, 0, s, s))
        p.end()
        ic.addPixmap(pm)
    return ic


#: the mark's ink square and paper airfoil (resources/aerolume.svg, the website's favicon colours)
APP_ICON_INK, APP_ICON_PAPER = "#0e0e0e", "#f6f6f6"


def app_icon_svg(name: str | None = None) -> str:
    """The mark's SVG for a theme: ink on paper in light, inverted on the dark theme.

    On the near-black dark theme an ink square would vanish into the page and leave a floating
    airfoil, so the in-app mark (Start page, splash) inverts there, as the website's filled
    button does (light fill, dark glyph). The taskbar / file icon is always the ink original.
    """
    src = APP_ICON_PATH.read_text(encoding="utf-8")
    if is_dark(name):
        dark = SEMANTIC_DARK
        src = (src.replace(APP_ICON_INK, "@ink@").replace(APP_ICON_PAPER, dark["primary_fg"])
               .replace("@ink@", dark["primary"]))
    return src


def app_icon_pixmap(size: int, dpr: float = 1.0, name: str | None = None) -> QPixmap:
    """The application mark at one size (Start page, splash) for the active (or named) theme."""
    px = max(1, round(size * dpr))
    pm = QPixmap(px, px)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(QByteArray(app_icon_svg(name).encode("utf-8"))).render(p, QRectF(0, 0, px, px))
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


def _qss_image(name: str, colour: str) -> str:
    """Write a tinted copy of an icon to a cache folder and return a QSS ``url()`` path.

    QSS sub-control images (check marks, combo arrows) must be files; tinting
    them per theme keeps the arrows the same colour as the text around them.
    The file name includes a hash of the SVG, so a redrawn icon never picks up
    a stale cached copy.
    """
    import hashlib

    src = _svg_source(name).replace("currentColor", colour)
    digest = hashlib.sha1(src.encode()).hexdigest()[:10]
    folder = Path(tempfile.gettempdir()) / "aerolume-theme"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}-{digest}.svg"
    if not path.exists():
        path.write_text(src, encoding="utf-8")
    return path.as_posix()


# ---------------------------------------------------------------------------
# style sheet
# ---------------------------------------------------------------------------

_QSS = Template(r"""
* { outline: 0; }
QWidget { color: $fg; font-family: $ui_stack; font-size: 9pt; }
QMainWindow, QDialog { background: $bg; }
QMainWindow::separator { background: $bg; width: 4px; height: 4px; }
QMainWindow::separator:hover { background: $line_strong; }
QStackedWidget, QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }
QAbstractScrollArea { border: none; }
QLabel { background: transparent; }
QLabel:disabled { color: $faint; }
QDialog QLabel { color: $fg; }

/* ---------- typography hooks ---------- */
QLabel[role="title"] { font-size: 14pt; font-weight: 500; color: $fg; }
QLabel[role="h1"], QLabel[role="hero"] { font-size: 11pt; font-weight: 500; color: $fg; }
QLabel[role="h2"] { font-size: 10pt; font-weight: 500; color: $fg; }
QLabel[role="h3"] { font-size: 9pt; font-weight: 600; color: $fg; }
QLabel[role="secondary"], QLabel[role="pg-label"], QLabel[role="eyebrow"] { color: $fg2; }
QLabel[role="muted"] { color: $muted; }
QLabel[role="small"] { color: $muted; font-size: 8pt; }
QLabel[role="mono"] { font-family: $mono_stack; }
*[mono="true"] { font-family: $mono_stack; font-size: 9pt; }
*[mono="small"] { font-family: $mono_stack; font-size: 8pt; }
QLabel[role="kpi-value"] { font-size: 11pt; font-weight: 400; color: $fg; }
QLabel[role="kpi-label"] { color: $fg2; font-weight: 400; }
QLabel[role="kpi-delta-good"] { color: $success; }
QLabel[role="kpi-delta-bad"] { color: $danger; }
QLabel[role="badge"], QLabel[role="badge-ok"], QLabel[role="badge-warn"], QLabel[role="badge-bad"],
QLabel[role="badge-accent"], QLabel[role="status-good"], QLabel[role="status-warn"],
QLabel[role="status-bad"] {
    background: transparent; border: none; padding: 0; color: $fg2; font-weight: 400;
}
QLabel[role="badge-ok"], QLabel[role="status-good"] { color: $success; }
QLabel[role="badge-warn"], QLabel[role="status-warn"] { color: $warn; }
QLabel[role="badge-bad"], QLabel[role="status-bad"] { color: $danger; }
QLabel[role="badge-accent"] { color: $accent; }

/* ---------- surfaces ---------- */
QFrame[card="true"] { background: $surface_2; border: 1px solid $line; border-radius: ${radius_lg}px; }
QFrame[card="true"][interactive="true"]:hover { border-color: $line_strong; background: $hover; }
QFrame[card="true"] QLabel { border: none; }
QFrame[panel="true"] { background: $surface_2; border: none; }
QFrame[surface="true"] { background: $surface; border: 1px solid $line; }
QFrame[chrome="toolbar"] { background: $surface_2; border: none; border-bottom: 1px solid $line; }
QFrame[chrome="rail"], QFrame[chrome="topbar"] { background: $bg; border: none; }
QFrame[chrome="page"], QFrame[chrome="dock"] { background: $surface_2; border: 1px solid $line; }
QFrame[chrome="dock"][collapsed="true"] { border: none; background: transparent; }
QFrame[chrome="dock-header"] { background: $bg_subtle; border: 1px solid $line; }
QFrame[chrome="dock-header"][collapsed="true"] { background: $bg_subtle; }
QFrame[chrome="dock-header"] QLabel[role="dock-title"] { color: $fg; font-weight: 600; }
QFrame[chrome="dock-header"]:hover { border-color: $line_strong; }
QFrame[pill] { background: transparent; border: none; }
QFrame[divider="true"] { background: $line; border: none; max-height: 1px; min-height: 1px; }
QFrame[divider="vertical"] { background: $line; border: none; max-width: 1px; min-width: 1px; }
QFrame[notice="info"] { background: $primary_soft; border: none; border-left: 3px solid $accent; }
QFrame[notice="good"] { background: $success_soft; border: none; border-left: 3px solid $success; }
QFrame[notice="warn"] { background: $warn_soft; border: none; border-left: 3px solid $warn; }
QFrame[notice="bad"] { background: $danger_soft; border: none; border-left: 3px solid $danger; }
QFrame[notice] QLabel { color: $fg; }
QFrame[notice="good"] QLabel[role="notice-title"] { color: $success; font-weight: 600; }
QFrame[notice="warn"] QLabel[role="notice-title"] { color: $warn; font-weight: 600; }
QFrame[notice="bad"] QLabel[role="notice-title"] { color: $danger; font-weight: 600; }
QFrame[notice="info"] QLabel[role="notice-title"] { color: $fg; font-weight: 600; }
QFrame[callout="true"] { background: $surface; border: 1px solid $line_strong; border-radius: ${radius_lg}px; }
QFrame[callout="true"] QLabel { color: $fg; }
QFrame[splash="true"] { background: $surface; border: 1px solid $line_strong; }

/* ---------- buttons (24 px): the website's two buttons. Secondary is an outline in the control
   grey that turns ink on hover; primary is filled ink (light ink on the dark theme) ---------- */
QPushButton {
    background: transparent; color: $fg; border: 1px solid $control; border-radius: ${radius}px;
    padding: 2px 10px; min-height: 18px;
}
QPushButton:hover { border-color: $fg; }
QPushButton:pressed { background: $pressed; border-color: $fg; }
QPushButton:focus { border-color: $focus; }
QPushButton:disabled { color: $faint; border-color: $line_strong; background: transparent; }
QPushButton:checked { background: $primary_soft; border-color: $fg; }
QPushButton::menu-indicator { image: url("$img_chevron_down"); subcontrol-position: right center;
    subcontrol-origin: padding; width: 12px; right: 4px; }
QPushButton[variant="primary"] { background: $primary; color: $primary_fg; border: 1px solid $primary; }
QPushButton[variant="primary"]:hover { background: $primary_hover; border-color: $primary_hover; }
QPushButton[variant="primary"]:pressed { background: $primary_pressed; border-color: $primary_pressed; }
/* focus: the website's offset ring (outline in the text colour, a gap of page between it and the
   button) in one widget. A $fg border on an ink fill would change nothing, so the fill is clipped
   to the content box and the 1 px of padding around it shows the panel: an ink ring, one pixel of
   paper, then the ink button. The geometry comes from the resting state, so nothing moves.
   (A 3 px double border was tried first: Qt blends its middle pixel to mid grey.) */
QPushButton[variant="primary"]:focus { border: 1px solid $fg; padding: 1px; background-clip: content; }
QPushButton[variant="primary"]:disabled { background: $surface_2; color: $faint; border-color: $line_strong; }
QPushButton[variant="ghost"] { background: transparent; border: 1px solid transparent; color: $fg2; }
QPushButton[variant="ghost"]:hover { background: $hover; color: $fg; }
QPushButton[variant="ghost"]:pressed { background: $pressed; }
QPushButton[variant="ghost"]:focus { border-color: $focus; }
QPushButton[variant="ghost"]:disabled { color: $faint; background: transparent; }
QPushButton[variant="danger"] { background: transparent; color: $danger; border: 1px solid $control; }
QPushButton[variant="danger"]:hover { background: $danger; color: $danger_fg; border-color: $danger; }
QPushButton[variant="danger"]:focus { border-color: $focus; }
QPushButton[variant="danger"]:disabled { color: $faint; border-color: $line_strong; background: transparent; }
/* links are ink with an underline, as on the website (where every text link is the text colour,
   underlined); hovering greys the text */
QPushButton[variant="link"], QToolButton[variant="link"] {
    background: transparent; border: none; color: $fg; padding: 1px 0; text-align: left; min-height: 18px;
    text-decoration: underline; }
QPushButton[variant="link"]:hover, QToolButton[variant="link"]:hover {
    color: $fg2; text-decoration: underline; background: transparent; }
QPushButton[variant="link"]:focus, QToolButton[variant="link"]:focus { color: $fg; background: $hover; }
QPushButton[variant="link"]:disabled, QToolButton[variant="link"]:disabled { color: $faint; text-decoration: none; }

QToolButton { background: transparent; border: 1px solid transparent; border-radius: ${radius}px;
    padding: 2px; color: $fg; }
QToolButton:hover { background: $hover; }
QToolButton:pressed { background: $pressed; }
/* checked = soft fill inside a 1 px ink border. The fill alone measured 1.07-1.46:1 against the
   panels (critique of v4), under the 3:1 WCAG 1.4.11 asks of a state indicator; the ink border is
   >= 3:1 on every panel and on the fill itself (CONTRAST_RULES "checked indicator"). Focus on a
   checked button would then change nothing, so the fill is clipped to the content box: ink ring,
   the 2 px padding as a gap of panel, then the fill (the website's offset focus ring, in one widget,
   with no geometry change). */
QToolButton:checked { background: $primary_soft; border-color: $fg; }
QToolButton:focus { border-color: $focus; }
QToolButton:checked:focus { background-clip: content; }
QToolButton:disabled { color: $faint; }
QToolButton::menu-indicator { image: none; }
QToolButton[popupMode="1"] { padding-right: 12px; }
QToolButton::menu-button { border: none; width: 12px; }
QToolButton::menu-arrow { image: url("$img_chevron_down"); width: 10px; height: 10px; }
QToolButton[dropdown="true"] { padding-right: 16px; }
QToolButton[dropdown="true"]::menu-indicator { image: url("$img_chevron_down");
    subcontrol-position: right center; subcontrol-origin: padding; width: 10px; height: 10px; right: 3px; }
/* Filter toggles stay flat in both themes: on = normal text, off = muted text (the icon goes
   faint too, see messages.py). A fill for "on" read as a row of chips in the dark theme, where
   $hover is far from the title-row background, while it vanished in the light one. */
QToolButton[variant="toggle"] { padding: 0 6px; border: 1px solid transparent; background: transparent; }
QToolButton[variant="toggle"]:checked { background: transparent; color: $fg; }
QToolButton[variant="toggle"]:!checked { color: $muted; }
QToolButton[variant="toggle"]:hover { background: $hover; }
QToolButton[variant="toggle"]:pressed { background: $pressed; }
QToolButton[variant="toggle"]:focus { border-color: $focus; }
/* Viewport layer toggles (view3d_panel.FlowToggles): on = soft fill with an accent outline, so an
   "on" layer reads like the checked item of the View segmented control beside it. */
QToolButton[variant="layer"] { padding: 0 8px; border: 1px solid transparent; border-radius: ${radius}px;
    background: transparent; color: $fg2; }
QToolButton[variant="layer"]:hover { background: $hover; color: $fg; }
QToolButton[variant="layer"]:pressed { background: $pressed; }
QToolButton[variant="layer"]:checked { background: $primary_soft; border-color: $fg; color: $fg; }
/* no vertical padding to make a gap from: a focused checked layer keeps its full fill */
QToolButton[variant="layer"]:checked:focus { background-clip: border; }
QToolButton[variant="layer"]:disabled { color: $faint; }
QToolButton[variant="layer"]:checked:disabled { background: transparent; border-color: $line_strong;
    color: $muted; }
QToolButton[role="section"] { border: none; border-radius: 0; padding: 3px 0; font-weight: 600;
    color: $fg; text-align: left; }
QToolButton[role="section"]:hover { background: transparent; color: $fg2; }
QToolButton[role="section"]:checked { background: transparent; }

/* segmented control (widgets.common.SegmentedControl, param_form.SegmentedButtons). The frame's
   outline is painted by the widget (theme.paint_segmented_frame), not by a QSS border: a QSS border
   insets the frame's contents by 1 px, so a frame squeezed to 24 px (a tool strip, an inspector
   row) left 22 px for the 24 px segments and clipped them; the bottom of the checked segment's ink
   border went missing. Now the segments fill the frame, and each draws its box 1 px inside its own
   rectangle (margin), so the outline shows round them and the whole 24 px stays the hit target. */
QFrame[segmented="true"] { background: transparent; border: none; }
/* 1 px margin + 2 x 2 px padding + 18 px content + 1 = 24 px (WCAG 2.2 SC 2.5.8, v3.4 audit).
   The segments are rounded one step tighter than the frame (4 px inside 6 px), so the checked
   segment's box follows the frame's corner instead of poking a square corner through it */
QFrame[segmented="true"] QPushButton { background: transparent; border: none; border-radius: ${radius_seg}px;
    margin: 1px; padding: 2px 10px; color: $fg2; min-height: 18px; }
QFrame[segmented="true"] QPushButton:hover { background: $hover; color: $fg; }
/* checked = soft fill and a 1 px ink border (>= 3:1, WCAG 1.4.11; the fill alone was 1.1-1.5:1);
   focus = a 2 px ring, so a focused checked segment still differs from a checked one. Border and
   padding always add up to the resting 2 px / 10 px, so the segment never changes size. */
QFrame[segmented="true"] QPushButton:checked { background: $primary_soft; color: $fg; border: 1px solid $fg;
    padding: 1px 9px; }
QFrame[segmented="true"] QPushButton:focus { border: 2px solid $focus; padding: 0 8px; }

/* ---------- inputs (24 px): outlined in $control, 3:1 on every surface (WCAG 1.4.11) ---------- */
QLineEdit, QAbstractSpinBox, QComboBox {
    background: $surface; color: $fg; border: 1px solid $control; border-radius: ${radius}px;
    padding: 1px 6px; min-height: 20px; selection-background-color: $primary_soft; selection-color: $fg;
}
QPlainTextEdit, QTextEdit {
    background: $surface; color: $fg; border: 1px solid $line_strong; border-radius: 0;
    padding: 2px; selection-background-color: $primary_soft; selection-color: $fg;
}
QLineEdit:hover, QAbstractSpinBox:hover, QComboBox:hover { border-color: $control_hover; }
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {
    border-color: $focus; }
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {
    color: $faint; border-color: $line_strong; background: $surface_2; }
QLineEdit[invalid="true"], QAbstractSpinBox[invalid="true"] { border-color: $danger; }
QLineEdit[readOnly="true"] { background: $surface_2; border-color: $line_strong; }
QTextBrowser { background: $surface; border: 1px solid $line; padding: 8px; }

QComboBox { padding-right: 20px; combobox-popup: 0; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 18px;
    border: none; }
QComboBox::down-arrow { image: url("$img_chevron_down"); width: 12px; height: 12px; }
QComboBox::down-arrow:disabled { image: url("$img_chevron_down_faint"); }
QComboBox QAbstractItemView { background: $surface; border: 1px solid $line_strong; padding: 2px 0;
    selection-background-color: $primary_soft; selection-color: $fg; outline: 0; }
QComboBox QAbstractItemView::item { min-height: 22px; padding: 0 8px; }
QComboBox QAbstractItemView::item:hover { background: $hover; }
QComboBox QAbstractItemView::item:selected { background: $primary_soft; color: $fg; }

QAbstractSpinBox { padding-right: 18px; }
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { subcontrol-origin: border; width: 16px;
    border: none; background: transparent; }
/* the arrows sit 2 px in from the right edge, inside the 18 px of padding (more padding clipped the
   numbers of the fixed-width boxes in the physics editor), their hover fill rounded to follow the
   6 px corners */
QAbstractSpinBox::up-button { subcontrol-position: top right; margin: 1px 2px 0 0; border-top-right-radius: 5px; }
QAbstractSpinBox::down-button { subcontrol-position: bottom right; margin: 0 2px 1px 0;
    border-bottom-right-radius: 5px; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background: $hover; }
QAbstractSpinBox::up-arrow { image: url("$img_chevron_up"); width: 10px; height: 10px; }
QAbstractSpinBox::down-arrow { image: url("$img_chevron_down"); width: 10px; height: 10px; }
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off { image: url("$img_chevron_up_faint"); }
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {
    image: url("$img_chevron_down_faint"); }

/* 24 px tall rows: the whole label is the hit target (WCAG 2.2 SC 2.5.8); at the indicator's own
   height a check box was 16 px tall */
QCheckBox, QRadioButton { spacing: 6px; background: transparent; min-height: 24px; }
QCheckBox:disabled, QRadioButton:disabled { color: $faint; }
QCheckBox::indicator, QRadioButton::indicator { width: 13px; height: 13px; border: 1px solid $control;
    background: $surface; }
QCheckBox::indicator { border-radius: ${radius_sm}px; }
QRadioButton::indicator { border-radius: 7px; }
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: $fg; }
QCheckBox::indicator:focus, QRadioButton::indicator:focus { border-color: $focus; }
QCheckBox::indicator:checked { background: $primary; border-color: $primary; image: url("$img_check"); }
QCheckBox::indicator:indeterminate { background: $primary; border-color: $primary;
    image: url("$img_minus"); }
QRadioButton::indicator:checked { background: $surface; border-color: $primary; image: url("$img_dot_ink"); }
/* checked + focused indicators: the 13 px content shrinks by what the thicker frame adds, so the
   indicator stays 15 px and the label beside it does not move when focus arrives */
QRadioButton::indicator:checked:focus { border: 2px solid $focus; width: 11px; height: 11px; }
/* a checked box is filled with ink, the focus colour, so a $focus border on it changes nothing (the
   v3.3 audit found the checked "Heat transfer" box unchanged by focus): it gets the primary
   button's offset ring instead, an ink edge with one pixel of the panel inside it */
QCheckBox::indicator:checked:focus, QCheckBox::indicator:indeterminate:focus {
    border: 1px solid $focus; padding: 1px; background-clip: content; width: 11px; height: 11px; }
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled { background: $surface_2; border-color: $line_strong; }
QCheckBox::indicator:checked:disabled { image: url("$img_check_faint"); }

QSlider { background: transparent; min-height: 20px; }
QSlider::groove:horizontal { height: 3px; background: $control; border-radius: 1px; }
QSlider::sub-page:horizontal { background: $fg; border-radius: 1px; }
QSlider::add-page:horizontal { background: $control; border-radius: 1px; }
QSlider::handle:horizontal { width: 12px; height: 12px; margin: -5px 0; border-radius: 6px;
    background: $surface; border: 1px solid $control_hover; }
QSlider::handle:horizontal:hover { border-color: $fg; }
/* focus: the handle's ring turns $focus and doubles (a slider has no border of its own to colour) */
QSlider::handle:horizontal:focus, QSlider::handle:vertical:focus { border: 2px solid $focus; }
QSlider::groove:vertical { width: 3px; background: $control; border-radius: 1px; }
QSlider::sub-page:vertical { background: $control; }
QSlider::add-page:vertical { background: $fg; }
QSlider::handle:vertical { width: 12px; height: 12px; margin: 0 -5px; border-radius: 6px;
    background: $surface; border: 1px solid $control_hover; }
QSlider::sub-page:horizontal:disabled, QSlider::add-page:vertical:disabled { background: $faint; }
QSlider::handle:disabled { border-color: $line_strong; background: $surface_2; }

QProgressBar { background: $line; border: none; border-radius: 0; max-height: 4px; min-height: 4px;
    text-align: center; color: transparent; }
QProgressBar::chunk { background: $accent; }

/* ---------- tabs: flat, 2 px ink underline (the website marks its current page in ink) ---------- */
QTabWidget::pane { border: none; border-top: 1px solid $line; top: -1px; }
QTabWidget::tab-bar { left: 0; }
QTabBar { background: transparent; qproperty-drawBase: 0; }
QTabBar::tab { background: transparent; color: $fg2; padding: 4px 12px 4px 12px; margin: 0;
    border: none; border-bottom: 2px solid transparent; min-height: 16px; }
QTabBar::tab:hover { color: $fg; background: $hover; }
QTabBar::tab:selected { color: $fg; border-bottom: 2px solid $fg; background: transparent; }
QTabBar::tab:disabled { color: $faint; }
QTabBar::close-button { image: url("$img_close"); subcontrol-position: right; }
QTabBar::close-button:hover { background: $hover; border-radius: ${radius_sm}px; }
QDockWidget QTabBar::tab, QMainWindow > QTabBar::tab { padding: 3px 10px; }

/* ---------- scrollbars: slim, the handle in $control so it reaches 3:1 (WCAG 1.4.11) on every
   surface; $line_strong, used up to v3.0, measured 1.7-1.9:1 ---------- */
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:vertical { background: $control; border-radius: ${radius_sm}px; min-height: 28px; margin: 2px; }
QScrollBar::handle:horizontal { background: $control; border-radius: ${radius_sm}px; min-width: 28px; margin: 2px; }
QScrollBar::handle:hover { background: $control_hover; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; border: none; background: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }
QAbstractScrollArea::corner { background: transparent; }

/* ---------- item views ---------- */
QTableView, QTreeView, QListView {
    background: $surface; alternate-background-color: $surface_2; border: 1px solid $line;
    gridline-color: $line; selection-background-color: $primary_soft; selection-color: $fg;
}
/* Borderless views keep a transparent 1 px border so focus can colour it without moving the
   content: the v3.3 keyboard audit found the Project tree and the Start page lists unchanged
   by keyboard focus until an item was current (WCAG 2.4.7). */
QAbstractItemView[flat="true"] { border: 1px solid transparent; }
QAbstractItemView[pane="true"] { border: 1px solid transparent; background: $surface_2;
    alternate-background-color: $surface; }
QTableView:focus, QTreeView:focus, QListView:focus { border-color: $focus; }
/* a frameless scroll area that stays a Tab stop (read-only content that overflows, see
   common.unfocus_scroll_areas) gets the same transparent border, coloured on focus */
QScrollArea[scrollstop="true"] { border: 1px solid transparent; }
QScrollArea[scrollstop="true"]:focus { border-color: $focus; }
QTableView::item, QTreeView::item { padding: 2px 4px; border: none; min-height: 18px; }
QListView::item { padding: 3px 6px; }
QTableView::item:hover, QTreeView::item:hover, QListView::item:hover { background: $hover; }
QTableView::item:selected, QTreeView::item:selected, QListView::item:selected {
    background: $primary_soft; color: $fg; }
/* Check boxes inside item views (Compare / Report simulation lists) are drawn by the view, not by a
   QCheckBox, so they need their own rules; without them Fusion draws an unchecked box that vanishes
   on the dark panel and a white-with-black-tick box that clashes with the QCheckBoxes beside it. */
QAbstractItemView::indicator { width: 13px; height: 13px; border: 1px solid $control;
    border-radius: ${radius_sm}px; background: $surface; }
QAbstractItemView::indicator:hover { border-color: $fg; }
QAbstractItemView::indicator:checked { background: $primary; border-color: $primary;
    image: url("$img_check"); }
QAbstractItemView::indicator:indeterminate { background: $primary; border-color: $primary;
    image: url("$img_minus"); }
QAbstractItemView::indicator:disabled { background: $surface_2; border-color: $line_strong; }
QAbstractItemView::indicator:checked:disabled { image: url("$img_check_faint"); }
QTreeView::branch { background: transparent; }
QTreeView::branch:has-children:!has-siblings:closed, QTreeView::branch:closed:has-children:has-siblings {
    image: url("$img_chevron_right"); }
QTreeView::branch:open:has-children:!has-siblings, QTreeView::branch:open:has-children:has-siblings {
    image: url("$img_chevron_down"); }
/* table headers have no fill of their own and a stronger rule under them, like the website's tables */
QHeaderView { background: $surface; border: none; }
QHeaderView::section { background: $surface; color: $fg2; padding: 3px 6px; border: none;
    border-bottom: 1px solid $line_strong; border-right: 1px solid $line; font-weight: 400; }
QHeaderView::section:last { border-right: none; }
QHeaderView::section:hover { background: $hover; }
QHeaderView::up-arrow { image: url("$img_chevron_up"); width: 10px; height: 10px; }
QHeaderView::down-arrow { image: url("$img_chevron_down"); width: 10px; height: 10px; }
QTableCornerButton::section { background: $surface; border: none; border-bottom: 1px solid $line_strong; }

/* ---------- menus, tooltips ---------- */
QMenuBar { background: $bg; color: $fg; border: none; padding: 1px 2px; }
QMenuBar::item { background: transparent; padding: 3px 8px; border-radius: ${radius_sm}px; }
QMenuBar::item:selected { background: $hover; }
QMenuBar::item:pressed { background: $pressed; }
QMenu { background: $surface; border: 1px solid $line_strong; padding: 3px 0; }
QMenu::item { padding: 4px 24px 4px 26px; color: $fg; background: transparent; }
QMenu::item:selected { background: $primary_soft; }
QMenu::item:disabled { color: $faint; }
QMenu::icon { padding-left: 6px; }
QMenu::separator { height: 1px; background: $line; margin: 3px 0; }
QMenu::section { color: $muted; padding: 4px 26px 2px 26px; background: transparent; }
QMenu::indicator { width: 13px; height: 13px; left: 6px; }
QMenu::indicator:checked { image: url("$img_check_text"); }
QMenu::right-arrow { image: url("$img_chevron_right"); width: 10px; height: 10px; right: 6px; }
QToolTip { background: $surface; color: $fg; border: 1px solid $line_strong; padding: 3px 6px; }

/* ---------- chrome ---------- */
QToolBar { background: $bg; border: none; spacing: 2px; padding: 2px 4px; }
QToolBar::separator { background: $line; width: 1px; margin: 4px 6px; }
/* 18 px content + 2 x 2 px padding + 2 x 1 px border = 24 px tall, at least 24 px wide (WCAG 2.2
   SC 2.5.8): with the 16 px icons alone they were 22 px, the account button 35 x 22 (v3.4 audit) */
QToolBar QToolButton { padding: 2px 5px; color: $fg; min-height: 18px; min-width: 12px; }
QToolBar QToolButton:checked { background: $primary_soft; border-color: $fg; }
QStatusBar { background: $bg; color: $fg2; border-top: 1px solid $line; min-height: 22px; }
QStatusBar::item { border: none; }
QStatusBar QLabel { color: $fg2; padding: 0 8px; }
QStatusBar QLabel[sep="true"] { border-left: 1px solid $line; }
QSplitter::handle { background: $line; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }
QSplitter::handle:hover { background: $control; }
QGroupBox { background: transparent; border: none; border-top: 1px solid $line;
    margin-top: 14px; padding: 8px 0 0 0; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 0; padding: 0 6px 0 0;
    color: $fg; }
QDialogButtonBox QPushButton { min-width: 72px; }
QDockWidget { color: $fg2; titlebar-close-icon: url("$img_close"); titlebar-normal-icon: url("$img_copy"); }
QDockWidget::title { background: $bg_subtle; padding: 4px 6px; text-align: left; }
QDockWidget::close-button, QDockWidget::float-button { border: none; background: transparent; padding: 1px;
    icon-size: 12px; }
QDockWidget::close-button:hover, QDockWidget::float-button:hover { background: $hover; }
QMainWindow > QTabBar { background: $bg; }
QMainWindow > QTabBar::tab { background: $bg_subtle; color: $fg2; border: 1px solid $line;
    border-top: none; padding: 3px 12px; }
QMainWindow > QTabBar::tab:selected { background: $surface_2; color: $fg; border-bottom: 2px solid $fg; }
""")


def _font_stack(primary: str, fallback: str) -> str:
    """A QSS ``font-family`` list: the bundled face, then the platform face for missing glyphs."""
    fams = [primary] if primary == fallback else [primary, fallback]
    return ", ".join(f'"{f}"' for f in fams)


def stylesheet(name: str | None = None) -> str:
    """The complete application style sheet for a theme."""
    t = tokens(name)
    values = dict(t)
    values.pop("series")
    values.update(
        ui=font_family(), mono=mono_family(),
        ui_stack=_font_stack(font_family(), fallback_family()),
        mono_stack=_font_stack(mono_family(), mono_fallback_family()),
        img_chevron_down=_qss_image("chevron-down", t["fg2"]),
        img_chevron_down_faint=_qss_image("chevron-down", t["faint"]),
        img_chevron_up=_qss_image("chevron-up", t["fg2"]),
        img_chevron_up_faint=_qss_image("chevron-up", t["faint"]),
        img_chevron_right=_qss_image("chevron-right", t["fg2"]),
        img_check=_qss_image("check", t["primary_fg"]),
        img_check_text=_qss_image("check", t["fg"]),
        img_check_faint=_qss_image("check", t["muted"]),
        img_minus=_qss_image("minus", t["primary_fg"]),
        img_dot_accent=_qss_image("dot", t["accent"]),
        img_dot_ink=_qss_image("dot", t["fg"]),
        radius=RADIUS, radius_lg=RADIUS_LG, radius_sm=RADIUS_SM, radius_seg=RADIUS - 2,
        img_close=_qss_image("close", t["fg2"]),
        img_copy=_qss_image("copy", t["fg2"]),
    )
    return _QSS.substitute(values)


def palette(name: str | None = None) -> QPalette:
    """A QPalette matching the tokens, for the parts Qt draws natively (text cursors,
    native dialogs, item-view highlight fallbacks, rich-text links)."""
    t = tokens(name)
    pal = QPalette()
    C = QPalette.ColorRole
    G = QPalette.ColorGroup
    pal.setColor(C.Window, QColor(t["bg"]))
    pal.setColor(C.WindowText, QColor(t["fg"]))
    pal.setColor(C.Base, QColor(t["surface"]))
    pal.setColor(C.AlternateBase, QColor(t["surface_2"]))
    pal.setColor(C.Text, QColor(t["fg"]))
    pal.setColor(C.Button, QColor(t["surface"]))
    pal.setColor(C.ButtonText, QColor(t["fg"]))
    pal.setColor(C.BrightText, QColor(t["danger"]))
    pal.setColor(C.Highlight, QColor(t["primary_soft"]))
    pal.setColor(C.HighlightedText, QColor(t["fg"]))
    pal.setColor(C.ToolTipBase, QColor(t["surface"]))
    pal.setColor(C.ToolTipText, QColor(t["fg"]))
    pal.setColor(C.PlaceholderText, QColor(t["muted"]))
    # rich-text links are ink, as the website's are (v4); they keep Qt's underline
    pal.setColor(C.Link, QColor(t["fg"]))
    pal.setColor(C.LinkVisited, QColor(t["fg"]))
    pal.setColor(C.Accent, QColor(t["accent"]))
    pal.setColor(C.Light, QColor(t["surface"]))
    pal.setColor(C.Midlight, QColor(t["line"]))
    pal.setColor(C.Mid, QColor(t["line_strong"]))
    pal.setColor(C.Dark, QColor(t["line_strong"]))
    pal.setColor(C.Shadow, QColor(t["line_strong"]))
    for role in (C.WindowText, C.Text, C.ButtonText):
        pal.setColor(G.Disabled, role, QColor(t["faint"]))
    return pal


PLOT_GRID_ALPHA = 0.15


def plot_style(name: str | None = None) -> dict:
    """Publication-style plot settings derived from the tokens, for pyqtgraph pages.

    ``background`` (surface), ``axis`` (fg2, 1 px), ``text`` (axis labels),
    ``grid_alpha`` (~0.15), ``series`` (the six plot colours), ``band``
    (the light grey averaging-window fill) and ``legend_bg`` / ``legend_border``.
    """
    t = tokens(name)
    return {
        "background": t["surface"], "axis": t["fg2"], "text": t["fg2"], "grid_alpha": PLOT_GRID_ALPHA,
        "series": list(t["series"]), "band": rgba(t["muted"], 0.12),
        "legend_bg": t["surface"], "legend_border": t["line"],
    }


def configure_pyqtgraph(name: str | None = None) -> None:
    """Point pyqtgraph's global defaults at the theme (new plots pick them up).

    Plots that already exist keep their colours; pages that own plots should
    restyle them from :func:`tokens` / :func:`plot_style` in a slot connected to
    :data:`changed`.
    """
    try:
        import pyqtgraph as pg
    except ImportError:     # pragma: no cover - pyqtgraph is a hard dependency, but never crash the theme
        return
    t = tokens(name)
    pg.setConfigOptions(background=t["surface"], foreground=t["fg2"], antialias=True)


def apply(app: QApplication | None = None, name: str | None = None, *, font: str | None = None) -> None:
    """Apply a theme to the whole application and announce it on :data:`changed`.

    ``name`` is "light", "dark" or "system" (follow the operating system,
    re-applied when it changes); None keeps the current theme.

    ``font`` ("bundled" | "system") switches the interface font; None keeps
    the current choice. Uses the Fusion base style so the style sheet renders
    identically on Windows, macOS and Linux (the native styles ignore parts of
    QSS).
    """
    global _current, _font_choice, _follow_system, _scheme_hooked
    app = app or QApplication.instance()
    if name in THEME_CHOICES:
        _follow_system = name == "system"
    name = resolve(name)
    _current = name
    if font in FONT_CHOICES:
        _font_choice = font
    if app is None:
        return
    if _follow_system and not _scheme_hooked:
        # Follow the OS while "system" is chosen; the slot checks the flag, so
        # choosing Light or Dark later stops it without disconnecting.
        try:
            app.styleHints().colorSchemeChanged.connect(_on_system_scheme)
            _scheme_hooked = True
        except AttributeError:      # pragma: no cover - Qt < 6.5
            pass
    ensure_fonts()
    sheet = stylesheet(name)
    if app.property(_FUSION_PROP) is not True:
        # Once a style sheet is set, app.style() is Qt's private style-sheet
        # wrapper whose name() is "", so asking it "are you Fusion?" always
        # said no and re-ran setStyle (0.2 s of re-polishing) on every apply.
        # A flag on the application remembers it instead.
        app.setStyle("Fusion")
        app.setProperty(_FUSION_PROP, True)
    if app.styleSheet() != sheet:
        _swap_stylesheet(app, sheet, palette(name), ui_font())
    else:
        app.setFont(ui_font())
        app.setPalette(palette(name))
    _relayout_rich_text_links(app, name)
    configure_pyqtgraph(name)
    changed.emit(name)


_FUSION_PROP = "aerolumeFusionBase"

#: the theme the rich-text labels' links were last laid out in (see :func:`_relayout_rich_text_links`)
_links_theme: str | None = None


def _relayout_rich_text_links(app: QApplication, name: str) -> None:
    """Re-parse every rich-text QLabel that holds a link, so the link takes the new theme's colour.

    Why: Qt's HTML importer bakes the palette's ``Link`` colour into the label's text document
    when the text is set; a later ``setPalette`` updates ``palette().link()`` but not the
    drawn link. Measured before this pass: after light to dark the Assistant's "Script panel"
    link stayed #0e0e0e on #111111 (about 1.1:1). Setting the same text again is a no-op in
    QLabel (it compares the strings), so the text is cleared and set back. The base-class
    methods are called on purpose: Python subclasses (``ElidedLabel``) override ``setText``
    to keep a full text, which a round trip through their override would replace with the
    elided one. Runs only when the theme really changed: about 2,000 widgets, a few ms.
    """
    global _links_theme
    if name == _links_theme:
        return
    _links_theme = name
    for w in app.allWidgets():
        if not isinstance(w, QLabel) or w.textFormat() == Qt.TextFormat.PlainText:
            continue
        text = QLabel.text(w)
        if "<a " not in text:
            continue
        QLabel.setText(w, "")
        QLabel.setText(w, text)


def _swap_stylesheet(app: QApplication, sheet: str, pal: QPalette, font: QFont) -> None:
    """Replace the application style sheet in two *linear* passes instead of one quadratic one.

    Why: Qt's own update path for an application that already has a style sheet
    (``QStyleSheetStyle::repolish(qApp)``) walks every polished widget and then,
    for each one, recurses into its children again, so a widget is re-polished
    once per ancestor. Measured on the full window after visiting every page
    (1,933 widgets, 116 visible): 21,803 StyleChange and 43,600 palette / font
    change events, 2.1 s. Clearing the sheet first and setting the new one goes
    through ``QApplication::setStyle`` twice instead, which polishes each widget
    exactly once per pass: 4,294 StyleChange events, 0.27 s, the same final
    state (Qt's documented "sheet removed" and "sheet set the first time"
    paths). Nothing paints in between: there is no event-loop turn between the
    two calls. Font and palette are set while no sheet is active so they are
    propagated once, not re-resolved through the sheet for every widget.
    """
    if app.styleSheet():
        app.setStyleSheet("")
    app.setFont(font)
    app.setPalette(pal)
    app.setStyleSheet(sheet)


def toggle() -> str:
    """Switch between light and dark (stops following the OS); returns the new theme name."""
    apply(None, "light" if _current == "dark" else "dark")
    return _current


def repolish(widget: QWidget) -> None:
    """Re-evaluate the style sheet after changing a dynamic property at runtime."""
    st = widget.style()
    st.unpolish(widget)
    st.polish(widget)
    widget.update()


def set_prop(widget: QWidget, name: str, value: str) -> None:
    """Set a style-hook property (``role``, ``variant``, ...) and repolish if it changed."""
    if widget.property(name) != value:
        widget.setProperty(name, value)
        repolish(widget)


def paint_segmented_frame(widget: QWidget) -> None:
    """The outline and fill of a segmented control, painted under its segments.

    Why a paint and not the style sheet: a QSS border on the frame insets its contents by
    1 px, which clipped the 24 px segments wherever the frame was given only 24 px (see the
    ``QFrame[segmented]`` rule). ``control`` outline (>= 3:1 on every panel), ``surface`` fill,
    the website's 6 px radius.
    """
    p = QPainter(widget)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(qcolor("control"), 1.0))
    p.setBrush(qcolor("surface"))
    p.drawRoundedRect(QRectF(widget.rect()).adjusted(0.5, 0.5, -0.5, -0.5), RADIUS, RADIUS)
    p.end()


def paint_focus_ring(painter: QPainter, rect: QRectF, radius: float = RADIUS, width: float = 1.0) -> None:
    """The shared focus ring for custom-painted widgets: ``width`` px of the focus colour, just inside ``rect``.

    Widgets with a border use 1 px (the border changes colour, as the style sheet does for
    inputs). Borderless ones -- a group header is text on the panel -- use 2 px: a 1 px ring
    around text with nothing else changing was measured too faint to find in the v3.3
    keyboard audit (WCAG 2.4.7), and 2 px matches the primary button's ring.
    """
    painter.save()
    pen = QPen(qcolor("focus"))
    pen.setWidthF(width)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    inset = width / 2
    painter.drawRoundedRect(rect.adjusted(inset, inset, -inset, -inset), radius, radius)
    painter.restore()


__all__ = [
    "ALIASES",
    "BUNDLED_MONO_FAMILY",
    "BUNDLED_UI_FAMILY",
    "CONTRAST_RULES",
    "CONTROL_H",
    "DARK",
    "DEFAULT_THEME",
    "FONT_CHOICES",
    "FONT_PT",
    "FONT_PX",
    "LIGHT",
    "MIN_TARGET",
    "PALETTES",
    "RADIUS",
    "RADIUS_LG",
    "RADIUS_SM",
    "ROW_H",
    "SCRIM_ALPHA",
    "SEMANTIC_DARK",
    "SEMANTIC_LIGHT",
    "SERIES",
    "SERIES_DARK",
    "SPACING",
    "THEMES",
    "TOKEN_KEYS",
    "app_icon",
    "app_icon_pixmap",
    "app_icon_svg",
    "apply",
    "bundled_fonts_available",
    "changed",
    "configure_pyqtgraph",
    "contrast_ratio",
    "contrast_report",
    "current",
    "ensure_fonts",
    "fallback_family",
    "font_choice",
    "font_family",
    "icon",
    "icon_names",
    "is_dark",
    "mix",
    "mono_family",
    "mono_font",
    "paint_focus_ring",
    "paint_segmented_frame",
    "palette",
    "plot_style",
    "preference",
    "qcolor",
    "relative_luminance",
    "render_icon",
    "repolish",
    "resolve",
    "rgba",
    "semantic_tokens",
    "set_prop",
    "signals",
    "status_colour",
    "stylesheet",
    "system_theme",
    "toggle",
    "tokens",
    "ui_font",
    "use_mono",
]

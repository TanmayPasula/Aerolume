"""The v3 design tokens: every pairing the UI draws meets its WCAG 2.x contrast, in both themes.

The token contrast test DESIGN.md section 13 asks for. The
contrast maths is re-implemented here from the WCAG 2.2 definition rather than
taken from ``theme.contrast_ratio``, so a bug in the theme's helper cannot pass
its own test. Runs without a QApplication except where fonts are checked.
"""
from __future__ import annotations

import os
import re

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from aerolume.app import theme

HEX = re.compile(r"^#[0-9a-f]{6}$")
REQUIRED = ("bg", "bg_subtle", "surface", "surface_2", "fg", "muted", "faint", "line", "line_strong",
            "control", "primary", "primary_hover", "primary_fg", "primary_soft", "accent", "success",
            "success_fg", "success_soft", "warn", "warn_soft", "danger", "danger_fg", "danger_soft",
            "focus", "series", "field_bg", "field_grid", "field_text", "field_solid", "field_solid_line")


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    r, g, b = (_lin(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.mark.parametrize("name", theme.THEMES)
def test_design_13_token_set_is_complete_and_opaque(name):
    sem = theme.semantic_tokens(name)
    for key in REQUIRED:
        assert key in sem, key
    for key, value in sem.items():
        values = value if key == "series" else [value]
        assert all(HEX.match(v) for v in values), (key, value)
    assert len(sem["series"]) == 6 and len(set(sem["series"])) == 6


@pytest.mark.parametrize("name", theme.THEMES)
def test_every_contrast_rule_holds(name):
    sem = theme.semantic_tokens(name)
    failures = []
    for fg, bg, need, why in theme.CONTRAST_RULES:
        r = ratio(sem[fg], sem[bg])
        if r + 1e-9 < need:
            failures.append(f"{fg} on {bg}: {r:.2f} < {need} ({why})")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("name", theme.THEMES)
def test_the_rules_cover_text_controls_and_status(name):
    """The rule list itself must keep the three families the brief names."""
    rules: dict = {}
    for fg, bg, need, _ in theme.CONTRAST_RULES:      # a pair may serve twice (text, checked border):
        rules[(fg, bg)] = max(need, rules.get((fg, bg), 0))     # the strictest rule is what it meets
    for fg in ("fg", "fg2", "muted"):
        for bg in ("bg", "surface", "surface_2", "bg_subtle"):
            assert rules.get((fg, bg), 0) >= 4.5, (fg, bg)
    for bg in ("bg", "surface", "surface_2"):
        assert rules.get(("control", bg), 0) >= 3.0
        assert rules.get(("focus", bg), 0) >= 3.0
    for s in ("success", "warn", "danger"):
        assert rules.get((s, f"{s}_soft"), 0) >= 4.5
        assert rules.get((f"{s}_fg", s), 0) >= 4.5
    assert rules.get(("primary_fg", "primary"), 0) >= 4.5


@pytest.mark.parametrize("name", theme.THEMES)
def test_plot_series_are_graphical_objects_at_3_to_1(name):
    sem = theme.semantic_tokens(name)
    for c in sem["series"]:
        assert ratio(c, sem["surface"]) >= 3.0, c
        assert ratio(c, sem["field_bg"]) >= 3.0, c


@pytest.mark.parametrize("name", theme.THEMES)
def test_text_ladder_is_ordered(name):
    """fg > fg2 > muted > faint against the surface, and lines are lighter than controls."""
    sem = theme.semantic_tokens(name)
    steps = [ratio(sem[k], sem["surface"]) for k in ("fg", "fg2", "muted", "faint")]
    assert steps == sorted(steps, reverse=True), steps
    assert ratio(sem["line"], sem["surface"]) < ratio(sem["control"], sem["surface"])
    assert ratio(sem["line"], sem["surface"]) < 3.0       # decorative: never mistaken for a control edge


def _qss_rule(sheet: str, selector_start: str) -> str:
    """The declaration block of the first QSS rule whose selector list starts with ``selector_start``."""
    i = sheet.index(selector_start)
    return sheet[sheet.index("{", i) + 1:sheet.index("}", i)]


@pytest.mark.parametrize("name", theme.THEMES)
def test_style_sheet_pairs_found_by_the_v31_audit_meet_their_rule(name):
    """Three pairs the sheet drew outside the rule list (hovered links, scroll-bar handles, the
    primary button's focus ring) measured under their WCAG minimum; each now uses a token the
    rules cover, and this pins the sheet to it so a later edit cannot quietly undo the fix.

    v4 (the website's language): links are ink and hover to ``fg2`` (a text token, >= 4.5:1 on
    every background by the rules); the primary button is filled ink, so its focus ring is the
    website's offset ring, an ``fg`` border with the panel showing in a 1 px gap inside it
    (``background-clip: content``), where a plain ``fg`` border would vanish into the fill."""
    sem = theme.semantic_tokens(name)
    sheet = theme.stylesheet(name)
    link_hover = _qss_rule(sheet, 'QPushButton[variant="link"]:hover')
    assert f"color: {sem['fg2']}" in link_hover
    handle = _qss_rule(sheet, "QScrollBar::handle:vertical")
    assert f"background: {sem['control']}" in handle
    focus = _qss_rule(sheet, 'QPushButton[variant="primary"]:focus')
    assert f"border: 1px solid {sem['fg']}" in focus and "background-clip: content" in focus
    assert "padding: 1px" in focus
    for bg in ("bg", "bg_subtle", "surface", "surface_2"):
        assert ratio(sem["fg2"], sem[bg]) >= 4.5, bg
        assert ratio(sem["control"], sem[bg]) >= 3.0, bg
    for bg in ("bg", "surface", "surface_2"):
        assert ratio(sem["fg"], sem[bg]) >= 3.0, bg


@pytest.mark.parametrize("name", theme.THEMES)
def test_palette_speaks_the_websites_tokens(name):
    """v4 maps the website's CSS custom properties (website/assets/site.css) onto the app tokens.

    The site's ink, its secondary grey, the filled button and the focus colour are the app's in
    both themes; in light the side panels are the site's page colour, the panel headers its
    code-block grey, the wells its white frames and the control outline its control border. The
    chrome is neutral: no token the chrome paints has a colour cast (only the data marks, status
    colours and plot series carry hue)."""
    sem = theme.semantic_tokens(name)
    site = theme.SITE_LIGHT if name == "light" else theme.SITE_DARK
    css = (theme.RESOURCES.parents[3] / "website" / "assets" / "site.css")
    if css.exists():                     # the table in theme.py is the site's, value for value
        text = css.read_text(encoding="utf-8")
        block = text.split(":root {", 1)[1] if name == "light" else text.split(':root[data-theme="dark"] {', 1)[1]
        block = block.split("}", 1)[0]
        for key, var in (("bg", "--bg"), ("surface", "--surface"), ("code_bg", "--code-bg"), ("fg", "--fg"),
                         ("muted", "--muted"), ("line_strong", "--line-strong"), ("btn_bg", "--btn-bg"),
                         ("focus", "--focus")):
            assert f"{var}: {site[key]};" in block, (name, var)
    assert sem["fg"] == site["fg"] and sem["fg2"] == site["muted"]
    assert sem["primary"] == site["btn_bg"] and sem["focus"] == site["focus"]
    if name == "light":
        assert sem["surface_2"] == site["bg"] and sem["bg_subtle"] == site["code_bg"]
        assert sem["surface"] == site["surface"] and sem["muted"] == site["faint"]
        assert sem["control"] == site["line_strong"]
    for key in ("bg", "bg_subtle", "surface", "surface_2", "fg", "fg2", "muted", "faint", "line", "line_strong",
                "control", "control_hover", "primary", "primary_hover", "primary_pressed", "primary_fg",
                "primary_soft", "focus", "hover", "pressed", "grid", "field_bg", "field_grid", "field_text",
                "field_solid", "field_solid_line"):
        h = sem[key].lstrip("#")
        assert h[0:2] == h[2:4] == h[4:6], (name, key, sem[key])      # neutral grey


def test_theme_helper_agrees_with_the_definition():
    for a, b in (("#000000", "#ffffff"), ("#2463c4", "#eceef1"), ("#7e848f", "#2d2f34")):
        assert abs(theme.contrast_ratio(a, b) - ratio(a, b)) < 1e-9
    assert abs(ratio("#000000", "#ffffff") - 21.0) < 1e-9
    report = theme.contrast_report("light")
    assert len(report) == len(theme.CONTRAST_RULES)
    assert all(measured >= need for _fg, _bg, measured, need, _why in report)


@pytest.mark.parametrize("name", theme.THEMES)
def test_legacy_names_alias_the_v3_tokens(name):
    t = theme.tokens(name)
    for old, new in theme.ALIASES.items():
        assert t[old] == t[new], (old, new)
    for key in ("bg", "panel", "surface", "border", "border_strong", "text", "text2", "muted", "accent",
                "accent_soft", "good", "warn", "bad", "hover", "pressed", "faint", "accent2", "accent_fill",
                "accent_hover", "accent_pressed", "on_accent", "grid", "bg2", "panel2", "input", "border2",
                "plot_bg", "selection", "focus", "good_soft", "warn_soft", "bad_soft", "accent2_soft"):
        assert HEX.match(t[key]), (key, t[key])


def test_bundled_fonts_load_and_are_the_default(qapp_tokens):
    from PySide6.QtGui import QFont, QFontDatabase, QFontInfo
    assert theme.ensure_fonts()
    fams = set(QFontDatabase.families())
    assert theme.BUNDLED_UI_FAMILY in fams and theme.BUNDLED_MONO_FAMILY in fams
    assert {"Regular", "SemiBold"} <= set(QFontDatabase.styles(theme.BUNDLED_UI_FAMILY))
    theme.apply(qapp_tokens, "light", font="bundled")
    assert theme.font_family() == theme.BUNDLED_UI_FAMILY
    assert qapp_tokens.font().families()[0] == theme.BUNDLED_UI_FAMILY
    # a real SemiBold face, not a synthesised bold of Regular
    f = theme.ui_font(weight=QFont.Weight.DemiBold)
    assert QFontInfo(f).styleName() == "SemiBold"
    # the platform face is the glyph fallback for Greek, which the Latin subsets lack
    assert len(theme.ui_font().families()) == 2
    assert '"Atkinson Hyperlegible Next", "' in theme.stylesheet("light")
    theme.apply(qapp_tokens, "light", font="system")
    try:
        assert theme.font_family() != theme.BUNDLED_UI_FAMILY
        assert theme.BUNDLED_UI_FAMILY not in theme.stylesheet("light")
    finally:
        theme.apply(qapp_tokens, "light", font="bundled")


def test_font_licences_ship_with_the_fonts():
    for stem in ("AtkinsonHyperlegibleNext", "AtkinsonHyperlegibleMono"):
        lic = theme.FONT_DIR / f"OFL-{stem}.txt"
        assert lic.exists(), lic
        assert "SIL Open Font License" in lic.read_text(encoding="utf-8")
    for name in theme.BUNDLED_FONT_FILES:
        assert (theme.FONT_DIR / name).stat().st_size > 10_000, name
    readme = (theme.FONT_DIR / "README.md").read_text(encoding="utf-8")
    for name in theme.BUNDLED_FONT_FILES:
        assert name in readme, name


@pytest.fixture(scope="module")
def qapp_tokens():
    from aerolume.app.main import create_app
    return create_app(["aerolume-token-tests"])


def test_use_mono_wins_over_the_style_sheet(qapp_tokens):
    from PySide6.QtWidgets import QPlainTextEdit
    theme.apply(qapp_tokens, "light", font="bundled")
    plain = QPlainTextEdit()
    plain.setFont(theme.mono_font())            # the trap: the sheet's QWidget font rule wins over setFont
    plain.ensurePolished()
    assert plain.font().families()[0] != theme.mono_family()
    box = theme.use_mono(QPlainTextEdit())
    box.ensurePolished()
    assert box.font().families()[0] == theme.mono_family() and box.font().pointSizeF() == 9.0
    small = theme.use_mono(QPlainTextEdit(), small=True)
    small.ensurePolished()
    assert small.font().families()[0] == theme.mono_family() and small.font().pointSizeF() == 8.0


def _opaque_rgb(qicon, state) -> set[tuple[int, int, int]]:
    """RGB of the fully opaque pixels of ``qicon`` drawn at 32 px (the stroke cores, no anti-aliasing)."""
    import numpy as np
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QIcon, QImage

    img = qicon.pixmap(QSize(32, 32), QIcon.Mode.Normal, state).toImage().convertToFormat(
        QImage.Format.Format_ARGB32)
    a = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())
    px = a[:, :img.width() * 4].reshape(img.height(), img.width(), 4)       # B, G, R, A
    core = px[px[..., 3] == 255]
    return {(int(r), int(g), int(b)) for b, g, r, _a in core}


@pytest.mark.parametrize("name", theme.THEMES)
def test_checked_icons_are_ink_in_every_icon_engine(qapp_tokens, name):
    """v4: a checked chrome icon is ink (``ACTIVE_ICON_TOKEN``), whichever engine draws it.

    The app has four icon engines: the theme's bundled set, the hand-drawn glyphs of the
    parameter forms (Points, Dimensions), the live-edit tools and the 3D view's camera presets.
    Only the first applied the mapping at first, so the Geometry toolbar showed a blue checked
    "Dimensions" beside an ink checked "Grid". Unchecked icons keep the quiet ``text2``.
    """
    from PySide6.QtGui import QColor, QIcon

    from aerolume.app.widgets.draw_tools import tool_icon
    from aerolume.app.widgets.param_form import svg_icon
    from aerolume.app.widgets.view3d import preset_icon

    theme.apply(qapp_tokens, name)
    try:
        t = theme.tokens()
        ink, quiet = QColor(t["fg"]).getRgb()[:3], QColor(t["text2"]).getRgb()[:3]
        icons = {"theme": theme.icon("grid", "text2", active="accent"),
                 "glyph": svg_icon("dimension", active="accent"),
                 "draw tool": tool_icon("brush"), "3D preset": preset_icon("iso")}
        for label, ic in icons.items():
            on, off = _opaque_rgb(ic, QIcon.State.On), _opaque_rgb(ic, QIcon.State.Off)
            assert on == {ink}, (name, label, on)
            assert off == {quiet}, (name, label, off)
    finally:
        theme.apply(qapp_tokens, "light")


def test_docs_state_the_true_number_of_contrast_rules():
    """docs/ui_guidelines.md and CHANGELOG.md said "68 pairings" when CONTRAST_RULES held 73 rules (over 68
    distinct colour pairs): the counts in the docs are pinned to the code (release gate 3, 2 November 2026)."""
    import re
    from pathlib import Path
    rules = len(theme.CONTRAST_RULES)
    pairs = len({(fg, bg) for fg, bg, _need, _why in theme.CONTRAST_RULES})
    repo = Path(__file__).resolve().parent.parent
    for name in ("docs/ui_guidelines.md", "CHANGELOG.md"):
        text = " ".join((repo / name).read_text(encoding="utf-8").split())
        found = re.findall(r"(\d+) (?:measured )?rules over (\d+) colour pairings", text)
        assert found, name
        assert all((int(r), int(p)) == (rules, pairs) for r, p in found), (name, found, rules, pairs)
        assert not re.search(r"\d+ (?:measured )?pairings\b", text), name

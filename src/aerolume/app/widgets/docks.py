"""Dockable side panels with saved workspaces (Lumen's layout, on top of QDockWidget).

Every side panel of the window - Project, Properties, Messages, Assistant,
Script, Flow visualisation, Diagnostics - is a :class:`DockPanel`, and one
:class:`DockManager` per window decides where each one sits. What a user can
do with a panel (docs/ui_guidelines.md section 3b):

* **collapse** it to its header (the chevron, a double-click on the header, or
  the header menu), so the rest of the rail gets the room;
* **reorder** it within its rail (header menu: Move up / Move down, or drag the
  header; Qt's own dock drag still works and is read back);
* **move** it between the left, right and bottom rails (header menu);
* **hide** it (the close button or the header menu) and bring it back from
  View ▸ Panels (each panel also has an Alt+<n> shortcut);
* run a rail in **solo mode**: opening one panel collapses the others there;
* **resize** a rail by dragging its edge; double-clicking the gap between a rail
  and the page puts the rails back to the workspace's widths;
* toggle a panel's effect with the **switch in its header** when the panel has
  one (:meth:`DockPanel.enable_switch`): off keeps every setting, so a whole
  group can be compared on and off and restored exactly.

**Keyboard.** The header's collapse, options and hide buttons are not Tab
stops: with them in the chain, Tab crossed 29 stops of panel chrome before it
reached the Setup form (v3.4 audit, WCAG 2.4.3). F6 / Shift+F6 cycle the
window's regions (the window does that, see ``MainWindow.cycle_region``) and
land on a panel's header, where Enter or Space collapses or expands it,
Shift+F10 or the Menu key opens its options menu, Delete hides it, and Tab
goes on into the panel's content.

**Workspaces.** The arrangement (which panels show, in which rail and order,
collapsed or not, solo flags, rail sizes) is a *workspace*. Four presets ship
(:data:`WORKSPACES`): Default, Analysis, Live and Scripting. Changes are kept
in the workspace that is current - each remembers its own layout across
restarts in the settings file (``workspaces/<name>``, JSON) - and View ▸
Workspace ▸ Reset puts the preset back. :meth:`DockManager.capture` /
:meth:`DockManager.apply` are an exact round trip.

A workspace is a tool for a task, and Scripting is not a study's starting
layout: a "Heated room" study created from Scripting opened with an empty
script under a Setup page squeezed to 300 px (v3.4 audit). A new study opens in
the user's last other workspace (:meth:`DockManager.study_workspace`).

**Narrow windows.** Below :data:`COMPACT_WIDTH` the right rail has no room
beside the page, so its panels fold into the left rail as tabs; they go back
where they were once the window is wider again (the assignment is
remembered, not lost).

Between :data:`COMPACT_WIDTH` and :data:`RAIL_CAP_WIDTH` each side rail is
capped at :data:`RAIL_CAP_SHARE` (28 %) of the window width. Without the cap a
panel's content minimum decided the rail: at 1280 px the Live workspace's
right rail grew to 400 px and left the page's 3D view about 350 px wide, too
narrow for its own toolbar. A panel's explicit minimum (the Assistant's 320
px) still wins over the cap.

**Contents.** The window gives the shell's own panels their widgets once
(:meth:`DockManager.set_content`). A page may also put widgets into the
Properties, Flow visualisation and Diagnostics panels by implementing::

    def dock_panels(self) -> dict[str, QWidget]:   # keys: "properties", "flowvis", "diagnostics"

The window asks once, when the page is built; each widget is shown while that
page is current, and a panel with nothing to show for the current page is
unavailable (hidden, its View entry disabled with the reason). Any widget can
reach the panels with :func:`manager_for`, e.g. to add a header switch::

    panel = docks.manager_for(self).panel("diagnostics")
    panel.enable_switch(True, "Record diagnostics")
    panel.switchToggled.connect(self.set_recording)

Panels never float (``DockWidgetFloatable`` is off): a floating panel on a
second screen survives a restart badly and confuses the solo and compact
rules; the rails cover every layout the workspaces need.
"""
from __future__ import annotations

import contextlib
import copy
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from PySide6.QtCore import QEvent, QObject, QPoint, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QContextMenuEvent, QKeyEvent, QMouseEvent, QPainter
from PySide6.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from .common import EmptyState, IconButton, Switch, label

AREAS = ("left", "right", "bottom")
_QT_AREA = {"left": Qt.DockWidgetArea.LeftDockWidgetArea, "right": Qt.DockWidgetArea.RightDockWidgetArea,
            "bottom": Qt.DockWidgetArea.BottomDockWidgetArea}
_AREA_OF = {v: k for k, v in _QT_AREA.items()}
AREA_TEXT = {"left": "left rail", "right": "right rail", "bottom": "bottom rail"}

#: below this window width the right rail folds into the left one (tabs); back above + hysteresis
COMPACT_WIDTH = 1180
COMPACT_HYSTERESIS = 40
#: panels that need no project: they stay on screen on the Start page when wanted. The toolbar's
#: Assistant and Script toggles used to light up there while both panels were hidden, so a script
#: started from a saved Scripting workspace ran where nobody could see it.
TOOL_WINDOWS = ("assistant", "script")
#: below this window width each side rail is capped at RAIL_CAP_SHARE of the window width
RAIL_CAP_WIDTH = 1400
RAIL_CAP_SHARE = 0.28
_QWIDGETSIZE_MAX = 16777215
#: width of the gap between a side rail and the page (``QMainWindow::separator`` in theme.py)
_SEPARATOR_PX = 4


@dataclass(frozen=True)
class PanelSpec:
    """One side panel: key, title, icon, default rail, Qt object name, shortcut, what it is for.

    ``object_name`` is what QMainWindow, older settings files and tests know
    the dock by ("projectDock"); ``page_content`` marks panels whose widgets
    come from the current page (:meth:`DockManager.set_page_content`).
    """

    key: str
    title: str
    icon: str
    area: str
    object_name: str
    shortcut: str = ""
    description: str = ""
    min_width: int = 0
    page_content: bool = False


PANELS: tuple[PanelSpec, ...] = (
    PanelSpec("project", "Project", "project", "left", "projectDock", "Alt+1",
              "Designs and their simulations: what the pages work on"),
    PanelSpec("messages", "Messages", "messages", "bottom", "messagesDock", "Alt+2",
              "Planner notes, warnings, errors and the run log"),
    PanelSpec("assistant", "Assistant", "assistant", "right", "assistantDock", "Alt+3",
              "Plan, run and analyse experiments with Claude or a local model", min_width=320),
    PanelSpec("script", "Script", "code", "bottom", "scriptDock", "Alt+4",
              "Python against the scripting API, with inline figures"),
    PanelSpec("properties", "Properties", "settings", "right", "propertiesDock", "Alt+5",
              "The current step's properties, when the step puts them in a panel", page_content=True),
    PanelSpec("flowvis", "Flow visualisation", "wind", "right", "flowvisDock", "Alt+6",
              "Smoke, particles and streamlines of the flow on screen", page_content=True),
    PanelSpec("diagnostics", "Diagnostics", "gauge", "bottom", "diagnosticsDock", "Alt+7",
              "Divergence, change per step, CFL number, time step and solver residual", page_content=True),
)
SPEC = {s.key: s for s in PANELS}
KEY_OF_OBJECT = {s.object_name: s.key for s in PANELS}

#: the preset workspaces: panel -> (shown, rail, collapsed), rail order, rail sizes (px), solo rails
WORKSPACES: dict[str, dict] = {
    "Default": {
        "panels": {"project": [True, "left", False], "properties": [True, "right", False],
                   "assistant": [False, "right", False], "flowvis": [False, "right", False],
                   "messages": [True, "bottom", True], "script": [False, "bottom", False],
                   "diagnostics": [False, "bottom", False]},
        "order": {"left": ["project"], "right": ["properties", "assistant", "flowvis"],
                  "bottom": ["script", "messages", "diagnostics"]},
        "sizes": {"left": 230, "right": 340, "bottom": 170}, "solo": [],
    },
    "Analysis": {
        "panels": {"project": [True, "left", False], "properties": [True, "right", False],
                   "assistant": [True, "right", False], "flowvis": [False, "right", False],
                   "messages": [True, "bottom", False], "script": [False, "bottom", False],
                   "diagnostics": [False, "bottom", False]},
        "order": {"left": ["project"], "right": ["properties", "assistant", "flowvis"],
                  "bottom": ["messages", "script", "diagnostics"]},
        "sizes": {"left": 230, "right": 380, "bottom": 170}, "solo": ["right"],
    },
    # rev 2: Diagnostics moved from the right rail to the bottom one. Its content (the Latest list
    # beside the history plots) needs 592 px of width: in the right rail it took 596 of 1600 px and left
    # the 3D Simulate page 798 px, and at 1280 px the capped rail hid the plots behind a sideways scroll
    # bar (measured, v3.1 audit). Under the page it gets the window's width and 180 px of height.
    "Live": {
        "panels": {"project": [True, "left", False], "properties": [True, "left", False],
                   "assistant": [False, "right", False], "flowvis": [True, "right", False],
                   "messages": [True, "bottom", True], "script": [False, "bottom", False],
                   "diagnostics": [True, "bottom", False]},
        "order": {"left": ["project", "properties"], "right": ["flowvis", "assistant"],
                  "bottom": ["diagnostics", "messages", "script"]},
        "sizes": {"left": 220, "right": 330, "bottom": 180}, "solo": [], "rev": 2,
    },
    # rev 2: the bottom rail is a share of the window height, not pixels. Stored as 380 px (and
    # re-captured from whatever the rail had, e.g. 173 px after a session at 1280 x 800), the
    # Script panel showed three editor lines at 1600 x 1000 while the page kept 70 % of the
    # height (v3.3 audit).
    # rev 3: 40 %, not 54 %. At 54 % (540 of 1000 px) the Setup page above kept about 300 px: a
    # 180 px 3D preview and truncated Estimate and Planner lists (v3.4 audit). 40 % (400 px)
    # still shows a script of about a dozen lines beside its output, and the page keeps 60 %.
    "Scripting": {
        "panels": {"project": [True, "left", False], "properties": [False, "right", False],
                   "assistant": [True, "right", False], "flowvis": [False, "right", False],
                   "messages": [True, "bottom", True], "script": [True, "bottom", False],
                   "diagnostics": [False, "bottom", False]},
        "order": {"left": ["project"], "right": ["assistant", "properties", "flowvis"],
                  "bottom": ["script", "messages", "diagnostics"]},
        "sizes": {"left": 230, "right": 360, "bottom": 400}, "shares": {"bottom": 0.40},
        "solo": [], "rev": 3,
    },
}
#: a bottom rail sized by share (``"shares"`` in a workspace) is never opened below this [px].
#: 300 px keeps 40 % at 1280 x 800 (320 px); the old 360 px floor was 45 % there.
BOTTOM_SHARE_MIN = 300
#: the range a stored bottom-rail share is accepted in (anything else falls back to the preset). A
#: drag may give the Script panel more than the preset's 40 %, but never more than 60 %: above that
#: the page under it falls below its own minimum height at 1000 px and Qt stops honouring either.
_SHARE_RANGE = (0.15, 0.6)
WORKSPACE_NAMES = tuple(WORKSPACES)
DEFAULT_WORKSPACE = "Default"
#: workspaces for a tool rather than a study: a new study never opens in one (study_workspace)
TOOL_WORKSPACES = ("Scripting",)
_SETTINGS_PREFIX = "workspaces/"


def preset_rev(name: str) -> int:
    """Revision of a preset workspace (1 unless the preset says otherwise)."""
    return int(WORKSPACES.get(name, {}).get("rev", 1))


def _normalise_layout(raw: dict | None, preset: dict, *, stored: bool = False) -> dict:
    """A complete, valid layout from a stored one (unknown keys dropped, missing ones from the preset).

    A settings file is user-editable and outlives versions; a bad entry must
    fall back to the preset rather than break the window. With ``stored`` (a
    layout read back from the settings file), one saved from an older revision
    of the preset (``rev``, default 1) is dropped for the new preset: a preset
    changes because the old arrangement was measured to fail (see
    ``WORKSPACES["Live"]``), and a user who had merely opened that workspace
    once would otherwise keep the failing layout for good.
    """
    out = copy.deepcopy(preset)
    out["rev"] = int(preset.get("rev", 1))
    if not isinstance(raw, dict):
        return out
    if stored:
        try:
            raw_rev = int(raw.get("rev", 1))
        except (TypeError, ValueError):
            raw_rev = 1
        if raw_rev < out["rev"]:
            return out
    panels = raw.get("panels")
    if isinstance(panels, dict):
        for key, val in panels.items():
            if key in SPEC and isinstance(val, (list, tuple)) and len(val) == 3 and val[1] in AREAS:
                out["panels"][key] = [bool(val[0]), str(val[1]), bool(val[2])]
    order = raw.get("order")
    if isinstance(order, dict):
        for area in AREAS:
            keys = order.get(area)
            if isinstance(keys, list):
                out["order"][area] = [k for k in keys if k in SPEC]
    sizes = raw.get("sizes")
    if isinstance(sizes, dict):
        for area in AREAS:
            v = sizes.get(area)
            if isinstance(v, (int, float)) and 60 <= v <= 4000:
                out["sizes"][area] = int(v)
    shares = raw.get("shares")
    if isinstance(shares, dict) and "shares" in preset:
        v = shares.get("bottom")
        if isinstance(v, (int, float)) and _SHARE_RANGE[0] <= v <= _SHARE_RANGE[1]:
            out["shares"]["bottom"] = float(v)
    solo = raw.get("solo")
    if isinstance(solo, list):
        out["solo"] = [a for a in solo if a in AREAS]
    # every panel in exactly one order list, the one of its rail
    for area in AREAS:
        out["order"][area] = [k for k in out["order"][area] if out["panels"][k][1] == area]
    for key, (_shown, area, _c) in out["panels"].items():
        if key not in out["order"][area]:
            out["order"][area].append(key)
    return out


# ---------------------------------------------------------------------------
# the header
# ---------------------------------------------------------------------------

class DockHeader(QFrame):
    """A panel header: collapse chevron, title, optional extras, optional switch, menu, close.

    Mouse presses fall through to the dock (QDockWidget drags a panel by its
    title widget when the press is not accepted), so dragging the header
    still moves the panel. A double-click collapses or expands it; a right
    click opens the same menu as the ⋯ button. Qt lays a title widget out at
    its size-hint height, hence :meth:`sizeHint`.

    The three chrome buttons are out of the Tab chain (see the module
    docstring); the header itself is the keyboard's way in. It takes focus
    only when asked (F6 region cycling calls :meth:`focus_header`), never on a
    click, so dragging a panel never steals focus from the editor beside it.
    """

    HEIGHT = 28
    collapseRequested = Signal()
    menuRequested = Signal(QPoint)

    def __init__(self, dock: QDockWidget, title: str = "") -> None:
        super().__init__(dock)
        self.dock = dock
        self.setProperty("chrome", "dock-header")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(2)
        lay.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.collapse_button = IconButton("chevron-down", "Collapse the panel", icon_size=14)
        self.collapse_button.clicked.connect(self.collapseRequested.emit)
        lay.addWidget(self.collapse_button)
        self.title = label(title or dock.windowTitle(), "dock-title")
        self.title.setMinimumWidth(0)
        lay.addWidget(self.title)
        self._stretch = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        lay.addItem(self._stretch)
        self.switch: Switch | None = None            # the one on screen
        self._switches: list[Switch] = []
        self.menu_button = IconButton("more", "Panel options", icon_size=14)
        self.menu_button.clicked.connect(
            lambda: self.menuRequested.emit(self.menu_button.mapToGlobal(self.menu_button.rect().bottomLeft())))
        lay.addWidget(self.menu_button)
        self.close_button = IconButton("close", f"Hide the {dock.windowTitle()} panel", icon_size=14)
        self.close_button.clicked.connect(dock.close)
        lay.addWidget(self.close_button)
        for b in (self.collapse_button, self.menu_button, self.close_button):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # NoFocus keeps clicks and Tab off the header; setFocus() from F6 still works
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._name_header(title or dock.windowTitle())
        self.setFixedHeight(self.HEIGHT)
        dock.windowTitleChanged.connect(self._retitle)
        self.set_collapsed_look(False)

    def _retitle(self, text: str) -> None:
        self.title.setText(text)
        self.close_button.setToolTip(f"Hide the {text} panel")
        self.close_button.setAccessibleName(f"Hide the {text} panel")
        self._name_header(text)

    def _name_header(self, text: str) -> None:
        self.setAccessibleName(f"{text} panel")
        self.setAccessibleDescription("Enter collapses or expands, Shift+F10 opens the panel options, "
                                      "Delete hides the panel, Tab moves into it")

    # -- keyboard ----------------------------------------------------------------
    def focus_header(self) -> None:
        """Give the header keyboard focus (F6): it shows a focus ring and takes the keys below."""
        self.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.update()

    def keyPressEvent(self, e: QKeyEvent) -> None:
        key, mods = e.key(), e.modifiers()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.collapseRequested.emit()
        elif key == Qt.Key.Key_Menu or (key == Qt.Key.Key_F10 and mods & Qt.KeyboardModifier.ShiftModifier):
            self.menuRequested.emit(self.mapToGlobal(self.menu_button.geometry().bottomLeft()))
        elif key == Qt.Key.Key_Delete:
            self.dock.close()
        else:
            super().keyPressEvent(e)
            return
        e.accept()

    def focusInEvent(self, e) -> None:
        super().focusInEvent(e)
        self.update()

    def focusOutEvent(self, e) -> None:
        super().focusOutEvent(e)
        self.update()

    def paintEvent(self, e) -> None:
        super().paintEvent(e)
        if self.hasFocus():
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            # 2 px: the header has no border of its own to recolour (theme.paint_focus_ring)
            theme.paint_focus_ring(p, QRectF(self.rect()).adjusted(1, 1, -1, -1), theme.RADIUS, 2.0)
            p.end()

    def sizeHint(self) -> QSize:
        return QSize(super().sizeHint().width(), self.HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return QSize(super().minimumSizeHint().width(), self.HEIGHT)

    def set_collapsed_look(self, collapsed: bool) -> None:
        self.collapse_button.set_icon("chevron-right" if collapsed else "chevron-down")
        name = self.dock.windowTitle()
        tip = f"Expand the {name} panel" if collapsed else f"Collapse the {name} panel"
        self.collapse_button.setToolTip(tip)
        self.collapse_button.setAccessibleName(tip)
        theme.set_prop(self, "collapsed", "true" if collapsed else "false")

    # -- extra widgets (the Messages filters live here) ------------------------------
    def add_widget(self, w: QWidget) -> QWidget:
        """Put a compact widget (filters, a tool button) in the header, after the title."""
        lay = self.layout()
        lay.insertSpacing(lay.indexOf(self.title) + 1, 12)
        lay.insertWidget(lay.indexOf(self.title) + 2, w)
        return w

    def add_expanding(self, w: QWidget) -> QWidget:
        """Give ``w`` the free width of the row, ahead of the stretch before the buttons.

        Used for the newest-message line of a collapsed Messages panel: it takes
        whatever room the counters leave and elides rather than pushing the
        buttons off the row. The stretch (factor 0) stays, so the buttons keep
        to the right edge while ``w`` is hidden.
        """
        lay = self.layout()
        i = lay.indexOf(self._stretch)
        lay.insertSpacing(i, 12)
        lay.insertWidget(i + 1, w, 1)
        lay.insertSpacing(i + 2, 8)
        return w

    def add_button(self, b: QWidget) -> QWidget:
        """Put a small button just before the menu button."""
        lay = self.layout()
        lay.insertWidget(lay.indexOf(self.menu_button), b)
        return b

    def set_switch(self, switch: Switch | None) -> None:
        """Show ``switch`` before the menu button (None shows none); others stay, hidden.

        A panel can hold one switch per content (the window's Diagnostics and
        the Simulate page's own): the header shows the one of the content on
        screen. :meth:`remove_switch` deletes one.
        """
        if switch is not None and switch not in self._switches:
            lay = self.layout()
            lay.insertWidget(lay.indexOf(self.menu_button), switch)
            self._switches.append(switch)
        for sw in self._switches:
            sw.setVisible(sw is switch)
        self.switch = switch

    def remove_switch(self, switch: Switch) -> None:
        if switch in self._switches:
            self._switches.remove(switch)
        if switch is self.switch:
            self.switch = None
        switch.setParent(None)
        switch.deleteLater()

    # -- mouse -------------------------------------------------------------------
    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.collapseRequested.emit()
            e.accept()
            return
        super().mouseDoubleClickEvent(e)

    def contextMenuEvent(self, e: QContextMenuEvent) -> None:
        self.menuRequested.emit(e.globalPos())
        e.accept()


# ---------------------------------------------------------------------------
# the panel
# ---------------------------------------------------------------------------

class _RailScroll(QScrollArea):
    """The panel content's holder: transparent, and scrolls sideways only when the rail is capped.

    Qt never lets a dock narrower than its content's minimum width, so a cap
    (:meth:`DockManager.cap_rails`) on a rail holding a 400 px-minimum panel
    did nothing (measured: the Live workspace's Diagnostics stayed 403 px at
    1280 px). While :attr:`cap` is set the holder reports at most the cap as
    its minimum width and the content scrolls horizontally instead of being
    clipped; with no cap it reports the content's own minimum, exactly as
    before. Vertical sizing is always the content's (no vertical scroll bar:
    each panel scrolls its own lists).
    """

    def __init__(self, inner: QWidget) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Not a Tab stop: the panel's own widgets take focus (and are scrolled into view). As a
        # stop it had no name and was announced as nothing (v3.3 keyboard audit, WCAG 4.1.2).
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setWidget(inner)
        self.viewport().setAutoFillBackground(False)
        self.cap: int | None = None
        inner.installEventFilter(self)
        if isinstance(inner, QStackedWidget):
            inner.currentChanged.connect(lambda _i: self.updateGeometry())

    def eventFilter(self, obj, e) -> bool:
        # A scroll area's size hint normally ignores its content, so Qt never tells the layout
        # above when the content's minimum changes; this one reports the content's, so it must
        # pass the change on, or the dock keeps the minimum of an empty stack (measured: an open
        # Diagnostics panel squeezed to its 31 px header at 1280 x 800).
        if obj is self.widget() and e.type() == QEvent.Type.LayoutRequest:
            self.updateGeometry()
        return super().eventFilter(obj, e)

    def set_cap(self, cap: int | None) -> None:
        if cap != self.cap:
            self.cap = cap
            self.updateGeometry()

    def _content_min(self) -> QSize:
        return self.widget().minimumSizeHint().expandedTo(self.widget().minimumSize())

    def minimumSizeHint(self) -> QSize:
        m = self._content_min()
        if self.cap is None or m.width() <= self.cap:
            return m
        return QSize(self.cap, m.height() + self.horizontalScrollBar().sizeHint().height())

    def sizeHint(self) -> QSize:
        hint = self.widget().sizeHint().expandedTo(self._content_min())
        if self.cap is not None and hint.width() > self.cap:
            return QSize(self.cap, hint.height() + self.horizontalScrollBar().sizeHint().height())
        return hint


class DockPanel(QDockWidget):
    """One side panel: a :class:`DockHeader` over a content stack (global and per-page widgets).

    ``collapsedChanged(bool)`` reports a collapse or expand; ``switchToggled(bool)``
    the header switch. ``collapse_handler`` (a callable taking the wanted
    collapsed state) lets the window take over collapsing for a panel with its
    own rules - the Messages log keeps per-window-size heights.
    """

    collapsedChanged = Signal(bool)
    switchToggled = Signal(bool)

    def __init__(self, spec: PanelSpec, parent: QWidget | None = None) -> None:
        super().__init__(spec.title, parent)
        self.spec = spec
        self.setObjectName(spec.object_name)
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable
                         | QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.header = DockHeader(self, spec.title)
        self.setTitleBarWidget(self.header)
        self.frame = QFrame()
        self.frame.setProperty("chrome", "dock")
        fl = QVBoxLayout(self.frame)
        fl.setContentsMargins(1, 0, 1, 1)
        fl.setSpacing(0)
        self.stack = QStackedWidget()
        self.scroll = _RailScroll(self.stack)
        fl.addWidget(self.scroll)
        self.setWidget(self.frame)
        #: the minimum width asked for in code (spec or window); Qt also writes the content's minimum
        #: into minimumWidth(), which a rail cap must be able to override
        self.declared_min_width = 0
        if spec.min_width:
            self.setMinimumWidth(spec.min_width)
        # Qt's dock layout hands a rail's spare height only to items that expand vertically: with the
        # default Preferred policy a lone panel expanded after a collapse kept its minimum height and
        # left the rest of the rail empty
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._global: QWidget | None = None
        self._pages: dict[str, QWidget] = {}
        self._page = ""
        self._collapsed = False
        self._expanded_height = 0
        self._switches: dict[int, tuple[Switch, QObject | None]] = {}   # content id -> (switch, mirror)
        self.collapse_handler: Callable[[bool], None] | None = None
        self.setToolTip("")
        self.header.setToolTip(spec.description)

    def setMinimumWidth(self, w: int) -> None:      # Qt override: remember what code asked for
        self.declared_min_width = int(w)
        super().setMinimumWidth(w)

    # -- content -------------------------------------------------------------------
    def set_content(self, w: QWidget) -> QWidget:
        """The panel's own widget (shown on every page that does not put its own here)."""
        if self._global is not None and self._global is not w:
            self.stack.removeWidget(self._global)
        self._global = w
        if self.stack.indexOf(w) < 0:
            self.stack.addWidget(w)
        self._sync_stack()
        return w

    def set_page_content(self, page: str, w: QWidget | None) -> None:
        """What the panel shows while ``page`` is current (None clears it)."""
        old = self._pages.pop(page, None)
        if old is not None and old is not w:
            self.stack.removeWidget(old)
        if w is not None:
            self._pages[page] = w
            if self.stack.indexOf(w) < 0:
                self.stack.addWidget(w)
        self._sync_stack()

    def show_page(self, page: str) -> None:
        self._page = page
        self._sync_stack()

    def _sync_stack(self) -> None:
        w = self.content()
        if w is not None:
            self.stack.setCurrentWidget(w)
        if hasattr(self, "_switches"):
            self._sync_switch()

    def content(self) -> QWidget | None:
        """The widget shown now: the current page's, else the panel's own, else None."""
        return self._pages.get(self._page) or self._global

    def available(self) -> bool:
        """Whether the panel has something to show on the current page."""
        return self.content() is not None

    def unavailable_reason(self) -> str:
        if self.spec.page_content:
            return f"This step has no {self.spec.title.lower()} panel"
        return f"The {self.spec.title} panel is not part of this build"

    # -- collapse --------------------------------------------------------------------
    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool, *, emit: bool = True) -> None:
        """Fold the panel to its header row (the rail gives the room to its neighbours) or open it.

        The content is squeezed to 0 px rather than hidden: QDockWidget lays its
        title widget out at the size-hint width when the content is hidden,
        which crushed the header's extras (the Messages counters) to a few
        characters.
        """
        collapsed = bool(collapsed)
        if collapsed == self._collapsed:
            self.header.set_collapsed_look(collapsed)
            return
        if collapsed and self.isVisible() and self.height() > DockHeader.HEIGHT + 20:
            self._expanded_height = self.height()
        self._collapsed = collapsed
        self.scroll.setVisible(not collapsed)
        self.frame.setMinimumHeight(0)
        self.frame.setMaximumHeight(0 if collapsed else 16777215)
        theme.set_prop(self.frame, "collapsed", "true" if collapsed else "false")
        if collapsed:
            self.setMinimumHeight(0)
            self.setMaximumHeight(DockHeader.HEIGHT)
        else:
            self.setMaximumHeight(16777215)
            # the folded state left an explicit 31 px minimum behind (Qt keeps the minimum it
            # clamped to the header height), which then overrode the content's minimum: an
            # opened Diagnostics panel was squeezed back to its header row at 1280 x 800
            self.setMinimumHeight(0)
        self.header.set_collapsed_look(collapsed)
        if emit:
            self.collapsedChanged.emit(collapsed)

    def expanded_height(self) -> int:
        """The height the panel had before it was last collapsed (0 when unknown)."""
        return self._expanded_height

    def request_collapse(self, collapsed: bool) -> None:
        """A user's collapse / expand: through the window's handler when it has one."""
        if self.collapse_handler is not None:
            self.collapse_handler(bool(collapsed))
        else:
            self.set_collapsed(collapsed)

    # -- header switch ---------------------------------------------------------------
    def enable_switch(self, checked: bool = True, tooltip: str = "",
                      on_toggled: Callable[[bool], None] | None = None, *,
                      content: QWidget | None = None) -> Switch:
        """Show an on/off switch in the header for the panel's effect (Lumen's A/B switch).

        Off must keep every value of the panel, so turning it on again gives the
        exact state back. Returns the switch; ``switchToggled`` also reports it.

        The switch belongs to ``content`` - by default the content on screen
        when it is enabled (a page calls this from ``on_shown``) - and the
        header shows it only while that content is current: the Simulate page's
        Record switch never stands in the header over the window's own
        Diagnostics. Enabling again for the same content replaces its switch.
        When ``on_toggled`` is a button's ``setChecked``, :meth:`set_switch_checked`
        called from that button's signal sets this switch, whichever is on screen.
        """
        key = content if content is not None else self.content()
        old = self._switches.pop(self._key(key), None)
        if old is not None:
            self.header.remove_switch(old[0])
        sw = Switch(checked, tooltip or f"Turn the {self.spec.title.lower()} on or off")
        sw.toggled.connect(self.switchToggled.emit)
        if on_toggled is not None:
            sw.toggled.connect(on_toggled)
        mirror = getattr(on_toggled, "__self__", None)
        self._switches[self._key(key)] = (sw, mirror if isinstance(mirror, QObject) else None)
        self._sync_switch()
        return sw

    @staticmethod
    def _key(w: QWidget | None) -> int:
        return 0 if w is None else id(w)

    def _current_switch(self) -> Switch | None:
        entry = self._switches.get(self._key(self.content())) or self._switches.get(0)
        return entry[0] if entry is not None else None

    def _sync_switch(self) -> None:
        self.header.set_switch(self._current_switch())

    def disable_switch(self, content: QWidget | None = None) -> None:
        """Remove the switch of ``content`` (default: the content on screen)."""
        entry = self._switches.pop(self._key(content if content is not None else self.content()), None)
        if entry is not None:
            self.header.remove_switch(entry[0])
        self._sync_switch()

    def switch_checked(self) -> bool | None:
        """The state of the switch on screen (None when the current content has none)."""
        sw = self._current_switch()
        return None if sw is None else sw.isChecked()

    def set_switch_checked(self, on: bool) -> None:
        """Set the switch on screen - or, as a slot of a mirrored button, that button's switch."""
        src = self.sender()
        sw = None
        if src is not None:
            sw = next((s for s, m in self._switches.values() if m is src), None)
        sw = sw or self._current_switch()
        if sw is not None:
            sw.setChecked(bool(on))


# ---------------------------------------------------------------------------
# the manager
# ---------------------------------------------------------------------------

_MANAGERS: list[DockManager] = []


def manager_for(widget: QWidget | None) -> DockManager | None:
    """The dock manager of the window ``widget`` is in (None outside a managed window).

    Falls back to the only manager when the widget is not parented yet (a page
    being built), which is the common case for a single-window app.
    """
    win = widget.window() if widget is not None else None
    for m in _MANAGERS:
        try:
            if m.window is win:
                return m
        except RuntimeError:
            continue
    alive = []
    for m in _MANAGERS:
        with contextlib.suppress(RuntimeError):
            m.window.objectName()
            alive.append(m)
    return alive[-1] if len(alive) == 1 else None


class DockManager(QObject):
    """Owns the panels of one QMainWindow: placement, visibility, collapse, solo, workspaces.

    ``settings`` is an object with ``value(key, default)`` / ``set_value(key, v)``
    (``app.settings.Settings``); None keeps workspaces in memory only (tests).
    The manager's model - which panel is wanted, its rail, order and collapse
    state - is the source of truth; Qt's dock layout is rebuilt from it.
    """

    layoutChanged = Signal()
    workspaceChanged = Signal(str)
    panelVisibilityChanged = Signal(str, bool)      # key, shown now
    switchToggled = Signal(str, bool)               # key, on

    def __init__(self, window: QMainWindow, settings=None, *, persist: bool = True) -> None:
        super().__init__(window)
        self.window = window
        self.settings = settings
        self.persist = persist and settings is not None
        self.panels: dict[str, DockPanel] = {}
        self.actions: dict[str, QAction] = {}
        self._wanted: dict[str, bool] = {}
        self._area: dict[str, str] = {}
        self._order: dict[str, list[str]] = {a: [] for a in AREAS}
        self._solo: set[str] = set()
        self._sizes: dict[str, int] = {}
        #: rails sized as a share of the window height (``"shares"`` of the workspace), and the
        #: bottom rail's size when a mouse press began (a drag of its edge sets a new share)
        self._shares: dict[str, float] = {}
        self._press_bottom: int | None = None
        self._shown = True                  # False on the Start page: every panel steps aside
        self._page = ""
        self._compact = False
        self._folded: list[str] = []        # right-rail panels folded into the left one (compact)
        self._applying = 0
        self.workspace = DEFAULT_WORKSPACE
        #: the last workspace that is not a tool workspace (a new study opens there, study_workspace)
        self.last_study_workspace = DEFAULT_WORKSPACE
        self._layouts: dict[str, dict] = {n: copy.deepcopy(p) for n, p in WORKSPACES.items()}
        if self.persist:
            for n in WORKSPACE_NAMES:
                self._layouts[n] = self._load(n)
            name = str(self.settings.value(_SETTINGS_PREFIX + "current", DEFAULT_WORKSPACE) or "")
            self.workspace = name if name in WORKSPACES else DEFAULT_WORKSPACE
            last = str(self.settings.value(_SETTINGS_PREFIX + "last_study", "") or "")
            if last in WORKSPACES and last not in TOOL_WORKSPACES:
                self.last_study_workspace = last
        if self.workspace not in TOOL_WORKSPACES:
            self.last_study_workspace = self.workspace
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self._remember_layout)
        for spec in PANELS:
            self._add_panel(spec)
        self._apply_layout(self._layouts[self.workspace], resize=False)
        window.installEventFilter(self)
        _MANAGERS.append(self)
        self.destroyed.connect(lambda *_: _MANAGERS.remove(self) if self in _MANAGERS else None)

    # -- construction -------------------------------------------------------------------
    def _add_panel(self, spec: PanelSpec) -> DockPanel:
        p = DockPanel(spec, self.window)
        self.panels[spec.key] = p
        self.window.addDockWidget(_QT_AREA[spec.area], p)
        p.hide()
        p.header.collapseRequested.connect(lambda k=spec.key: self.toggle_collapsed(k, user=True))
        p.header.menuRequested.connect(lambda pos, k=spec.key: self.panel_menu(k).exec(pos))
        p.header.close_button.clicked.connect(lambda _c=False, k=spec.key: self.hide_panel(k))
        p.collapsedChanged.connect(lambda c, k=spec.key: self._collapsed_changed(k, c))
        p.switchToggled.connect(lambda on, k=spec.key: self.switchToggled.emit(k, on))
        p.dockLocationChanged.connect(lambda a, k=spec.key: self._dock_moved(k, a))
        a = QAction(f"{spec.title} panel", self.window)
        a.setCheckable(True)
        if spec.shortcut:
            a.setShortcut(spec.shortcut)
        a.setStatusTip(spec.description)
        a.setToolTip(spec.description)
        a.triggered.connect(lambda checked, k=spec.key: self.set_wanted(k, checked, raise_=True))
        self.window.addAction(a)
        self.actions[spec.key] = a
        return p

    def panel(self, key: str) -> DockPanel:
        return self.panels[key]

    def keys(self) -> list[str]:
        return list(self.panels)

    def set_content(self, key: str, w: QWidget) -> QWidget:
        """Give a panel its own widget (the shell's Project tree, Messages, Assistant, Script)."""
        self.panels[key].set_content(w)
        self._apply_visibility()
        return w

    def set_page_content(self, page: str, key: str, w: QWidget | None) -> None:
        """What a page shows in the Properties, Flow visualisation or Diagnostics panel."""
        self.panels[key].set_page_content(page, w)
        self._apply_visibility()

    def adopt_page_panels(self, page_key: str, page: QWidget) -> list[str]:
        """Ask a page for its ``dock_panels()`` and install them; returns the keys it filled."""
        fn = getattr(page, "dock_panels", None)
        if not callable(fn):
            return []
        try:
            panels = fn() or {}
        except Exception:  # noqa: BLE001 - a page bug must not break the window's layout
            return []
        filled = []
        for key, w in dict(panels).items():
            if key in self.panels and isinstance(w, QWidget):
                self.panels[key].set_page_content(page_key, w)
                filled.append(key)
        self._apply_visibility()
        return filled

    def show_page(self, page: str) -> None:
        """The window switched pages: panels show the page's widgets (or step aside)."""
        self._page = page
        for p in self.panels.values():
            p.show_page(page)
        self._apply_visibility()
        self.cap_rails()                    # the page guard depends on the page's minimum width

    # -- visibility ----------------------------------------------------------------------
    def wanted(self, key: str) -> bool:
        """Whether the user wants the panel shown (it may still be hidden: Start page, no content)."""
        return self._wanted.get(key, False)

    def is_shown(self, key: str) -> bool:
        return not self.panels[key].isHidden()

    def set_wanted(self, key: str, on: bool, *, raise_: bool = False) -> None:
        """Show or hide a panel at the user's request (remembered in the workspace)."""
        self._wanted[key] = bool(on)
        if on and key in self._folded and not self._compact:
            self._folded.remove(key)
        self._apply_visibility()
        p = self.panels[key]
        if on and raise_ and not p.isHidden():
            if p.is_collapsed():
                self.set_collapsed(key, False, user=True)
            p.raise_()
        self._changed()

    def show_panel(self, key: str, *, raise_: bool = True) -> None:
        self.set_wanted(key, True, raise_=raise_)

    def hide_panel(self, key: str) -> None:
        self.set_wanted(key, False)

    def set_workspace_visible(self, visible: bool) -> None:
        """Hide the project panels (the Start page) or bring back the user's choice.

        The :data:`TOOL_WINDOWS` (Assistant, Script) keep the user's choice even
        then: neither needs a project, and a lit toolbar toggle must mean a
        panel on screen.
        """
        self._shown = bool(visible)
        self._apply_visibility()

    def workspace_visible(self) -> bool:
        return self._shown

    def _apply_visibility(self) -> None:
        self._applying += 1
        try:
            for key, p in self.panels.items():
                show = ((self._shown or key in TOOL_WINDOWS) and self._wanted.get(key, False)
                        and p.available())
                if show != (not p.isHidden()):
                    p.setVisible(show)
                    self.panelVisibilityChanged.emit(key, show)
                a = self.actions.get(key)
                if a is not None:
                    a.setChecked(self._wanted.get(key, False))
                    a.setEnabled(p.available())
                    tip = p.spec.description if p.available() else p.unavailable_reason()
                    a.setToolTip(tip)
                    a.setStatusTip(tip)
        finally:
            self._applying -= 1

    # -- collapse and solo -----------------------------------------------------------------
    def is_collapsed(self, key: str) -> bool:
        return self.panels[key].is_collapsed()

    def set_collapsed(self, key: str, collapsed: bool, *, user: bool = False) -> None:
        p = self.panels[key]
        if user:
            p.request_collapse(collapsed)
        else:
            p.set_collapsed(collapsed)

    def toggle_collapsed(self, key: str, *, user: bool = False) -> None:
        self.set_collapsed(key, not self.is_collapsed(key), user=user)

    def _collapsed_changed(self, key: str, collapsed: bool) -> None:
        p = self.panels[key]
        area = self._area.get(key, p.spec.area)
        if not collapsed:
            if area in self._solo and not self._applying:
                for other in self._visible_order(area):
                    if other != key and not self.panels[other].is_collapsed():
                        self.panels[other].request_collapse(True)
            if p.collapse_handler is None and p.isVisible():
                QTimer.singleShot(0, self, lambda: self._restore_height(key))
        self._changed()

    def _restore_height(self, key: str) -> None:
        """Give an expanded panel its height back.

        Qt's dock layout remembers the 28 px a collapsed panel had as the
        item's size; a panel alone in a side rail then stays at its minimum
        height however the window or ``resizeDocks`` is told (measured: a lone
        dock re-expanded to 134 of 766 px in plain Qt). Re-adding the dock to
        its rail drops the stale size, and its width is put back after. With
        neighbours, a vertical ``resizeDocks`` to the remembered height works.
        """
        p = self.panels[key]
        if not p.isVisible() or p.is_collapsed():
            return
        area = self.area_of(key)
        side = area in ("left", "right") and not (self._compact and key in self._folded)
        visible = [k for k in self._visible_order(area) if not self.panels[k].isHidden()]
        if side and visible == [key]:
            width = p.width()
            self._applying += 1
            try:
                self.window.addDockWidget(_QT_AREA[area], p, Qt.Orientation.Vertical)
            finally:
                self._applying -= 1
            QTimer.singleShot(0, self, lambda: self.window.resizeDocks([p], [width], Qt.Orientation.Horizontal))
            return
        h = p.expanded_height()
        if h:
            self.window.resizeDocks([p], [h], Qt.Orientation.Vertical)

    def is_solo(self, area: str) -> bool:
        return area in self._solo

    def set_solo(self, area: str, on: bool) -> None:
        """Solo mode for a rail: at most one expanded panel there (the first one keeps open)."""
        if on:
            self._solo.add(area)
            opened = [k for k in self._visible_order(area) if not self.panels[k].is_collapsed()]
            for k in opened[1:]:
                self.panels[k].request_collapse(True)
        else:
            self._solo.discard(area)
        self._changed()

    # -- placement -------------------------------------------------------------------------
    def area_of(self, key: str) -> str:
        return self._area.get(key, SPEC[key].area)

    def order(self, area: str) -> list[str]:
        """The panels of a rail, top to bottom (hidden ones included)."""
        return list(self._order[area])

    def move_to(self, key: str, area: str, index: int | None = None) -> None:
        """Put a panel in another rail (at ``index`` in its order; default: the end)."""
        if area not in AREAS:
            raise ValueError(f"unknown rail {area!r}")
        old = self.area_of(key)
        if key in self._order[old]:
            self._order[old].remove(key)
        dest = self._order[area]
        dest.insert(len(dest) if index is None else max(0, min(index, len(dest))), key)
        self._area[key] = area
        if key in self._folded:
            self._folded.remove(key)
        self._relayout({old, area})
        self._changed()

    def move_by(self, key: str, step: int) -> None:
        """Move a panel up (-1) or down (+1) within its rail."""
        area = self.area_of(key)
        order = self._order[area]
        i = order.index(key)
        j = max(0, min(len(order) - 1, i + step))
        if i == j:
            return
        order.insert(j, order.pop(i))
        self._relayout({area})
        self._changed()

    def _visible_order(self, area: str) -> list[str]:
        return [k for k in self._order[area] if self._wanted.get(k) and self.panels[k].available()]

    def _relayout(self, areas: Iterable[str]) -> None:
        """Rebuild Qt's stacking of the given rails from the model (vertical stacks, model order)."""
        self._applying += 1
        try:
            for area in areas:
                keys = self._order[area]
                if self._compact and area == "left":
                    keys = [k for k in keys if k not in self._folded]
                prev = None
                for k in keys:
                    p = self.panels[k]
                    if prev is None:
                        self.window.addDockWidget(_QT_AREA[area], p, Qt.Orientation.Vertical)
                    else:
                        self.window.splitDockWidget(self.panels[prev], p, Qt.Orientation.Vertical)
                    prev = k
            if self._compact:
                self._fold_into_left()
            self._apply_visibility()
            self.cap_rails()
        finally:
            self._applying -= 1

    def _dock_moved(self, key: str, qt_area) -> None:
        """A native drag moved a panel: read the new rail and order back into the model."""
        if self._applying or qt_area not in _AREA_OF:
            return
        area = _AREA_OF[qt_area]
        old = self.area_of(key)
        if key in self._order[old]:
            self._order[old].remove(key)
        self._area[key] = area
        self._order[area].append(key)
        self.cap_rails()
        QTimer.singleShot(0, self, lambda: self._sync_order_from_geometry(area))
        self._changed()

    def _sync_order_from_geometry(self, area: str) -> None:
        def pos(k: str) -> tuple[int, int]:
            p = self.panels[k]
            return (p.y(), p.x()) if not p.isHidden() else (10 ** 6, 0)
        self._order[area].sort(key=pos)

    # -- rail sizes -------------------------------------------------------------------------
    def rail_size(self, area: str) -> int:
        """Width (left / right) or height (bottom) of a rail now, from its first visible panel."""
        keys = self._visible_order(area)
        if not keys:
            return self._sizes.get(area, 0)
        p = self.panels[keys[0]]
        return p.width() if area in ("left", "right") else sum(
            self.panels[k].height() for k in keys if not self.panels[k].is_collapsed())

    def bottom_share(self) -> float | None:
        """The share of the window height the current workspace gives its bottom rail (None: pixels)."""
        return self._shares.get("bottom")

    def rail_target(self, area: str) -> int | None:
        """The size the workspace asks for a rail now: its share of the window height, or its pixels."""
        share = self._shares.get(area)
        if share is not None and area == "bottom" and self.window.height() > 0:
            return max(BOTTOM_SHARE_MIN, round(share * self.window.height()))
        return self._sizes.get(area)

    def set_rail_size(self, area: str, size: int) -> None:
        keys = [k for k in self._visible_order(area) if not self.panels[k].isHidden()]
        if not keys:
            self._sizes[area] = int(size)
            return
        if area in ("left", "right"):
            self.window.resizeDocks([self.panels[keys[0]]], [int(size)], Qt.Orientation.Horizontal)
        else:
            open_keys = [k for k in keys if not self.panels[k].is_collapsed()]
            if open_keys:
                self.window.resizeDocks([self.panels[open_keys[0]]], [int(size)], Qt.Orientation.Vertical)
        self._sizes[area] = int(size)

    def reset_rail_sizes(self) -> None:
        """The rails back to the current workspace's preset widths (double-click on a rail edge)."""
        preset = WORKSPACES[self.workspace]["sizes"]
        for area in AREAS:
            if area != "bottom":
                self.set_rail_size(area, preset[area])
        self._changed()

    # -- workspaces ---------------------------------------------------------------------------
    def capture(self) -> dict:
        """The current arrangement as a workspace layout (JSON-safe)."""
        panels = {k: [bool(self._wanted.get(k, False)), self.area_of(k), bool(self.panels[k].is_collapsed())]
                  for k in self.panels}
        sizes = dict(self._layouts.get(self.workspace, WORKSPACES[DEFAULT_WORKSPACE])["sizes"])
        if self.window.isVisible() and self._shown:
            for area in AREAS:
                if any(not self.panels[k].isHidden() for k in self._order[area]):
                    v = self.rail_size(area)
                    if v >= 60:
                        sizes[area] = int(v)
        out = {"panels": panels, "order": {a: list(self._order[a]) for a in AREAS}, "sizes": sizes,
               "solo": sorted(self._solo), "rev": preset_rev(self.workspace)}
        if self._shares:
            # A share-sized rail is not learnt from its geometry: a page's minimum height squeezes
            # it in a small window, and that squeezed size would stick. Only a drag sets a share.
            out["shares"] = dict(self._shares)
            if "bottom" in self._shares:
                out["sizes"]["bottom"] = int(WORKSPACES[self.workspace]["sizes"]["bottom"])
        return out

    def apply(self, layout: dict, *, resize: bool = True) -> None:
        """Arrange the panels as ``layout`` says (a :meth:`capture` result or a preset)."""
        self._apply_layout(_normalise_layout(layout, WORKSPACES.get(self.workspace, WORKSPACES["Default"])),
                           resize=resize)
        self._changed(save=False)

    def _apply_layout(self, layout: dict, *, resize: bool = True) -> None:
        self._applying += 1
        try:
            if self._compact:
                self._unfold()
            for key, (shown, area, collapsed) in layout["panels"].items():
                self._wanted[key] = shown
                self._area[key] = area
            self._order = {a: list(layout["order"][a]) for a in AREAS}
            self._solo = set(layout.get("solo", []))
            self._sizes = dict(layout["sizes"])
            self._shares = dict(layout.get("shares") or {})
            for key, (_shown, _area, collapsed) in layout["panels"].items():
                p = self.panels[key]
                if p.collapse_handler is not None:
                    p.collapse_handler(collapsed)
                else:
                    p.set_collapsed(collapsed, emit=False)
            # placed after the fold states: re-adding every dock drops the stale item sizes Qt keeps for
            # a panel that was collapsed in the previous workspace (see _restore_height)
            self._relayout(AREAS)
            if self._compact or (self.window.isVisible() and self.window.width() < COMPACT_WIDTH):
                self._compact = False
                self.update_compact(self.window.width())
        finally:
            self._applying -= 1
        if resize:
            QTimer.singleShot(0, self, self._apply_sizes)

    def _apply_sizes(self) -> None:
        for area in AREAS:
            target = self.rail_target(area)
            if target is not None:
                keep = self._sizes.get(area)
                self.set_rail_size(area, target)
                if area in self._shares and keep is not None:
                    self._sizes[area] = keep        # the stored pixels stay the preset's

    def apply_sizes_now(self) -> None:
        """Size the rails to the workspace's widths (the window calls it once it is laid out)."""
        self._apply_sizes()

    def set_workspace(self, name: str) -> None:
        """Switch workspaces: the current arrangement is kept in the old one, the new one is applied."""
        if name not in WORKSPACES:
            raise ValueError(f"unknown workspace {name!r}")
        if name == self.workspace:
            return
        self._remember_layout()
        self.workspace = name
        if name not in TOOL_WORKSPACES:
            self.last_study_workspace = name
        self._apply_layout(self._layouts[name])
        if self.persist:
            self.settings.set_value(_SETTINGS_PREFIX + "current", name)
            self.settings.set_value(_SETTINGS_PREFIX + "last_study", self.last_study_workspace)
        self.workspaceChanged.emit(name)
        self.layoutChanged.emit()

    def study_workspace(self) -> str:
        """The workspace a new study opens in: the current one, unless it is a tool workspace
        (Scripting), then the last other one the user had (Default when there was none)."""
        if self.workspace not in TOOL_WORKSPACES:
            return self.workspace
        return self.last_study_workspace if self.last_study_workspace in WORKSPACES else DEFAULT_WORKSPACE

    def headers_in_order(self) -> list[DockHeader]:
        """Headers of the panels on screen, rail by rail (left, right, bottom) in rail order:
        the stops F6 visits after the page."""
        out = []
        for area in AREAS:
            for k in self._order[area]:
                p = self.panels[k]
                if p.isVisible() and not p.isFloating():
                    out.append(p.header)
        return out

    def reset_workspace(self, name: str | None = None) -> None:
        """Put a workspace back to its preset (the current one by default)."""
        name = name or self.workspace
        self._layouts[name] = copy.deepcopy(WORKSPACES[name])
        if self.persist:
            self.settings.set_value(_SETTINGS_PREFIX + name, json.dumps(self._layouts[name]))
        if name == self.workspace:
            self._apply_layout(self._layouts[name])
            self.layoutChanged.emit()
            self.workspaceChanged.emit(name)

    def layout_of(self, name: str) -> dict:
        """The stored layout of a workspace (the current one includes unsaved changes)."""
        if name == self.workspace:
            return self.capture()
        return copy.deepcopy(self._layouts[name])

    def _load(self, name: str) -> dict:
        raw = self.settings.value(_SETTINGS_PREFIX + name, "")
        data = None
        if isinstance(raw, str) and raw.strip():
            with contextlib.suppress(ValueError):
                data = json.loads(raw)
        return _normalise_layout(data, WORKSPACES[name], stored=True)

    def _remember_layout(self) -> None:
        """Keep the current arrangement in the current workspace (and the settings file)."""
        if not self._shown:                 # the Start page hides everything: nothing to learn there
            return
        self._layouts[self.workspace] = self.capture()
        if self.persist:
            self.settings.set_value(_SETTINGS_PREFIX + self.workspace, json.dumps(self._layouts[self.workspace]))
            self.settings.set_value(_SETTINGS_PREFIX + "current", self.workspace)

    def save(self) -> None:
        """Write the current workspace now (window close)."""
        self._save_timer.stop()
        self._remember_layout()

    def _changed(self, *, save: bool = True) -> None:
        if self._applying:
            return
        if save:
            self._save_timer.start()
        self.layoutChanged.emit()

    # -- compact (narrow window) ---------------------------------------------------------------
    def is_compact(self) -> bool:
        return self._compact

    def update_compact(self, width: int) -> None:
        """Fold the right rail into the left one below :data:`COMPACT_WIDTH`; unfold above it."""
        if not self._compact and width < COMPACT_WIDTH:
            self._compact = True
            self._folded = [k for k in self._order["right"]]
            self._fold_into_left()
            self._apply_visibility()
            self.layoutChanged.emit()
        elif self._compact and width >= COMPACT_WIDTH + COMPACT_HYSTERESIS:
            self._unfold()
            self._apply_visibility()
            self.layoutChanged.emit()

    def _fold_into_left(self) -> None:
        self._applying += 1
        try:
            left = [k for k in self._order["left"] if k not in self._folded]
            anchor = self.panels[left[0]] if left else None
            for k in self._folded:
                p = self.panels[k]
                if p.is_collapsed():
                    p.set_collapsed(False, emit=False)     # a tab cannot be folded to its header
                if anchor is None:
                    self.window.addDockWidget(_QT_AREA["left"], p)
                    anchor = p
                else:
                    self.window.tabifyDockWidget(anchor, p)
            if left:
                self.panels[left[0]].raise_()
        finally:
            self._applying -= 1

    def _unfold(self) -> None:
        self._applying += 1
        try:
            self._compact = False
            self._folded = []
            self._relayout(["left", "right"])
        finally:
            self._applying -= 1

    def rail_cap(self, width: int | None = None) -> int | None:
        """The widest a side rail may be at window ``width`` (None: no cap at or above RAIL_CAP_WIDTH)."""
        width = self.window.width() if width is None else int(width)
        return round(RAIL_CAP_SHARE * width) if width < RAIL_CAP_WIDTH else None

    def _rail_of(self, key: str) -> str:
        """The rail a panel is in on screen (a folded right-rail panel sits in the left one)."""
        return "left" if key in self._folded else self.area_of(key)

    def natural_min_width(self, key: str) -> int:
        """The width a panel needs uncapped: its content's minimum, its header's, or the declared one."""
        p = self.panels[key]
        content = p.scroll._content_min().width() + 2          # the frame's 1 px side margins
        return max(content, p.header.minimumSizeHint().width(), p.declared_min_width)

    def page_min_width(self) -> int:
        """The minimum width of the page on screen, with the chrome around it (step bar frame).

        Not the central widget's own minimum: the pages sit in a QStackedWidget,
        whose minimum is the *largest* page's (measured: 1041 px, the 2D Geometry
        page, while Results needs 253 px), so guarding with it would pin the rails
        on every page to what the widest one needs.
        """
        central = self.window.centralWidget()
        if central is None:
            return 0

        def need(w: QWidget) -> int:
            return w.minimumSizeHint().expandedTo(w.minimumSize()).width()

        total = need(central)
        stack = getattr(self.window, "stack", None)
        if not isinstance(stack, QStackedWidget) or not central.isAncestorOf(stack):
            stack = next(iter(central.findChildren(QStackedWidget)), None)
        page = stack.currentWidget() if stack is not None else None
        if page is None:
            return total
        # margins and frames between central and stack: from the geometry once laid out. The
        # difference of the two minimums is not reliable since the window's page stack reports
        # the current page's minimum (main_window._PageStack): right after a page's minimum
        # changes, the stack's is already new while the central widget's cached one is not.
        if central.width() > 0 and stack.width() > 0:
            chrome = max(0, central.width() - stack.width())
        else:
            chrome = max(0, total - need(stack))
        return chrome + need(page)

    def page_guard_caps(self, width: int | None = None) -> dict[str, int | None]:
        """Per side rail, the widest it may be so the page keeps its own minimum width (None: no limit).

        Above :data:`RAIL_CAP_WIDTH` no share cap applies, so a wide panel
        content decided the rail alone: with Diagnostics moved to the right rail
        at 1400-1520 px its 596 px left the 3D Simulate page 598-640 px against
        a 711 px minimum, and the page was cut off (measured). The guard leaves
        room for the page's minimum (:meth:`page_min_width`) and the other rail's
        natural minimum; Qt then shares out the rest. :meth:`cap_rails` never
        applies it below a panel's header or declared minimum, nor below the
        workspace's own width for that rail: the guard is for content minimums
        far above the preset, not for shrinking the presets on a wide page (a
        rail held under its preset would also be saved that narrow).
        """
        width = self.window.width() if width is None else int(width)
        caps: dict[str, int | None] = {"left": None, "right": None}
        if self.window.centralWidget() is None or not self._shown:
            return caps
        need = self.page_min_width()
        shown = [k for k in self.panels if not self.panels[k].isHidden() and self._rail_of(k) in caps]
        mins = {side: max((self.natural_min_width(k) for k in shown if self._rail_of(k) == side), default=0)
                for side in caps}
        seps = _SEPARATOR_PX * sum(1 for side in caps if mins[side])
        for side, other in (("left", "right"), ("right", "left")):
            room = width - need - mins[other] - seps
            if mins[side] and mins[side] > room:
                caps[side] = max(room, 0)
        return caps

    def cap_rails(self, width: int | None = None) -> None:
        """Apply :meth:`rail_cap` and :meth:`page_guard_caps` as the maximum width of side-rail panels.

        Bottom-rail panels are never capped (they span the window). Called on
        every window resize, after a relayout (a panel moved between rails
        changes which cap applies to it) and on a page switch (the page's
        minimum width is part of the guard).
        """
        cap = self.rail_cap(width)
        guard = self.page_guard_caps(width)
        preset = WORKSPACES.get(self.workspace, WORKSPACES[DEFAULT_WORKSPACE])["sizes"]
        for key, p in self.panels.items():
            rail = self._rail_of(key)
            side = rail in ("left", "right")
            limits = []
            if side and cap is not None:
                limits.append(max(cap, p.declared_min_width))
            if side and guard.get(rail) is not None:
                limits.append(max(guard[rail], p.declared_min_width, p.header.minimumSizeHint().width(),
                                  int(preset.get(rail, 0))))
            capped = bool(limits)
            limit = min(limits) if capped else _QWIDGETSIZE_MAX
            p.scroll.set_cap(limit - 2 if capped else None)     # the frame's 1 px side margins
            if p.maximumWidth() != limit:
                p.setMaximumWidth(limit)
                if not capped and p.declared_min_width:
                    # a cap below the declared minimum made Qt lower the minimum too: put it back
                    QWidget.setMinimumWidth(p, p.declared_min_width)

    def folded(self) -> list[str]:
        """Right-rail panels currently folded into the left rail as tabs."""
        return list(self._folded)

    # -- menus ---------------------------------------------------------------------------------
    def panel_menu(self, key: str) -> QMenu:
        """The header menu of a panel: collapse, reorder, move, solo, hide."""
        p = self.panels[key]
        area = self.area_of(key)
        m = QMenu(self.window)
        m.addAction("Expand" if p.is_collapsed() else "Collapse",
                    lambda: self.toggle_collapsed(key, user=True)).setEnabled(key not in self._folded)
        m.addSeparator()
        order = self._order[area]
        up = m.addAction("Move up", lambda: self.move_by(key, -1))
        up.setEnabled(key in order and order.index(key) > 0)
        down = m.addAction("Move down", lambda: self.move_by(key, 1))
        down.setEnabled(key in order and order.index(key) < len(order) - 1)
        for dest in AREAS:
            if dest != area:
                m.addAction(f"Move to the {AREA_TEXT[dest]}", lambda d=dest: self.move_to(key, d))
        m.addSeparator()
        solo = m.addAction(f"Solo mode in the {AREA_TEXT[area]}")
        solo.setCheckable(True)
        solo.setChecked(self.is_solo(area))
        solo.setToolTip("Opening one panel in this rail collapses the others")
        solo.triggered.connect(lambda on: self.set_solo(area, on))
        m.addSeparator()
        m.addAction(theme.icon("close"), "Hide panel", lambda: self.hide_panel(key))
        return m

    def fill_panels_menu(self, menu: QMenu) -> None:
        """Add the "Panels" section of View: one checkable entry per panel."""
        menu.addSection("Panels")
        for spec in PANELS:
            menu.addAction(self.actions[spec.key])

    def fill_workspace_menu(self, menu: QMenu) -> None:
        """The View ▸ Workspace menu: the four workspaces (radio), then Reset."""
        menu.clear()
        for name in WORKSPACE_NAMES:
            a = menu.addAction(name, lambda n=name: self.set_workspace(n))
            a.setCheckable(True)
            a.setChecked(name == self.workspace)
            a.setStatusTip(WORKSPACE_HINTS.get(name, ""))
        menu.addSeparator()
        menu.addAction(f"Reset the {self.workspace} workspace", lambda: self.reset_workspace()).setStatusTip(
            "Put the panels of this workspace back where they started")
        menu.addAction("Reset rail widths", self.reset_rail_sizes)

    # -- events ------------------------------------------------------------------------------------
    def eventFilter(self, obj, e) -> bool:
        if obj is self.window:
            et = e.type()
            if et == QEvent.Type.Resize:
                self.update_compact(self.window.width())
                self.cap_rails(self.window.width())
            elif et == QEvent.Type.MouseButtonDblClick and self._on_rail_edge(e.position().toPoint()):
                self.reset_rail_sizes()
                return True
            elif et == QEvent.Type.MouseButtonPress and "bottom" in self._shares:
                self._press_bottom = self.rail_size("bottom")
            elif et == QEvent.Type.MouseButtonRelease:
                # the window's own mouse events are its dock separators: a rail was resized
                if self._press_bottom is not None:
                    QTimer.singleShot(0, self, self._learn_bottom_share)
                self._changed()
        return super().eventFilter(obj, e)

    def _learn_bottom_share(self) -> None:
        """A drag of the bottom rail's edge in a share-sized workspace sets its new share."""
        before, self._press_bottom = self._press_bottom, None
        h = self.window.height()
        if before is None or "bottom" not in self._shares or h <= 0:
            return
        now = self.rail_size("bottom")
        if abs(now - before) > 4:
            lo, hi = _SHARE_RANGE
            self._shares["bottom"] = min(hi, max(lo, now / h))
            self._changed()

    def _on_rail_edge(self, pos: QPoint) -> bool:
        """Whether a window-level click landed on the gap between a side rail and the page."""
        central = self.window.centralWidget()
        if central is None:
            return False
        g = central.geometry()
        return g.top() <= pos.y() <= g.bottom() and (abs(pos.x() - g.left()) <= 6 or abs(pos.x() - g.right()) <= 6)


#: one line per workspace, for the menu's status tips and the docs
WORKSPACE_HINTS = {
    "Default": "Project on the left, the log folded at the bottom: the guided workflow",
    "Analysis": "The Assistant and the open log beside the results",
    "Live": "Flow visualisation beside a live run and Diagnostics under it, a narrower Project panel",
    "Scripting": "The Script panel takes 40 % of the window height, the Assistant on the right",
}


def placeholder(title: str, text: str) -> QWidget:
    """A plain empty state for a panel whose content could not be built (the reason in one line)."""
    return EmptyState("warning", title, text)


__all__ = [
    "AREAS",
    "BOTTOM_SHARE_MIN",
    "COMPACT_WIDTH",
    "DEFAULT_WORKSPACE",
    "PANELS",
    "RAIL_CAP_SHARE",
    "RAIL_CAP_WIDTH",
    "SPEC",
    "TOOL_WINDOWS",
    "WORKSPACES",
    "WORKSPACE_HINTS",
    "WORKSPACE_NAMES",
    "DockHeader",
    "DockManager",
    "DockPanel",
    "PanelSpec",
    "manager_for",
    "placeholder",
]

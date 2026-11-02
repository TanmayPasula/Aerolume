"""The Account entry of the shell: a top-right button with a menu (DESIGN.md section 11).

:func:`install` puts an :class:`AccountButton` at the right end of the main
window's menu bar (or in a slot the window offers through
``add_corner_widget(widget)``) and returns it. The button shows

* a plain person glyph when no Supabase project is connected (the default;
  the menu says the app works offline and how to connect one),
* "Sign in" when a project is connected and nobody is signed in,
* a 22 px avatar (picture, or initials) and the display name, elided, when signed in.

The menu is rebuilt every time it opens, so it always reflects the current
state: Sign in..., Create account... / Profile..., Cloud projects...,
Share result..., Account settings..., Sign out.

All network work runs on a worker thread through :class:`AccountController`,
which hands results back to the GUI thread with a queued signal. Nothing here
blocks the event loop, and nothing runs at all until the user opens the menu,
except reading the stored session from disk.
"""
from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QSize, Qt, QTimer, Signal, Slot
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMenuBar,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from .. import theme

PROVIDER_NAMES = {"email": "Email and password", "github": "GitHub", "google": "Google"}
#: opens the Account menu from anywhere in the window (Ctrl+Shift+A is the Assistant)
SHORTCUT = "Ctrl+Shift+U"
#: the person glyph (matches the 16 px line icon set)
GLYPH_SIZE = 16
#: the signed-in avatar disc: big enough for legible initials, small enough for the menu-bar row
AVATAR_SIZE = 22
#: the display name beside the avatar is elided beyond this width
NAME_MAX_PX = 160


def _tok(key: str) -> str:
    """A theme colour by token name (v3 names such as ``text2`` included, through the aliases).

    Every colour comes from the theme's token tables: the v3 hex fallbacks this used to carry
    would have drawn the old blue-grey look had a token gone missing. The light table answers
    a key the active table lacks, so the avatar still draws."""
    t = theme.tokens()
    return str(t.get(key) or theme.tokens("light")[key])


class AccountController(QObject):
    """Owns the :class:`~aerolume.accounts.service.AccountService` for the GUI.

    ``run(fn, on_done, on_error)`` executes ``fn()`` on a small thread pool and
    calls ``on_done(result)`` or ``on_error(exc)`` on the GUI thread. ``changed``
    re-emits the service's events (signed_in, signed_out, profile, config,
    refreshed) on the GUI thread; ``busyChanged`` tells views to disable
    controls while work is in flight.
    """

    changed = Signal(str)
    busyChanged = Signal(bool)
    _deliver = Signal(object, object, object, object)     # (on_done, on_error, result, error)
    _event = Signal(str)

    def __init__(self, service: Any = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._service = None
        self._unsub: Callable[[], None] | None = None
        self._pool = DaemonExecutor("aerolume-account")
        self._running = 0
        self.avatar: QImage | None = None
        self._deliver.connect(self._on_deliver, Qt.ConnectionType.QueuedConnection)
        self._event.connect(self._on_event, Qt.ConnectionType.QueuedConnection)
        if service is not None:
            self.set_service(service)

    # -- service ------------------------------------------------------------------
    @property
    def service(self):
        if self._service is None:
            from ...accounts import get_service
            self.set_service(get_service())
        return self._service

    def set_service(self, service) -> None:
        if self._unsub is not None:
            self._unsub()
        self._service = service
        self.avatar = None
        self._unsub = service.subscribe(self._event.emit)
        self.changed.emit("config")

    @Slot(str)
    def _on_event(self, event: str) -> None:
        if event in ("signed_out", "config", "signed_in"):
            self.avatar = None
        self.changed.emit(event)

    # -- work ---------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._running > 0

    def run(self, fn: Callable[[], Any], on_done: Callable[[Any], None] | None = None,
            on_error: Callable[[BaseException], None] | None = None) -> concurrent.futures.Future:
        self._running += 1
        if self._running == 1:
            self.busyChanged.emit(True)

        def job():
            try:
                result = fn()
            except BaseException as exc:
                self._deliver.emit(on_done, on_error, None, exc)
                raise
            self._deliver.emit(on_done, on_error, result, None)
            return result
        return self._pool.submit(job)

    @Slot(object, object, object, object)
    def _on_deliver(self, on_done, on_error, result, error) -> None:
        self._running = max(0, self._running - 1)
        if self._running == 0:
            self.busyChanged.emit(False)
        try:
            if error is not None:
                if on_error is not None:
                    on_error(error)
            elif on_done is not None:
                on_done(result)
        except RuntimeError:            # the dialog that asked was closed meanwhile
            pass

    def load_avatar(self) -> None:
        """Fetch the profile picture in the background (the button repaints when it arrives)."""
        svc = self.service
        if not svc.signed_in:
            return

        def fetch():
            svc.get_profile()
            return svc.avatar_bytes()

        def done(data):
            img = QImage.fromData(data) if data else None
            self.avatar = img if img is not None and not img.isNull() else None
            self.changed.emit("avatar")
        self.run(fetch, done, lambda _e: None)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class DaemonExecutor:
    """Runs each account job on its own daemon thread and returns a :class:`concurrent.futures.Future`.

    Why not ``ThreadPoolExecutor``: its workers are joined when the interpreter exits, so a request
    stuck on a server that accepts connections but never answers (a captive portal, a stalled proxy)
    kept the closed app alive for the HTTP timeout -- 30 s per request, two for the avatar -- while
    it still held the single-instance lock, and starting Aerolume again in that minute found it
    "already running". Account calls are idempotent network requests whose results nobody waits
    for once the window is gone, so daemon threads, which exit with the process, are the right kind.
    """

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._count = 0
        self._closed = False

    def submit(self, fn: Callable[[], Any]) -> concurrent.futures.Future:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        if self._closed:
            fut.cancel()
            return fut

        def body() -> None:
            if not fut.set_running_or_notify_cancel():
                return
            try:
                fut.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 - handed to the future, like an executor does
                fut.set_exception(exc)
        self._count += 1
        threading.Thread(target=body, name=f"{self._prefix}-{self._count}", daemon=True).start()
        return fut

    def shutdown(self, wait: bool = False, cancel_futures: bool = True) -> None:
        """Accept no new jobs; running ones finish (or end with the process)."""
        self._closed = True


_controller: AccountController | None = None


def controller() -> AccountController:
    """The application-wide controller (created on first use)."""
    global _controller
    if _controller is None:
        _controller = AccountController(parent=QApplication.instance())
    return _controller


def set_controller(ctrl: AccountController | None) -> None:
    """Replace the application-wide controller (tests point it at a mock-backed service)."""
    global _controller
    _controller = ctrl


# ---------------------------------------------------------------------------
# avatar pictures
# ---------------------------------------------------------------------------

def avatar_pixmap(size: int, initials: str = "", image: QImage | None = None, *, dpr: float = 1.0) -> QPixmap:
    """A round avatar: the picture when there is one, else initials on a neutral disc.

    Neutral on purpose (the ``pressed`` grey with text-colour letters): the accent
    colour is reserved for selection and the primary action.
    """
    px = QPixmap(int(size * dpr), int(size * dpr))
    px.setDevicePixelRatio(dpr)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    rect = QRectF(0.5, 0.5, size - 1, size - 1)
    path = QPainterPath()
    path.addEllipse(rect)
    if image is not None and not image.isNull():
        p.setClipPath(path)
        side = min(image.width(), image.height())
        src = QRectF((image.width() - side) / 2, (image.height() - side) / 2, side, side)
        p.drawImage(rect, image, src)
        p.setClipping(False)
        p.setPen(QPen(QColor(_tok("border_strong")), 1))
        p.drawEllipse(rect)
    else:
        p.setPen(QPen(QColor(_tok("border_strong")), 1))
        p.setBrush(QColor(_tok("pressed")))
        p.drawEllipse(rect)
        f = theme.ui_font(max(9, round(size * 0.45)), QFont.Weight.DemiBold)
        p.setFont(f)
        p.setPen(QColor(_tok("text")))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, initials or "?")
    p.end()
    return px


def person_pixmap(size: int, colour: str, *, dpr: float = 1.0) -> QPixmap:
    """A 1.25 px line person glyph (head + shoulders), matching the 16 px icon set."""
    px = QPixmap(int(size * dpr), int(size * dpr))
    px.setDevicePixelRatio(dpr)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 16.0
    pen = QPen(QColor(colour), 1.25 * s)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(8 * s, 5.5 * s), 2.75 * s, 2.75 * s)
    body = QPainterPath()
    body.moveTo(2.75 * s, 14 * s)
    body.cubicTo(3.2 * s, 10.6 * s, 5.3 * s, 9.4 * s, 8 * s, 9.4 * s)
    body.cubicTo(10.7 * s, 9.4 * s, 12.8 * s, 10.6 * s, 13.25 * s, 14 * s)
    p.drawPath(body)
    p.end()
    return px


# ---------------------------------------------------------------------------
# the button and its menu
# ---------------------------------------------------------------------------

class _MenuHeader(QWidget):
    """Non-interactive first rows of the menu: who is signed in, and where."""

    def __init__(self, lines: list[tuple[str, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 6)
        lay.setSpacing(2)
        self.setMinimumWidth(280)
        for text, role in lines:
            lab = QLabel(text)
            lab.setProperty("role", role)
            lab.setTextFormat(Qt.TextFormat.PlainText)
            lab.setWordWrap(True)
            lay.addWidget(lab)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)


class AccountButton(QToolButton):
    """Top-right Account button. ``menu_actions()`` lists what the open menu offers (tests)."""

    #: class default: Qt sends StyleChange from inside __init__ (setStyleSheet), before it is set
    _sync_queued = False

    def __init__(self, main_window: QWidget | None = None, state: Any = None,
                 ctrl: AccountController | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.main_window, self.state = main_window, state
        self.ctrl = ctrl or controller()
        self.setObjectName("accountButton")
        self.setAutoRaise(True)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        # Not in the Tab chain: as the menu bar's corner widget it would be the first stop and
        # take the window's initial focus. Keyboard users open it with SHORTCUT (see install()).
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setIconSize(QSize(GLYPH_SIZE, GLYPH_SIZE))
        self._menu = QMenu(self)
        self._menu.setObjectName("accountMenu")
        self._menu.aboutToShow.connect(self._rebuild_menu)
        self.setMenu(self._menu)
        # No vertical padding or border: the 22 px avatar must fit the menu bar's
        # own row height, or the bar grows by a few pixels at sign-in and the
        # whole window below it jumps.
        self.setStyleSheet("QToolButton#accountButton::menu-indicator { image: none; width: 0; }"
                           "QToolButton#accountButton { padding: 0 6px; margin: 0 2px; border: none;"
                           " border-radius: 3px; }")
        self.ctrl.changed.connect(self._on_changed)
        theme.changed.connect(self._on_theme)      # bound slot: disconnected when the button dies
        self._avatar_asked = False
        self._sync_queued = False
        self.refresh()

    # -- appearance ---------------------------------------------------------------
    def refresh(self) -> None:
        """Icon, text and tooltip for the current state, drawn in the current theme's colours."""
        svc = self.ctrl.service
        dpr = self.devicePixelRatioF() or 1.0
        glyph = QIcon(person_pixmap(GLYPH_SIZE, _tok("text2"), dpr=dpr))
        if not svc.configured:
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            self.setIcon(glyph)
            self.setIconSize(QSize(GLYPH_SIZE, GLYPH_SIZE))
            self.setText("")
            tip = "Account: not connected. Aerolume works fully offline."
        elif not svc.signed_in:
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.setIcon(glyph)
            self.setIconSize(QSize(GLYPH_SIZE, GLYPH_SIZE))
            self.setText("Sign in")
            tip = f"Sign in to {svc.config.server_name}"
        else:
            user = svc.user
            # Avatar plus the name: initials alone in a small disc are hard to read and
            # do not say whose account it is at a glance.
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.setIcon(QIcon(avatar_pixmap(AVATAR_SIZE, user.initials, self.ctrl.avatar, dpr=dpr)))
            self.setIconSize(QSize(AVATAR_SIZE, AVATAR_SIZE))
            self.setText(self.fontMetrics().elidedText(user.label, Qt.TextElideMode.ElideRight, NAME_MAX_PX))
            tip = f"Signed in as {user.label}" + (f" ({user.email})" if user.display_name else "")
            if not self._avatar_asked:
                self._avatar_asked = True
                self.ctrl.load_avatar()
        self.setToolTip(f"{tip}\nAccount menu: {SHORTCUT}")
        self.setAccessibleName("Account")
        self.setAccessibleDescription(tip)
        self._fix_own_size()
        self._queue_sync()

    def _fix_own_size(self) -> None:
        """Pin the button to its size hint, at one height for every state.

        One height (at least :data:`theme.MIN_TARGET`, 24 px for WCAG 2.2 SC 2.5.8;
        the avatar's, or a text line plus 4 px when that is taller) keeps the menu
        bar the same height signed in or out. The release audit found the button
        at 35x22 px: it passed only through the spacing exception. Done at once in
        :meth:`refresh`, so the bar's first layout already sees the final size.
        """
        hint = self.sizeHint()
        size = QSize(max(hint.width(), theme.MIN_TARGET),
                     max(theme.MIN_TARGET, AVATAR_SIZE, self.fontMetrics().height() + 4))
        if self.minimumSize() != size or self.maximumSize() != size:
            self.setFixedSize(size)

    def _queue_sync(self) -> None:
        if not self._sync_queued:
            self._sync_queued = True
            QTimer.singleShot(0, self, self._sync_geometry)

    def _sync_geometry(self) -> None:
        """Give the button (and its holder in the menu bar) the size it asks for.

        QMenuBar places its corner widget only when the bar itself is laid out
        (resize, setCornerWidget); it does not follow later size-hint changes.
        A theme switch re-polishes the bar before this button, so the bar
        measured the corner with the old style and squeezed it to 24 px wide:
        the 16 px glyph then painted as a 4 px square and the avatar at 8 px.
        Fixing the sizes and re-seating the corner after the style settles
        keeps the button exactly at its size hint in every state and theme.
        """
        self._sync_queued = False
        self._fix_own_size()
        holder = self.parentWidget()
        if holder is None or holder.layout() is None:
            return
        holder.layout().activate()
        hsize = holder.layout().totalSizeHint()
        moved = holder.size() != hsize          # before setFixedSize, which resizes in place
        if holder.minimumSize() != hsize or holder.maximumSize() != hsize:
            holder.setFixedSize(hsize)
        bar = holder.parentWidget()
        if moved and isinstance(bar, QMenuBar) and bar.cornerWidget(Qt.Corner.TopRightCorner) is holder:
            bar.setCornerWidget(holder, Qt.Corner.TopRightCorner)      # the bar re-lays its corner
            holder.show()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.FontChange):
            self._queue_sync()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._queue_sync()

    @Slot(str)
    def _on_theme(self, _name: str) -> None:
        self.refresh()

    @Slot(str)
    def _on_changed(self, event: str) -> None:
        if event in ("signed_in", "signed_out", "config"):
            self._avatar_asked = False
        self.refresh()

    # -- menu -----------------------------------------------------------------------
    def _add(self, text: str, slot: Callable[[], None], *, enabled: bool = True, tip: str = "") -> None:
        a = self._menu.addAction(text)
        a.setEnabled(enabled)
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        a.triggered.connect(lambda _=False: slot())

    def _header(self, lines: list[tuple[str, str]]) -> None:
        wa = QWidgetAction(self._menu)
        wa.setDefaultWidget(_MenuHeader(lines))
        wa.setEnabled(False)
        self._menu.addAction(wa)
        self._menu.addSeparator()

    def _rebuild_menu(self) -> None:
        m = self._menu
        m.clear()
        m.setToolTipsVisible(True)
        svc = self.ctrl.service
        if not svc.configured:
            self._header([("Not connected", "h3"),
                          (("Aerolume works fully offline. Connect a Supabase project to sign in, keep "
                            "cloud copies of projects and share results."), "muted")])
            self._add("Connect a project…", self.open_settings)
            self._add("About accounts", self.open_docs)
            return
        if not svc.signed_in:
            self._header([("Not signed in", "h3"), (svc.config.server_label, "muted")])
            self._add("Sign in…", lambda: self.open_sign_in("sign_in"))
            self._add("Create account…", lambda: self.open_sign_in("sign_up"))
            m.addSeparator()
            self._add("Account settings…", self.open_settings)
            self._add("About accounts", self.open_docs)
            return
        user = svc.user
        lines = [(user.label, "h3")]
        if user.display_name:
            lines.append((user.email, "secondary"))
        lines.append((svc.config.server_label, "muted"))
        self._header(lines)
        self._add("Profile…", self.open_profile)
        self._add("Cloud projects…", self.open_cloud)
        has_project = self.state is not None and getattr(self.state, "project", None) is not None
        self._add("Share result…", self.open_share, enabled=has_project,
                  tip="" if has_project else "Open a project first.")
        m.addSeparator()
        self._add("Account settings…", self.open_settings)
        self._add("Sign out", self.sign_out)

    def menu_actions(self) -> list[str]:
        """Texts of the menu's actions as they are now (tests and the MCP bridge)."""
        self._rebuild_menu()
        return [a.text() for a in self._menu.actions() if a.text()]

    # -- actions --------------------------------------------------------------------
    def _parent(self) -> QWidget:
        return self.main_window or self.window()

    def open_sign_in(self, mode: str = "sign_in"):
        from ..dialogs.account_dialogs import SignInDialog
        dlg = SignInDialog(self.ctrl, mode=mode, parent=self._parent(), on_settings=self.open_settings)
        dlg.open()
        return dlg

    def open_profile(self):
        from ..dialogs.account_dialogs import ProfileDialog
        dlg = ProfileDialog(self.ctrl, parent=self._parent())
        dlg.open()
        return dlg

    def open_cloud(self):
        from ..dialogs.account_dialogs import CloudProjectsDialog
        dlg = CloudProjectsDialog(self.ctrl, state=self.state, main_window=self.main_window, parent=self._parent())
        dlg.open()
        return dlg

    def open_share(self):
        from ..dialogs.account_dialogs import ShareDialog
        dlg = ShareDialog(self.ctrl, state=self.state, parent=self._parent())
        dlg.open()
        return dlg

    def open_settings(self):
        from ..dialogs.account_dialogs import AccountSettingsDialog
        dlg = AccountSettingsDialog(self.ctrl, parent=self._parent())
        dlg.open()
        return dlg

    def open_docs(self) -> None:
        from ..dialogs.account_dialogs import open_accounts_doc
        open_accounts_doc(self._parent())

    def sign_out(self) -> None:
        svc = self.ctrl.service
        self.ctrl.run(svc.sign_out, lambda _r: self._status("Signed out", 4000),
                      lambda e: self._status(f"Sign-out: {e}", 6000))

    def _status(self, text: str, ms: int) -> None:
        if self.state is not None and hasattr(self.state, "statusMessage"):
            self.state.statusMessage.emit(text, ms)


def install(main_window: QWidget, state: Any = None, *, ctrl: AccountController | None = None) -> AccountButton:
    """Add the Account button to the shell and return it.

    Placement: ``main_window.add_corner_widget(widget)`` when the window offers
    a header slot, else the right end of its menu bar. Calling it twice returns
    the button that is already there.
    """
    existing = getattr(main_window, "account_button", None)
    if isinstance(existing, AccountButton):
        return existing
    # Parent the holder at once: QMenuBar.setCornerWidget does not transfer ownership in
    # PySide, so a parentless holder would be deleted with its Python wrapper when this
    # function returns, taking the button with it (seen: "C++ object already deleted").
    bar = main_window.menuBar() if hasattr(main_window, "menuBar") else None
    holder = QWidget(bar if bar is not None else main_window)
    lay = QHBoxLayout(holder)
    lay.setContentsMargins(0, 0, 4, 0)
    lay.setSpacing(0)
    button = AccountButton(main_window, state, ctrl, holder)
    lay.addWidget(button)
    adder = getattr(main_window, "add_corner_widget", None)
    if callable(adder):
        adder(holder)
    elif bar is not None:
        bar.setCornerWidget(holder, Qt.Corner.TopRightCorner)
    main_window.account_button = button
    main_window.account_corner = holder         # keeps the wrapper (and so the widget) alive
    if isinstance(main_window, QWidget):
        act = QAction("Account menu", main_window)
        act.setObjectName("actAccountMenu")
        act.setShortcut(QKeySequence(SHORTCUT))
        act.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        act.triggered.connect(lambda _=False: button.showMenu())
        main_window.addAction(act)
        main_window.act_account_menu = act
    return button

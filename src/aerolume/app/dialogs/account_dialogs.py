"""Account dialogs: sign in / create account, profile, cloud projects, share links, settings.

Design rules (docs/ui_guidelines.md, DESIGN.md 11 and 13):

* Offline is a first-class state, not an error. With no Supabase project
  connected every dialog says so plainly, says that Aerolume works fully
  without an account, and offers one way forward (Account settings).
* Say what leaves the computer before it leaves: the settings page lists it,
  the share dialog says who can open a link and for how long, the cloud list
  says which copy is newer. Nothing is uploaded, replaced or deleted without
  an explicit button press, and replacing never destroys unsynced work.
* Restrained: property-grid style forms, one primary button per view, status as
  one line of text (error, progress or done) under the form, no illustrations.
* Accessible: every field has a visible label and an accessible name, errors
  are text (not colour alone) and move focus to the field to fix, the default
  button is the primary action, Esc cancels and stops any waiting sign-in.

Every network call runs through :class:`~aerolume.app.widgets.account_menu.AccountController`
on a worker thread; the dialogs only react to its results.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QBuffer, QByteArray, QEvent, QIODevice, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication, QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..widgets.account_menu import PROVIDER_NAMES, AccountController, avatar_pixmap, controller
from ..widgets.common import SPACING, SegmentedControl, button, label

DOCS_HINT = "docs/accounts.md"


# ---------------------------------------------------------------------------
# small building blocks
# ---------------------------------------------------------------------------

class _Form(QFormLayout):
    """A form whose row labels are buddies of their fields: screen readers read the label
    with the field, and clicking the label focuses the field."""

    def addRow(self, *args) -> None:
        super().addRow(*args)
        if len(args) == 2 and isinstance(args[0], QLabel) and isinstance(args[1], QWidget) and args[0].text():
            args[0].setBuddy(args[1])


def _form() -> QFormLayout:
    f = _Form()
    f.setHorizontalSpacing(3 * SPACING)
    f.setVerticalSpacing(2 * SPACING)
    f.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return f


def _field(name: str, example: str = "", *, password: bool = False) -> QLineEdit:
    """A labelled line edit. ``example`` becomes the placeholder and must be an example
    value only ("name@example.org"), never an instruction: placeholder text disappears
    on the first keystroke and is drawn at about 3.2:1, below WCAG 1.4.3. Instructions
    go in a persistent :func:`_hint` instead."""
    e = QLineEdit()
    e.setAccessibleName(name)
    if example:
        e.setPlaceholderText(example)
    if password:
        e.setEchoMode(QLineEdit.EchoMode.Password)
    return e


def _hint(field: QWidget, text: str) -> QLabel:
    """Persistent helper text for ``field``, placed on the form row beneath it.

    Drawn with the ``muted`` token (measured 5.8:1 light, above 4.5:1 dark), it stays
    visible while the user types (WCAG 3.3.2), and it is the field's accessible
    description so screen readers read it with the label.
    """
    lab = label(text, "muted", wrap=True)
    field.setAccessibleDescription(text)
    return lab


def _row(*widgets: QWidget, stretch_at: int | None = None) -> QHBoxLayout:
    h = QHBoxLayout()
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(2 * SPACING)
    for i, w in enumerate(widgets):
        if stretch_at == i:
            h.addStretch(1)
        h.addWidget(w)
    if stretch_at is not None and stretch_at >= len(widgets):
        h.addStretch(1)
    return h


def _when(iso: str) -> str:
    """ISO timestamp -> "18 Oct 2026, 14:05" in local time ("" when missing or unreadable)."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso)          # Python 3.11 reads "Z" and offsets
    except ValueError:
        return iso
    if dt.tzinfo is not None:
        dt = dt.astimezone()
    return dt.strftime("%d %b %Y, %H:%M").lstrip("0")


def _size(n: int) -> str:
    if n <= 0:
        return ""
    if n < 1024 * 1024:
        return f"{max(1, round(n / 1024))} kB"
    return f"{n / 1024 / 1024:.1f} MB"


class StatusLine(QLabel):
    """One line of status under a form: progress (muted), done (good) or an error (bad).

    The text says what happened; colour only repeats it. Errors are announced to
    screen readers through the accessible description.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.kind = ""
        self.clear()

    def _set(self, text: str, role: str, kind: str) -> None:
        self.kind = kind
        self.setText(text)
        theme.set_prop(self, "role", role)
        self.setAccessibleDescription(text)
        self.setVisible(bool(text))

    def info(self, text: str) -> None:
        self._set(text, "muted", "info")

    def ok(self, text: str) -> None:
        self._set(text, "status-good", "ok")

    def error(self, text: str) -> None:
        self._set(text, "status-bad", "error")

    def clear(self) -> None:
        self._set("", "muted", "")


def _error_text(exc: BaseException) -> str:
    from ...accounts import AccountError
    if isinstance(exc, AccountError):
        return str(exc)
    return f"Unexpected error ({type(exc).__name__}). Details are in the log."


class _Pages(QWidget):
    """One visible page at a time, sized to *that* page.

    Not a QStackedWidget: its layout reports the tallest page for height-for-width,
    so a short page (offline, "check your email") would sit in a window sized for
    the sign-up form. Hidden widgets take no space in a box layout, which is
    exactly the behaviour wanted here.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._pages: list[QWidget] = []
        self._current = -1

    def addWidget(self, w: QWidget) -> int:
        self._pages.append(w)
        self._lay.addWidget(w)
        w.setVisible(False)
        if self._current < 0:
            self.setCurrentIndex(0)
        return len(self._pages) - 1

    def setCurrentIndex(self, i: int) -> None:
        self._current = i
        for k, w in enumerate(self._pages):
            w.setVisible(k == i)

    def currentIndex(self) -> int:
        return self._current

    def currentWidget(self) -> QWidget | None:
        return self._pages[self._current] if 0 <= self._current < len(self._pages) else None


class _AccountDialog(QDialog):
    """Common frame: 16 px margins, a status line, busy handling, Esc stops waiting work."""

    def __init__(self, ctrl: AccountController | None, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctrl = ctrl or controller()
        self.svc = self.ctrl.service
        self.setWindowTitle(title)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(4 * SPACING, 4 * SPACING, 4 * SPACING, 3 * SPACING)
        self.lay.setSpacing(3 * SPACING)
        self.status = StatusLine()
        self._busy_widgets: list[QWidget] = []
        self._refocus: QWidget | None = None      # focused before a busy period (see _set_busy)
        self._user_acted = False                  # a key / click in this dialog while busy
        self._watching = False                    # the busy-time event filter is installed
        self.fit_height = True          # dialogs with tables set False and size themselves
        self._min_width = 0             # the explicit minimum width (setMinimumWidth below)
        self._fitting = False

    def setMinimumWidth(self, w: int) -> None:      # Qt override
        """Remember the explicit minimum width: :meth:`fit` owns the minimum size afterwards."""
        self._min_width = w
        super().setMinimumWidth(w)

    # -- sizing ---------------------------------------------------------------------
    # The window height must follow the content: pages differ in height, the
    # provider buttons arrive from the network after the dialog is shown, a
    # status line appears and disappears, the sign-up rows show and hide, and a
    # theme switch changes font metrics. Sizing once in showEvent left the
    # layout squeezed into a stale height (rows drawn over each other, the
    # provider buttons clipped). So the dialog re-fits on every LayoutRequest,
    # which Qt posts to the window whenever any of those things happens.
    #
    # Why not QLayout.SetMinimumSize: it pins the minimum to totalMinimumSize(),
    # which measures wrapped labels with Qt's one-line/golden-ratio heuristics,
    # not at the width the dialog has (a band of empty space under short pages,
    # too little height under long ones), and it overrides the explicit minimum
    # width. totalHeightForWidth() at the real width is the exact answer.

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.fit()

    def hideEvent(self, event) -> None:
        """A dialog closed while busy stops watching input (see :meth:`_set_busy`)."""
        if self._watching:
            QApplication.instance().removeEventFilter(self)
            self._watching = False
        super().hideEvent(event)

    def event(self, e) -> bool:
        handled = super().event(e)
        # Qt has already re-activated the layout (QLayout.widgetEvent runs first).
        if e.type() == QEvent.Type.LayoutRequest and self.isVisible():
            self.fit()
        return handled

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width() and self.isVisible():
            self.fit()                  # narrower means more wrapped lines: a taller minimum

    def needed_height(self, width: int | None = None) -> int:
        """The content's height at ``width`` (default: the current width)."""
        w = max(width if width is not None else self.width(), self._min_width)
        return self.lay.totalHeightForWidth(w)

    def fit(self) -> None:
        """Make the window exactly as tall as its content at the current width.

        The minimum height follows too, so neither the user nor the window
        manager can squeeze the rows into each other; the minimum width is the
        explicit one (or the layout's, when that is wider).
        """
        if not self.fit_height or self._fitting:
            return
        self._fitting = True
        try:
            self.lay.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
            min_w = max(self._min_width, self.lay.totalMinimumSize().width())
            w = max(self.width(), min_w)
            h = self.needed_height(w)
            if h <= 0:
                return
            if (self.minimumWidth(), self.minimumHeight()) != (min_w, h):
                QDialog.setMinimumSize(self, min_w, h)
            if (self.width(), self.height()) != (w, h):
                self.resize(w, h)
        finally:
            self._fitting = False

    def run(self, fn, on_done=None, on_error=None, *, progress: str = ""):
        """Run ``fn`` on the worker thread with this dialog's controls disabled meanwhile."""
        if progress:
            self.status.info(progress)
        self._set_busy(True)

        def done(result):
            self._set_busy(False)
            if on_done is not None:
                on_done(result)

        def failed(exc):
            self._set_busy(False)
            if on_error is not None:
                on_error(exc)
            else:
                self.status.error(_error_text(exc))
        return self.ctrl.run(fn, done, failed)

    def _set_busy(self, busy: bool) -> None:
        """Disable the controls while work runs, and give keyboard focus back afterwards.

        Disabling the focused field makes Qt move focus to the next enabled
        widget (or, before the window shows, lets activation pick one), and
        nothing moved it back: measured, the sign-in dialog opened with focus on
        the mode switch instead of Email (the provider list loads while the form
        is disabled), and the profile dialog on "Use on this computer". So the
        focused widget is remembered and restored when the work ends - unless
        the user pressed a key or clicked in this dialog meanwhile, in which
        case wherever they put focus is theirs. Watching input events (an
        application event filter that exists only while busy) is the one
        signal that tells a user's focus move from Qt's.
        """
        if busy and self._refocus is None:
            fw = self.focusWidget()
            self._refocus = fw if fw in self._busy_widgets else None
            self._user_acted = False
            app = QApplication.instance()
            if app is not None and not self._watching:
                app.installEventFilter(self)
                self._watching = True
        for w in self._busy_widgets:
            w.setEnabled(not busy)
        if busy:
            self.setCursor(Qt.CursorShape.BusyCursor)
            return
        self.unsetCursor()
        if self._watching:
            QApplication.instance().removeEventFilter(self)
            self._watching = False
        target, self._refocus = self._refocus, None
        if target is not None and not self._user_acted and target.isEnabled() and target.isVisibleTo(self):
            target.setFocus()

    def eventFilter(self, obj, event) -> bool:      # Qt override; installed only while busy
        if (event.type() in (QEvent.Type.KeyPress, QEvent.Type.MouseButtonPress)
                and isinstance(obj, QWidget) and obj.window() is self):
            self._user_acted = True
        return False

    def confirm(self, title: str, text: str, ok_text: str, *, danger: bool = False) -> bool:
        """A yes/no question with a verb on the button (tests replace this method)."""
        box = QMessageBox(QMessageBox.Icon.Warning if danger else QMessageBox.Icon.Question, title, text,
                          parent=self)
        ok = box.addButton(ok_text, QMessageBox.ButtonRole.AcceptRole)
        if danger:
            ok.setProperty("variant", "danger")
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Cancel if danger else ok)
        box.exec()
        return box.clickedButton() is ok


def _not_connected_panel(on_settings, on_docs) -> QWidget:
    """The offline state: plain statement, what an account would add, one way forward."""
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(2 * SPACING)
    v.addWidget(label("Accounts are not set up", "h3"))
    v.addWidget(label("Aerolume works fully offline: projects, simulations and reports stay on this "
                      "computer and need no account.", "secondary", wrap=True))
    v.addWidget(label("An account adds cloud copies of projects and read-only share links. It needs a "
                      "Supabase project, which you create and own; supabase/README.md explains the "
                      "steps.", "secondary", wrap=True))
    v.addLayout(_row(button("Account settings…", "primary", on_click=on_settings),
                     button("About accounts", "ghost", on_click=on_docs), stretch_at=2))
    v.addStretch(1)
    return w


def open_accounts_doc(parent: QWidget | None = None) -> None:
    """Show docs/accounts.md in the in-app viewer (or the system viewer as a fallback)."""
    here = Path(__file__).resolve()
    candidates = [here.parents[4] / "docs" / "accounts.md", here.parents[2] / "docs" / "accounts.md"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        QMessageBox.information(parent, "Documentation not installed",
                                "The accounts guide is docs/accounts.md in the source distribution.")
        return
    try:
        from ..main_window import DocViewer
        DocViewer(path, "Aerolume — Accounts", parent).exec()
    except Exception:  # noqa: BLE001 - the shell may be mid-refactor; the file still opens
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


# ---------------------------------------------------------------------------
# sign in / create account
# ---------------------------------------------------------------------------

class SignInDialog(_AccountDialog):
    """Sign in, create an account, email link or code, password reset, GitHub / Google.

    Pages: ``offline`` (no project connected), ``form``, ``email`` (check your
    email: link or code), ``browser`` (waiting for the provider), ``password``
    (set a new password after a reset). ``page()`` names the current one.
    """

    signedIn = Signal()

    def __init__(self, ctrl: AccountController | None = None, *, mode: str = "sign_in",
                 parent: QWidget | None = None, on_settings=None) -> None:
        super().__init__(ctrl, "Create account" if mode == "sign_up" else "Sign in", parent)
        self.setMinimumWidth(440)
        self._on_settings = on_settings
        self.pending = None
        self._completing = False
        self.stack = _Pages()
        self._slots: dict[str, tuple[QVBoxLayout, int]] = {}
        self._pages: dict[str, int] = {}
        for name, build in (("offline", self._build_offline), ("form", self._build_form),
                            ("email", self._build_email), ("browser", self._build_browser),
                            ("password", self._build_password)):
            self._pages[name] = self.stack.addWidget(build())
        self.lay.addWidget(self.stack)
        self.footer = label("", "small", wrap=True)
        self.lay.addWidget(self.footer)
        self.mode.changed.connect(self._set_mode)
        self._set_mode(mode if mode in ("sign_in", "sign_up") else "sign_in")
        self.mode.set_current(self._mode)
        if not self.svc.configured:
            self._show("offline")
            self.footer.setVisible(False)       # an empty label would still take a row of spacing
        else:
            self._show("form")
            self.footer.setText(f"{self.svc.config.server_label}\nProjects stay on this computer unless you "
                                "upload them.")
            self.run(self.svc.auth_providers, self._set_providers, lambda _e: self._set_providers(()))
        QShortcut(QKeySequence.StandardKey.Cancel, self, activated=self.reject)

    # -- pages --------------------------------------------------------------------
    def _build_offline(self) -> QWidget:
        return _not_connected_panel(self._open_settings, lambda: open_accounts_doc(self))

    def _build_form(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(3 * SPACING)
        self.mode = SegmentedControl([("sign_in", "Sign in"), ("sign_up", "Create account")], "sign_in")
        self.mode.setAccessibleName("Sign in or create an account")
        v.addWidget(self.mode)
        f = _form()
        self.email = _field("Email", "name@example.org")
        self.email.setInputMethodHints(Qt.InputMethodHint.ImhEmailCharactersOnly)
        self.password = _field("Password", password=True)
        self.name = _field("Display name")
        self.name_label = label("Display name", "pg-label")
        labels = [label("Email", "pg-label"), label("Password", "pg-label")]
        # The label column keeps the width of its longest label in both modes, so the
        # fields do not jump sideways when "Display name" shows or hides.
        self.name_label.ensurePolished()
        for lab in labels:
            lab.setMinimumWidth(self.name_label.sizeHint().width())
        f.addRow(labels[0], self.email)
        f.addRow(labels[1], self.password)
        self.password_hint = _hint(self.password, "At least 8 characters.")
        f.addRow(QLabel(""), self.password_hint)
        f.addRow(self.name_label, self.name)
        self.name_hint = _hint(self.name, "Optional. Shown on results you share.")
        f.addRow(QLabel(""), self.name_hint)
        self.form_layout = f
        v.addLayout(f)
        self.submit = button("Sign in", "primary", on_click=self.submit_form)
        self.submit.setDefault(True)
        self.magic = button("Email me a sign-in link", "link", on_click=self.send_link)
        self.forgot = button("Forgot password?", "link", on_click=self.send_reset)
        self._slots['form'] = (v, v.count())
        v.addLayout(_row(self.submit, self.magic, self.forgot, stretch_at=3))
        self.providers_box = QWidget()
        pv = QVBoxLayout(self.providers_box)
        pv.setContentsMargins(0, SPACING, 0, 0)
        pv.setSpacing(2 * SPACING)
        line = QFrame()
        line.setProperty("divider", "true")
        line.setFixedHeight(1)
        pv.addWidget(line)
        self.provider_buttons: dict[str, QPushButton] = {}
        prow = QHBoxLayout()
        prow.setSpacing(2 * SPACING)
        for key in ("github", "google"):
            b = button(f"Continue with {PROVIDER_NAMES[key]}", on_click=lambda k=key: self.start_oauth(k))
            b.setToolTip(f"Opens {PROVIDER_NAMES[key]} in your web browser. Return here when you are done.")
            self.provider_buttons[key] = b
            prow.addWidget(b)
        prow.addStretch(1)
        pv.addLayout(prow)
        self.providers_box.setVisible(False)
        v.addWidget(self.providers_box)
        v.addStretch(1)
        self._busy_widgets += [self.email, self.password, self.name, self.submit, self.magic, self.forgot,
                               self.mode, *self.provider_buttons.values()]
        self.email.returnPressed.connect(self.password.setFocus)
        self.password.returnPressed.connect(self.submit_form)
        return w

    def _build_email(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(3 * SPACING)
        self.email_title = label("Check your email", "h3")
        v.addWidget(self.email_title)
        self.email_text = label("", "secondary", wrap=True)
        v.addWidget(self.email_text)
        f = _form()
        self.code = _field("Code from the email")
        self.code.setMaxLength(12)
        self.code.setInputMethodHints(Qt.InputMethodHint.ImhDigitsOnly)
        self.code.setFont(theme.mono_font())
        self.code.returnPressed.connect(self.verify_code)
        f.addRow(label("Code", "pg-label"), self.code)
        self.code_hint = _hint(self.code, "The 6-digit code in the same email.")
        f.addRow(QLabel(""), self.code_hint)
        v.addLayout(f)
        self.verify = button("Verify", "primary", on_click=self.verify_code)
        self.resend = button("Send again", "link", on_click=self.resend_email)
        self.back = button("Back", "ghost", on_click=self.back_to_form)
        self._slots['email'] = (v, v.count())
        v.addLayout(_row(self.verify, self.resend, self.back, stretch_at=2))
        v.addStretch(1)
        self._busy_widgets += [self.code, self.verify, self.resend]
        return w

    def _build_browser(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(3 * SPACING)
        self.browser_title = label("Continue in your browser", "h3")
        v.addWidget(self.browser_title)
        self.browser_text = label("", "secondary", wrap=True)
        v.addWidget(self.browser_text)
        self.reopen = button("Open the page again", "link", on_click=self._reopen_browser)
        self.cancel_wait = button("Cancel", on_click=self.back_to_form)
        self._slots['browser'] = (v, v.count())
        v.addLayout(_row(self.cancel_wait, self.reopen, stretch_at=2))
        v.addStretch(1)
        return w

    def _build_password(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(3 * SPACING)
        v.addWidget(label("Set a new password", "h3"))
        v.addWidget(label("You are signed in with the reset link. Choose the password for your next "
                          "sign-in.", "secondary", wrap=True))
        f = _form()
        self.new_pw = _field("New password", password=True)
        self.new_pw2 = _field("Repeat the new password", password=True)
        f.addRow(label("New password", "pg-label"), self.new_pw)
        f.addRow(label("Repeat", "pg-label"), self.new_pw2)
        f.addRow(QLabel(""), _hint(self.new_pw, "At least 8 characters."))
        v.addLayout(f)
        self.save_pw = button("Save password", "primary", on_click=self.save_password)
        self.new_pw2.returnPressed.connect(self.save_password)
        self._slots['password'] = (v, v.count())
        v.addLayout(_row(self.save_pw, stretch_at=1))
        v.addStretch(1)
        self._busy_widgets += [self.new_pw, self.new_pw2, self.save_pw]
        return w

    def _show(self, name: str) -> None:
        self.stack.setCurrentIndex(self._pages[name])
        if name in self._slots:
            lay, idx = self._slots[name]
            lay.insertWidget(idx, self.status)
        self.stack.updateGeometry()
        if self.isVisible():
            self.fit()                  # each page has its own height
        self.status.clear()
        focus = {"form": self.email if not self.email.text() else self.password, "email": self.code,
                 "password": self.new_pw, "browser": self.cancel_wait}.get(name)
        if focus is not None:
            focus.setFocus()

    def page(self) -> str:
        idx = self.stack.currentIndex()
        return next(k for k, v in self._pages.items() if v == idx)

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        up = mode == "sign_up"
        self.setWindowTitle("Create account" if up else "Sign in")
        self.submit.setText("Create account" if up else "Sign in")
        self.form_layout.setRowVisible(self.name, up)
        self.form_layout.setRowVisible(self.password_hint, up)
        self.form_layout.setRowVisible(self.name_hint, up)
        # The 8-character rule describes the password only while creating one.
        self.password.setAccessibleDescription(self.password_hint.text() if up else "")
        self.forgot.setVisible(not up)
        self.magic.setText("Email me a sign-up link" if up else "Email me a sign-in link")
        self.status.clear()

    def _set_providers(self, providers) -> None:
        for key, b in self.provider_buttons.items():
            b.setVisible(key in providers)
        self.providers_box.setVisible(bool(providers))

    def _open_settings(self) -> None:
        self.reject()
        if self._on_settings is not None:
            self._on_settings()
        else:
            AccountSettingsDialog(self.ctrl, parent=self.parentWidget()).open()

    # -- validation -----------------------------------------------------------
    def _need_email(self) -> str | None:
        from ...accounts.service import valid_email
        email = self.email.text().strip()
        if not valid_email(email):
            self.status.error("Enter a valid email address.")
            self.email.setFocus()
            theme.set_prop(self.email, "invalid", "true")
            return None
        theme.set_prop(self.email, "invalid", "false")
        return email

    # -- actions --------------------------------------------------------------------
    def submit_form(self) -> None:
        email = self._need_email()
        if email is None:
            return
        pw = self.password.text()
        if self._mode == "sign_in":
            if not pw:
                self.status.error("Enter your password.")
                self.password.setFocus()
                return
            self.run(lambda: self.svc.sign_in(email, pw), lambda _u: self._finish(), self._form_error,
                     progress="Signing in…")
            return
        if len(pw) < 8:
            self.status.error("Use at least 8 characters for the password.")
            self.password.setFocus()
            return
        name = self.name.text().strip()

        def done(res):
            result, pending = res
            if result.user is not None:
                self._finish()
                return
            self._await_email(pending, "signup")
        self.run(lambda: self.svc.sign_up(email, pw, name), done, self._form_error,
                 progress="Creating the account…")

    def send_link(self) -> None:
        email = self._need_email()
        if email is None:
            return
        create = self._mode == "sign_up"
        self.run(lambda: self.svc.begin_email_link(email, create_user=create),
                 lambda p: self._await_email(p, "email"), self._form_error, progress="Sending the email…")

    def send_reset(self) -> None:
        email = self._need_email()
        if email is None:
            return
        self.run(lambda: self.svc.begin_password_reset(email), lambda p: self._await_email(p, "recovery"),
                 self._form_error, progress="Sending the email…")

    def resend_email(self) -> None:
        kind = getattr(self.pending, "kind", "email")
        self._drop_pending()
        if kind == "recovery":
            self.send_reset()
        else:
            self.send_link()

    def start_oauth(self, provider: str) -> None:
        def started(pending):
            self.pending = pending
            name = PROVIDER_NAMES.get(provider, provider)
            self.browser_title.setText("Continue in your browser")
            self.browser_text.setText(f"{name} opened in your web browser. Sign in there; this window "
                                      "finishes by itself when you return. If no page opened, use the "
                                      "link below.")
            self._show("browser")
            self._wait_pending(pending)
        self.run(lambda: self.svc.begin_oauth(provider), started, self._form_error,
                 progress=f"Opening {PROVIDER_NAMES.get(provider, provider)}…")

    def _reopen_browser(self) -> None:
        if self.pending is not None and self.pending.url:
            QDesktopServices.openUrl(QUrl(self.pending.url))

    def _await_email(self, pending, kind: str) -> None:
        self.pending = pending
        email = pending.email
        if kind == "signup":
            self.email_title.setText("Confirm your email")
            text = (f"If {email} is new, we sent it a confirmation link. Open the link on this computer "
                    "to finish, or type the code from the email. If the address already has an "
                    "account, go back and sign in.")
        elif kind == "recovery":
            self.email_title.setText("Reset your password")
            text = (f"If {email} has an account, we sent it a reset link. Open it on this computer, or "
                    "type the code from the email. You then choose a new password.")
        else:
            self.email_title.setText("Check your email")
            text = (f"We sent a sign-in link to {email}. Open it on this computer to finish, or type the "
                    "code from the email.")
        self.email_text.setText(text)
        self.code.clear()
        self._show("email")
        self._wait_pending(pending)

    def _wait_pending(self, pending) -> None:
        """Wait for the browser callback in the background (does not disable the page)."""
        def done(_user):
            if pending is self.pending:
                self._finish()

        def failed(exc):
            from ...accounts import CancelledError
            if pending is not self.pending or self._completing or isinstance(exc, CancelledError):
                return
            self._show("form")
            self._form_error(exc)
        self.ctrl.run(lambda: pending.wait(900), done, failed)

    def verify_code(self) -> None:
        code = self.code.text().strip()
        if not code:
            self.status.error("Enter the code from the email.")
            self.code.setFocus()
            return
        pending = self.pending
        if pending is None:
            return
        self._completing = True

        def failed(exc):
            # A wrong code leaves the link listener running: the email link still works.
            self._completing = False
            self.status.error(_error_text(exc))
            self.code.setFocus()
            self.code.selectAll()
        self.run(lambda: pending.complete_with_code(code), lambda _u: self._finish(), failed,
                 progress="Checking the code…")

    def save_password(self) -> None:
        a, b = self.new_pw.text(), self.new_pw2.text()
        if len(a) < 8:
            self.status.error("Use at least 8 characters for the password.")
            self.new_pw.setFocus()
            return
        if a != b:
            self.status.error("The two passwords differ.")
            self.new_pw2.setFocus()
            return
        self.run(lambda: self.svc.set_password(a), lambda _r: self._done_accept(), progress="Saving…")

    def back_to_form(self) -> None:
        self._drop_pending()
        self._show("form")

    def _form_error(self, exc: BaseException) -> None:
        self.status.error(_error_text(exc))
        code = getattr(exc, "code", "")
        if code in ("invalid_credentials", "missing_password", "weak_password"):
            self.password.setFocus()
            self.password.selectAll()
        elif code in ("invalid_email", "user_already_exists", "email_exists"):
            self.email.setFocus()

    def _finish(self) -> None:
        self._completing = False
        self.pending = None
        if self.svc.recovery_pending:
            self._show("password")
            return
        self._done_accept()

    def _done_accept(self) -> None:
        self.signedIn.emit()
        self.accept()

    def _drop_pending(self) -> None:
        p, self.pending = self.pending, None
        if p is not None:
            p.cancel()

    def done(self, result: int) -> None:
        self._drop_pending()
        super().done(result)


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------

def image_to_avatar_png(data: bytes, side: int = 256) -> bytes:
    """Any image Qt can read -> a centred square PNG of ``side`` px (well under the 1 MB limit)."""
    img = QImage.fromData(data)
    if img.isNull():
        raise ValueError("not an image")
    s = min(img.width(), img.height())
    img = img.copy((img.width() - s) // 2, (img.height() - s) // 2, s, s)
    img = img.scaled(side, side, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return bytes(ba.data())


class ProfileDialog(_AccountDialog):
    """Display name, affiliation, picture, and the preferences saved with the account."""

    def __init__(self, ctrl: AccountController | None = None, parent: QWidget | None = None) -> None:
        super().__init__(ctrl, "Profile", parent)
        self.setMinimumWidth(500)
        if not self.svc.signed_in:
            self.lay.addWidget(label("Not signed in", "h3"))
            self.lay.addWidget(label("Sign in from the Account menu to edit your profile.", "secondary"))
            box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            box.rejected.connect(self.reject)
            self.lay.addWidget(box)
            return
        user = self.svc.user
        top = QHBoxLayout()
        top.setSpacing(4 * SPACING)
        pic = QVBoxLayout()
        pic.setSpacing(SPACING)
        self.avatar = QLabel()
        self.avatar.setFixedSize(64, 64)
        self.avatar.setAccessibleName("Profile picture")
        pic.addWidget(self.avatar, 0, Qt.AlignmentFlag.AlignHCenter)
        self.change_pic = button("Change…", "link", on_click=self.choose_picture)
        self.remove_pic = button("Remove", "link", on_click=self.remove_picture)
        pic.addWidget(self.change_pic, 0, Qt.AlignmentFlag.AlignHCenter)
        pic.addWidget(self.remove_pic, 0, Qt.AlignmentFlag.AlignHCenter)
        pic.addStretch(1)
        top.addLayout(pic)
        f = _form()
        self.display_name = _field("Display name")
        self.display_name.setMaxLength(80)
        self.affiliation = _field("Affiliation")
        self.affiliation.setMaxLength(120)
        f.addRow(label("Display name", "pg-label"), self.display_name)
        f.addRow(QLabel(""), _hint(self.display_name, "Shown on results you share."))
        f.addRow(label("Affiliation", "pg-label"), self.affiliation)
        f.addRow(QLabel(""), _hint(self.affiliation, "Optional. University, company or lab."))
        email = label(user.email, "secondary")
        email.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        f.addRow(label("Email", "pg-label"), email)
        f.addRow(label("Signed in with", "pg-label"), label(PROVIDER_NAMES.get(user.provider, user.provider),
                                                           "secondary"))
        top.addLayout(f, 1)
        self.lay.addLayout(top)

        self.lay.addWidget(label("Preferences saved with your account", "h3"))
        pf = _form()
        self.units = SegmentedControl([("SI", "SI"), ("imperial", "Imperial")], "SI")
        self.units.setAccessibleName("Units")
        self.theme_pref = SegmentedControl([("light", "Light"), ("dark", "Dark")], "light")
        self.theme_pref.setAccessibleName("Theme")
        pf.addRow(label("Units", "pg-label"), self.units)
        pf.addRow(label("Theme", "pg-label"), self.theme_pref)
        self.lay.addLayout(pf)
        self.apply_here = button("Use on this computer", "link", on_click=self.apply_to_this_computer)
        self.apply_here.setToolTip("Switch this computer's units and theme to the values above.")
        self.lay.addLayout(_row(label("Used as defaults when you sign in on another computer.", "muted"),
                                self.apply_here, stretch_at=1))
        self.lay.addWidget(self.status)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.save_button = box.button(QDialogButtonBox.StandardButton.Save)
        self.save_button.setProperty("variant", "primary")
        self.sign_out_all = button("Sign out of all devices…", "ghost", on_click=self.sign_out_everywhere)
        box.addButton(self.sign_out_all, QDialogButtonBox.ButtonRole.ResetRole)
        box.accepted.connect(self.save)
        box.rejected.connect(self.reject)
        self.lay.addWidget(box)
        self._busy_widgets += [self.display_name, self.affiliation, self.change_pic, self.remove_pic,
                               self.save_button, self.units, self.theme_pref, self.sign_out_all,
                               # before the profile loads it would apply the placeholder SI / light
                               self.apply_here]
        self._paint_avatar()
        self.ctrl.changed.connect(self._on_changed)
        self.display_name.setFocus()        # the first thing to edit; restored after the load below
        self.run(self.svc.get_profile, self._fill, progress="Loading the profile…")

    def _paint_avatar(self) -> None:
        user = self.svc.user
        dpr = self.devicePixelRatioF() or 1.0
        self.avatar.setPixmap(avatar_pixmap(64, user.initials if user else "?", self.ctrl.avatar, dpr=dpr))
        self.remove_pic.setVisible(self.ctrl.avatar is not None)

    def _on_changed(self, event: str) -> None:
        if event == "avatar":
            self._paint_avatar()

    def _fill(self, prof) -> None:
        self.status.clear()
        self.profile = prof
        self.display_name.setText(prof.display_name)
        self.affiliation.setText(prof.affiliation)
        prefs = prof.preferences or {}
        from .. import settings as app_settings
        s = app_settings.get()
        self.units.set_current(prefs.get("unit_system", s.unit_system))
        self.theme_pref.set_current(prefs.get("theme", s.theme))
        if prof.avatar_path and self.ctrl.avatar is None:
            self.ctrl.load_avatar()

    def values(self) -> dict[str, Any]:
        """What Save sends. Unknown preference keys written by newer versions are kept."""
        prof = getattr(self, "profile", None)
        prefs = dict(prof.preferences) if prof is not None else {}
        prefs.update(unit_system=self.units.current(), theme=self.theme_pref.current())
        return {"display_name": self.display_name.text().strip(),
                "affiliation": self.affiliation.text().strip(), "preferences": prefs}

    def save(self) -> None:
        v = self.values()
        self.run(lambda: self.svc.update_profile(**v), lambda _p: self.accept(), progress="Saving…")

    def choose_picture(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a profile picture", "",
                                              "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self.set_picture_file(path)

    def set_picture_file(self, path: str | Path) -> None:
        try:
            png = image_to_avatar_png(Path(path).read_bytes())
        except (OSError, ValueError):
            self.status.error("That file is not a picture Aerolume can read. Use PNG or JPEG.")
            return

        def done(_prof):
            self.ctrl.avatar = QImage.fromData(png)
            self.ctrl.changed.emit("avatar")
            self.status.ok("Picture saved.")
        self.run(lambda: self.svc.set_avatar(png), done, progress="Uploading the picture…")

    def remove_picture(self) -> None:
        def done(_prof):
            self.ctrl.avatar = None
            self.ctrl.changed.emit("avatar")
            self.status.ok("Picture removed.")
        self.run(self.svc.remove_avatar, done, progress="Removing the picture…")

    def apply_to_this_computer(self) -> None:
        from .. import settings as app_settings
        s = app_settings.get()
        s.unit_system = self.units.current()
        if s.theme != self.theme_pref.current():
            s.theme = self.theme_pref.current()
            theme.apply(None, s.theme)
        self.status.ok("This computer now uses these units and theme.")

    def sign_out_everywhere(self) -> None:
        if not self.confirm("Sign out of all devices",
                            "Sign out here and on every other computer where you are signed in? Local "
                            "projects are not affected.", "Sign out everywhere", danger=True):
            return
        self.run(lambda: self.svc.sign_out(everywhere=True), lambda _r: self.accept(),
                 progress="Signing out…")


# ---------------------------------------------------------------------------
# cloud projects
# ---------------------------------------------------------------------------

_STATUS_ROLE = {"synced": "status-good", "local_changes": "secondary", "remote_changes": "secondary",
                "conflict": "status-warn", "remote_only": "muted", "local_only": "muted",
                "remote_missing": "status-warn"}
_STATUS_TOKEN = {"synced": "good", "conflict": "warn", "remote_missing": "warn", "remote_only": "muted",
                 "local_only": "muted"}
_STATUS_TIP = {
    "synced": "The cloud copy has the same content as this folder.",
    "local_changes": "This folder changed since it was last uploaded. Upload to update the cloud copy.",
    "remote_changes": "Another computer uploaded a newer version. Download to update this folder.",
    "conflict": "This folder and the cloud copy both changed since they were last synced.",
    "remote_only": "In the cloud, not on this computer. Download to get a local copy.",
    "local_only": "On this computer only.",
    "remote_missing": "This folder was linked to a cloud copy that has since been deleted.",
}


def _parts_text(parts: int, part_bytes: int) -> str:
    return f" in {parts} parts of up to {_size(part_bytes)}" if parts > 1 else ""


def size_summary(st, preview, part_bytes: int) -> tuple[str, str]:
    """(summary, problem) for the selected Cloud projects row; problem is "" when there is none.

    Answers "how big is this, and why" before anything is uploaded: the exact
    packed size (what the upload sends, not the size on disk), how many parts it
    goes up in, and the largest runs or folders, largest first. Before this the
    size appeared only in the error after an upload was refused.
    """
    lines: list[str] = []
    problem = ""
    c = getattr(st, "cloud", None)
    if preview is not None and st.local_path:
        runs = f"{preview.runs} run{'s' if preview.runs != 1 else ''}"
        if st.status == "synced" and c is not None:
            lines.append(f"Cloud copy: {_size(c.size_bytes)}{_parts_text(c.part_count, part_bytes)}, the same as "
                         f"this folder ({runs}).")
        else:
            lines.append(f"Uploading sends {_size(preview.size_bytes)}"
                         f"{_parts_text(preview.parts, preview.part_bytes)} ({runs}; "
                         f"{_size(preview.disk_bytes)} on disk).")
        groups = [g for g in preview.groups if g.size_bytes > 0]
        if groups:
            top, rest = groups[:4], groups[4:]
            items = [f"{g.label} {_size(g.size_bytes)}" for g in top]
            if rest:
                items.append(f"{len(rest)} more, {_size(sum(g.size_bytes for g in rest))}")
            lines.append("Largest: " + " · ".join(items) + ".")
        if preview.over_limit:
            mb = preview.max_project_bytes // (1024 * 1024)
            problem = (f"This is more than the server's project limit of {mb} MB. "
                       "On a paid plan raise \"max_project_mb\" in account.json (About accounts).")
    elif c is not None:
        when = _when(c.updated_at)
        lines.append(f"Cloud copy: {_size(c.size_bytes)}{_parts_text(c.part_count, part_bytes)}"
                     + (f", uploaded {when}." if when else "."))
    return "\n".join(lines), problem


class CloudProjectsDialog(_AccountDialog):
    """Cloud copies of projects with their sync status; upload, download, delete."""

    COLUMNS = ("Project", "Status", "Size", "Updated", "Folder on this computer")
    #: Progress lines from the worker ("Uploading part 2 of 3 ..."); queued to the GUI thread.
    progress = Signal(str)

    def __init__(self, ctrl: AccountController | None = None, *, state: Any = None, main_window: Any = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(ctrl, "Cloud projects", parent)
        self.state, self.main_window = state, main_window
        self.fit_height = False
        self.resize(820, 440)
        self.download_dir: Path | None = None       # tests: skip the folder dialog
        self.rows: list = []
        self.previews: dict[str, Any] = {}           # local folder -> UploadPreview
        self.progress.connect(self._on_progress)
        if not self.svc.configured or not self.svc.signed_in:
            if not self.svc.configured:
                self.lay.addWidget(_not_connected_panel(self._settings, lambda: open_accounts_doc(self)), 1)
            else:
                self.lay.addWidget(label("Not signed in", "h3"))
                self.lay.addWidget(label("Sign in from the Account menu to see your cloud projects.",
                                         "secondary"))
                self.lay.addStretch(1)
            box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            box.rejected.connect(self.reject)
            self.lay.addWidget(box)
            return
        self.lay.addWidget(label(f"Copies of your projects on {self.svc.config.server_name}. You always work in the "
                                 "local folder; uploading replaces the cloud copy, downloading never "
                                 "overwrites unsynced work.", "secondary", wrap=True))
        self.table = QTreeWidget()
        self.table.setRootIsDecorated(False)
        self.table.setUniformRowHeights(True)
        self.table.setAlternatingRowColors(False)
        self.table.setHeaderLabels(list(self.COLUMNS))
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAccessibleName("Cloud projects")
        # Folder paths share a long prefix (C:\Users\<name>\Documents\...); eliding at the
        # end hid the one part that tells two rows apart, the project folder's name.
        self.table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        hdr = self.table.header()
        hdr.setStretchLastSection(True)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        for col, w in ((0, 220), (1, 130), (2, 70), (3, 140)):
            self.table.setColumnWidth(col, w)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.itemDoubleClicked.connect(lambda *_: self.open_selected())
        self.lay.addWidget(self.table, 1)
        self.empty = label("No cloud projects yet. Open a project and choose Upload current project.", "muted")
        self.empty.setVisible(False)
        self.lay.addWidget(self.empty)
        # Size of the selected row before anything is sent: packed size, parts, largest runs.
        self.details = label("", "secondary", wrap=True)
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details.setAccessibleName("Size of the selected project")
        self.details.setVisible(False)
        self.lay.addWidget(self.details)
        self.details_problem = label("", "status-bad", wrap=True)
        self.details_problem.setVisible(False)
        self.lay.addWidget(self.details_problem)
        self.upload = button("Upload current project", "primary", on_click=self.upload_current)
        self.upload_sel = button("Upload", on_click=self.upload_selected)
        self.download = button("Download", on_click=self.download_selected)
        self.open_btn = button("Open", on_click=self.open_selected)
        self.delete = button("Delete from cloud…", "danger", on_click=self.delete_selected)
        self.refresh_btn = button("Refresh", "ghost", on_click=self.refresh)
        self.lay.addLayout(_row(self.upload, self.upload_sel, self.download, self.open_btn, self.delete,
                                self.refresh_btn, stretch_at=5))
        self.lay.addWidget(self.status)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        self.lay.addWidget(box)
        self._busy_widgets += [self.upload, self.upload_sel, self.download, self.delete, self.refresh_btn,
                               self.open_btn]
        self._update_buttons()
        # Item colours are concrete QColors, not stylesheet tokens, so they must be re-read when
        # the theme changes (Preferences, or the OS switching under "System"): measured before
        # this, the light-theme green/grey left on the dark surface drew at 2.2:1.
        theme.changed.connect(self._recolour)      # bound slot: disconnected when the dialog dies
        self.refresh()

    def _settings(self) -> None:
        self.reject()
        AccountSettingsDialog(self.ctrl, parent=self.parentWidget()).open()

    # -- data -----------------------------------------------------------------------
    def _current_path(self) -> Path | None:
        proj = getattr(self.state, "project", None) if self.state is not None else None
        return Path(proj.path) if proj is not None else None

    def refresh(self, *, keep_status: bool = False) -> None:
        cur = self._current_path()
        svc = self.svc

        def load():
            from ...accounts import AccountError
            rows = svc.sync_overview([cur] if cur else [])
            previews = {}
            for st in rows:          # local work only (file sizes, exports deflated once, cached by mtime)
                if st.local_path and st.local_path not in previews:
                    try:
                        previews[st.local_path] = svc.upload_preview(st.local_path)
                    except (OSError, ValueError, AccountError):
                        pass         # an unreadable folder still lists; it just shows no size
            return rows, previews
        self.run(load, lambda r: self._fill(*r, keep_status=keep_status),
                 progress="" if keep_status else "Reading the cloud…")

    def _fill(self, rows, previews: dict | None = None, *, keep_status: bool = False) -> None:
        if not keep_status:
            self.status.clear()
        previous = self.selected()
        self.rows = list(rows)
        self.previews = dict(previews or {})
        self.table.clear()
        cur = self._current_path()
        for st in self.rows:
            folder = st.local_path or "—"
            if cur is not None and st.local_path and Path(st.local_path) == cur:
                folder += "  (open)"
            pv = self.previews.get(st.local_path) if st.local_path else None
            if st.cloud and (st.status == "synced" or pv is None):
                size, size_tip = _size(st.cloud.size_bytes), "Packed size of the cloud copy."
            elif pv is not None:
                size, size_tip = _size(pv.size_bytes), "Packed size this folder would upload."
            else:
                size, size_tip = "", ""
            when = _when(st.cloud.updated_at) if st.cloud else ""
            it = QTreeWidgetItem([st.name, st.label, size, when, folder])
            it.setData(0, Qt.ItemDataRole.UserRole, st)
            it.setToolTip(1, _STATUS_TIP.get(st.status, ""))
            it.setToolTip(2, size_tip)
            it.setToolTip(4, st.local_path)
            it.setTextAlignment(2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.addTopLevelItem(it)
        self._recolour()
        self.empty.setVisible(not self.rows)
        if self.rows:
            self.table.setCurrentItem(self.table.topLevelItem(self._row_to_select(previous, cur)))
        self._update_buttons()

    def _row_to_select(self, previous, cur: Path | None) -> int:
        """Keep the row the user had selected across a refresh; else the open project's folder; else the first.

        The size details describe the selected row, so the first-row default put a
        cloud-only copy's size where the open project's (the one Upload acts on) belonged.
        """
        def same(a, b) -> bool:
            if a.local_path or b.local_path:
                return bool(a.local_path and b.local_path and Path(a.local_path) == Path(b.local_path))
            return bool(a.cloud and b.cloud and a.cloud.id == b.cloud.id)
        if previous is not None:
            for i, st in enumerate(self.rows):
                if same(st, previous):
                    return i
        if cur is not None:
            for i, st in enumerate(self.rows):
                if st.local_path and Path(st.local_path) == cur:
                    return i
        return 0

    def _set_busy(self, busy: bool) -> None:
        """Busy re-enables every action afterwards; put back the ones the selection rules out
        (measured: after a refused upload, Open and Upload were enabled for a cloud-only row)."""
        super()._set_busy(busy)
        if not busy and hasattr(self, "table"):
            self._update_buttons()

    def _recolour(self, _theme: str = "") -> None:
        """Status text in the current theme's status colour (each token passes 4.5:1 on its surface)."""
        t = theme.tokens()
        for i in range(self.table.topLevelItemCount()):
            it = self.table.topLevelItem(i)
            st = it.data(0, Qt.ItemDataRole.UserRole)
            tok = _STATUS_TOKEN.get(getattr(st, "status", ""))
            if tok:
                it.setForeground(1, theme.qcolor(t.get(tok, tok)))

    def selected(self):
        it = self.table.currentItem() if hasattr(self, "table") else None
        return it.data(0, Qt.ItemDataRole.UserRole) if it is not None else None

    def _update_buttons(self) -> None:
        st = self.selected()
        cur = self._current_path()
        self.upload.setEnabled(cur is not None)
        self.upload.setToolTip("" if cur is not None else "Open a project first.")
        self.upload_sel.setEnabled(bool(st and st.local_path and st.status in (
            "local_only", "local_changes", "conflict", "remote_missing")))
        self.download.setEnabled(bool(st and st.cloud and st.status in ("remote_only", "remote_changes",
                                                                        "conflict", "synced")))
        self.download.setText("Update local folder" if st and st.status == "remote_changes" else "Download")
        self.open_btn.setEnabled(bool(st and st.local_path and self.main_window is not None
                                      and hasattr(self.main_window, "open_project_path")))
        self.delete.setEnabled(bool(st and st.cloud))
        self._show_details(st)

    def _show_details(self, st) -> None:
        text, problem = ("", "") if st is None else size_summary(
            st, self.previews.get(st.local_path) if st.local_path else None, self.svc.config.part_bytes)
        self.details.setText(text)
        self.details.setAccessibleDescription(text)
        self.details.setVisible(bool(text))
        self.details_problem.setText(problem)
        self.details_problem.setAccessibleDescription(problem)
        self.details_problem.setVisible(bool(problem))

    def _on_progress(self, text: str) -> None:
        if text:
            self.status.info(text + "…")

    def _report(self, text: str) -> None:
        """Worker-side progress callback: hand the line to the GUI thread (dialog may be gone)."""
        try:
            self.progress.emit(text)
        except RuntimeError:            # the dialog was closed; the upload carries on
            pass

    # -- actions --------------------------------------------------------------------
    def upload_current(self) -> None:
        cur = self._current_path()
        if cur is None:
            return
        try:
            if self.state is not None:
                self.state.save()        # upload what the user sees, not the last autosave
        except Exception as exc:  # noqa: BLE001
            self.status.error(f"Could not save the project first: {exc}")
            return
        self._upload(cur)

    def upload_selected(self) -> None:
        st = self.selected()
        if st and st.local_path:
            self._upload(Path(st.local_path))

    def _upload(self, folder: Path, force: bool = False) -> None:
        from ...accounts import SyncConflictError

        def failed(exc):
            if isinstance(exc, SyncConflictError):
                self._resolve_conflict(folder, exc)
            else:
                if getattr(exc, "code", "") == "too_large":
                    self.details_problem.setVisible(False)     # the status line now says it in full
                self.status.error(_error_text(exc))
        def done(st):
            c = st.cloud
            if c is None:
                self.status.ok(f"Uploaded {st.name}.")
            else:
                self.status.ok(f"Uploaded {st.name}: {_size(c.size_bytes)}"
                               f"{_parts_text(c.part_count, self.svc.config.part_bytes)}.")
            self.refresh(keep_status=True)
        self.run(lambda: self.svc.upload_project(folder, force=force, progress=self._report), done, failed,
                 progress=f"Preparing {folder.name}…")

    def ask_conflict(self, both: bool) -> str:
        """"upload", "download" or "" (cancel). A method so tests can answer it."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Cloud copy changed")
        if both:
            box.setText("This folder and the cloud copy both changed since they were last synced.")
            box.setInformativeText("Choose which version to keep. Downloading puts the cloud version in a "
                                   "new folder, so this folder is not changed.")
        else:
            box.setText("The cloud copy is newer than this folder.")
            box.setInformativeText("Another computer uploaded changes. Download them, or replace the cloud "
                                   "copy with this folder.")
        up = box.addButton("Replace cloud copy with mine", QMessageBox.ButtonRole.DestructiveRole)
        down = box.addButton("Download cloud version", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(down)
        box.exec()
        clicked = box.clickedButton()
        return "upload" if clicked is up else "download" if clicked is down else ""

    def _resolve_conflict(self, folder: Path, exc) -> None:
        local_changed = True
        link = self.svc.link_status(folder)
        if link is not None:
            from ...accounts import cloud
            local_changed = cloud.content_hash(folder) != link.base_hash
        choice = self.ask_conflict(local_changed)
        if choice == "upload":
            self._upload(folder, force=True)
        elif choice == "download":
            if local_changed:
                self._download(exc.project_id, folder.parent, replace=None)
            else:
                self._download(exc.project_id, None, replace=folder)
        else:
            self.status.info("Nothing was changed.")

    def download_selected(self) -> None:
        st = self.selected()
        if not st or not st.cloud:
            return
        if st.status == "remote_changes" and st.local_path:
            if self._is_open(st.local_path):
                self.status.error("This project is open. Close it (File > Close project), then update it.")
                return
            if not self.confirm("Update local folder", f"Replace {st.local_path} with the newer cloud "
                                "version? The folder has no changes that were not uploaded.", "Update"):
                return
            self._download(st.cloud.id, None, replace=Path(st.local_path))
            return
        parent = self.download_dir
        if parent is None:
            from .. import settings as app_settings
            start = str(app_settings.default_projects_dir())
            chosen = QFileDialog.getExistingDirectory(self, "Download into folder", start)
            if not chosen:
                return
            parent = Path(chosen)
        self._download(st.cloud.id, parent, replace=None)

    def _download(self, project_id: str, parent: Path | None, replace: Path | None) -> None:
        def done(path):
            self.last_download = Path(path)
            self.status.ok(f"Downloaded to {path}.")
            self.refresh(keep_status=True)
        self.run(lambda: self.svc.download_project(project_id, parent, replace=replace, progress=self._report),
                 done, progress="Downloading…")

    def _is_open(self, path: str) -> bool:
        cur = self._current_path()
        return cur is not None and Path(path).resolve() == cur.resolve()

    def open_selected(self) -> None:
        st = self.selected()
        if st and st.local_path and self.main_window is not None and hasattr(self.main_window, "open_project_path"):
            self.accept()
            self.main_window.open_project_path(st.local_path)

    def delete_selected(self) -> None:
        st = self.selected()
        if not st or not st.cloud:
            return
        if not self.confirm("Delete cloud copy", f"Delete the cloud copy of {st.name}? Folders on this "
                            "computer are not changed. This cannot be undone.", "Delete cloud copy",
                            danger=True):
            return
        self.run(lambda: self.svc.delete_cloud_project(st.cloud.id),
                 lambda _r: (self.status.ok(f"Deleted the cloud copy of {st.name}."),
                             self.refresh(keep_status=True)),
                 progress="Deleting…")


# ---------------------------------------------------------------------------
# share links
# ---------------------------------------------------------------------------

EXPIRY_CHOICES = ((7, "7 days"), (30, "30 days"), (90, "90 days"), (365, "1 year"))


class ShareDialog(_AccountDialog):
    """Create a read-only, expiring link to a report; list and revoke earlier links."""

    def __init__(self, ctrl: AccountController | None = None, *, state: Any = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(ctrl, "Share result", parent)
        self.state = state
        self.fit_height = False
        self.resize(640, 480)
        self.shares: list = []
        project = getattr(state, "project", None) if state is not None else None
        if not self.svc.configured or not self.svc.signed_in or project is None:
            if not self.svc.configured:
                self.lay.addWidget(_not_connected_panel(self._settings, lambda: open_accounts_doc(self)), 1)
            else:
                self.lay.addWidget(label("Not signed in" if not self.svc.signed_in else "No project open", "h3"))
                self.lay.addWidget(label("Sign in from the Account menu to share results."
                                         if not self.svc.signed_in else "Open a project with a report first.",
                                         "secondary"))
                self.lay.addStretch(1)
            box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            box.rejected.connect(self.reject)
            self.lay.addWidget(box)
            return
        self.project = project
        reports = sorted(Path(project.path, "reports").glob("*.htm*"), key=lambda p: p.stat().st_mtime,
                         reverse=True) if Path(project.path, "reports").is_dir() else []
        f = _form()
        self.report = QComboBox()
        self.report.setAccessibleName("Report")
        for p in reports:
            self.report.addItem(p.name, str(p))
        f.addRow(label("Report", "pg-label"), self.report)
        self.title = _field("Title")
        self.title.setMaxLength(200)
        self.title.setText(getattr(project, "name", ""))
        f.addRow(label("Title", "pg-label"), self.title)
        f.addRow(QLabel(""), _hint(self.title, "Shown to people who open the link."))
        self.with_pdf = QCheckBox("Include the PDF version")
        f.addRow(QLabel(""), self.with_pdf)
        # Without a PDF the box would be a greyed-out control whose reason is disabled-grey
        # text (2.2:1) or a tooltip keyboard users never see; a muted line (>= 4.5:1) says it.
        self.no_pdf = label("No PDF of this report to include. Export one on the Report step.", "muted",
                            wrap=True)
        f.addRow(QLabel(""), self.no_pdf)
        self.pdf_form = f
        self.expiry = QComboBox()
        self.expiry.setAccessibleName("Link expires after")
        for days, text in EXPIRY_CHOICES:
            self.expiry.addItem(text, days)
        self.expiry.setCurrentIndex(1)
        f.addRow(label("Expires after", "pg-label"), self.expiry)
        self.lay.addLayout(f)
        self.lay.addWidget(label("Anyone with the link can open these files until it expires. Revoking "
                                 "deletes them from the cloud and ends the link at once.", "muted", wrap=True))
        self.create = button("Create link", "primary", on_click=self.create_link)
        self.lay.addLayout(_row(self.create, stretch_at=1))
        self.link = QLineEdit()
        self.link.setReadOnly(True)
        self.link.setAccessibleName("Share link")
        self.link.setPlaceholderText("The link appears here")
        self.copy = button("Copy link", icon="copy", on_click=lambda: self._copy(self.link.text()))
        self.copy.setEnabled(False)
        self.lay.addLayout(_row(self.link, self.copy))
        self.lay.addWidget(self.status)
        if not reports:
            self.status.error("This project has no HTML report yet. Create one on the Report step "
                              "(format HTML), then share it.")
            self.create.setEnabled(False)
        self.report.currentIndexChanged.connect(self._update_pdf)
        self._update_pdf()

        self.lay.addWidget(label("Your links", "h3"))
        self.table = QTreeWidget()
        self.table.setRootIsDecorated(False)
        self.table.setHeaderLabels(["Title", "Created", "Expires"])
        self.table.setAccessibleName("Your share links")
        self.table.setColumnWidth(0, 260)
        self.table.setColumnWidth(1, 150)
        self.table.itemSelectionChanged.connect(self._update_list_buttons)
        self.lay.addWidget(self.table, 1)
        # An empty table alone reads as "still loading" or "broken"; say what it will hold.
        self.no_links = label("No share links yet. Links you create appear here until they expire "
                              "or you revoke them.", "muted", wrap=True)
        self.no_links.setVisible(False)
        self.lay.addWidget(self.no_links)
        self.copy_old = button("Copy link", on_click=self.copy_selected)
        self.revoke = button("Revoke…", "danger", on_click=self.revoke_selected)
        self.lay.addLayout(_row(self.copy_old, self.revoke, stretch_at=2))
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        self.lay.addWidget(box)
        self._busy_widgets += [self.create, self.revoke, self.copy_old]
        self._update_list_buttons()
        self.run(self.svc.list_shares, self._fill_shares)

    def _settings(self) -> None:
        self.reject()
        AccountSettingsDialog(self.ctrl, parent=self.parentWidget()).open()

    def _update_pdf(self) -> None:
        path = self.report.currentData()
        pdf = Path(path).with_suffix(".pdf") if path else None
        ok = bool(pdf and pdf.is_file())
        self.with_pdf.setEnabled(ok)
        self.with_pdf.setChecked(ok)
        self.pdf_form.setRowVisible(self.with_pdf, ok)
        self.pdf_form.setRowVisible(self.no_pdf, not ok)

    def files(self) -> list[Path]:
        path = self.report.currentData()
        if not path:
            return []
        out = [Path(path)]
        if self.with_pdf.isChecked() and Path(path).with_suffix(".pdf").is_file():
            out.append(Path(path).with_suffix(".pdf"))
        return out

    def create_link(self) -> None:
        files = self.files()
        if not files:
            return
        title, days = self.title.text().strip(), int(self.expiry.currentData())
        pid = ""
        link = self.svc.link_status(self.project.path)
        if link is not None:
            pid = link.project_id

        def done(share):
            self.link.setText(share.url)
            self.link.setCursorPosition(0)          # show where the link starts, not its signature
            self.copy.setEnabled(True)
            self.status.ok(f"Link created. It expires on {_when(share.expires_at)}.")
            self.run(self.svc.list_shares, self._fill_shares)
        self.run(lambda: self.svc.create_share(files, title=title, project_id=pid, expires_days=days), done,
                 progress="Uploading the report…")

    def _fill_shares(self, shares) -> None:
        self.shares = list(shares)
        self.table.clear()
        for s in self.shares:
            it = QTreeWidgetItem([s.title, _when(s.created_at), _when(s.expires_at)])
            it.setData(0, Qt.ItemDataRole.UserRole, s)
            self.table.addTopLevelItem(it)
        self.no_links.setVisible(not self.shares)
        self._update_list_buttons()

    def _selected_share(self):
        it = self.table.currentItem()
        return it.data(0, Qt.ItemDataRole.UserRole) if it is not None else None

    def _update_list_buttons(self) -> None:
        has = self._selected_share() is not None
        self.copy_old.setEnabled(has)
        self.revoke.setEnabled(has)

    def _copy(self, text: str) -> None:
        if text:
            QGuiApplication.clipboard().setText(text)
            self.status.ok("Link copied.")

    def copy_selected(self) -> None:
        s = self._selected_share()
        if s is not None:
            self._copy(s.url)

    def revoke_selected(self) -> None:
        s = self._selected_share()
        if s is None or not self.confirm("Revoke link", f"Revoke the link to “{s.title}”? Its files "
                                         "are deleted from the cloud and the link stops working.",
                                         "Revoke link", danger=True):
            return
        self.run(lambda: self.svc.revoke_share(s.token),
                 lambda _r: (self.status.ok("Link revoked."), self.run(self.svc.list_shares, self._fill_shares)),
                 progress="Revoking…")


# ---------------------------------------------------------------------------
# settings (Account settings; also meant for Preferences > Account)
# ---------------------------------------------------------------------------

class AccountSettingsPage(QWidget):
    """The Account tab of Preferences: which Supabase project, a connection test, and what is sent.

    :meth:`apply` writes ``account.json`` (aerolume.accounts.config) and
    returns True when the connection changed. The key field starts empty and
    shows the saved key masked beneath it, like the Assistant tab's API key.
    """

    def __init__(self, ctrl: AccountController | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from ...accounts import config as C
        self.C = C
        self.ctrl = ctrl or controller()
        self.cfg = C.load_config()
        self.last_error = ""
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2 * SPACING, 0, 0)
        lay.setSpacing(3 * SPACING)
        lay.addWidget(label("Optional. Aerolume works fully offline without an account. Connect a Supabase "
                            "project to sign in, keep cloud copies of projects and share results; "
                            "supabase/README.md shows how to create one.", "secondary", wrap=True))
        f = _form()
        self.url = _field("Project URL", "https://<project-ref>.supabase.co")
        self.url.setText(self.cfg.url)
        f.addRow(label("Project URL", "pg-label"), self.url)
        # No placeholder: the instruction is the persistent key_hint below the field.
        self.key = _field("Anon or publishable key", password=True)
        f.addRow(label("Anon key", "pg-label"), self.key)
        self.key_hint = label("", "muted")
        f.addRow(QLabel(""), self.key_hint)
        lay.addLayout(f)
        self.test = button("Test connection", on_click=self.test_connection)
        self.disconnect = button("Disconnect", "ghost", on_click=self.clear)
        self.disconnect.setEnabled(self.cfg.source == "file")       # nothing saved: nothing to disconnect
        lay.addLayout(_row(self.test, self.disconnect, stretch_at=2))
        self.status = StatusLine()
        lay.addWidget(self.status)
        if C.env_override():
            for w in (self.url, self.key, self.disconnect):
                w.setEnabled(False)
            self.key_hint.setText(f"Set by {C.ENV_URL} and {C.ENV_KEY} in the environment.")
        elif self.cfg.anon_key:
            self.key_hint.setText(f"Saved key: {C.mask_key(self.cfg.anon_key)}. Paste a new one to replace it.")
        else:
            self.key_hint.setText("Project Settings > API in the Supabase dashboard. Never the service_role key.")
        self.key.setAccessibleDescription(self.key_hint.text())
        problem = C.config_problem()
        if problem:
            self.status.error(problem)
        elif self.cfg.configured:
            # Say where things stand before anything is tested: a settings page that shows a URL
            # but not whether you are signed in to it leaves the one question unanswered.
            svc = self.ctrl.service
            user = svc.user if svc.config.url == self.cfg.url else None
            self.status.info(f"Signed in as {user.email}." if user is not None
                             else "Not signed in. Sign in from the Account button at the top right.")

        lay.addWidget(label("What leaves this computer", "h3"))
        from ...accounts.session import SESSION_FILE
        from ...project.recent import config_dir
        store = "encrypted for your Windows account" if sys.platform == "win32" else "readable only by you"
        facts = [
            "Nothing while you are signed out.",
            "Signed in: your email address and the profile you fill in.",
            "A project only when you upload it, and a report only when you create a share link.",
            f"Sign-in tokens are stored {store} in {config_dir() / SESSION_FILE}.",
        ]
        bullets = QVBoxLayout()
        bullets.setSpacing(SPACING)
        for text in facts:
            row = label("•  " + text, "secondary", wrap=True)
            row.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            bullets.addWidget(row)
        lay.addLayout(bullets)
        lay.addStretch(1)

    def minimumSizeHint(self) -> QSize:      # Qt override
        """Minimum height = the content's height at the page's current width.

        The wrapped labels use a Minimum vertical policy, so the layout's own
        minimum adds up their narrow-column (golden-ratio) heights: 492 px for
        a page that needs 300 px at 560 px wide. As a Preferences tab that
        stretched the whole dialog, and on the other tabs left a band of empty
        space. The real height at the real width is the honest minimum.
        """
        base = super().minimumSizeHint()
        lay = self.layout()
        if lay is None or not lay.hasHeightForWidth():
            return base
        w = max(self.width(), base.width())
        return QSize(base.width(), lay.totalHeightForWidth(w))

    def values(self) -> tuple[str, str]:
        key = self.key.text().strip() or (self.cfg.anon_key if self.cfg.source == "file" else "")
        return self.url.text().strip(), key

    def clear(self) -> None:
        self.url.clear()
        self.key.clear()
        self.cfg = self.C.AccountConfig()
        self.key_hint.setText("Disconnected when you press OK. Local projects are not affected.")

    def test_connection(self) -> None:
        url, key = self.values()
        try:
            url = self.C.normalise_url(url)
        except Exception as exc:  # noqa: BLE001
            self.status.error(str(exc))
            return
        if not url or not key:
            self.status.error("Enter the project URL and the anon key first.")
            return
        from ...accounts import AccountService
        from ...accounts.service import SCHEMA_VERSION
        cfg = self.C.AccountConfig(url, key, "test")
        svc = AccountService(cfg)
        self.status.info("Testing…")
        self.test.setEnabled(False)

        def done(rep):
            self.test.setEnabled(True)
            self.report = rep
            extra = ""
            # An older schema (a project set up for an earlier Aerolume) is not "connected, all
            # good": uploads in parts would fail, so it is reported as the error it will become.
            good = bool(rep.key_accepted and rep.schema_version and rep.schema_version >= SCHEMA_VERSION)
            if good:
                names = ", ".join(PROVIDER_NAMES[p] for p in rep.providers) or "none"
                extra = f" Browser sign-in: {names}. New accounts: {'allowed' if rep.signup_enabled else 'off'}."
            (self.status.ok if good else self.status.error)(rep.message + extra)

        def failed(exc):
            self.test.setEnabled(True)
            self.status.error(_error_text(exc))
        self.ctrl.run(svc.test_connection, done, failed)

    def apply(self) -> bool:
        """Save the connection; True when it changed (the service then switches project).

        An invalid entry is not saved: :attr:`last_error` holds the reason (also shown
        in the status line) and the previous connection stays in force.
        """
        self.last_error = ""
        if self.C.env_override():
            return False
        url, key = self.values()
        before = self.C.load_config()
        try:
            cfg = self.C.save_config(url, key)
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            self.status.error(self.last_error)
            return False
        changed = (cfg.url, cfg.anon_key) != (before.url, before.anon_key)
        if changed:
            self.ctrl.service.set_config(cfg)
        return changed


class AccountSettingsDialog(_AccountDialog):
    """:class:`AccountSettingsPage` on its own (from the Account menu), applied on OK."""

    def __init__(self, ctrl: AccountController | None = None, parent: QWidget | None = None) -> None:
        super().__init__(ctrl, "Account settings", parent)
        self.setMinimumWidth(560)
        self.page = AccountSettingsPage(self.ctrl)
        self.lay.addWidget(self.page, 1)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        box.button(QDialogButtonBox.StandardButton.Ok).setProperty("variant", "primary")
        box.accepted.connect(self._ok)
        box.rejected.connect(self.reject)
        self.lay.addWidget(box)

    def _ok(self) -> None:
        self.page.apply()
        if not self.page.last_error:
            self.accept()


__all__ = ["AccountSettingsDialog", "AccountSettingsPage", "CloudProjectsDialog", "ProfileDialog", "ShareDialog",
           "SignInDialog", "StatusLine", "image_to_avatar_png", "open_accounts_doc"]


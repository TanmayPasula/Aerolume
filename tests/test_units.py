"""Unit parsing: commas are read only where they have one meaning; non-finite values are refused.

"2,5" in the speed box used to become 25 m/s and "0,25 m" a 25 m chord: every
comma was dropped as a thousands separator, and the 10x-100x values passed the
boxes' range checks.
"""
from __future__ import annotations

import math
import os

import pytest

from aerolume.core import units


@pytest.mark.parametrize("text, kind, si", [
    ("2,5", "speed", 2.5),                  # a decimal comma
    ("0,25 m", "length", 0.25),
    ("0,5", "angle", 0.5),
    ("0,250", "length", 0.25),              # a leading zero is never a thousands group
    ("1234,567", "length", 1234.567),       # not a valid grouping: decimal
    (",5", "length", 0.5),
    ("2,5e3", "length", 2500.0),
    ("1,234.5", "length", 1234.5),          # thousands groups with a decimal point
    ("12,345,678", "length", 12345678.0),
    ("120 km/h", "speed", 120 / 3.6),
    ("4.5 ft", "length", 4.5 * 0.3048),
    ("-3", "angle", -3.0),
])
def test_unambiguous_commas_and_units(text, kind, si):
    assert units.parse(text, kind) == pytest.approx(si)


@pytest.mark.parametrize("text", ["1,500", "1,500 m", "12,345"])
def test_a_comma_before_exactly_three_digits_is_refused_as_ambiguous(text):
    """'1,500' is fifteen hundred in English and one and a half in German: no guess."""
    with pytest.raises(ValueError, match="ambiguous"):
        units.parse(text, "length")


@pytest.mark.parametrize("text", ["1,23.4", "1.234,5", "1,2,3", "2,", "1e400", "nan", "inf", "-inf", "abc"])
def test_malformed_and_non_finite_values_are_refused(text):
    with pytest.raises(ValueError):
        units.parse(text, "length")


def test_non_finite_numbers_are_refused_too():
    with pytest.raises(ValueError):
        units.parse(float("inf"), "speed")
    with pytest.raises(ValueError):
        units.parse(float("nan"), "speed")


def test_text_written_into_an_editable_field_parses_back():
    """sig() groups thousands for reading; fields use grouping=False, which round-trips."""
    assert units.sig(1234.5) == "1,234"
    assert units.sig(1234.5, grouping=False) == "1234"
    for v in (0.00123, 1.5, 999.9, 1234.5, 98765.0, 4.5e6):
        text = units.sig(v, 4, grouping=False)
        assert math.isclose(units.parse(text, "length"), float(text), rel_tol=0, abs_tol=0)


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    return widgets.QApplication.instance() or widgets.QApplication([])


@pytest.mark.gui
def test_quantity_spin_box_reads_a_decimal_comma(qapp):
    """The real Setup widget: '2,5' typed into the speed box commits 2.5 m/s, '1,500' is not accepted."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QValidator
    from PySide6.QtTest import QTest

    from aerolume.app.widgets.param_form import QuantitySpinBox
    box = QuantitySpinBox("speed", "m/s", minimum=0.01, maximum=400.0, show_suffix=False)
    got = []
    box.siValueChanged.connect(got.append)
    box.lineEdit().selectAll()
    QTest.keyClicks(box.lineEdit(), "2,5")
    QTest.keyClick(box.lineEdit(), Qt.Key.Key_Return)
    assert box.si_value() == pytest.approx(2.5) and got and got[-1] == pytest.approx(2.5)
    assert box.validate("1,500", 0)[0] == QValidator.State.Intermediate
    assert box.validate("2,5", 0)[0] == QValidator.State.Acceptable


@pytest.mark.gui
def test_unit_line_edit_round_trips_its_own_text(qapp):
    """A value of 1234.5 m is shown as '1234 m' (no grouping), which parses back unchanged."""
    from aerolume.app.widgets.common import UnitLineEdit
    e = UnitLineEdit("length", "m", 1234.5)
    assert "," not in e.text()
    e._commit()
    assert e.is_valid() and e.value() == pytest.approx(1234.0, abs=1.0)

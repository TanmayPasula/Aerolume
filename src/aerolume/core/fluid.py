"""Fluid properties: air from the ISA or a temperature, water, custom fluids.

The ISA here is the 1976 standard troposphere + lower stratosphere (to 20 km),
which covers every road vehicle and almost every aircraft section study.
Air viscosity follows Sutherland's law; water viscosity follows the Vogel
equation fitted to IAPWS data (better than 1 % from 0 to 100 C).
"""
from __future__ import annotations

import math
import re

from ..model import Fluid

R_AIR = 287.05287          # J/(kg K)
GAMMA_AIR = 1.4
G0 = 9.80665
T0, P0 = 288.15, 101325.0
LAPSE = -0.0065            # K/m, troposphere
H_TROPOPAUSE = 11000.0


def sutherland_viscosity(temperature_k: float) -> float:
    """Dynamic viscosity of air [Pa s] from Sutherland's law."""
    mu_ref, t_ref, s = 1.716e-5, 273.15, 110.4
    return mu_ref * (temperature_k / t_ref) ** 1.5 * (t_ref + s) / (temperature_k + s)


def isa(altitude_m: float) -> tuple[float, float, float]:
    """(temperature K, pressure Pa, density kg/m3) of the ISA at a geopotential altitude."""
    h = float(altitude_m)
    if h < -1000.0 or h > 20000.0:
        raise ValueError("ISA model here covers -1 km to 20 km")
    if h <= H_TROPOPAUSE:
        t = T0 + LAPSE * h
        p = P0 * (t / T0) ** (-G0 / (LAPSE * R_AIR))
    else:
        t11 = T0 + LAPSE * H_TROPOPAUSE
        p11 = P0 * (t11 / T0) ** (-G0 / (LAPSE * R_AIR))
        t = t11
        p = p11 * math.exp(-G0 * (h - H_TROPOPAUSE) / (R_AIR * t11))
    return t, p, p / (R_AIR * t)


def air_isa(altitude_m: float = 0.0) -> Fluid:
    """Air at an ISA altitude."""
    t, _, rho = isa(altitude_m)
    label = "sea level" if altitude_m == 0 else f"{altitude_m:,.0f} m"
    return Fluid(name=f"Air, ISA {label}", density=rho, viscosity=sutherland_viscosity(t))


def air(temperature_c: float = 15.0, pressure_pa: float = P0) -> Fluid:
    """Dry air at a temperature and pressure."""
    t = temperature_c + 273.15
    rho = pressure_pa / (R_AIR * t)
    return Fluid(name=f"Air, {temperature_c:g} C", density=rho, viscosity=sutherland_viscosity(t))


def water(temperature_c: float = 20.0) -> Fluid:
    """Fresh water at a temperature (0..100 C)."""
    t = temperature_c + 273.15
    mu = 2.414e-5 * 10.0 ** (247.8 / (t - 140.0))
    # Kell (1975) density polynomial, kg/m3
    c = temperature_c
    rho = (999.83952 + 16.945176 * c - 7.9870401e-3 * c**2 - 46.170461e-6 * c**3
           + 105.56302e-9 * c**4 - 280.54253e-12 * c**5) / (1 + 16.879850e-3 * c)
    return Fluid(name=f"Water, {temperature_c:g} C", density=rho, viscosity=mu)


def speed_of_sound(temperature_k: float) -> float:
    return math.sqrt(GAMMA_AIR * R_AIR * temperature_k)


PRESETS = {
    "Air, ISA sea level": lambda: air_isa(0.0),
    "Air, 20 C": lambda: air(20.0),
    "Air, ISA 1,000 m": lambda: air_isa(1000.0),
    "Air, ISA 3,000 m": lambda: air_isa(3000.0),
    "Water, 20 C": lambda: water(20.0),
}


#: A temperature at the end of a fluid name written with a bare "C" ("Air, 20 C"): the spelling
#: :func:`air` and :func:`water` have always used. A name already reading "°C" does not match.
_BARE_CELSIUS_RE = re.compile(r"(\d) C$")


def display_name(name: str) -> str:
    """A fluid name as people should read it: "Air, 20 C" becomes "Air, 20 °C".

    The plain "C" spelling stays in :class:`Fluid` names because saved projects store that name and
    :data:`PRESETS` is keyed by it, so only text shown to people (the app, reports, run messages)
    goes through here. Idempotent: a name that already reads "°C" and names without a temperature
    are returned unchanged."""
    return _BARE_CELSIUS_RE.sub(r"\1 °C", str(name))

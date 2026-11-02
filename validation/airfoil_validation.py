"""Validation of the viscous-inviscid airfoil solver against wind-tunnel data.

Run ``python validation/airfoil_validation.py`` to recompute every case, print
the comparison tables used in docs/validation_airfoil.md and render the
figures to docs/img/airfoil_*.png (matplotlib Figure + Agg canvas, no pyplot).
``tests/test_airfoil_validation.py`` imports the case functions and reference
data from here and asserts on the same numbers, so the document and the tests
cannot drift apart.

Reference data (only the handful of values compared are transcribed):

* NACA 0012, Re = 6e6, M = 0.15, transition tripped with 80-grit roughness:
  C. L. Ladson, "Effects of independent variation of Mach and Reynolds numbers
  on the low-speed aerodynamic characteristics of the NACA 0012 airfoil
  section", NASA TM-4074, 1988 (tabulated cl, cd; as distributed by the NASA
  Turbulence Modeling Resource, file CLCD_Ladson_expdata.dat).
* NACA 0012, Re = 3e6, tripped: D. Gregory and C. L. O'Reilly, "Low-speed
  aerodynamic characteristics of NACA 0012 aerofoil section, including the
  effects of upper-surface roughness simulating hoar frost", ARC R&M 3726,
  1970 (cl digitised from the report's plot by the NASA Turbulence Modeling
  Resource, file CL_Gregory_expdata.dat; ~0.01 reading accuracy).
* NACA 0012, Re = 6e6, smooth model (free transition): I. H. Abbott and A. E.
  von Doenhoff, "Theory of Wing Sections", Dover 1959, Appendix IV (cl and the
  cd-cl drag polar, digitised from the book's plots by the NASA Turbulence
  Modeling Resource, files 0012.abbottdata.cl.dat / .cd.dat; "digitizing only
  approximate").
* NACA 2412 and 4412, Re ~ 6e6, smooth: Abbott & von Doenhoff Appendix IV
  section-characteristic plots, read as summarised in Marchman, "Aerodynamics
  and Aircraft Performance" 3rd ed., Appendix (zero-lift angle, cl at zero
  angle, cl_max) and Anderson, "Fundamentals of Aerodynamics" (thin-airfoil
  examples; cm_c/4). These are plot readings, good to about +-0.3 deg,
  +-0.03 in cl and +-0.01 in cm.

Code-to-code reference: XFOIL 6.99 polars of NACA 0012 / 2412 / 4412 at Re 1e6
and 3e6, run on the identical coordinates (XFOIL_REF below). Wind-tunnel data
say whether the *model* is right; XFOIL says whether the *implementation*
reproduces the model, point by point, including the non-linear stretch of
the lift curve where a surface's transition point runs off the trailing edge.

Modelling choices: free transition uses Ncrit = 9 (a low-turbulence tunnel
such as the NACA LTPT/TDT); tripped cases force transition at x/c = 0.05 on
both surfaces (grit strips sit close to the leading edge); Ladson's Mach 0.15
is included through the Karman-Tsien correction.
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aerolume.geometry.airfoils import naca4
from aerolume.model import PolarResult
from aerolume.solvers.airfoil import polar

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"

# ---------------------------------------------------------------------------
# reference data
# ---------------------------------------------------------------------------

#: Ladson NASA TM-4074, NACA 0012, Re 6e6, M 0.15, 80-grit trip: (alpha deg, cl, cd)
LADSON_0012_RE6E6 = np.array([
    (-4.04, -0.4417, 0.00871),
    (-2.14, -0.2385, 0.00800),
    (-0.05, -0.0126, 0.00809),
    (2.05, 0.2125, 0.00816),
    (4.04, 0.4316, 0.00823),
    (6.09, 0.6546, 0.00885),
    (8.30, 0.8873, 0.01050),
    (10.12, 1.0707, 0.01201),
    (12.12, 1.2605, 0.01332),
    (14.22, 1.4365, 0.01625),
    (16.30, 1.5739, 0.02218),
    (17.13, 1.6116, 0.02560),
    (18.02, 0.9967, 0.18785),       # stalled
])

#: Gregory & O'Reilly R&M 3726, NACA 0012, Re 3e6, tripped: (alpha deg, cl), digitised
GREGORY_0012_RE3E6 = np.array([
    (1.97, 0.231),
    (5.94, 0.658),
    (9.04, 0.985),
    (12.13, 1.293),
    (14.13, 1.447),
    (16.10, 1.561),
    (17.11, 0.957),                 # stalled
])

#: Abbott & von Doenhoff, NACA 0012, Re 6e6, smooth: (alpha deg, cl), digitised
ABBOTT_0012_RE6E6_CL = np.array([
    (1.97, 0.216),
    (4.88, 0.552),
    (7.96, 0.871),
    (10.19, 1.121),
    (13.11, 1.363),
    (16.38, 1.596),
])

#: Abbott & von Doenhoff, NACA 0012, Re 6e6, smooth: (cl, cd) drag polar, digitised
ABBOTT_0012_RE6E6_CD = np.array([
    (0.021, 0.00593),
    (0.263, 0.00601),
    (0.462, 0.00660),
    (0.662, 0.00779),
    (0.878, 0.00916),
    (1.079, 0.01130),
    (1.261, 0.01335),
])

#: Abbott & von Doenhoff App. IV (plot readings), Re ~ 6e6, smooth
CAMBERED_REF = {
    "2412": {"alpha_l0": -2.0, "cl0": 0.25, "clmax": 1.7, "cm": -0.05},
    "4412": {"alpha_l0": -4.0, "cl0": 0.40, "clmax": 1.65, "cm": -0.10},
}

#: XFOIL 6.99 (M. Drela, MIT; the stock Windows binary) on *exactly* the geometry Aerolume's
#: viscous solver analyses: ``xfoil_coords(code)`` = naca4(code, 200) through prepare_coords with
#: the trailing edge opened to solver.MIN_TE_GAP. XFOIL commands: LOAD, PPAR N 200, OPER,
#: VISC <Re>, ITER 300, Ncrit 9 (default), PACC, ASEQ 0 12 0.5, INIT, ASEQ 0 -4 -0.5.
#: Angles at which XFOIL did not converge are absent. Columns: (alpha deg, cl, cd, cm,
#: xtr top, xtr bottom). Regenerate with ``python validation/airfoil_validation.py --xfoil PATH``.
#: This is a code-to-code reference: it checks that the solver reproduces the XFOIL model,
#: including the lift-curve kink where the pressure-side transition point runs off the trailing
#: edge (docs/validation_airfoil.md, "Cross-check against XFOIL").
XFOIL_REF = {
    ("0012", 1e6): np.array([
        (-4, -0.4200, 0.00724, -0.0077, 0.9608, 0.2577), (-3.5, -0.3647, 0.00674, -0.0075, 0.9445, 0.3137),
        (-3, -0.3117, 0.00633, -0.0067, 0.9227, 0.3694), (-2.5, -0.2590, 0.00600, -0.0058, 0.8960, 0.4244),
        (-1.5, -0.1546, 0.00555, -0.0036, 0.8261, 0.5329), (-1, -0.1029, 0.00543, -0.0024, 0.7838, 0.5857),
        (0, -0.0000, 0.00533, 0.0000, 0.6885, 0.6886), (0.5, 0.0513, 0.00535, 0.0013, 0.6379, 0.7373),
        (1, 0.1029, 0.00543, 0.0025, 0.5858, 0.7838), (1.5, 0.1546, 0.00555, 0.0036, 0.5328, 0.8261),
        (2.5, 0.2590, 0.00600, 0.0058, 0.4245, 0.8961), (3, 0.3117, 0.00633, 0.0067, 0.3692, 0.9227),
        (3.5, 0.3647, 0.00674, 0.0075, 0.3135, 0.9446), (4, 0.4199, 0.00724, 0.0078, 0.2578, 0.9608),
        (4.5, 0.4790, 0.00781, 0.0070, 0.2037, 0.9727), (5, 0.5434, 0.00842, 0.0050, 0.1541, 0.9801),
        (5.5, 0.6100, 0.00907, 0.0025, 0.1134, 0.9861), (6, 0.6800, 0.00970, -0.0009, 0.0841, 0.9895),
        (6.5, 0.7475, 0.01034, -0.0037, 0.0642, 0.9939), (7, 0.8159, 0.01094, -0.0067, 0.0517, 0.9971),
        (7.5, 0.8826, 0.01159, -0.0095, 0.0430, 1.0000), (8, 0.9223, 0.01218, -0.0065, 0.0378, 1.0000),
        (8.5, 0.9608, 0.01284, -0.0033, 0.0338, 1.0000), (9, 0.9998, 0.01347, -0.0001, 0.0307, 1.0000),
        (9.5, 1.0385, 0.01418, 0.0030, 0.0281, 1.0000), (10, 1.0756, 0.01508, 0.0062, 0.0258, 1.0000),
        (10.5, 1.1157, 0.01583, 0.0089, 0.0240, 1.0000), (11, 1.1533, 0.01691, 0.0117, 0.0224, 1.0000),
        (11.5, 1.1937, 0.01784, 0.0139, 0.0210, 1.0000), (12, 1.2269, 0.01932, 0.0169, 0.0197, 1.0000),
    ]),
    ("0012", 3e6): np.array([
        (-4, -0.4281, 0.00610, -0.0047, 0.8671, 0.1516), (-3.5, -0.3751, 0.00584, -0.0040, 0.8312, 0.1900),
        (-3, -0.3220, 0.00562, -0.0033, 0.7912, 0.2320), (-2.5, -0.2689, 0.00543, -0.0026, 0.7477, 0.2779),
        (-2, -0.2155, 0.00528, -0.0020, 0.7025, 0.3248), (-1, -0.1080, 0.00509, -0.0010, 0.6096, 0.4205),
        (-0.5, -0.0540, 0.00504, -0.0005, 0.5627, 0.4679), (0, -0.0000, 0.00503, 0.0000, 0.5156, 0.5157),
        (0.5, 0.0540, 0.00505, 0.0005, 0.4677, 0.5623), (1, 0.1080, 0.00509, 0.0010, 0.4207, 0.6092),
        (1.5, 0.1617, 0.00517, 0.0015, 0.3726, 0.6561), (2, 0.2155, 0.00528, 0.0020, 0.3249, 0.7025),
        (2.5, 0.2688, 0.00543, 0.0027, 0.2781, 0.7481), (3, 0.3220, 0.00562, 0.0033, 0.2322, 0.7910),
        (3.5, 0.3750, 0.00584, 0.0040, 0.1900, 0.8312), (4, 0.4280, 0.00610, 0.0047, 0.1518, 0.8672),
        (4.5, 0.4809, 0.00639, 0.0054, 0.1195, 0.8986), (5, 0.5339, 0.00671, 0.0061, 0.0935, 0.9250),
        (5.5, 0.5868, 0.00706, 0.0068, 0.0738, 0.9461), (6, 0.6399, 0.00742, 0.0075, 0.0594, 0.9634),
        (6.5, 0.6934, 0.00780, 0.0081, 0.0486, 0.9768), (7, 0.7509, 0.00823, 0.0076, 0.0404, 0.9852),
        (7.5, 0.8129, 0.00868, 0.0062, 0.0338, 0.9896), (8, 0.8769, 0.00915, 0.0042, 0.0292, 0.9924),
        (8.5, 0.9384, 0.00965, 0.0027, 0.0255, 0.9953), (9, 1.0026, 0.01019, 0.0006, 0.0225, 0.9968),
        (9.5, 1.0655, 0.01075, -0.0013, 0.0201, 0.9985), (10, 1.1277, 0.01130, -0.0031, 0.0185, 0.9999),
        (10.5, 1.1669, 0.01189, -0.0000, 0.0170, 1.0000), (11, 1.2073, 0.01240, 0.0029, 0.0161, 1.0000),
        (11.5, 1.2472, 0.01307, 0.0057, 0.0148, 1.0000), (12, 1.2894, 0.01369, 0.0080, 0.0141, 1.0000),
    ]),
    ("2412", 1e6): np.array([
        (-4, -0.1857, 0.00753, -0.0569, 0.8847, 0.1335), (-3.5, -0.1319, 0.00720, -0.0563, 0.8643, 0.1692),
        (-3, -0.0783, 0.00692, -0.0557, 0.8411, 0.2115), (-2.5, -0.0248, 0.00666, -0.0550, 0.8155, 0.2612),
        (-1.5, 0.0812, 0.00622, -0.0535, 0.7577, 0.3838), (-1, 0.1328, 0.00594, -0.0525, 0.7256, 0.4800),
        (-0.5, 0.1838, 0.00571, -0.0514, 0.6928, 0.5862), (0, 0.2347, 0.00555, -0.0502, 0.6586, 0.6816),
        (0.5, 0.2851, 0.00545, -0.0487, 0.6251, 0.7702), (1, 0.3351, 0.00541, -0.0471, 0.5919, 0.8520),
        (1.5, 0.3866, 0.00551, -0.0457, 0.5602, 0.9165), (2, 0.4440, 0.00573, -0.0456, 0.5285, 0.9562),
        (2.5, 0.5077, 0.00603, -0.0471, 0.4968, 0.9776), (3, 0.5797, 0.00634, -0.0506, 0.4632, 0.9876),
        (3.5, 0.6541, 0.00667, -0.0547, 0.4271, 0.9946), (4, 0.7316, 0.00704, -0.0596, 0.3832, 1.0000),
        (4.5, 0.7764, 0.00745, -0.0575, 0.3396, 1.0000), (5, 0.8196, 0.00797, -0.0550, 0.2907, 1.0000),
        (5.5, 0.8616, 0.00859, -0.0524, 0.2384, 1.0000), (6, 0.9029, 0.00931, -0.0497, 0.1867, 1.0000),
        (6.5, 0.9440, 0.01008, -0.0470, 0.1408, 1.0000), (7, 0.9864, 0.01087, -0.0446, 0.1044, 1.0000),
        (7.5, 1.0303, 0.01164, -0.0425, 0.0788, 1.0000), (8, 1.0749, 0.01240, -0.0407, 0.0612, 1.0000),
        (8.5, 1.1194, 0.01319, -0.0389, 0.0494, 1.0000), (9, 1.1638, 0.01398, -0.0371, 0.0415, 1.0000),
        (9.5, 1.2069, 0.01483, -0.0352, 0.0357, 1.0000), (10, 1.2488, 0.01571, -0.0332, 0.0315, 1.0000),
        (10.5, 1.2888, 0.01665, -0.0309, 0.0282, 1.0000), (11, 1.3240, 0.01773, -0.0279, 0.0257, 1.0000),
        (11.5, 1.3564, 0.01886, -0.0246, 0.0236, 1.0000), (12, 1.3862, 0.02014, -0.0212, 0.0219, 1.0000),
    ]),
    ("2412", 3e6): np.array([
        (-4, -0.1958, 0.00622, -0.0542, 0.7830, 0.0866), (-3.5, -0.1411, 0.00605, -0.0538, 0.7541, 0.1077),
        (-3, -0.0866, 0.00591, -0.0534, 0.7234, 0.1334), (-2.5, -0.0320, 0.00578, -0.0530, 0.6918, 0.1649),
        (-2, 0.0227, 0.00567, -0.0526, 0.6591, 0.2012), (-1.5, 0.0772, 0.00558, -0.0523, 0.6265, 0.2424),
        (-1, 0.1319, 0.00551, -0.0519, 0.5937, 0.2885), (-0.5, 0.1864, 0.00545, -0.0515, 0.5621, 0.3405),
        (0, 0.2408, 0.00536, -0.0512, 0.5320, 0.4059), (0.5, 0.2946, 0.00522, -0.0507, 0.5037, 0.4968),
        (1, 0.3485, 0.00513, -0.0503, 0.4765, 0.5825), (1.5, 0.4024, 0.00507, -0.0498, 0.4505, 0.6610),
        (2, 0.4561, 0.00505, -0.0492, 0.4240, 0.7383), (2.5, 0.5088, 0.00507, -0.0485, 0.3939, 0.8145),
        (3, 0.5604, 0.00516, -0.0474, 0.3578, 0.8846), (3.5, 0.6111, 0.00539, -0.0461, 0.3172, 0.9380),
        (4, 0.6643, 0.00575, -0.0455, 0.2726, 0.9704), (4.5, 0.7224, 0.00625, -0.0461, 0.2236, 0.9868),
        (5, 0.7880, 0.00685, -0.0485, 0.1726, 0.9936), (5.5, 0.8567, 0.00748, -0.0517, 0.1271, 0.9976),
        (6, 0.9214, 0.00809, -0.0540, 0.0923, 0.9999), (6.5, 0.9648, 0.00857, -0.0515, 0.0721, 1.0000),
        (7, 1.0084, 0.00905, -0.0491, 0.0571, 1.0000), (7.5, 1.0534, 0.00953, -0.0470, 0.0465, 1.0000),
        (8, 1.0998, 0.01002, -0.0453, 0.0385, 1.0000), (8.5, 1.1475, 0.01053, -0.0439, 0.0326, 1.0000),
        (9, 1.1953, 0.01107, -0.0426, 0.0279, 1.0000), (9.5, 1.2431, 0.01163, -0.0413, 0.0245, 1.0000),
        (10, 1.2904, 0.01222, -0.0401, 0.0218, 1.0000), (10.5, 1.3366, 0.01285, -0.0387, 0.0195, 1.0000),
        (11, 1.3813, 0.01356, -0.0371, 0.0175, 1.0000), (11.5, 1.4258, 0.01422, -0.0355, 0.0163, 1.0000),
        (12, 1.4670, 0.01505, -0.0335, 0.0148, 1.0000),
    ]),
    ("4412", 1e6): np.array([
        (-3.5, 0.0957, 0.00753, -0.1043, 0.7972, 0.0948), (-3, 0.1500, 0.00734, -0.1038, 0.7761, 0.1170),
        (-2.5, 0.2040, 0.00719, -0.1033, 0.7531, 0.1451), (-2, 0.2580, 0.00707, -0.1027, 0.7286, 0.1789),
        (-1, 0.3653, 0.00690, -0.1016, 0.6767, 0.2669), (-0.5, 0.4186, 0.00684, -0.1010, 0.6498, 0.3265),
        (0, 0.4711, 0.00668, -0.1003, 0.6234, 0.4275), (0.5, 0.5208, 0.00625, -0.0992, 0.5977, 0.6550),
        (1, 0.5660, 0.00583, -0.0966, 0.5740, 0.8809), (1.5, 0.6450, 0.00596, -0.1014, 0.5500, 0.9921),
        (2, 0.7083, 0.00620, -0.1030, 0.5290, 1.0000), (2.5, 0.7579, 0.00642, -0.1017, 0.5104, 1.0000),
        (3.5, 0.8592, 0.00691, -0.0994, 0.4764, 1.0000), (4, 0.9105, 0.00717, -0.0985, 0.4584, 1.0000),
        (4.5, 0.9617, 0.00746, -0.0976, 0.4385, 1.0000), (5, 1.0125, 0.00779, -0.0966, 0.4157, 1.0000),
        (5.5, 1.0633, 0.00815, -0.0957, 0.3892, 1.0000), (6, 1.1125, 0.00861, -0.0946, 0.3553, 1.0000),
        (6.5, 1.1591, 0.00925, -0.0930, 0.3130, 1.0000), (7, 1.2029, 0.01009, -0.0911, 0.2615, 1.0000),
        (7.5, 1.2429, 0.01116, -0.0887, 0.2028, 1.0000), (8, 1.2799, 0.01239, -0.0858, 0.1471, 1.0000),
        (8.5, 1.3155, 0.01362, -0.0827, 0.1042, 1.0000), (9, 1.3483, 0.01477, -0.0790, 0.0759, 1.0000),
        (9.5, 1.3807, 0.01589, -0.0755, 0.0580, 1.0000), (10, 1.4115, 0.01709, -0.0718, 0.0464, 1.0000),
        (10.5, 1.4421, 0.01832, -0.0684, 0.0389, 1.0000), (11, 1.4706, 0.01971, -0.0649, 0.0335, 1.0000),
        (11.5, 1.4969, 0.02129, -0.0614, 0.0296, 1.0000), (12, 1.5203, 0.02314, -0.0580, 0.0266, 1.0000),
    ]),
    ("4412", 3e6): np.array([
        (-4, 0.0348, 0.00626, -0.1032, 0.7197, 0.0574), (-3.5, 0.0901, 0.00616, -0.1030, 0.6955, 0.0677),
        (-3, 0.1453, 0.00607, -0.1027, 0.6702, 0.0829), (-2.5, 0.2005, 0.00601, -0.1024, 0.6440, 0.1007),
        (-2, 0.2557, 0.00597, -0.1021, 0.6176, 0.1225), (-1.5, 0.3109, 0.00593, -0.1019, 0.5920, 0.1493),
        (-1, 0.3661, 0.00593, -0.1016, 0.5672, 0.1791), (-0.5, 0.4212, 0.00592, -0.1014, 0.5437, 0.2155),
        (0, 0.4766, 0.00591, -0.1012, 0.5224, 0.2565), (0.5, 0.5319, 0.00591, -0.1011, 0.5032, 0.3049),
        (1, 0.5871, 0.00587, -0.1009, 0.4859, 0.3710), (1.5, 0.6409, 0.00565, -0.1006, 0.4697, 0.5303),
        (2, 0.6943, 0.00545, -0.1002, 0.4546, 0.6880), (2.5, 0.7453, 0.00518, -0.0992, 0.4405, 0.8685),
        (3, 0.8057, 0.00517, -0.1000, 0.4246, 0.9874), (3.5, 0.8767, 0.00543, -0.1035, 0.4025, 1.0000),
        (4, 0.9263, 0.00570, -0.1021, 0.3763, 1.0000), (4.5, 0.9758, 0.00606, -0.1009, 0.3437, 1.0000),
        (5, 1.0248, 0.00652, -0.0996, 0.3058, 1.0000), (5.5, 1.0730, 0.00708, -0.0983, 0.2620, 1.0000),
        (6, 1.1199, 0.00776, -0.0968, 0.2136, 1.0000), (6.5, 1.1653, 0.00857, -0.0951, 0.1625, 1.0000),
        (7, 1.2104, 0.00938, -0.0933, 0.1190, 1.0000), (8, 1.3008, 0.01089, -0.0901, 0.0643, 1.0000),
        (8.5, 1.3458, 0.01160, -0.0884, 0.0495, 1.0000), (9, 1.3900, 0.01228, -0.0867, 0.0395, 1.0000),
        (9.5, 1.4328, 0.01299, -0.0847, 0.0324, 1.0000), (10, 1.4718, 0.01371, -0.0821, 0.0272, 1.0000),
        (10.5, 1.5089, 0.01447, -0.0792, 0.0236, 1.0000), (11, 1.5445, 0.01531, -0.0762, 0.0208, 1.0000),
        (11.5, 1.5787, 0.01622, -0.0731, 0.0187, 1.0000), (12, 1.6109, 0.01726, -0.0700, 0.0170, 1.0000),
    ]),
}

NCRIT = 9.0
TRIP = 0.05
PANELS = 199


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

@dataclass
class Case:
    """One computed polar plus the derived numbers the docs and tests compare."""
    name: str
    code: str
    re: float
    result: PolarResult
    seconds: float
    table: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.table = self.result.table()

    def at(self, key: str, alpha: float | np.ndarray) -> np.ndarray:
        """Converged-point value of ``key`` linearly interpolated to ``alpha``."""
        t = self.table
        return np.interp(alpha, t["alpha"], t[key], left=np.nan, right=np.nan)

    def cd_at_cl(self, cl: float | np.ndarray) -> np.ndarray:
        """Cd interpolated at a lift coefficient, on the pre-stall branch."""
        t = self.table
        i = np.argsort(t["alpha"])
        c, d = t["cl"][i], t["cd"][i]
        k = int(np.argmax(c)) + 1
        return np.interp(cl, c[:k], d[:k], left=np.nan, right=np.nan)

    def lift_line(self, lo: float, hi: float) -> tuple[float, float]:
        """(slope per degree, zero-lift angle) of a straight-line fit of cl over [lo, hi] deg."""
        t = self.table
        m = (t["alpha"] >= lo - 1e-9) & (t["alpha"] <= hi + 1e-9)
        slope, icept = np.polyfit(t["alpha"][m], t["cl"][m], 1)
        return float(slope), float(-icept / slope)

    def clmax(self) -> tuple[float, float]:
        """(cl_max, alpha at cl_max) over the converged points."""
        t = self.table
        i = int(np.argmax(t["cl"]))
        return float(t["cl"][i]), float(t["alpha"][i])

    def n_failed(self) -> int:
        return sum(not p.converged for p in self.result.points)


def run(name: str, code: str, re: float, alphas, *, tripped: bool = False, mach: float = 0.0,
        panels: int = PANELS) -> Case:
    """Compute one polar (distributions dropped) and wrap it in a :class:`Case`."""
    xtr = TRIP if tripped else 1.0
    t0 = time.perf_counter()
    res = polar(naca4(code, 200), re, alphas, mach=mach, ncrit=NCRIT, xtr_upper=xtr, xtr_lower=xtr,
                panels=panels, distributions=False)
    return Case(name, code, re, res, time.perf_counter() - t0)


def case_0012_re6e6_tripped(**kw) -> Case:
    return run("NACA 0012, Re 6e6, M 0.15, tripped (Ladson)", "0012", 6e6,
               np.arange(-5.0, 18.01, 1.0), tripped=True, mach=0.15, **kw)


def case_0012_re3e6_tripped(**kw) -> Case:
    return run("NACA 0012, Re 3e6, tripped (Gregory & O'Reilly)", "0012", 3e6,
               np.arange(0.0, 18.01, 1.0), tripped=True, **kw)


def case_0012_re6e6_free(**kw) -> Case:
    return run("NACA 0012, Re 6e6, free transition (Abbott & von Doenhoff)", "0012", 6e6,
               np.arange(0.0, 18.01, 1.0), **kw)


def case_cambered(code: str, **kw) -> Case:
    return run(f"NACA {code}, Re 6e6, free transition (Abbott & von Doenhoff)", code, 6e6,
               np.arange(-6.0, 20.01, 1.0), **kw)


XFOIL_CASES = tuple((code, re) for code in ("0012", "2412", "4412") for re in (1e6, 3e6))


def xfoil_coords(code: str) -> np.ndarray:
    """The coordinates the viscous solver actually analyses for ``naca4(code, 200)``.

    prepare_coords output with the trailing edge opened to solver.MIN_TE_GAP,
    i.e. what ``polar`` passes to mfoil, so XFOIL gets the identical shape.
    """
    from aerolume.solvers.airfoil import prepare_coords
    from aerolume.solvers.airfoil.solver import MIN_TE_GAP, adapt_trailing_edge
    xy, _, info = prepare_coords(naca4(code, 200))
    return adapt_trailing_edge(xy, info["te_gap"], MIN_TE_GAP) if info["te_gap"] < MIN_TE_GAP else xy


def case_xfoil(code: str, re: float, step: float = 1.0, **kw) -> Case:
    """Free-transition polar, -4..12 deg, on the geometry of the XFOIL reference."""
    return run(f"NACA {code}, Re {re:.0e}, free transition (XFOIL cross-check)", code, re,
               np.round(np.arange(-4.0, 12.0 + 1e-9, step), 6), **kw)


def compare_xfoil(c: Case) -> np.ndarray:
    """Rows (alpha, cl_xf, cl, cd_xf, cd, cm_xf, cm, xtr_bot_xf, xtr_bot) at the angles both have."""
    ref = XFOIL_REF[(c.code, c.re)]
    t = c.table
    rows = []
    for a, cl, cd, cm, _xt, xb in ref:
        k = np.flatnonzero(np.abs(t["alpha"] - a) < 1e-6)
        if len(k):
            i = int(k[0])
            rows.append((a, cl, t["cl"][i], cd, t["cd"][i], cm, t["cm"][i], xb, t["xtr_lower"][i]))
    return np.array(rows)


def xfoil_slopes(c: Case) -> np.ndarray:
    """Rows (alpha_mid, dcl/dalpha XFOIL, dcl/dalpha Aerolume) over consecutive computed angles.

    Only intervals whose two end angles both exist in the XFOIL reference are
    returned, so the two slopes are over the same interval.
    """
    rows = compare_xfoil(c)
    out = []
    for r0, r1 in itertools.pairwise(rows):
        h = r1[0] - r0[0]
        t = c.table["alpha"]
        # consecutive in the computed sweep (no computed angle in between)
        if np.any((t > r0[0] + 1e-9) & (t < r1[0] - 1e-9)):
            continue
        out.append((0.5 * (r0[0] + r1[0]), (r1[1] - r0[1]) / h, (r1[2] - r0[2]) / h))
    return np.array(out)


def regenerate_xfoil_ref(xfoil: str, step: float = 0.5) -> str:
    """Run XFOIL on :func:`xfoil_coords` for every case; return the XFOIL_REF literal.

    Needs a local XFOIL executable (not a dependency of Aerolume); graphics are
    switched off (PLOP G F) and everything runs in a temporary directory.
    """
    import subprocess
    import tempfile
    lines = ["XFOIL_REF = {"]
    with tempfile.TemporaryDirectory() as tmp:
        for code, re in XFOIL_CASES:
            dat, pol = Path(tmp) / f"n{code}.dat", Path(tmp) / f"p{code}_{re:.0e}.txt"
            np.savetxt(dat, xfoil_coords(code), fmt="%.8f", header=f"NACA{code}", comments="")
            cmd = (f"PLOP\nG F\n\nLOAD {dat.name}\nPPAR\nN 200\n\n\nOPER\nVISC {re:.0f}\nITER 300\nPACC\n"
                   f"{pol.name}\n\nASEQ 0 12 {step}\nINIT\nASEQ 0 -4 {-step}\n\nQUIT\n")
            subprocess.run([xfoil], input=cmd, text=True, cwd=tmp, capture_output=True, timeout=900,
                           check=False)        # XFOIL exits 0 even on failures; the polar file tells
            rows: dict[float, tuple] = {}
            body = False
            for line in pol.read_text().splitlines():
                if line.strip().startswith("------"):
                    body = True
                elif body and line.strip():
                    a, cl, cd, _cdp, cm, xt, xb = (float(v) for v in line.split()[:7])
                    rows[a] = (a, cl, cd, cm, xt, xb)
            lines.append(f'    ("{code}", {re:.0e}): np.array(['.replace("e+06", "e6"))
            for r in sorted(rows.values()):
                lines.append(f"        ({r[0]:g}, {r[1]:.4f}, {r[2]:.5f}, {r[3]:.4f}, {r[4]:.4f}, {r[5]:.4f}),")
            lines.append("    ]),")
    lines.append("}")
    return "\n".join(lines)


def benchmark_polar(repeats: int = 1) -> tuple[float, int]:
    """Wall time of the reference performance case: NACA 2412, Re 1e6, 19 angles -4..14 deg."""
    best, conv = math.inf, 0
    xy = naca4("2412", 200)
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = polar(xy, 1e6, np.arange(-4.0, 14.01, 1.0))
        best = min(best, time.perf_counter() - t0)
        conv = sum(p.converged for p in r.points)
    return best, conv


# ---------------------------------------------------------------------------
# comparisons
# ---------------------------------------------------------------------------

def compare_ladson(c: Case) -> list[tuple[float, float, float, float, float]]:
    """Rows (alpha, cl_exp, cl_calc, cd_exp, cd_calc) at the pre-stall Ladson angles."""
    rows = []
    for a, cl, cd in LADSON_0012_RE6E6[:-1]:
        rows.append((a, cl, float(c.at("cl", a)), cd, float(c.at("cd", a))))
    return rows


def compare_gregory(c: Case) -> list[tuple[float, float, float]]:
    return [(a, cl, float(c.at("cl", a))) for a, cl in GREGORY_0012_RE3E6[:-1]]


def compare_abbott_0012(c: Case) -> tuple[list, list]:
    cl_rows = [(a, cl, float(c.at("cl", a))) for a, cl in ABBOTT_0012_RE6E6_CL]
    cd_rows = [(cl, cd, float(c.cd_at_cl(cl))) for cl, cd in ABBOTT_0012_RE6E6_CD]
    return cl_rows, cd_rows


def cambered_summary(c: Case) -> dict[str, float]:
    slope, a0 = c.lift_line(-2.0, 6.0)
    clmax, aclmax = c.clmax()
    return {
        "slope": slope, "alpha_l0": a0, "cl0": float(c.at("cl", 0.0)), "clmax": clmax,
        "alpha_clmax": aclmax, "cm": float(np.mean(c.at("cm", np.array([0.0, 2.0, 4.0])))),
        "cd_cl04": float(c.cd_at_cl(0.4)), "cd_cl08": float(c.cd_at_cl(0.8)),
    }


def ladson_slope() -> tuple[float, float]:
    """Measured lift slope [/deg] and zero-lift angle of Ladson's data over -4..8 deg."""
    d = LADSON_0012_RE6E6
    m = (d[:, 0] >= -4.1) & (d[:, 0] <= 8.4)
    s, i = np.polyfit(d[m, 0], d[m, 1], 1)
    return float(s), float(-i / s)


# ---------------------------------------------------------------------------
# figures (matplotlib Figure + Agg canvas only; never pyplot)
# ---------------------------------------------------------------------------

def _figure(ncols: int, width: float = 10.0, height: float = 4.0):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(width, height), dpi=110, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, ncols)
    for ax in np.atleast_1d(axes):
        ax.grid(True, color="#d8dde3", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    return fig, axes


CALC = "#0f7c8a"     # Aerolume teal
EXP = "#2b2b2b"
EXP2 = "#c0602b"


def fig_0012(c6t: Case, c3t: Case, c6f: Case, path: Path) -> None:
    fig, (a1, a2) = _figure(2)
    t = c6t.table
    a1.plot(t["alpha"], t["cl"], "-", color=CALC, lw=2, label="Aerolume, Re 6e6 tripped")
    t3 = c3t.table
    a1.plot(t3["alpha"], t3["cl"], "--", color=CALC, lw=1.4, label="Aerolume, Re 3e6 tripped")
    a1.plot(LADSON_0012_RE6E6[:, 0], LADSON_0012_RE6E6[:, 1], "o", color=EXP, ms=5,
            label="Ladson TM-4074, Re 6e6 (80 grit)")
    a1.plot(GREGORY_0012_RE3E6[:, 0], GREGORY_0012_RE3E6[:, 1], "s", mfc="none", color=EXP2, ms=5,
            label="Gregory & O'Reilly, Re 3e6")
    a1.set_xlabel("angle of attack [deg]")
    a1.set_ylabel("$c_l$")
    a1.set_title("NACA 0012 lift")
    a1.legend(fontsize=8, frameon=False, loc="upper left")
    a2.plot(t["cd"] * 1e4, t["cl"], "-", color=CALC, lw=2, label="Aerolume, tripped (x/c 0.05)")
    f = c6f.table
    a2.plot(f["cd"] * 1e4, f["cl"], ":", color=CALC, lw=2, label="Aerolume, free transition, Ncrit 9")
    pre = LADSON_0012_RE6E6[:-1]
    a2.plot(pre[:, 2] * 1e4, pre[:, 1], "o", color=EXP, ms=5, label="Ladson, tripped")
    a2.plot(ABBOTT_0012_RE6E6_CD[:, 1] * 1e4, ABBOTT_0012_RE6E6_CD[:, 0], "^", mfc="none", color=EXP2, ms=6,
            label="Abbott & von Doenhoff, smooth")
    a2.set_xlabel("$c_d$ [counts, 1e-4]")
    a2.set_ylabel("$c_l$")
    a2.set_title("NACA 0012 drag polar, Re 6e6")
    a2.set_xlim(0, 300)
    a2.legend(fontsize=8, frameon=False, loc="lower right")
    fig.savefig(path)


def fig_cambered(cases: dict[str, Case], path: Path) -> None:
    fig, (a1, a2, a3) = _figure(3, width=13.0)
    colors = {"2412": CALC, "4412": EXP2}
    for code, c in cases.items():
        t = c.table
        ref = CAMBERED_REF[code]
        col = colors[code]
        a1.plot(t["alpha"], t["cl"], "-", color=col, lw=2, label=f"Aerolume NACA {code}")
        a1.plot([ref["alpha_l0"], 0.0], [0.0, ref["cl0"]], "o", color=col, mfc="none", ms=7,
                label=f"A&vD {code}: $\\alpha_{{L0}}$, $c_l(0)$")
        a1.axhline(ref["clmax"], color=col, lw=0.8, ls=":")
        i = np.argsort(t["alpha"])
        k = int(np.argmax(t["cl"][i])) + 1
        a2.plot(t["cd"][i][:k] * 1e4, t["cl"][i][:k], "-", color=col, lw=2, label=f"NACA {code}")
        a3.plot(t["alpha"], t["cm"], "-", color=col, lw=2, label=f"Aerolume NACA {code}")
        a3.axhline(ref["cm"], color=col, lw=0.8, ls=":", label=f"A&vD {code} (plot reading)")
    a1.set_xlabel("angle of attack [deg]")
    a1.set_ylabel("$c_l$")
    a1.set_title("Lift, Re 6e6 (dotted: A&vD $c_{l,max}$)")
    a1.legend(fontsize=7, frameon=False, loc="lower right")
    a2.set_xlabel("$c_d$ [counts]")
    a2.set_ylabel("$c_l$")
    a2.set_title("Drag polar (predicted)")
    a2.set_xlim(0, 300)
    a2.legend(fontsize=8, frameon=False, loc="lower right")
    a3.set_xlabel("angle of attack [deg]")
    a3.set_ylabel("$c_{m,c/4}$")
    a3.set_title("Pitching moment")
    a3.set_ylim(-0.16, 0.02)
    a3.legend(fontsize=7, frameon=False, loc="lower right")
    fig.savefig(path)


def fig_xfoil(cases: dict[tuple[str, float], Case], path: Path) -> None:
    """Cl and Cm vs alpha: Aerolume lines, XFOIL 6.99 markers, one column per section."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(13.0, 7.0), dpi=110, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
    styles = {1e6: ("-", "o", CALC), 3e6: ("--", "s", EXP2)}
    for col, code in enumerate(("0012", "2412", "4412")):
        acl, acm = axes[0, col], axes[1, col]
        for ax in (acl, acm):
            ax.grid(True, color="#d8dde3", linewidth=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        for re in (1e6, 3e6):
            c = cases[(code, re)]
            ls, mk, color = styles[re]
            t = c.table
            ref = XFOIL_REF[(code, re)]
            acl.plot(t["alpha"], t["cl"], ls, color=color, lw=1.8, label=f"Aerolume, Re {re:.0e}")
            acl.plot(ref[:, 0], ref[:, 1], mk, color=EXP, mfc="none", ms=4, mew=0.8,
                     label=f"XFOIL 6.99, Re {re:.0e}")
            acm.plot(t["alpha"], t["cm"], ls, color=color, lw=1.8)
            acm.plot(ref[:, 0], ref[:, 3], mk, color=EXP, mfc="none", ms=4, mew=0.8)
        acl.set_title(f"NACA {code}, free transition, Ncrit 9")
        acl.set_ylabel("$c_l$")
        acm.set_ylabel("$c_{m,c/4}$")
        acm.set_xlabel("angle of attack [deg]")
        if col == 0:
            acl.legend(fontsize=7, frameon=False, loc="upper left")
    fig.savefig(path)


def fig_cp(path: Path) -> None:
    """Viscous vs inviscid Cp and Cf on NACA 2412 at 4 deg, Re 1e6 (illustrates the coupling)."""
    from aerolume.solvers.airfoil import analyze
    p = analyze(naca4("2412", 200), 1e6, 4.0)
    fig, (a1, a2) = _figure(2)
    a1.plot(p.x, p.cp_inviscid, "--", color=EXP, lw=1.2, label="inviscid (panel method)")
    a1.plot(p.x, p.cp, "-", color=CALC, lw=2, label="viscous (coupled)")
    a1.invert_yaxis()
    a1.set_xlabel("x/c")
    a1.set_ylabel("$C_p$")
    a1.set_title(f"NACA 2412, 4 deg, Re 1e6: $c_l$ = {p.cl:.3f}, $c_d$ = {p.cd:.5f}")
    a1.legend(fontsize=8, frameon=False)
    ile = int(np.argmin(p.x))
    a2.plot(p.x[:ile + 1], p.cf[:ile + 1], "-", color=CALC, lw=2, label="upper surface")
    a2.plot(p.x[ile:], p.cf[ile:], "-", color=EXP2, lw=1.5, label="lower surface")
    for xt, col in ((p.xtr_upper, CALC), (p.xtr_lower, EXP2)):
        if xt < 1:
            a2.axvline(xt, color=col, ls=":", lw=1)
    a2.set_xlabel("x/c")
    a2.set_ylabel("$c_f$")
    a2.set_title("Skin friction (dotted: transition)")
    a2.legend(fontsize=8, frameon=False)
    fig.savefig(path)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _fmt(v: float, nd: int = 4) -> str:
    return "n/c" if not math.isfinite(v) else f"{v:.{nd}f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-figures", action="store_true", help="skip rendering docs/img/airfoil_*.png")
    ap.add_argument("--no-bench", action="store_true", help="skip the performance benchmark")
    ap.add_argument("--xfoil-only", action="store_true", help="only the XFOIL cross-check section")
    ap.add_argument("--xfoil", metavar="PATH", help="regenerate XFOIL_REF with this XFOIL executable, print it "
                    "and exit")
    args = ap.parse_args(argv)

    out = sys.stdout
    if args.xfoil:
        print(regenerate_xfoil_ref(args.xfoil), file=out)
        return 0

    xf = {key: case_xfoil(*key, step=0.5) for key in XFOIL_CASES}
    print("## Cross-check against XFOIL 6.99 (identical geometry, free transition, Ncrit 9)\n", file=out)
    print("| section | Re | angles | max abs d cl | max abs d cm | max abs d cd | max abs d slope [/deg] | "
          "unconverged | wall |", file=out)
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|", file=out)
    for (code, re), c in xf.items():
        r = compare_xfoil(c)
        sl = xfoil_slopes(c)
        print(f"| {code} | {re:.0e} | {len(r)} | {np.max(np.abs(r[:, 2] - r[:, 1])):.4f} | "
              f"{np.max(np.abs(r[:, 6] - r[:, 5])):.4f} | {np.max(np.abs(r[:, 4] / r[:, 3] - 1)) * 100:.1f}% | "
              f"{np.max(np.abs(sl[:, 2] - sl[:, 1])):.4f} | {c.n_failed()} of {len(c.result.points)} | "
              f"{c.seconds:.0f} s |", file=out)
    c = xf[("2412", 1e6)]
    print("\nNACA 2412, Re 1e6 in detail:\n", file=out)
    print("| alpha | cl XFOIL | cl | cm XFOIL | cm | cd XFOIL | cd | xtr bottom XFOIL | xtr bottom |", file=out)
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|", file=out)
    for a, clx, cl, cdx, cd, cmx, cm, xbx, xb in compare_xfoil(c):
        if abs(a - round(a)) < 1e-9:
            print(f"| {a:g} | {clx:.4f} | {cl:.4f} | {cmx:.4f} | {cm:.4f} | {cdx:.5f} | {cd:.5f} | {xbx:.3f} | "
                  f"{xb:.3f} |", file=out)
    for n in c.result.notes:
        if "trailing edge between" in n:
            print(f"\nnote: {n}", file=out)
    if not args.no_figures:
        IMG.mkdir(parents=True, exist_ok=True)
        fig_xfoil(xf, IMG / "airfoil_xfoil.png")
    if args.xfoil_only:
        return 0
    print(file=out)

    c6t = case_0012_re6e6_tripped()
    c3t = case_0012_re3e6_tripped()
    c6f = case_0012_re6e6_free()
    cam = {code: case_cambered(code) for code in ("2412", "4412")}

    print("## NACA 0012, Re 6e6, M 0.15, tripped vs Ladson (NASA TM-4074)\n", file=out)
    print("| alpha | cl exp | cl calc | d cl | cd exp | cd calc | d cd |", file=out)
    print("|---:|---:|---:|---:|---:|---:|---:|", file=out)
    for a, cle, clc, cde, cdc in compare_ladson(c6t):
        print(f"| {a:.2f} | {cle:.4f} | {_fmt(clc)} | {clc - cle:+.3f} | {cde:.5f} | {_fmt(cdc, 5)} | "
              f"{(cdc / cde - 1) * 100:+.0f}% |", file=out)
    s_exp, a0_exp = ladson_slope()
    s_c, a0_c = c6t.lift_line(-4.0, 8.0)
    print(f"\nlift slope -4..8 deg: exp {s_exp:.4f}/deg (alpha_L0 {a0_exp:+.2f}), calc {s_c:.4f}/deg "
          f"(alpha_L0 {a0_c:+.2f}); cl_max calc {c6t.clmax()[0]:.3f} at {c6t.clmax()[1]:g} deg "
          f"(exp 1.612 at 17.1 deg, stalled at 18.0); wall {c6t.seconds:.1f} s, "
          f"{c6t.n_failed()} unconverged\n", file=out)

    print("## NACA 0012, Re 3e6, tripped vs Gregory & O'Reilly (R&M 3726)\n", file=out)
    print("| alpha | cl exp | cl calc | d cl |", file=out)
    print("|---:|---:|---:|---:|", file=out)
    for a, cle, clc in compare_gregory(c3t):
        print(f"| {a:.2f} | {cle:.3f} | {_fmt(clc, 3)} | {clc - cle:+.3f} |", file=out)
    print(f"\ncl_max calc {c3t.clmax()[0]:.3f} at {c3t.clmax()[1]:g} deg (exp ~1.56 at 16.1 deg); "
          f"wall {c3t.seconds:.1f} s, {c3t.n_failed()} unconverged\n", file=out)

    print("## NACA 0012, Re 6e6, free transition vs Abbott & von Doenhoff (smooth)\n", file=out)
    cl_rows, cd_rows = compare_abbott_0012(c6f)
    print("| alpha | cl exp | cl calc | d cl |", file=out)
    print("|---:|---:|---:|---:|", file=out)
    for a, cle, clc in cl_rows:
        print(f"| {a:.2f} | {cle:.3f} | {_fmt(clc, 3)} | {clc - cle:+.3f} |", file=out)
    print("\n| cl | cd exp | cd calc | d cd |", file=out)
    print("|---:|---:|---:|---:|", file=out)
    for cl, cde, cdc in cd_rows:
        print(f"| {cl:.3f} | {cde:.5f} | {_fmt(cdc, 5)} | {(cdc / cde - 1) * 100:+.0f}% |", file=out)
    print(f"\nwall {c6f.seconds:.1f} s, {c6f.n_failed()} unconverged\n", file=out)

    print("## NACA 2412 / 4412, Re 6e6, free transition vs Abbott & von Doenhoff\n", file=out)
    print("| section | quantity | A&vD | calc |", file=out)
    print("|---|---|---:|---:|", file=out)
    for code, c in cam.items():
        s = cambered_summary(c)
        ref = CAMBERED_REF[code]
        print(f"| {code} | zero-lift angle [deg] | {ref['alpha_l0']:+.1f} | {s['alpha_l0']:+.2f} |", file=out)
        print(f"| {code} | cl at 0 deg | {ref['cl0']:.2f} | {s['cl0']:.3f} |", file=out)
        print(f"| {code} | lift slope -2..6 deg [/deg] | (~0.105) | {s['slope']:.4f} |", file=out)
        print(f"| {code} | cl_max (alpha) | ~{ref['clmax']:.2f} | {s['clmax']:.3f} ({s['alpha_clmax']:g} deg) |",
              file=out)
        print(f"| {code} | cm c/4 (0-4 deg) | {ref['cm']:+.2f} | {s['cm']:+.4f} |", file=out)
        print(f"| {code} | cd at cl 0.4 / 0.8 | - | {s['cd_cl04']:.5f} / {s['cd_cl08']:.5f} |", file=out)
        print(f"| {code} | wall time, unconverged | | {c.seconds:.1f} s, {c.n_failed()} of "
              f"{len(c.result.points)} |", file=out)

    if not args.no_bench:
        secs, conv = benchmark_polar()
        print(f"\n## Performance\n\nNACA 2412, Re 1e6, 19 angles (-4..14 deg), 199 panels: {secs:.2f} s, "
              f"{conv}/19 converged\n", file=out)

    if not args.no_figures:
        IMG.mkdir(parents=True, exist_ok=True)
        fig_0012(c6t, c3t, c6f, IMG / "airfoil_0012.png")
        fig_cambered(cam, IMG / "airfoil_cambered.png")
        fig_cp(IMG / "airfoil_cp_2412.png")
        print(f"figures written to {IMG}", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

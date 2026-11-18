# XFOIL hinge moment: what does it contain? (M4 step 3)

Produced by `tests/diagnostics/xfoil_hinge_calibration.py` (XFOIL 6.99, documented commands only:
`SAVE`, `FNEW`, `FMOM`, `CPWR`, `DUMP`, `PACC`). Flapped sections made by `Airfoil.with_flap`
(hinge 0.75 c, half thickness), PANE 160 nodes. All hinge moments in XFOIL's sign (positive =
trailing edge up, counter-clockwise); Foil-Lab's `ch` is the negative of these.

Columns: `FMOM` is the value XFOIL prints and writes as `Chinge`; "pressure only (mid)" is our
integral of XFOIL's own `Cp` (from `CPWR`) over XFOIL's own panels (from `SAVE`) with x aft of the
hinge, both arm terms; "x arm only" keeps only the chordwise arm, (x - x_h) f_y; "friction" is the
wall shear of `DUMP` (`tau = Cf Ue^2`, along the flow), chordwise arm only.

Findings:

* **Friction does not explain the gap.** Without friction (Re = 0) XFOIL's value is still 3 % (no
  deflection) to 8 % (20 deg) above our pressure integral on the same `Cp`. The friction moment is at
  most 3e-5, an order smaller than the gap at deflection and of either sign.
* **The `Cf` convention of `DUMP` is not the one assumed here**: the friction drag from `Cf Ue^2`
  differs from the polar's `CD - CDp` by up to 40 %, from `Cf` alone by 20-60 %; the last two columns
  show it. The friction numbers are therefore not reliable and prove nothing either way.
* **Undeflected flap: XFOIL's value is the chordwise-arm-only moment** of the pressure force on the
  panels aft of the hinge: 0.001632 against 0.001626 (NACA 0012, 3 deg), 0.005274 against 0.005252
  (NACA 2412). Dropping the vertical arm term -(y - y_h) f_x is the only variant that matched.
* **Deflected flap: not explained.** With the same rule the 10 deg values are 0.010865 (XFOIL
  0.011408) and 0.013999 (0.015014), and at 20 deg 0.017943 (0.020218): which panels XFOIL counts as
  the flap once the flap is rotated, and how, is not recoverable from the documentation or from
  these runs. The pressure force components XFOIL prints (x-Force, y-Force) differ from every panel
  sum tried, so its flap region or axes differ from ours.

Consequence: XFOIL's `Chinge` is a reference of unknown definition at the 3-8 % level (more with
deflection). Comparisons of Foil-Lab's `ch` against it carry that uncertainty; it is reported next to
every comparison and is not tuned away.

| section | flap deg | alpha | Re | FMOM | pressure only (mid) | pressure, x arm only | friction (Cf*Ue^2) | x arm only + friction | CM check (polar / mid / lin) | CD-CDp check (polar / Cf Ue^2 / Cf) |
|---|---|---|---|---|---|---|---|---|---|---|
| NACA 0012 | 0 | 3 | 0 | 0.001632 | 0.001586 | 0.001626 | 0.000000 | 0.001626 | -0.0042 / -0.0042 / -0.0042 | - |
| NACA 0012 | 0 | 3 | 1e+06 | 0.000254 | 0.000220 | 0.000239 | -0.000007 | 0.000232 | 0.0049 / 0.0048 / 0.0049 | 0.00552 / 0.00570 / 0.00459 |
| NACA 0012 | 0 | 6 | 1e+06 | 0.002460 | 0.002374 | 0.002446 | -0.000008 | 0.002439 | -0.0043 / -0.0043 / -0.0043 | 0.00748 / 0.00896 / 0.00578 |
| NACA 0012 | 10 | 3 | 0 | 0.011408 | 0.010628 | 0.010865 | 0.000000 | 0.010865 | -0.1275 / -0.1274 / -0.1274 | - |
| NACA 0012 | 10 | 3 | 1e+06 | 0.007080 | 0.006506 | 0.006693 | -0.000017 | 0.006676 | -0.0934 / -0.0934 / -0.0933 | 0.00961 / 0.01043 / 0.00673 |
| NACA 0012 | 10 | 6 | 1e+06 | 0.006993 | 0.006498 | 0.006649 | -0.000014 | 0.006635 | -0.0814 / -0.0814 / -0.0814 | 0.01205 / 0.01318 / 0.00701 |
| NACA 0012 | 20 | 3 | 0 | 0.020218 | 0.018769 | 0.017943 | 0.000000 | 0.017943 | -0.2462 / -0.2461 / -0.2460 | - |
| NACA 0012 | 20 | 3 | 1e+06 | 0.011131 | 0.010275 | 0.009373 | -0.000026 | 0.009347 | -0.1391 / -0.1392 / -0.1391 | 0.01819 / 0.01293 / 0.00702 |
| NACA 0012 | 20 | 6 | 1e+06 | 0.011271 | 0.010478 | 0.009477 | -0.000025 | 0.009452 | -0.1232 / -0.1233 / -0.1232 | 0.02255 / 0.01520 / 0.00690 |
| NACA 2412 | 0 | 3 | 0 | 0.005274 | 0.005086 | 0.005252 | 0.000000 | 0.005252 | -0.0600 / -0.0600 / -0.0600 | - |
| NACA 2412 | 0 | 3 | 1e+06 | 0.004316 | 0.004142 | 0.004289 | -0.000015 | 0.004274 | -0.0542 / -0.0542 / -0.0542 | 0.00542 / 0.00575 / 0.00438 |
| NACA 2412 | 0 | 6 | 1e+06 | 0.004685 | 0.004471 | 0.004635 | -0.000010 | 0.004625 | -0.0495 / -0.0495 / -0.0495 | 0.00751 / 0.00875 / 0.00556 |
| NACA 2412 | 10 | 3 | 0 | 0.015014 | 0.013956 | 0.013999 | 0.000000 | 0.013999 | -0.1828 / -0.1827 / -0.1827 | - |
| NACA 2412 | 10 | 3 | 1e+06 | 0.008771 | 0.008011 | 0.008059 | -0.000021 | 0.008038 | -0.1324 / -0.1325 / -0.1324 | 0.00978 / 0.00922 / 0.00587 |
| NACA 2412 | 10 | 6 | 1e+06 | 0.008589 | 0.007953 | 0.007906 | -0.000017 | 0.007889 | -0.1160 / -0.1161 / -0.1160 | 0.01380 / 0.01372 / 0.00705 |
| NACA 2412 | 20 | 3 | 0 | 0.023602 | 0.021699 | 0.020053 | 0.000000 | 0.020053 | -0.2999 / -0.2997 / -0.2996 | - |
| NACA 2412 | 20 | 3 | 1e+06 | 0.013409 | 0.012407 | 0.010751 | -0.000030 | 0.010721 | -0.1698 / -0.1699 / -0.1698 | 0.02160 / 0.01233 / 0.00665 |
| NACA 2412 | 20 | 6 | 1e+06 | 0.013363 | 0.012500 | 0.010721 | -0.000028 | 0.010693 | -0.1504 / -0.1505 / -0.1505 | 0.02703 / 0.01578 / 0.00706 |

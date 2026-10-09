# Suspected misprints in Williams, ARC R&M 3717 (1973), Table 2

The CSV files in this folder are kept exactly as printed. The entries below are SUSPECTED
misprints in the original report, not corrections; no value has been changed.

| File | Row | Printed page (PDF page) | Printed xi, eta, Cp | Suspicion |
|---|---|---|---|---|
| `table2a_main.csv` | 17 | 20 (21) | 0.41297, -0.07214, 0.58264 | eta is 9.0e-4 off a quartic through its eight neighbours (other mid-chord rows 1e-4 or less); the fit gives -0.07124, the printed value with two digits swapped. A panel solution on the tabulated geometry shows a Cp zigzag of -0.0096 / +0.0259 / -0.0095 at rows 16-18 while the tabulated Cp is smooth. |
| `table2b_main.csv` | 16 | 22 (23) | 0.45790, -0.06936, 0.26996 | eta is 2.7e-4 off the same kind of fit (-0.06963); smaller Cp zigzag (+0.0066 / -0.0114 / +0.0073 at rows 15-17). |

Both values were re-read on 300-dpi renders of the page images and are printed as transcribed.
Evidence: DECISIONS.md, M2, "Option (b) results".

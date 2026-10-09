# Benchmark after M2 against the v1.3.1 baseline

Tables only (DECISIONS.md, M2 Q11). M2: commit `60c5da6`, one inviscid core (linear-strength vortex panels, Drela 1989 trailing-edge treatment) under the unchanged v1.3.1 viscous methods. M1: `docs/baseline_v1.3.1.md` (unchanged). Same adapter, rules and XFOIL references (6.99, PANE 160); errors against viscous XFOIL in the attached-flow range; 'not conv.' = excluded / compared. Inviscid has no drag.

Clark Y and E387: the M2 canonical files differ from M1 (arc-length repanelling), and so do their references; their before/after differences include that geometry change. E387 with a 0.25 % gap is new in M2. The VII methods are unchanged until M3.

Accuracy targets (SPEC.md 4.1, validated range, open trailing edge): mean |dCl| <= 0.03, mean relative |dCd| <= 10 %, mean |dCm| <= 0.01.

### Open trailing edge, validated range pooled (NACA 0012, 2412, 4412, 23012, Clark Y)

| Re | method | not conv. M1 | mean dCl M1 | mean rel dCd M1 | mean dCm M1 | not conv. M2 | mean dCl M2 | mean rel dCd M2 | mean dCm M2 |
|---|---|---|---|---|---|---|---|---|---|
| 5e5-6e6 pooled | inviscid | 0/330 | 0.0678 | - | 0.0117 | 0/330 | 0.0738 | - | 0.0136 |
| 5e5-6e6 pooled | one-way (default) | 0/330 | 0.0678 | 70.3 % | 0.0117 | 0/330 | 0.0738 | 70.7 % | 0.0136 |
| 5e5-6e6 pooled | transpiration VII | 64/330 | 0.0446 | 7.1 % | 0.0093 | 61/330 | 0.0386 | 8.0 % | 0.0082 |

Targets after M2 (pooled, open trailing edge, five sections above):

| method | mean dCl | mean rel dCd | mean dCm | targets |
|---|---|---|---|---|
| inviscid | 0.0738 | - | 0.0136 | Cl not met, Cm not met |
| one-way (default) | 0.0738 | 70.7 % | 0.0136 | Cl not met, Cd not met, Cm not met |
| transpiration VII | 0.0386 | 8.0 % | 0.0082 | Cl not met, Cd met, Cm met |

### Open trailing edge by Reynolds number

| Re | method | not conv. M1 | mean dCl M1 | mean rel dCd M1 | mean dCm M1 | not conv. M2 | mean dCl M2 | mean rel dCd M2 | mean dCm M2 |
|---|---|---|---|---|---|---|---|---|---|
| 50000 (not validated) | inviscid | 0/63 | 0.2119 | - | 0.0236 | 0/64 | 0.2156 | - | 0.0247 |
| 50000 (not validated) | one-way (default) | 0/63 | 0.2119 | 75.4 % | 0.0236 | 0/64 | 0.2156 | 76.0 % | 0.0247 |
| 50000 (not validated) | transpiration VII | 28/63 | 0.1563 | 28.8 % | 0.0251 | 25/64 | 0.1743 | 31.8 % | 0.0264 |
| 50000 (not validated) | Newton VII | 8/9 | 0.1765 | 17.7 % | 0.0378 | 7/9 | 0.2175 | 46.0 % | 0.0350 |
| 100000 (not validated) | inviscid | 0/67 | 0.0982 | - | 0.0154 | 0/67 | 0.1010 | - | 0.0162 |
| 100000 (not validated) | one-way (default) | 0/67 | 0.0982 | 69.4 % | 0.0154 | 0/67 | 0.1010 | 69.9 % | 0.0162 |
| 100000 (not validated) | transpiration VII | 24/67 | 0.1247 | 20.2 % | 0.0236 | 21/67 | 0.1228 | 19.4 % | 0.0235 |
| 200000 (not validated) | inviscid | 0/70 | 0.0654 | - | 0.0116 | 0/70 | 0.0681 | - | 0.0123 |
| 200000 (not validated) | one-way (default) | 0/70 | 0.0654 | 64.9 % | 0.0116 | 0/70 | 0.0681 | 65.4 % | 0.0123 |
| 200000 (not validated) | transpiration VII | 19/70 | 0.0873 | 7.6 % | 0.0173 | 19/70 | 0.0832 | 7.4 % | 0.0167 |
| 200000 (not validated) | Newton VII | 5/10 | 0.0840 | 3.6 % | 0.0158 | 5/10 | 0.1010 | 6.4 % | 0.0220 |
| 500000 | inviscid | 0/70 | 0.0555 | - | 0.0108 | 0/70 | 0.0592 | - | 0.0115 |
| 500000 | one-way (default) | 0/70 | 0.0555 | 63.1 % | 0.0108 | 0/70 | 0.0592 | 63.5 % | 0.0115 |
| 500000 | transpiration VII | 14/70 | 0.0552 | 6.9 % | 0.0114 | 13/70 | 0.0521 | 8.7 % | 0.0109 |
| 1e+06 | inviscid | 0/78 | 0.0625 | - | 0.0113 | 0/78 | 0.0704 | - | 0.0131 |
| 1e+06 | one-way (default) | 0/78 | 0.0625 | 65.9 % | 0.0113 | 0/78 | 0.0704 | 66.6 % | 0.0131 |
| 1e+06 | transpiration VII | 13/78 | 0.0445 | 6.3 % | 0.0091 | 16/78 | 0.0376 | 6.3 % | 0.0077 |
| 3e+06 | inviscid | 0/88 | 0.0745 | - | 0.0124 | 0/88 | 0.0803 | - | 0.0145 |
| 3e+06 | one-way (default) | 0/88 | 0.0745 | 73.1 % | 0.0124 | 0/88 | 0.0803 | 73.4 % | 0.0145 |
| 3e+06 | transpiration VII | 19/88 | 0.0422 | 7.0 % | 0.0088 | 15/88 | 0.0368 | 8.1 % | 0.0080 |
| 6e+06 | inviscid | 0/94 | 0.0750 | - | 0.0120 | 0/94 | 0.0812 | - | 0.0146 |
| 6e+06 | one-way (default) | 0/94 | 0.0750 | 76.8 % | 0.0120 | 0/94 | 0.0812 | 77.0 % | 0.0146 |
| 6e+06 | transpiration VII | 18/94 | 0.0390 | 8.2 % | 0.0084 | 17/94 | 0.0312 | 8.8 % | 0.0068 |

### E387 with a 0.25 % trailing-edge gap (new in M2; open trailing edge)

| Re | method | not conv. M1 | mean dCl M1 | mean rel dCd M1 | mean dCm M1 | not conv. M2 | mean dCl M2 | mean rel dCd M2 | mean dCm M2 |
|---|---|---|---|---|---|---|---|---|---|
| 50000 (not validated) | inviscid | - | - | - | - | 0/14 | 0.1896 | - | 0.0190 |
| 50000 (not validated) | one-way (default) | - | - | - | - | 0/14 | 0.1896 | 67.9 % | 0.0190 |
| 50000 (not validated) | transpiration VII | - | - | - | - | 0/14 | 0.1197 | 46.1 % | 0.0287 |
| 100000 (not validated) | inviscid | - | - | - | - | 0/14 | 0.0523 | - | 0.0122 |
| 100000 (not validated) | one-way (default) | - | - | - | - | 0/14 | 0.0523 | 52.6 % | 0.0122 |
| 100000 (not validated) | transpiration VII | - | - | - | - | 0/14 | 0.0543 | 23.4 % | 0.0188 |
| 200000 (not validated) | inviscid | - | - | - | - | 0/13 | 0.0368 | - | 0.0062 |
| 200000 (not validated) | one-way (default) | - | - | - | - | 0/13 | 0.0368 | 40.5 % | 0.0062 |
| 200000 (not validated) | transpiration VII | - | - | - | - | 0/13 | 0.0248 | 11.9 % | 0.0062 |
| 500000 | inviscid | - | - | - | - | 0/13 | 0.0345 | - | 0.0063 |
| 500000 | one-way (default) | - | - | - | - | 0/13 | 0.0345 | 36.2 % | 0.0063 |
| 500000 | transpiration VII | - | - | - | - | 0/13 | 0.0102 | 15.6 % | 0.0022 |
| 1e+06 | inviscid | - | - | - | - | 0/13 | 0.0302 | - | 0.0052 |
| 1e+06 | one-way (default) | - | - | - | - | 0/13 | 0.0302 | 36.6 % | 0.0052 |
| 1e+06 | transpiration VII | - | - | - | - | 0/13 | 0.0080 | 9.6 % | 0.0017 |
| 3e+06 | inviscid | - | - | - | - | 0/14 | 0.0279 | - | 0.0044 |
| 3e+06 | one-way (default) | - | - | - | - | 0/14 | 0.0279 | 42.3 % | 0.0044 |
| 3e+06 | transpiration VII | - | - | - | - | 0/14 | 0.0065 | 10.7 % | 0.0017 |
| 6e+06 | inviscid | - | - | - | - | 0/16 | 0.0316 | - | 0.0046 |
| 6e+06 | one-way (default) | - | - | - | - | 0/16 | 0.0316 | 49.9 % | 0.0046 |
| 6e+06 | transpiration VII | - | - | - | - | 0/16 | 0.0066 | 11.2 % | 0.0018 |
| 5e5-6e6 pooled | inviscid | - | - | - | - | 0/56 | 0.0310 | - | 0.0051 |
| 5e5-6e6 pooled | one-way (default) | - | - | - | - | 0/56 | 0.0310 | 41.7 % | 0.0051 |
| 5e5-6e6 pooled | transpiration VII | - | - | - | - | 0/56 | 0.0078 | 11.7 % | 0.0018 |

### E387, sharp trailing edge

| Re | method | not conv. M1 | mean dCl M1 | mean rel dCd M1 | mean dCm M1 | not conv. M2 | mean dCl M2 | mean rel dCd M2 | mean dCm M2 |
|---|---|---|---|---|---|---|---|---|---|
| 50000 (not validated) | inviscid | 0/11 | 0.2780 | - | 0.0128 | 0/13 | 0.2861 | - | 0.0195 |
| 50000 (not validated) | one-way (default) | 0/11 | 0.2780 | 74.8 % | 0.0128 | 0/13 | 0.2861 | 72.2 % | 0.0195 |
| 50000 (not validated) | transpiration VII | 3/11 | 0.2581 | 55.8 % | 0.0232 | 2/13 | 0.2202 | 49.8 % | 0.0240 |
| 100000 (not validated) | inviscid | 0/12 | 0.0364 | - | 0.0088 | 0/12 | 0.0395 | - | 0.0086 |
| 100000 (not validated) | one-way (default) | 0/12 | 0.0364 | 59.3 % | 0.0088 | 0/12 | 0.0395 | 55.8 % | 0.0086 |
| 100000 (not validated) | transpiration VII | 2/12 | 0.0687 | 29.1 % | 0.0230 | 0/12 | 0.0481 | 25.3 % | 0.0178 |
| 200000 (not validated) | inviscid | 0/12 | 0.0335 | - | 0.0068 | 0/13 | 0.0443 | - | 0.0085 |
| 200000 (not validated) | one-way (default) | 0/12 | 0.0335 | 49.1 % | 0.0068 | 0/13 | 0.0443 | 47.2 % | 0.0085 |
| 200000 (not validated) | transpiration VII | 1/12 | 0.0510 | 18.1 % | 0.0138 | 0/13 | 0.0350 | 11.2 % | 0.0086 |
| 500000 | inviscid | 0/12 | 0.0478 | - | 0.0089 | 0/13 | 0.0450 | - | 0.0083 |
| 500000 | one-way (default) | 0/12 | 0.0478 | 45.8 % | 0.0089 | 0/13 | 0.0450 | 44.4 % | 0.0083 |
| 500000 | transpiration VII | 3/12 | 0.0177 | 52.6 % | 0.0032 | 0/13 | 0.0142 | 8.7 % | 0.0029 |
| 1e+06 | inviscid | 0/12 | 0.0456 | - | 0.0084 | 0/13 | 0.0431 | - | 0.0079 |
| 1e+06 | one-way (default) | 0/12 | 0.0456 | 53.5 % | 0.0084 | 0/13 | 0.0431 | 46.1 % | 0.0079 |
| 1e+06 | transpiration VII | 1/12 | 0.0205 | 27.0 % | 0.0042 | 0/13 | 0.0081 | 10.2 % | 0.0018 |
| 3e+06 | inviscid | 0/13 | 0.0371 | - | 0.0067 | 0/14 | 0.0444 | - | 0.0080 |
| 3e+06 | one-way (default) | 0/13 | 0.0371 | 61.7 % | 0.0067 | 0/14 | 0.0444 | 54.5 % | 0.0080 |
| 3e+06 | transpiration VII | 0/13 | 0.0085 | 11.6 % | 0.0017 | 0/14 | 0.0036 | 11.2 % | 0.0007 |
| 6e+06 | inviscid | 0/12 | 0.0286 | - | 0.0054 | 0/16 | 0.0530 | - | 0.0094 |
| 6e+06 | one-way (default) | 0/12 | 0.0286 | 64.9 % | 0.0054 | 0/16 | 0.0530 | 61.8 % | 0.0094 |
| 6e+06 | transpiration VII | 0/12 | 0.0050 | 10.2 % | 0.0011 | 0/16 | 0.0046 | 11.8 % | 0.0008 |
| 5e5-6e6 pooled | inviscid | 0/49 | 0.0397 | - | 0.0073 | 0/56 | 0.0467 | - | 0.0085 |
| 5e5-6e6 pooled | one-way (default) | 0/49 | 0.0397 | 56.6 % | 0.0073 | 0/56 | 0.0467 | 52.3 % | 0.0085 |
| 5e5-6e6 pooled | transpiration VII | 4/49 | 0.0124 | 23.2 % | 0.0024 | 0/56 | 0.0074 | 10.6 % | 0.0015 |

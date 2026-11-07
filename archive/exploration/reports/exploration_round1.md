# Exploration round 1: entity-specific scene structure (England, training matches only)

Held-out log-likelihood gain per outcome, x1000 nats. Two folds alternate matches by kickoff.
scene = population with the variable vs without; style = entity's constant zone style;
interaction = entity-specific response to the variable beyond population response and constant style.
Brackets: 95% bootstrap CI over held-out matches. * = CI above 0.

## team_what

| entity | n | style (zone only) | phase | prev_event | time_bin | score_state | venue | opponent_class |
|---|---|---|---|---|---|---|---|---|
| Manchester City | 23418 | +45.1 | -0.13 [-0.89, +0.51] | -0.69 [-2.07, +0.40] | -0.29 [-0.91, +0.28] | +0.07 [-0.63, +0.71] | -0.16 [-0.43, +0.08] | -0.19 [-0.56, +0.19] |
| Liverpool | 18683 | +7.5 | +0.16 [-0.37, +0.78] | +0.25 [-0.07, +0.53] | +0.27 [+0.07, +0.47]* | +0.55 [-0.80, +2.08] | -0.04 [-0.51, +0.46] | -2.83 [-5.82, -0.20] |
| Burnley | 11950 | +18.3 | +0.98 [+0.13, +1.90]* | +1.14 [+0.42, +2.08]* | -0.09 [-0.29, +0.12] | +0.53 [-0.48, +1.59] | -0.51 [-1.35, +0.24] | +0.13 [-0.11, +0.38] |
| Manchester United | 16812 | -2.6 | +0.04 [-0.21, +0.27] | -0.13 [-0.31, +0.05] | -0.03 [-0.23, +0.17] | +0.07 [-0.21, +0.38] | -0.13 [-0.59, +0.33] | +0.59 [-0.05, +1.41] |

Population scene gain (same entities, x1000 nats):

| entity | phase | prev_event | time_bin | score_state | venue | opponent_class |
|---|---|---|---|---|---|---|
| Manchester City | +64.12 | +64.50 | -2.66 | -4.86 | -0.92 | +1.03 |
| Liverpool | +71.11 | +76.05 | +0.16 | -0.00 | -0.89 | +1.06 |
| Burnley | +93.00 | +115.42 | -0.35 | -0.01 | +0.09 | +3.67 |
| Manchester United | +66.54 | +71.57 | -0.11 | +0.10 | -0.19 | +3.45 |

- Liverpool / time_bin: final third, right, second_half_to_75, touch: observed 0.088, additive 0.055, population 0.055 (n=533, z=+3.4)
- Liverpool / time_bin: final third, centre, second_half_to_75, shot: observed 0.244, additive 0.197, population 0.206 (n=472, z=+2.5)
- Liverpool / time_bin: middle third, right, second_half_to_75, short_pass: observed 0.689, additive 0.727, population 0.680 (n=854, z=-2.5)
- Burnley / phase: own third, right, settled, clearance: observed 0.017, additive 0.001, population 0.001 (n=121, z=+5.6)
- Burnley / phase: middle third, left, settled, head_pass: observed 0.022, additive 0.008, population 0.006 (n=627, z=+3.9)
- Burnley / phase: final third, left, settled, cross: observed 0.244, additive 0.180, population 0.135 (n=398, z=+3.3)
- Burnley / prev_event: middle third, centre, own, clearance: observed 0.002, additive 0.000, population 0.000 (n=999, z=+3.5)
- Burnley / prev_event: final third, centre, opp_pass, long_pass: observed 0.100, additive 0.027, population 0.024 (n=60, z=+3.5)
- Burnley / prev_event: middle third, left, opp_set_piece, touch: observed 0.152, additive 0.047, population 0.065 (n=46, z=+3.4)

## player_what

| entity | n | style (zone only) | phase | prev_event | time_bin | score_state | venue | opponent_class |
|---|---|---|---|---|---|---|---|---|
| K. De Bruyne (Manchester City, MD) | 2439 | +10.2 | -0.03 [-0.28, +0.20] | +0.05 [-0.10, +0.21] | +0.02 [-0.53, +0.51] | -0.38 [-1.79, +0.78] | +0.07 [-0.19, +0.28] | +0.24 [+0.01, +0.47]* |
| Mohamed Salah (Liverpool, FW) | 1231 | -32.2 | +0.16 [-0.30, +0.66] | -0.41 [-0.71, -0.13] | -0.02 [-0.33, +0.33] | -0.04 [-0.41, +0.36] | +0.01 [-0.40, +0.43] | +0.03 [-0.23, +0.29] |
| H. Kane (Tottenham Hotspur, FW) | 1022 | -4.5 | +0.17 [-0.18, +0.50] | -0.20 [-0.43, -0.00] | -0.22 [-0.48, +0.03] | -0.08 [-0.32, +0.16] | +0.48 [-2.40, +3.26] | -0.28 [-0.57, +0.00] |
| N. Kante (Chelsea, MD) | 1720 | -25.5 | -0.27 [-0.47, -0.05] | -0.53 [-0.78, -0.25] | -0.17 [-0.44, +0.10] | -0.09 [-0.28, +0.08] | -0.16 [-0.38, +0.09] | -0.54 [-0.98, -0.08] |
| B. Mee (Burnley, DF) | 1086 | +18.2 | +0.12 [-0.49, +0.80] | +0.07 [-0.15, +0.27] | +0.11 [-0.13, +0.32] | +0.02 [-0.16, +0.20] | -0.08 [-0.47, +0.27] | -0.04 [-0.41, +0.31] |

Population scene gain (same entities, x1000 nats):

| entity | phase | prev_event | time_bin | score_state | venue | opponent_class |
|---|---|---|---|---|---|---|
| K. De Bruyne (Manchester City, MD) | +38.51 | +42.35 | -8.08 | +1.15 | -0.44 | -0.87 |
| Mohamed Salah (Liverpool, FW) | +7.99 | +8.74 | -6.13 | -9.50 | -1.02 | -1.80 |
| H. Kane (Tottenham Hotspur, FW) | +2.93 | -2.50 | -11.07 | -10.32 | -3.66 | -3.48 |
| N. Kante (Chelsea, MD) | +30.96 | +40.74 | -0.54 | -1.77 | -1.32 | +1.39 |
| B. Mee (Burnley, DF) | +168.63 | +236.57 | -11.10 | -4.60 | -6.51 | +1.64 |

- K. De Bruyne (Manchester City, MD) / opponent_class: final third, right, big_six, short_pass: observed 0.400, additive 0.537, population 0.462 (n=85, z=-2.5)
- K. De Bruyne (Manchester City, MD) / opponent_class: middle third, right, big_six, head_pass: observed 0.045, additive 0.016, population 0.042 (n=112, z=+2.3)
- K. De Bruyne (Manchester City, MD) / opponent_class: final third, left, big_six, shot: observed 0.065, additive 0.020, population 0.027 (n=46, z=+2.2)

## player_who

| entity | n | style (zone only) | phase | prev_event | time_bin | score_state | venue | opponent_class |
|---|---|---|---|---|---|---|---|---|
| K. De Bruyne (Manchester City, MD) | 22101 | +6.3 | +0.99 [+0.24, +1.77]* | +0.07 [-0.15, +0.28] | -0.12 [-0.32, +0.06] | -0.02 [-0.15, +0.09] | -0.00 [-0.20, +0.21] | -0.38 [-0.65, -0.11] |
| Mohamed Salah (Liverpool, FW) | 15575 | +30.8 | +0.12 [-0.26, +0.48] | +0.05 [-0.00, +0.12] | +0.34 [-0.04, +0.80] | -0.40 [-0.86, +0.04] | +0.07 [-0.08, +0.24] | -0.08 [-0.27, +0.11] |
| H. Kane (Tottenham Hotspur, FW) | 17358 | +3.3 | -0.11 [-0.33, +0.07] | -0.29 [-0.60, +0.02] | +0.07 [-0.04, +0.21] | +0.44 [-0.25, +1.19] | +0.40 [+0.01, +0.77]* | -2.80 [-3.67, -1.84] |
| N. Kante (Chelsea, MD) | 15301 | +0.1 | -0.02 [-0.54, +0.47] | -0.09 [-0.20, +0.02] | -0.88 [-1.92, +0.15] | -0.37 [-0.89, +0.12] | -0.26 [-1.32, +0.71] | -0.06 [-0.13, -0.00] |
| B. Mee (Burnley, DF) | 11003 | +37.9 | +3.59 [+2.14, +5.11]* | +2.02 [+1.22, +2.86]* | -0.82 [-1.77, -0.02] | -0.03 [-0.10, +0.04] | -0.21 [-0.35, -0.08] | +0.53 [-0.12, +1.15] |

Population scene gain (same entities, x1000 nats):

| entity | phase | prev_event | time_bin | score_state | venue | opponent_class |
|---|---|---|---|---|---|---|
| K. De Bruyne (Manchester City, MD) | +0.22 | -0.06 | +0.07 | -0.07 | +0.07 | +0.07 |
| Mohamed Salah (Liverpool, FW) | +2.31 | -0.39 | +0.00 | -0.34 | -0.19 | +0.56 |
| H. Kane (Tottenham Hotspur, FW) | +3.09 | +2.22 | -0.05 | -0.24 | +0.25 | +0.05 |
| N. Kante (Chelsea, MD) | +1.17 | +1.06 | +0.12 | -0.17 | +0.12 | +0.04 |
| B. Mee (Burnley, DF) | +4.08 | +2.63 | +0.01 | -0.14 | +0.01 | +0.57 |

- K. De Bruyne (Manchester City, MD) / phase: middle third, centre, settled, actor: observed 0.064, additive 0.091, population 0.159 (n=1906, z=-4.1)
- K. De Bruyne (Manchester City, MD) / phase: middle third, centre, early, actor: observed 0.127, additive 0.091, population 0.158 (n=1022, z=+4.1)
- K. De Bruyne (Manchester City, MD) / phase: middle third, left, early, actor: observed 0.120, additive 0.092, population 0.113 (n=1135, z=+3.3)
- H. Kane (Tottenham Hotspur, FW) / venue: middle third, right, away, actor: observed 0.056, additive 0.039, population 0.061 (n=1079, z=+2.8)
- H. Kane (Tottenham Hotspur, FW) / venue: middle third, right, home, actor: observed 0.025, additive 0.036, population 0.056 (n=1568, z=-2.4)
- H. Kane (Tottenham Hotspur, FW) / venue: final third, left, away, actor: observed 0.134, additive 0.107, population 0.152 (n=731, z=+2.4)
- B. Mee (Burnley, DF) / phase: own third, left, settled, actor: observed 0.575, additive 0.365, population 0.189 (n=174, z=+5.8)
- B. Mee (Burnley, DF) / phase: middle third, centre, first, actor: observed 0.145, additive 0.083, population 0.075 (n=317, z=+4.0)
- B. Mee (Burnley, DF) / phase: own third, left, first, actor: observed 0.252, additive 0.323, population 0.162 (n=516, z=-3.4)
- B. Mee (Burnley, DF) / prev_event: final third, left, opp_pass, actor: observed 0.087, additive 0.019, population 0.062 (n=69, z=+4.2)
- B. Mee (Burnley, DF) / prev_event: middle third, centre, opp_pass, actor: observed 0.174, additive 0.095, population 0.088 (n=201, z=+3.8)
- B. Mee (Burnley, DF) / prev_event: middle third, right, opp_pass, actor: observed 0.032, additive 0.009, population 0.109 (n=253, z=+3.7)

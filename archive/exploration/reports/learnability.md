# Round 5 (development, England training matches): can the differences be learned forward?

Estimation: each team's first k training matches (by date); evaluation: its 20th and later training matches
(152 team-matches). Gains: held-out log-likelihood per row, x1000 nats; CIs: match bootstrap. Interaction terms are
shrunk with lambda = phi / tau^2 from the variance-component test on the first 19 matches; style terms by inner CV.
Expected gain = sum of 0.5 x reliability x tau^2 x V on the evaluation rows. See src/learnability.py.

## Hyperparameters (first 19 matches)

| spec | tau | phi | p |
|---|---|---|---|
| team|long_pass | 0.083 | 1.01 | 0.010 |
| team|cross | 0.084 | 0.82 | 0.020 |
| team|back_pass | 0.059 | 0.85 | 0.010 |
| team|forward | 0.037 | 0.93 | 0.010 |
| who|slot | 0.054 | 0.88 | 0.010 |
| who|player_all | 0.083 | 0.92 | 0.010 |
| who|player_beyond | 0.070 | 1.07 | 0.010 |
| pwhat|back_pass|slot | 0.040 | 0.79 | 0.228 |
| pwhat|back_pass|player_all | 0.094 | 0.82 | 0.010 |
| pwhat|back_pass|player_beyond | 0.065 | 0.89 | 0.020 |
| pwhat|long_pass|slot | 0.111 | 0.77 | 0.010 |
| pwhat|long_pass|player_all | 0.097 | 0.81 | 0.020 |
| pwhat|long_pass|player_beyond | 0.000 | 0.91 | 0.822 |
| pwhat|forward|slot | 0.045 | 0.77 | 0.089 |
| pwhat|forward|player_all | 0.059 | 0.80 | 0.020 |
| pwhat|forward|player_beyond | 0.000 | 0.87 | 0.703 |

## Teams: team x steps on top of team style

| target | k = 5: gain [CI] (expected; median reliability) | k = 10: gain [CI] (expected; median reliability) | k = 15: gain [CI] (expected; median reliability) | k = 19: gain [CI] (expected; median reliability) |
|---|---|---|---|---|
| long_pass | +0.123 [+0.036, +0.212] (+0.052; 0.22) | +0.126 [+0.014, +0.245] (+0.085; 0.35) | +0.154 [+0.029, +0.282] (+0.107; 0.45) | +0.152 [+0.015, +0.293] (+0.122; 0.51) |
| cross | +0.038 [-0.002, +0.075] (+0.013; 0.08) | +0.032 [-0.027, +0.089] (+0.021; 0.14) | +0.034 [-0.030, +0.098] (+0.028; 0.20) | +0.052 [-0.011, +0.116] (+0.032; 0.24) |
| back_pass | -0.104 [-0.201, -0.011] (+0.058; 0.20) | -0.095 [-0.203, +0.010] (+0.090; 0.33) | -0.054 [-0.174, +0.072] (+0.111; 0.43) | -0.028 [-0.177, +0.107] (+0.124; 0.48) |
| forward | +0.019 [-0.028, +0.067] (+0.015; 0.11) | +0.088 [+0.025, +0.164] (+0.026; 0.21) | +0.075 [+0.007, +0.153] (+0.034; 0.28) | +0.097 [+0.018, +0.177] (+0.040; 0.33) |

Steps 0-2 at k = 19: mean absolute calibration error over team x step cells (>= 50 rows), team style vs + team x steps:

- long_pass (rate 9.5%): 1.40 -> 1.29 pp
- cross (rate 1.6%): 0.36 -> 0.37 pp
- back_pass (rate 13.0%): 1.57 -> 1.58 pp
- forward (rate 21.6%): 1.42 -> 1.37 pp

## Who performs the action (k = 19): placements, all relative to the population model

Cold rows: candidates with fewer than 5 estimation matches (10.8% of evaluation rows).

| model | gain [CI] | top-1 | top-1, steps 0-2 | cold gain | warm gain |
|---|---|---|---|---|---|
| population | 0 | 28.81% | 26.70% | 0 | 0 |
| slot_style | +0.587 [+0.433, +0.737] | 28.92% | 26.76% | +0.634 | +0.581 |
| slot_style+slot_steps | +0.631 [+0.467, +0.806] | 28.98% | 26.82% | +0.625 | +0.632 |
| player_style | +1.629 [+1.368, +1.891] | 29.47% | 27.41% | +0.120 | +1.811 |
| player_style+player_steps | +1.767 [+1.507, +2.041] | 29.61% | 27.44% | +0.107 | +1.967 |
| hier_style | +1.676 [+1.385, +1.978] | 29.46% | 27.27% | +0.686 | +1.796 |
| hier_style+slot_steps+player_steps | +1.828 [+1.523, +2.142] | 29.53% | 27.37% | +0.668 | +1.968 |

Steps terms on top of style, by history length (gain [CI]; top-1 change):

| chain | k = 5 | k = 10 | k = 15 | k = 19 |
|---|---|---|---|---|
| slot_style+slot_steps | -0.001 [-0.023, +0.022]; -0.01 pp | +0.001 [-0.025, +0.031]; -0.06 pp | +0.021 [-0.009, +0.053]; -0.05 pp | +0.045 [+0.016, +0.075]; +0.06 pp |
| player_style+player_steps | +0.074 [+0.041, +0.108]; -0.05 pp | +0.094 [+0.054, +0.133]; +0.13 pp | +0.114 [+0.076, +0.154]; +0.12 pp | +0.138 [+0.101, +0.176]; +0.14 pp |
| hier_style+slot_steps+player_steps | +0.052 [+0.026, +0.080]; -0.00 pp | +0.073 [+0.038, +0.112]; -0.03 pp | +0.115 [+0.079, +0.152]; +0.15 pp | +0.152 [+0.119, +0.185]; +0.07 pp |

Expected gain of the steps terms (slot; player) and median reliability: k = 5: +0.011, +0.042 (R 0.06, 0.14); k = 10: +0.019, +0.070 (R 0.11, 0.22); k = 15: +0.026, +0.091 (R 0.15, 0.29); k = 19: +0.030, +0.105 (R 0.19, 0.33)

## Players' action choice: steps terms on top of style (gain [CI])

| target | chain | k = 5 | k = 10 | k = 15 | k = 19 |
|---|---|---|---|---|---|
| back_pass | slot_style+slot_steps | -0.004 [-0.017, +0.010] | +0.026 [+0.002, +0.053] | +0.049 [+0.020, +0.081] | +0.024 [-0.005, +0.053] |
| back_pass | player_style+player_steps | +0.035 [-0.044, +0.108] | +0.107 [-0.003, +0.222] | +0.172 [+0.047, +0.292] | +0.238 [+0.115, +0.364] |
| back_pass | hier_style+slot_steps+player_steps | +0.010 [-0.030, +0.051] | +0.068 [+0.006, +0.130] | +0.107 [+0.044, +0.170] | +0.187 [+0.114, +0.258] |
| long_pass | slot_style+slot_steps | +0.053 [+0.000, +0.107] | +0.047 [-0.024, +0.114] | +0.143 [+0.039, +0.251] | +0.197 [+0.086, +0.313] |
| long_pass | player_style+player_steps | +0.121 [+0.083, +0.159] | +0.136 [+0.085, +0.193] | +0.177 [+0.115, +0.240] | +0.205 [+0.133, +0.278] |
| long_pass | hier_style+slot_steps+player_steps | +0.053 [+0.001, +0.101] | +0.047 [-0.020, +0.119] | +0.073 [-0.013, +0.158] | +0.119 [+0.025, +0.212] |
| forward | slot_style+slot_steps | +0.027 [+0.004, +0.050] | +0.025 [-0.005, +0.055] | +0.034 [-0.001, +0.067] | +0.067 [+0.030, +0.109] |
| forward | player_style+player_steps | +0.050 [+0.019, +0.083] | +0.049 [+0.008, +0.092] | +0.196 [+0.137, +0.253] | +0.255 [+0.189, +0.321] |
| forward | hier_style+slot_steps+player_steps | +0.015 [-0.006, +0.038] | +0.014 [-0.013, +0.041] | +0.021 [-0.012, +0.053] | +0.052 [+0.015, +0.091] |

Best chain at k = 19 (gain vs population): back_pass: slot_style+slot_steps +0.55, player_style+player_steps +1.55, hier_style+slot_steps+player_steps +1.05; long_pass: slot_style+slot_steps +3.94, player_style+player_steps +6.78, hier_style+slot_steps+player_steps +5.90; forward: slot_style+slot_steps +1.96, player_style+player_steps +3.43, hier_style+slot_steps+player_steps +2.72


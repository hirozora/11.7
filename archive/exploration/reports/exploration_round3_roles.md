# Round 3b: who performs the team's action x scene, under three population baselines (training matches only)

Players: 273 with one team and >= 15 matches. Rows: the team's open-play on-ball
actions while the player is on the pitch, in matches where his fine role is assigned (identical across baselines).
Baselines: wyscout4 = Wyscout's four roles; fine = depth relative to the team x lateral band, from the player's mean
event position in that match; fine_backline = fine x the team's back-line size in that match (bl2 / bl3).
Gains: held-out log-likelihood per opportunity, x1000 nats; interaction = the player's response to the variable beyond
the baseline population's response and his constant zone style. z_cal and 'replicated' as in round 2.

## Interaction gain by baseline

| variable | baseline | mean gain [entity-bootstrap CI] | share > 0 | mean forward | s | z_cal > 3 | replicated |
|---|---|---|---|---|---|---|---|
| phase | wyscout4 | +0.472 [+0.314, +0.625] | 0.65 | +0.113 | 4.03 | 15 | 15 |
| phase | fine | -0.019 [-0.091, +0.053] | 0.54 | +0.001 | 2.69 | 3 | 3 |
| phase | fine_backline | -0.013 [-0.085, +0.055] | 0.55 | -0.045 | 3.37 | 0 | 0 |
| steps_since_change | wyscout4 | +0.394 [+0.230, +0.560] | 0.68 | +0.334 | 3.54 | 15 | 13 |
| steps_since_change | fine | -0.084 [-0.217, +0.013] | 0.56 | -0.059 | 3.99 | 0 | 0 |
| steps_since_change | fine_backline | -0.075 [-0.201, +0.019] | 0.57 | -0.056 | 3.14 | 1 | 1 |
| prev_event | wyscout4 | -0.041 [-0.335, +0.148] | 0.52 | -0.409 | 4.00 | 4 | 4 |
| prev_event | fine | -0.082 [-0.142, -0.027] | 0.47 | -0.137 | 3.69 | 0 | 0 |
| prev_event | fine_backline | -0.094 [-0.156, -0.034] | 0.46 | -0.141 | 4.16 | 0 | 0 |
| opp_last_event | wyscout4 | -0.024 [-0.094, +0.046] | 0.51 | -0.033 | 3.14 | 3 | 3 |
| opp_last_event | fine | -0.096 [-0.151, -0.046] | 0.45 | -0.094 | 2.42 | 1 | 1 |
| opp_last_event | fine_backline | -0.089 [-0.138, -0.040] | 0.46 | -0.094 | 1.96 | 0 | 0 |
| possession_start | wyscout4 | -0.068 [-0.157, +0.020] | 0.50 | -0.073 | 3.33 | 1 | 1 |
| possession_start | fine | -0.170 [-0.238, -0.109] | 0.43 | -0.079 | 3.22 | 0 | 0 |
| possession_start | fine_backline | -0.181 [-0.251, -0.120] | 0.46 | -0.115 | 3.59 | 0 | 0 |

## Players replicated under wyscout4: the same gain under the fine baselines

| player | team | variable | wyscout4 gain (z_cal) | fine gain (z_cal) | fine_backline gain (z_cal) | share removed by fine |
|---|---|---|---|---|---|---|
| M. Yoshida | Southampton | phase | +2.20 (3.1) | +0.16 (0.3) | +0.04 (0.1) | 92.9% |
| M. Yoshida | Southampton | steps_since_change | +2.29 (3.4) | +0.25 (0.5) | +0.18 (0.3) | 89.2% |
| M. Özil | Arsenal | prev_event | +2.33 (4.4) | +0.01 (0.2) | +0.38 (0.8) | 99.5% |
| R. Bertrand | Southampton | phase | +3.08 (3.5) | +0.09 (0.3) | +0.17 (0.3) | 97.1% |
| W. Rooney | Everton | phase | +2.42 (4.3) | +0.74 (1.2) | +0.82 (1.6) | 69.6% |
| Marcos Alonso | Chelsea | phase | +4.02 (4.1) | +0.48 (0.8) | +0.39 (1.2) | 88.2% |
| K. Naughton | Swansea City | phase | +3.82 (3.8) | +0.80 (2.0) | +0.82 (1.2) | 79.2% |
| K. Naughton | Swansea City | steps_since_change | +3.78 (3.9) | +0.41 (0.8) | +0.52 (1.0) | 89.2% |
| S. Ward | Burnley | phase | +4.40 (3.0) | -0.01 (0.2) | +0.04 (0.2) | 100.1% |
| H. Maguire | Leicester City | phase | +3.30 (5.1) | +0.50 (1.1) | +0.44 (1.2) | 84.7% |
| H. Maguire | Leicester City | steps_since_change | +3.07 (7.0) | +0.25 (0.7) | +0.16 (0.6) | 91.7% |
| B. Mee | Burnley | phase | +3.28 (3.6) | +0.57 (1.0) | +0.56 (0.6) | 82.6% |
| L. Dunk | Brighton & Hove Albion | phase | +4.60 (3.1) | +0.19 (0.4) | +0.22 (0.4) | 95.9% |
| L. Dunk | Brighton & Hove Albion | steps_since_change | +5.39 (4.6) | +0.39 (1.0) | +0.41 (0.6) | 92.7% |
| L. Dunk | Brighton & Hove Albion | opp_last_event | +2.34 (3.5) | +0.30 (1.1) | +0.32 (0.9) | 87.3% |
| L. Dunk | Brighton & Hove Albion | possession_start | +2.58 (5.2) | +0.12 (0.3) | +0.09 (0.2) | 95.5% |
| D. Alli | Tottenham Hotspur | steps_since_change | +1.46 (3.6) | +0.24 (0.6) | +0.12 (0.4) | 83.4% |
| A. Rüdiger | Chelsea | steps_since_change | +1.59 (3.8) | +1.26 (1.5) | +0.71 (1.1) | 20.6% |
| Son Heung-Min | Tottenham Hotspur | phase | +1.34 (3.0) | +0.88 (2.8) | +0.91 (1.7) | 33.8% |
| C. Schindler | Huddersfield Town | phase | +6.33 (5.9) | +1.20 (2.5) | +1.17 (2.2) | 81.0% |
| C. Schindler | Huddersfield Town | steps_since_change | +6.91 (8.6) | +1.00 (1.0) | +0.80 (2.4) | 85.5% |
| A. Ogbonna | West Ham United | steps_since_change | +2.76 (4.2) | +0.04 (0.2) | +0.03 (0.2) | 98.7% |
| A. Lacazette | Arsenal | prev_event | +0.86 (4.1) | +0.90 (2.2) | +0.91 (1.8) | -4.9% |
| Azpilicueta | Chelsea | phase | +2.11 (4.9) | +0.84 (1.3) | +0.96 (1.6) | 60.4% |
| Azpilicueta | Chelsea | steps_since_change | +1.85 (3.3) | +0.30 (0.7) | +0.40 (1.1) | 83.8% |
| X. Shaqiri | Stoke City | prev_event | +2.22 (3.3) | +0.12 (0.2) | -0.18 (-0.3) | 94.7% |
| M. Jørgensen | Huddersfield Town | phase | +4.13 (5.5) | +0.84 (1.4) | +0.78 (1.3) | 79.7% |
| M. Jørgensen | Huddersfield Town | steps_since_change | +4.00 (6.2) | +0.91 (1.6) | +0.82 (1.2) | 77.2% |
| N. Aké | AFC Bournemouth | opp_last_event | +0.62 (3.0) | +0.02 (0.2) | -0.02 (0.3) | 96.9% |
| A. Christensen | Chelsea | phase | +4.91 (3.3) | +1.08 (1.6) | +1.18 (0.5) | 78.1% |
| E. Dier | Tottenham Hotspur | steps_since_change | +2.55 (6.7) | +0.10 (0.4) | +0.07 (0.3) | 96.1% |
| E. Dier | Tottenham Hotspur | opp_last_event | +1.11 (3.5) | -0.01 (0.2) | -0.10 (-0.2) | 100.6% |
| A. Mawson | Swansea City | phase | +2.80 (5.9) | -0.00 (0.2) | -0.00 (0.1) | 100.1% |
| A. Mawson | Swansea City | steps_since_change | +3.00 (3.1) | +0.01 (0.1) | +0.01 (0.2) | 99.7% |
| D. Sánchez | Tottenham Hotspur | steps_since_change | +2.66 (3.8) | +0.59 (0.7) | +0.58 (2.2) | 77.8% |
| D. Sánchez | Tottenham Hotspur | prev_event | +2.31 (4.0) | +0.49 (0.5) | +0.39 (0.4) | 78.9% |

Median share of the wyscout4 interaction gain removed by the fine baseline: 87.7%.

## Players replicated under fine

| player | team | role | variable | gain [CI] | forward | z_cal |
|---|---|---|---|---|---|---|
| T. Carroll | Swansea City | MD | opp_last_event | +0.87 [+0.24, +1.42] | +0.77 | 4.5 |
| C. Wood | Burnley | FW | phase | +3.36 [+0.78, +5.94] | +2.15 | 4.3 |
| C. Benteke | Crystal Palace | FW | phase | +1.65 [+0.53, +2.82] | +1.89 | 3.8 |
| S. Rondón | West Bromwich Albion | FW | phase | +2.10 [+0.41, +3.89] | +1.12 | 3.6 |

## Players replicated under fine_backline

| player | team | role | variable | gain [CI] | forward | z_cal |
|---|---|---|---|---|---|---|
| P. Jones | Manchester United | DF | steps_since_change | +1.95 [+0.65, +3.30] | +1.76 | 3.4 |

## Effect sizes on held-out rows (cells selected on the fitting half, |gamma| >= 0.3, >= 30 fitting rows)

up / down: cells where the player's interaction raises / lowers his probability of performing the action.
observed = share of these held-out team actions performed by the player; additive / interaction = mean predicted probability.
top-1 / top-3: the true actor among on-pitch teammates (teammates scored by their additive models), with the player scored
by his additive vs his interaction model.

| player | baseline | variable | cells | n | observed | additive | interaction | top-1 add -> int | top-3 add -> int |
|---|---|---|---|---|---|---|---|---|---|
| T. Carroll | fine | opp_last_event | up | 230 | 11.7% | 12.7% | 17.2% | 25.2% -> 25.7% | 71.7% -> 71.7% |
| T. Carroll | fine | opp_last_event | all | 11710 | 10.4% | 10.5% | 10.3% | 28.4% -> 28.4% | 67.4% -> 67.4% |
| S. Rondón | fine | phase | up | 409 | 13.4% | 9.0% | 11.8% | 26.2% -> 27.4% | 60.4% -> 61.4% |
| S. Rondón | fine | phase | down | 635 | 6.0% | 7.5% | 4.6% | 24.7% -> 24.7% | 62.2% -> 62.5% |
| S. Rondón | fine | phase | all | 9037 | 10.2% | 9.9% | 10.0% | 27.0% -> 27.1% | 63.3% -> 63.6% |
| C. Wood | fine | phase | up | 269 | 27.1% | 20.7% | 28.2% | 23.8% -> 27.5% | 59.5% -> 60.2% |
| C. Wood | fine | phase | down | 1064 | 3.7% | 6.4% | 4.1% | 29.9% -> 30.1% | 64.6% -> 64.5% |
| C. Wood | fine | phase | all | 5027 | 7.5% | 7.8% | 7.9% | 28.2% -> 28.4% | 63.3% -> 63.3% |
| C. Benteke | fine | phase | up | 427 | 7.7% | 7.8% | 10.6% | 25.1% -> 25.1% | 61.4% -> 61.1% |
| C. Benteke | fine | phase | down | 689 | 1.0% | 3.7% | 2.6% | 30.9% -> 30.9% | 64.6% -> 64.6% |
| C. Benteke | fine | phase | all | 9739 | 6.2% | 6.5% | 6.7% | 29.5% -> 29.6% | 65.1% -> 65.2% |
| P. Jones | fine_backline | steps_since_change | up | 316 | 21.5% | 17.0% | 23.8% | 20.6% -> 19.0% | 61.1% -> 62.7% |
| P. Jones | fine_backline | steps_since_change | down | 478 | 13.6% | 18.8% | 14.5% | 27.2% -> 27.2% | 61.1% -> 63.4% |
| P. Jones | fine_backline | steps_since_change | all | 12243 | 8.7% | 8.7% | 8.6% | 28.0% -> 28.0% | 64.9% -> 65.1% |
| L. Dunk | wyscout4 | opp_last_event | up | 372 | 15.3% | 11.8% | 18.5% | 32.8% -> 33.1% | 66.7% -> 65.9% |
| L. Dunk | wyscout4 | opp_last_event | down | 2912 | 1.4% | 1.7% | 0.9% | 30.9% -> 30.9% | 64.3% -> 64.4% |
| L. Dunk | wyscout4 | opp_last_event | all | 13710 | 10.0% | 10.2% | 10.0% | 29.1% -> 29.2% | 61.7% -> 61.7% |
| N. Aké | wyscout4 | opp_last_event | up | 162 | 24.7% | 13.6% | 19.5% | 25.3% -> 26.5% | 69.8% -> 70.4% |
| N. Aké | wyscout4 | opp_last_event | down | 339 | 8.3% | 9.1% | 6.8% | 25.1% -> 25.1% | 57.8% -> 57.8% |
| N. Aké | wyscout4 | opp_last_event | all | 13759 | 9.0% | 9.1% | 9.0% | 27.7% -> 28.0% | 62.8% -> 62.8% |
| E. Dier | wyscout4 | opp_last_event | up | 932 | 15.5% | 11.3% | 16.0% | 22.2% -> 22.7% | 60.0% -> 59.8% |
| E. Dier | wyscout4 | opp_last_event | down | 1272 | 16.2% | 19.9% | 15.0% | 25.5% -> 25.5% | 58.6% -> 59.9% |
| E. Dier | wyscout4 | opp_last_event | all | 18004 | 10.7% | 10.7% | 10.5% | 25.9% -> 25.9% | 59.8% -> 60.0% |
| M. Yoshida | wyscout4 | phase | up | 517 | 22.1% | 14.4% | 19.4% | 24.8% -> 24.8% | 68.7% -> 69.6% |
| M. Yoshida | wyscout4 | phase | down | 603 | 6.6% | 9.3% | 6.6% | 23.1% -> 23.1% | 56.6% -> 55.9% |
| M. Yoshida | wyscout4 | phase | all | 10329 | 9.8% | 10.0% | 10.0% | 24.9% -> 25.0% | 59.0% -> 59.0% |
| R. Bertrand | wyscout4 | phase | up | 1887 | 2.8% | 2.1% | 3.3% | 21.9% -> 21.9% | 56.6% -> 56.6% |
| R. Bertrand | wyscout4 | phase | down | 6774 | 1.8% | 3.0% | 1.6% | 24.7% -> 24.5% | 61.2% -> 61.2% |
| R. Bertrand | wyscout4 | phase | all | 14133 | 9.6% | 9.6% | 9.6% | 25.1% -> 24.9% | 60.4% -> 60.4% |
| W. Rooney | wyscout4 | phase | up | 2057 | 12.6% | 8.4% | 12.3% | 22.4% -> 22.4% | 53.3% -> 55.1% |
| W. Rooney | wyscout4 | phase | down | 1252 | 11.2% | 11.6% | 8.2% | 20.6% -> 20.3% | 48.6% -> 47.0% |
| W. Rooney | wyscout4 | phase | all | 9215 | 11.1% | 10.9% | 11.0% | 22.6% -> 22.6% | 54.1% -> 54.2% |
| Marcos Alonso | wyscout4 | phase | up | 2221 | 13.0% | 9.0% | 13.3% | 22.5% -> 23.6% | 57.4% -> 57.4% |
| Marcos Alonso | wyscout4 | phase | down | 6218 | 3.1% | 4.6% | 2.9% | 34.7% -> 34.5% | 68.0% -> 67.9% |
| Marcos Alonso | wyscout4 | phase | all | 16490 | 8.4% | 8.0% | 8.0% | 30.3% -> 30.3% | 64.0% -> 64.0% |
| K. Naughton | wyscout4 | phase | up | 1832 | 15.4% | 11.4% | 16.3% | 21.8% -> 25.5% | 59.3% -> 59.3% |
| K. Naughton | wyscout4 | phase | down | 4700 | 2.3% | 4.3% | 2.2% | 29.4% -> 30.4% | 68.8% -> 68.8% |
| K. Naughton | wyscout4 | phase | all | 13115 | 11.9% | 11.8% | 11.8% | 27.4% -> 28.4% | 63.5% -> 63.5% |
| S. Ward | wyscout4 | phase | up | 317 | 11.7% | 9.4% | 14.4% | 26.8% -> 24.3% | 61.8% -> 61.8% |
| S. Ward | wyscout4 | phase | down | 3472 | 1.1% | 2.4% | 1.1% | 27.2% -> 27.2% | 61.5% -> 61.6% |
| S. Ward | wyscout4 | phase | all | 7238 | 9.8% | 9.9% | 9.8% | 27.0% -> 27.4% | 61.7% -> 61.7% |
| H. Maguire | wyscout4 | phase | up | 599 | 25.7% | 14.9% | 23.2% | 28.9% -> 41.4% | 67.8% -> 67.8% |
| H. Maguire | wyscout4 | phase | down | 2441 | 6.2% | 7.5% | 5.4% | 28.8% -> 29.1% | 62.0% -> 61.9% |
| H. Maguire | wyscout4 | phase | all | 13548 | 10.5% | 10.6% | 10.5% | 27.6% -> 28.4% | 61.0% -> 61.0% |
| B. Mee | wyscout4 | phase | up | 1539 | 12.2% | 7.7% | 12.0% | 30.9% -> 30.9% | 61.9% -> 62.1% |
| B. Mee | wyscout4 | phase | down | 2715 | 4.7% | 7.3% | 4.6% | 27.4% -> 27.8% | 62.4% -> 62.4% |
| B. Mee | wyscout4 | phase | all | 11003 | 9.9% | 10.0% | 9.9% | 26.2% -> 26.5% | 59.4% -> 59.8% |
| L. Dunk | wyscout4 | phase | up | 1057 | 25.9% | 16.9% | 25.7% | 37.9% -> 37.9% | 68.1% -> 68.1% |
| L. Dunk | wyscout4 | phase | down | 2354 | 8.8% | 11.3% | 8.3% | 27.4% -> 27.4% | 60.9% -> 61.2% |
| L. Dunk | wyscout4 | phase | all | 13710 | 10.0% | 10.2% | 10.2% | 29.7% -> 29.6% | 61.9% -> 62.1% |
| Son Heung-Min | wyscout4 | phase | down | 322 | 22.4% | 28.3% | 21.5% | 22.0% -> 18.3% | 62.4% -> 62.4% |
| Son Heung-Min | wyscout4 | phase | all | 13768 | 8.5% | 8.5% | 8.5% | 26.4% -> 26.9% | 60.6% -> 60.6% |
| C. Schindler | wyscout4 | phase | up | 1572 | 18.1% | 12.0% | 18.8% | 30.7% -> 30.5% | 63.0% -> 63.4% |
| C. Schindler | wyscout4 | phase | down | 4328 | 6.4% | 8.9% | 6.1% | 25.0% -> 26.6% | 61.2% -> 61.7% |
| C. Schindler | wyscout4 | phase | all | 12174 | 9.0% | 9.1% | 9.1% | 26.9% -> 27.5% | 60.6% -> 61.0% |
| Azpilicueta | wyscout4 | phase | up | 397 | 17.1% | 15.7% | 19.2% | 30.2% -> 30.2% | 64.0% -> 64.0% |
| Azpilicueta | wyscout4 | phase | down | 661 | 10.7% | 16.3% | 12.1% | 38.1% -> 38.1% | 77.9% -> 77.5% |
| Azpilicueta | wyscout4 | phase | all | 17178 | 11.9% | 12.2% | 12.2% | 30.0% -> 30.0% | 63.6% -> 63.7% |
| M. Jørgensen | wyscout4 | phase | up | 1270 | 27.6% | 19.3% | 25.6% | 26.5% -> 29.8% | 63.2% -> 63.2% |
| M. Jørgensen | wyscout4 | phase | down | 1738 | 10.1% | 14.4% | 10.9% | 25.4% -> 27.2% | 54.3% -> 54.3% |
| M. Jørgensen | wyscout4 | phase | all | 13108 | 9.4% | 9.5% | 9.5% | 26.5% -> 27.1% | 60.1% -> 60.3% |
| A. Christensen | wyscout4 | phase | up | 966 | 17.8% | 12.5% | 18.8% | 31.6% -> 31.6% | 62.9% -> 64.0% |
| A. Christensen | wyscout4 | phase | down | 2438 | 7.6% | 11.2% | 7.0% | 33.5% -> 33.5% | 64.8% -> 64.7% |
| A. Christensen | wyscout4 | phase | all | 10304 | 9.2% | 9.8% | 9.6% | 29.5% -> 29.7% | 63.3% -> 63.5% |
| A. Mawson | wyscout4 | phase | up | 1599 | 14.8% | 10.1% | 15.1% | 31.6% -> 31.7% | 67.9% -> 68.2% |
| A. Mawson | wyscout4 | phase | down | 6217 | 3.2% | 4.5% | 2.8% | 28.0% -> 28.1% | 62.6% -> 62.7% |
| A. Mawson | wyscout4 | phase | all | 14220 | 10.0% | 10.1% | 10.1% | 27.3% -> 27.2% | 63.3% -> 63.2% |
| L. Dunk | wyscout4 | possession_start | up | 389 | 27.0% | 16.5% | 23.7% | 37.3% -> 37.3% | 63.8% -> 63.8% |
| L. Dunk | wyscout4 | possession_start | down | 893 | 15.6% | 20.8% | 15.4% | 24.4% -> 24.4% | 62.7% -> 62.7% |
| L. Dunk | wyscout4 | possession_start | all | 13710 | 10.0% | 10.2% | 10.1% | 29.3% -> 29.3% | 62.2% -> 62.2% |
| M. Özil | wyscout4 | prev_event | up | 47 | 2.1% | 5.3% | 7.2% | 14.9% -> 14.9% | 51.1% -> 51.1% |
| M. Özil | wyscout4 | prev_event | down | 1141 | 3.7% | 8.2% | 4.6% | 24.8% -> 25.0% | 54.4% -> 57.8% |
| M. Özil | wyscout4 | prev_event | all | 15120 | 12.4% | 12.3% | 12.5% | 24.1% -> 23.4% | 58.4% -> 58.6% |
| A. Lacazette | wyscout4 | prev_event | up | 749 | 2.3% | 1.7% | 2.4% | 21.1% -> 21.4% | 61.3% -> 61.5% |
| A. Lacazette | wyscout4 | prev_event | down | 910 | 4.0% | 3.5% | 2.4% | 30.1% -> 30.1% | 62.5% -> 62.5% |
| A. Lacazette | wyscout4 | prev_event | all | 15467 | 5.7% | 5.8% | 5.6% | 24.8% -> 24.9% | 60.4% -> 60.4% |
| X. Shaqiri | wyscout4 | prev_event | up | 328 | 4.6% | 7.6% | 10.5% | 29.0% -> 29.0% | 58.8% -> 56.7% |
| X. Shaqiri | wyscout4 | prev_event | down | 1486 | 4.5% | 7.3% | 3.9% | 25.8% -> 26.2% | 58.7% -> 59.6% |
| X. Shaqiri | wyscout4 | prev_event | all | 10670 | 11.0% | 10.9% | 11.0% | 26.1% -> 26.5% | 58.4% -> 58.4% |
| D. Sánchez | wyscout4 | prev_event | up | 751 | 23.6% | 15.0% | 19.3% | 32.1% -> 33.7% | 63.6% -> 64.2% |
| D. Sánchez | wyscout4 | prev_event | down | 4573 | 2.9% | 3.9% | 2.6% | 29.2% -> 29.8% | 62.1% -> 62.2% |
| D. Sánchez | wyscout4 | prev_event | all | 13371 | 9.9% | 9.9% | 9.9% | 26.1% -> 26.4% | 59.9% -> 60.0% |
| M. Yoshida | wyscout4 | steps_since_change | up | 534 | 24.0% | 16.2% | 22.1% | 25.7% -> 28.1% | 70.2% -> 70.8% |
| M. Yoshida | wyscout4 | steps_since_change | down | 348 | 4.0% | 5.3% | 3.6% | 27.3% -> 27.3% | 57.8% -> 57.8% |
| M. Yoshida | wyscout4 | steps_since_change | all | 10329 | 9.8% | 10.0% | 10.0% | 25.1% -> 25.3% | 58.9% -> 58.9% |
| K. Naughton | wyscout4 | steps_since_change | up | 2123 | 16.6% | 12.9% | 18.4% | 23.2% -> 26.2% | 57.7% -> 57.7% |
| K. Naughton | wyscout4 | steps_since_change | down | 5232 | 3.0% | 5.1% | 2.8% | 28.9% -> 29.9% | 67.4% -> 67.5% |
| K. Naughton | wyscout4 | steps_since_change | all | 13115 | 11.9% | 11.8% | 11.8% | 27.5% -> 28.3% | 63.5% -> 63.5% |
| H. Maguire | wyscout4 | steps_since_change | up | 1839 | 12.1% | 8.6% | 12.8% | 28.3% -> 31.6% | 61.0% -> 61.8% |
| H. Maguire | wyscout4 | steps_since_change | down | 2779 | 8.5% | 10.8% | 7.5% | 27.8% -> 28.8% | 65.1% -> 64.8% |
| H. Maguire | wyscout4 | steps_since_change | all | 13548 | 10.5% | 10.6% | 10.5% | 27.6% -> 28.3% | 61.3% -> 61.5% |
| L. Dunk | wyscout4 | steps_since_change | up | 1852 | 23.1% | 15.1% | 22.5% | 36.3% -> 37.1% | 68.8% -> 69.6% |
| L. Dunk | wyscout4 | steps_since_change | down | 4489 | 8.2% | 10.4% | 7.5% | 28.9% -> 28.9% | 62.3% -> 62.7% |
| L. Dunk | wyscout4 | steps_since_change | all | 13710 | 10.0% | 10.2% | 10.2% | 29.9% -> 30.0% | 62.1% -> 62.5% |
| D. Alli | wyscout4 | steps_since_change | down | 390 | 1.0% | 4.3% | 3.1% | 18.5% -> 18.5% | 57.4% -> 57.4% |
| D. Alli | wyscout4 | steps_since_change | all | 16883 | 8.8% | 9.0% | 8.9% | 25.6% -> 25.6% | 59.8% -> 60.3% |
| A. Rüdiger | wyscout4 | steps_since_change | all | 11794 | 10.9% | 11.2% | 11.3% | 30.3% -> 30.2% | 63.5% -> 63.6% |
| C. Schindler | wyscout4 | steps_since_change | up | 2131 | 17.4% | 11.0% | 17.1% | 31.9% -> 32.0% | 64.2% -> 65.7% |
| C. Schindler | wyscout4 | steps_since_change | down | 4337 | 6.0% | 8.8% | 5.9% | 24.6% -> 26.9% | 61.4% -> 61.7% |
| C. Schindler | wyscout4 | steps_since_change | all | 12174 | 9.0% | 9.1% | 9.2% | 26.9% -> 27.8% | 60.7% -> 61.0% |
| A. Ogbonna | wyscout4 | steps_since_change | up | 1167 | 12.3% | 10.9% | 15.5% | 26.0% -> 25.0% | 65.1% -> 64.1% |
| A. Ogbonna | wyscout4 | steps_since_change | down | 1688 | 2.9% | 5.4% | 3.2% | 25.8% -> 25.8% | 61.2% -> 61.0% |
| A. Ogbonna | wyscout4 | steps_since_change | all | 10497 | 8.5% | 8.7% | 8.6% | 25.3% -> 25.3% | 60.6% -> 60.6% |
| Azpilicueta | wyscout4 | steps_since_change | up | 538 | 42.0% | 41.9% | 51.2% | 42.0% -> 42.0% | 68.2% -> 68.2% |
| Azpilicueta | wyscout4 | steps_since_change | down | 579 | 16.1% | 20.2% | 15.1% | 34.2% -> 34.2% | 73.6% -> 73.6% |
| Azpilicueta | wyscout4 | steps_since_change | all | 17178 | 11.9% | 12.2% | 12.2% | 30.3% -> 30.3% | 63.5% -> 63.5% |
| M. Jørgensen | wyscout4 | steps_since_change | up | 904 | 28.1% | 18.1% | 25.4% | 26.0% -> 33.1% | 65.6% -> 65.6% |
| M. Jørgensen | wyscout4 | steps_since_change | down | 943 | 11.9% | 17.4% | 13.3% | 21.1% -> 24.6% | 52.1% -> 53.7% |
| M. Jørgensen | wyscout4 | steps_since_change | all | 13108 | 9.4% | 9.5% | 9.6% | 26.6% -> 27.3% | 60.1% -> 60.3% |
| E. Dier | wyscout4 | steps_since_change | up | 1468 | 17.6% | 13.9% | 19.5% | 25.2% -> 23.6% | 62.6% -> 61.6% |
| E. Dier | wyscout4 | steps_since_change | down | 1179 | 6.0% | 6.7% | 4.7% | 29.7% -> 29.4% | 61.7% -> 61.7% |
| E. Dier | wyscout4 | steps_since_change | all | 18004 | 10.7% | 10.7% | 10.7% | 25.8% -> 25.9% | 59.9% -> 60.2% |
| A. Mawson | wyscout4 | steps_since_change | up | 509 | 35.8% | 23.2% | 30.7% | 45.0% -> 45.0% | 71.1% -> 71.1% |
| A. Mawson | wyscout4 | steps_since_change | down | 693 | 13.6% | 15.1% | 11.5% | 28.9% -> 29.0% | 64.1% -> 64.1% |
| A. Mawson | wyscout4 | steps_since_change | all | 14220 | 10.0% | 10.1% | 10.1% | 27.4% -> 27.3% | 63.2% -> 63.1% |
| D. Sánchez | wyscout4 | steps_since_change | up | 830 | 19.4% | 15.1% | 21.5% | 28.3% -> 31.4% | 64.9% -> 64.9% |
| D. Sánchez | wyscout4 | steps_since_change | down | 1841 | 8.2% | 11.7% | 8.4% | 24.4% -> 23.9% | 61.7% -> 62.7% |
| D. Sánchez | wyscout4 | steps_since_change | all | 13371 | 9.9% | 9.9% | 9.9% | 26.1% -> 26.3% | 59.7% -> 59.8% |

## Team clustering of players' z_cal

| variable | baseline | eta squared | p | highest team means |
|---|---|---|---|---|
| phase | wyscout4 | 0.077 | 0.3368 | Chelsea +1.1, Huddersfield Town +1.1, Tottenham Hotspur +0.9 |
| phase | fine | 0.063 | 0.5947 | Chelsea +0.7, Burnley +0.6, Manchester United +0.5 |
| phase | fine_backline | 0.077 | 0.3258 | Chelsea +0.6, Burnley +0.5, Manchester United +0.4 |
| steps_since_change | wyscout4 | 0.077 | 0.3038 | Tottenham Hotspur +1.4, Huddersfield Town +1.3, Swansea City +1.0 |
| steps_since_change | fine | 0.083 | 0.2374 | Chelsea +0.3, Manchester United +0.3, Everton +0.2 |
| steps_since_change | fine_backline | 0.074 | 0.3958 | Manchester United +0.5, Chelsea +0.5, Everton +0.4 |


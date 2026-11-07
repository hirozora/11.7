# Exploration round 2: all teams and regular players (England, training matches only)

Entities: 20 teams; 273 players with one team and >= 15 matches (player_what also needs >= 500 actions); 13 players acted for two teams and are excluded.
Gains are held-out log-likelihood per outcome, x1000 nats. interaction = entity-specific response to the variable
beyond the population response and the entity's constant zone style (two folds alternating by kickoff).
forward = the same gain when fitting on the first 133 training matches and scoring the last 133.
z_cal = parametric-null z divided by the empirical null scale s (from z < 0). Expected z_cal > 3 under the null: 0.13% of entities.

## team_what

| variable | entities | mean gain [entity-bootstrap 95% CI] | share gain > 0 | mean forward | share forward > 0 | s | z_cal > 3 | replicated |
|---|---|---|---|---|---|---|---|---|
| opp_last_event | 20 | +0.002 [-0.281, +0.264] | 0.50 | +0.171 | 0.70 | 2.49 | 0 | 0 |
| possession_start | 20 | +0.069 [-0.153, +0.314] | 0.65 | -0.045 | 0.65 | 3.58 | 0 | 0 |
| steps_since_change | 20 | +0.280 [+0.158, +0.414] | 0.85 | +0.160 | 0.65 | 0.39 | 7 | 7 |

Replicated entities (z_cal > 3 and forward > 0):

| entity | team | role | variable | n | gain [CI] | forward | z_cal |
|---|---|---|---|---|---|---|---|
| Liverpool | Liverpool | all | steps_since_change | 18683 | +0.67 [+0.30, +1.20] | +0.49 | 13.7 |
| Brighton & Hove Albion | Brighton & Hove Albion | all | steps_since_change | 13710 | +0.55 [-0.69, +1.75] | +0.57 | 8.0 |
| Arsenal | Arsenal | all | steps_since_change | 20220 | +0.70 [+0.20, +1.56] | +0.07 | 6.4 |
| Chelsea | Chelsea | all | steps_since_change | 17911 | +1.02 [+0.46, +1.49] | +0.41 | 4.9 |
| Tottenham Hotspur | Tottenham Hotspur | all | steps_since_change | 19026 | +0.38 [+0.21, +0.56] | +0.52 | 4.9 |
| Burnley | Burnley | all | steps_since_change | 11950 | +0.62 [+0.04, +1.24] | +0.31 | 3.5 |
| Swansea City | Swansea City | all | steps_since_change | 14220 | +0.22 [+0.03, +0.41] | +0.15 | 3.1 |

- Liverpool / steps_since_change: own third, right, 6+, through_ball: observed 0.014, additive 0.001, population 0.000 (n=148, z=+5.1)
- Liverpool / steps_since_change: final third, right, 0, cross: observed 0.118, additive 0.060, population 0.067 (n=229, z=+3.7)
- Liverpool / steps_since_change: own third, left, 0, long_pass: observed 0.156, additive 0.105, population 0.108 (n=481, z=+3.6)
- Brighton & Hove Albion / steps_since_change: middle third, left, 6+, long_pass: observed 0.137, additive 0.089, population 0.066 (n=534, z=+3.9)
- Brighton & Hove Albion / steps_since_change: middle third, left, 0, short_pass: observed 0.487, additive 0.399, population 0.455 (n=448, z=+3.8)
- Brighton & Hove Albion / steps_since_change: middle third, left, 6+, short_pass: observed 0.727, additive 0.790, population 0.814 (n=534, z=-3.6)

## player_what

| variable | entities | mean gain [entity-bootstrap 95% CI] | share gain > 0 | mean forward | share forward > 0 | s | z_cal > 3 | replicated |
|---|---|---|---|---|---|---|---|---|
| opp_last_event | 232 | -0.875 [-1.284, -0.515] | 0.30 | -0.274 | 0.39 | 5.95 | 0 | 0 |
| possession_start | 232 | -0.753 [-1.027, -0.506] | 0.34 | -0.932 | 0.39 | 2.27 | 0 | 0 |
| steps_since_change | 232 | -0.878 [-1.292, -0.513] | 0.39 | -0.654 | 0.39 | 2.90 | 1 | 1 |

Replicated entities (z_cal > 3 and forward > 0):

| entity | team | role | variable | n | gain [CI] | forward | z_cal |
|---|---|---|---|---|---|---|---|
| J. Evans | West Bromwich Albion | DF | steps_since_change | 893 | +7.36 [-0.60, +15.05] | +0.34 | 3.6 |

- J. Evans / steps_since_change: own third, left, 0, short_pass: observed 0.482, additive 0.315, population 0.306 (n=110, z=+3.8)
- J. Evans / steps_since_change: own third, left, 2, touch: observed 0.156, additive 0.052, population 0.063 (n=32, z=+2.7)
- J. Evans / steps_since_change: own third, left, 0, touch: observed 0.045, additive 0.129, population 0.165 (n=110, z=-2.6)

Team clustering of players' z_cal (eta squared = share of variance between teams; permutation p):

| variable | eta squared | p | teams with highest mean z_cal |
|---|---|---|---|
| opp_last_event | 0.073 | 0.6677 | Manchester United +0.1, West Ham United +0.1, Brighton & Hove Albion +0.1 |
| possession_start | 0.095 | 0.2624 | Newcastle United +0.2, Leicester City +0.2, Huddersfield Town +0.2 |
| steps_since_change | 0.118 | 0.0825 | West Bromwich Albion +0.3, Brighton & Hove Albion +0.3, Southampton +0.2 |

## player_who

| variable | entities | mean gain [entity-bootstrap 95% CI] | share gain > 0 | mean forward | share forward > 0 | s | z_cal > 3 | replicated |
|---|---|---|---|---|---|---|---|---|
| opp_last_event | 273 | -0.055 [-0.118, +0.006] | 0.51 | -0.134 | 0.57 | 2.96 | 1 | 1 |
| possession_start | 273 | -0.120 [-0.207, -0.033] | 0.48 | -0.184 | 0.59 | 4.28 | 0 | 0 |
| steps_since_change | 273 | +0.308 [+0.167, +0.446] | 0.66 | +0.283 | 0.70 | 4.22 | 13 | 13 |

Replicated entities (z_cal > 3 and forward > 0):

| entity | team | role | variable | n | gain [CI] | forward | z_cal |
|---|---|---|---|---|---|---|---|
| L. Dunk | Brighton & Hove Albion | DF | steps_since_change | 13710 | +4.54 [+2.57, +6.40] | +5.19 | 9.3 |
| D. Sánchez | Tottenham Hotspur | DF | steps_since_change | 13407 | +3.29 [+1.80, +5.02] | +1.46 | 7.3 |
| S. Malone | Huddersfield Town | DF | steps_since_change | 5172 | +5.30 [+3.51, +6.94] | +6.49 | 7.2 |
| J. Hogg | Huddersfield Town | MD | steps_since_change | 7211 | +5.45 [+2.81, +7.91] | +5.70 | 6.3 |
| M. Jørgensen | Huddersfield Town | DF | steps_since_change | 13108 | +3.04 [+1.74, +4.37] | +1.53 | 5.5 |
| E. Dier | Tottenham Hotspur | MD | steps_since_change | 18060 | +2.41 [+1.08, +3.69] | +3.02 | 5.5 |
| Marcos Alonso | Chelsea | DF | steps_since_change | 16490 | +4.21 [+2.80, +5.63] | +4.94 | 5.5 |
| A. Mawson | Swansea City | DF | steps_since_change | 14220 | +4.32 [+2.81, +6.04] | +2.11 | 5.3 |
| H. Maguire | Leicester City | DF | steps_since_change | 13548 | +3.42 [+2.31, +4.62] | +3.54 | 5.0 |
| C. Schindler | Huddersfield Town | DF | steps_since_change | 12174 | +5.69 [+3.81, +7.62] | +3.98 | 4.9 |
| Marcos Alonso | Chelsea | DF | opp_last_event | 16490 | +1.81 [+0.89, +2.67] | +0.50 | 4.0 |
| D. Alli | Tottenham Hotspur | MD | steps_since_change | 16883 | +2.11 [+1.35, +2.92] | +1.24 | 3.5 |
| A. Ogbonna | West Ham United | DF | steps_since_change | 10497 | +3.31 [+2.08, +4.47] | +3.85 | 3.3 |
| Javi Manquillo | Newcastle United | DF | steps_since_change | 8486 | +1.63 [+0.67, +2.90] | +0.99 | 3.1 |

- L. Dunk / steps_since_change: own third, left, 6+, actor: observed 0.790, additive 0.413, population 0.196 (n=162, z=+9.7)
- L. Dunk / steps_since_change: middle third, left, 6+, actor: observed 0.266, additive 0.179, population 0.122 (n=534, z=+5.2)
- L. Dunk / steps_since_change: own third, left, 0, actor: observed 0.263, additive 0.360, population 0.163 (n=558, z=-4.7)
- D. Sánchez / steps_since_change: final third, left, 0, actor: observed 0.026, additive 0.003, population 0.065 (n=154, z=+5.6)
- D. Sánchez / steps_since_change: own third, centre, 0, actor: observed 0.201, additive 0.294, population 0.149 (n=403, z=-4.1)
- D. Sánchez / steps_since_change: middle third, left, 0, actor: observed 0.042, additive 0.016, population 0.102 (n=334, z=+3.9)
- S. Malone / steps_since_change: own third, left, 6+, actor: observed 0.029, additive 0.341, population 0.198 (n=69, z=-5.5)
- S. Malone / steps_since_change: own third, centre, 0, actor: observed 0.095, additive 0.058, population 0.149 (n=263, z=+2.6)
- S. Malone / steps_since_change: own third, centre, 6+, actor: observed 0.000, additive 0.051, population 0.132 (n=121, z=-2.5)
- J. Hogg / steps_since_change: own third, centre, 6+, actor: observed 0.371, additive 0.128, population 0.040 (n=151, z=+8.9)
- J. Hogg / steps_since_change: final third, centre, 0, actor: observed 0.286, additive 0.103, population 0.144 (n=63, z=+4.8)
- J. Hogg / steps_since_change: final third, right, 0, actor: observed 0.093, additive 0.023, population 0.128 (n=86, z=+4.4)
- M. Jørgensen / steps_since_change: middle third, right, 6+, actor: observed 0.308, additive 0.206, population 0.124 (n=490, z=+5.6)
- M. Jørgensen / steps_since_change: own third, right, 6+, actor: observed 0.588, additive 0.383, population 0.199 (n=153, z=+5.2)
- M. Jørgensen / steps_since_change: own third, left, 0, actor: observed 0.050, additive 0.026, population 0.163 (n=518, z=+3.4)
- E. Dier / steps_since_change: own third, right, 6+, actor: observed 0.245, additive 0.078, population 0.034 (n=139, z=+7.3)
- E. Dier / steps_since_change: middle third, right, 6+, actor: observed 0.197, additive 0.142, population 0.108 (n=990, z=+4.9)
- E. Dier / steps_since_change: middle third, centre, 6+, actor: observed 0.249, additive 0.201, population 0.147 (n=812, z=+3.4)
- Marcos Alonso / steps_since_change: middle third, left, 6+, actor: observed 0.146, additive 0.236, population 0.122 (n=758, z=-5.8)
- Marcos Alonso / steps_since_change: own third, left, 6+, actor: observed 0.083, additive 0.275, population 0.199 (n=180, z=-5.8)
- Marcos Alonso / steps_since_change: own third, left, 3-5, actor: observed 0.132, additive 0.240, population 0.171 (n=318, z=-4.5)
- A. Mawson / steps_since_change: own third, left, 6+, actor: observed 0.700, additive 0.437, population 0.197 (n=170, z=+6.9)
- A. Mawson / steps_since_change: middle third, right, 0, actor: observed 0.025, additive 0.005, population 0.103 (n=441, z=+6.0)
- A. Mawson / steps_since_change: own third, left, 0, actor: observed 0.280, additive 0.380, population 0.163 (n=422, z=-4.3)
- H. Maguire / steps_since_change: own third, left, 6+, actor: observed 0.659, additive 0.362, population 0.196 (n=135, z=+7.2)
- H. Maguire / steps_since_change: middle third, right, 1, actor: observed 0.036, additive 0.013, population 0.072 (n=358, z=+3.8)
- H. Maguire / steps_since_change: own third, left, 1, actor: observed 0.167, additive 0.260, population 0.131 (n=264, z=-3.4)
- C. Schindler / steps_since_change: own third, left, 6+, actor: observed 0.632, additive 0.368, population 0.198 (n=182, z=+7.4)
- C. Schindler / steps_since_change: own third, right, 0, actor: observed 0.084, additive 0.041, population 0.157 (n=489, z=+4.8)
- C. Schindler / steps_since_change: middle third, left, 6+, actor: observed 0.215, additive 0.137, population 0.123 (n=433, z=+4.7)
- Marcos Alonso / opp_last_event: own third, right, set_piece, actor: observed 0.061, additive 0.003, population 0.166 (n=66, z=+7.9)
- Marcos Alonso / opp_last_event: middle third, centre, set_piece, actor: observed 0.050, additive 0.007, population 0.071 (n=60, z=+3.8)
- Marcos Alonso / opp_last_event: own third, left, foul, actor: observed 0.056, additive 0.284, population 0.205 (n=54, z=-3.7)
- D. Alli / steps_since_change: middle third, right, 1, actor: observed 0.110, additive 0.052, population 0.130 (n=347, z=+4.8)
- D. Alli / steps_since_change: middle third, right, 6+, actor: observed 0.010, additive 0.043, population 0.108 (n=869, z=-4.7)
- D. Alli / steps_since_change: own third, centre, 0, actor: observed 0.045, additive 0.023, population 0.059 (n=509, z=+3.4)

Team clustering of players' z_cal (eta squared = share of variance between teams; permutation p):

| variable | eta squared | p | teams with highest mean z_cal |
|---|---|---|---|
| opp_last_event | 0.052 | 0.7906 | Huddersfield Town +0.4, Brighton & Hove Albion +0.4, Tottenham Hotspur +0.3 |
| possession_start | 0.102 | 0.0770 | Tottenham Hotspur +0.4, Stoke City +0.2, Swansea City +0.2 |
| steps_since_change | 0.116 | 0.0340 | Huddersfield Town +1.8, Tottenham Hotspur +1.4, Brighton & Hove Albion +1.2 |

## Round-1 entities in round 2

| analysis | entity | opp_last_event | possession_start | steps_since_change |
|---|---|---|---|---|
| player_what | H. Kane | -0.2 (-0.07) | -0.5 (-0.18) | +0.4 (-1.34) |
| player_what | B. Mee | +0.1 (+0.02) | +0.5 (-13.11) | +0.1 (+0.04) |
| player_what | N. Kanté | +0.0 (-0.00) | -0.2 (+0.38) | +0.1 (+0.07) |
| player_what | K. De Bruyne | +0.1 (-0.22) | -0.1 (-0.12) | +0.0 (-0.08) |
| player_what | Mohamed Salah | +0.0 (+0.30) | +0.1 (+0.28) | +0.1 (+0.22) |
| player_who | H. Kane | +1.0 (-0.34) | +0.4 (-0.27) | +0.3 (+0.28) |
| player_who | B. Mee | +0.5 (+0.05) | +1.5 (+1.15) | +2.8 (-0.85) |
| player_who | N. Kanté | -0.3 (+0.05) | +0.1 (-0.25) | +0.2 (-0.18) |
| player_who | K. De Bruyne | +0.0 (+0.29) | +0.2 (+0.15) | +1.9 (+0.31) |
| player_who | Mohamed Salah | +0.2 (+0.08) | +0.1 (+0.02) | +0.5 (+0.22) |
| team_what | Manchester United | -1.0 (-0.04) | +0.2 (-0.45) | +0.3 (-0.05) |
| team_what | Liverpool | +2.0 (+0.87) | +0.2 (+0.07) | +13.7 (+0.49) |
| team_what | Manchester City | +0.3 (+0.08) | +0.8 (+0.37) | +2.4 (+0.31) |
| team_what | Burnley | -0.1 (+0.25) | +0.2 (+0.12) | +3.5 (+0.31) |

Cells: z_cal (forward gain).

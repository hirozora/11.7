# Exploration round 2: all teams and regular players (England, training matches only)

Entities: 20 teams; 273 players with one team and >= 15 matches (player_what also needs >= 500 actions); 13 players acted for two teams and are excluded.
Gains are held-out log-likelihood per outcome, x1000 nats. interaction = entity-specific response to the variable
beyond the population response and the entity's constant zone style (two folds alternating by kickoff).
forward = the same gain when fitting on the first 133 training matches and scoring the last 133.
z_cal = parametric-null z divided by the empirical null scale s (from z < 0). Expected z_cal > 3 under the null: 0.13% of entities.

## team_what

| variable | entities | mean gain [entity-bootstrap 95% CI] | share gain > 0 | mean forward | share forward > 0 | s | z_cal > 3 | replicated |
|---|---|---|---|---|---|---|---|---|
| phase | 20 | +0.167 [-0.102, +0.406] | 0.65 | +0.054 | 0.55 | 4.16 | 0 | 0 |
| prev_event | 20 | +0.018 [-0.254, +0.275] | 0.50 | +0.169 | 0.80 | 3.19 | 1 | 1 |
| time_bin | 20 | +0.020 [-0.154, +0.194] | 0.50 | -0.154 | 0.45 | 2.54 | 0 | 0 |
| score_state | 20 | +0.241 [-0.141, +0.579] | 0.80 | -0.070 | 0.45 | 3.13 | 1 | 0 |
| venue | 20 | -0.161 [-0.370, +0.050] | 0.30 | -1.780 | 0.15 | 2.57 | 0 | 0 |
| opponent_class | 20 | -0.129 [-0.504, +0.161] | 0.60 | -0.166 | 0.50 | 8.20 | 0 | 0 |

Replicated entities (z_cal > 3 and forward > 0):

| entity | team | role | variable | n | gain [CI] | forward | z_cal |
|---|---|---|---|---|---|---|---|
| Brighton & Hove Albion | Brighton & Hove Albion | all | prev_event | 13710 | +0.98 [+0.20, +1.85] | +0.82 | 3.2 |

- Brighton & Hove Albion / prev_event: middle third, left, opp_pass, short_pass: observed 0.482, additive 0.338, population 0.404 (n=274, z=+5.0)
- Brighton & Hove Albion / prev_event: final third, centre, opp_other_on_ball, long_pass: observed 0.061, additive 0.005, population 0.011 (n=33, z=+4.5)
- Brighton & Hove Albion / prev_event: middle third, left, own, head_pass: observed 0.051, additive 0.033, population 0.021 (n=1816, z=+4.3)

## player_what

| variable | entities | mean gain [entity-bootstrap 95% CI] | share gain > 0 | mean forward | share forward > 0 | s | z_cal > 3 | replicated |
|---|---|---|---|---|---|---|---|---|
| phase | 232 | -0.586 [-0.889, -0.287] | 0.30 | -0.438 | 0.36 | 4.47 | 0 | 0 |
| prev_event | 232 | -0.462 [-0.706, -0.237] | 0.25 | -0.320 | 0.31 | 8.81 | 0 | 0 |
| time_bin | 232 | -0.527 [-0.707, -0.366] | 0.21 | -0.492 | 0.25 | 2.21 | 0 | 0 |
| score_state | 232 | -0.784 [-1.155, -0.475] | 0.25 | -1.465 | 0.23 | 11.55 | 0 | 0 |
| venue | 232 | -2.358 [-4.073, -1.005] | 0.19 | -10.825 | 0.22 | 4.07 | 0 | 0 |
| opponent_class | 232 | -1.461 [-2.596, -0.596] | 0.16 | -3.898 | 0.19 | 6.17 | 0 | 0 |

Team clustering of players' z_cal (eta squared = share of variance between teams; permutation p):

| variable | eta squared | p | teams with highest mean z_cal |
|---|---|---|---|
| phase | 0.090 | 0.3088 | Brighton & Hove Albion +0.2, West Bromwich Albion +0.2, Leicester City +0.2 |
| prev_event | 0.069 | 0.8091 | Leicester City +0.1, Manchester City +0.1, Southampton +0.0 |
| time_bin | 0.117 | 0.0845 | Leicester City +0.2, Swansea City +0.1, West Bromwich Albion +0.1 |
| score_state | 0.082 | 0.4353 | AFC Bournemouth +0.2, Huddersfield Town +0.0, Watford +0.0 |
| venue | 0.067 | 0.7336 | Liverpool +0.1, Leicester City +0.0, Stoke City +0.0 |
| opponent_class | 0.058 | 0.9425 | Liverpool +0.2, Newcastle United +0.0, Southampton +0.0 |

## player_who

| variable | entities | mean gain [entity-bootstrap 95% CI] | share gain > 0 | mean forward | share forward > 0 | s | z_cal > 3 | replicated |
|---|---|---|---|---|---|---|---|---|
| phase | 273 | +0.330 [+0.169, +0.476] | 0.67 | +0.223 | 0.67 | 6.75 | 6 | 6 |
| prev_event | 273 | -0.073 [-0.187, +0.027] | 0.55 | -0.253 | 0.55 | 5.60 | 1 | 1 |
| time_bin | 273 | -0.276 [-0.363, -0.199] | 0.29 | -0.216 | 0.41 | 4.03 | 0 | 0 |
| score_state | 273 | -0.430 [-0.628, -0.264] | 0.42 | -0.897 | 0.44 | 8.77 | 1 | 1 |
| venue | 273 | -0.766 [-1.058, -0.519] | 0.25 | -0.823 | 0.36 | 16.62 | 0 | 0 |
| opponent_class | 273 | -0.931 [-1.446, -0.532] | 0.30 | -0.512 | 0.41 | 19.02 | 0 | 0 |

Replicated entities (z_cal > 3 and forward > 0):

| entity | team | role | variable | n | gain [CI] | forward | z_cal |
|---|---|---|---|---|---|---|---|
| Marcos Alonso | Chelsea | DF | phase | 16490 | +4.06 [+2.82, +5.33] | +4.26 | 7.3 |
| A. Ogbonna | West Ham United | DF | score_state | 10497 | +5.38 [+2.08, +8.69] | +0.20 | 5.6 |
| B. Mee | Burnley | DF | phase | 11003 | +3.59 [+2.12, +5.19] | +2.46 | 3.9 |
| D. Sánchez | Tottenham Hotspur | DF | prev_event | 13407 | +1.61 [+0.58, +2.66] | +2.61 | 3.4 |
| E. Dier | Tottenham Hotspur | MD | phase | 18060 | +2.57 [+1.36, +3.71] | +2.08 | 3.2 |
| A. Mawson | Swansea City | DF | phase | 14220 | +3.18 [+1.15, +5.42] | +1.19 | 3.1 |
| L. Dunk | Brighton & Hove Albion | DF | phase | 13710 | +3.45 [+2.34, +4.65] | +3.66 | 3.1 |
| D. Sánchez | Tottenham Hotspur | DF | phase | 13407 | +2.41 [+0.69, +4.29] | +1.38 | 3.0 |

- Marcos Alonso / phase: own third, left, settled, actor: observed 0.101, additive 0.265, population 0.190 (n=348, z=-7.0)
- Marcos Alonso / phase: middle third, left, early, actor: observed 0.228, additive 0.167, population 0.083 (n=807, z=+4.6)
- Marcos Alonso / phase: own third, left, first, actor: observed 0.309, additive 0.229, population 0.162 (n=433, z=+4.0)
- A. Ogbonna / score_state: own third, left, trailing, actor: observed 0.371, additive 0.229, population 0.164 (n=272, z=+5.6)
- A. Ogbonna / score_state: middle third, left, trailing, actor: observed 0.171, additive 0.109, population 0.106 (n=703, z=+5.2)
- A. Ogbonna / score_state: own third, left, leading, actor: observed 0.084, additive 0.214, population 0.152 (n=190, z=-4.4)
- B. Mee / phase: own third, left, settled, actor: observed 0.575, additive 0.365, population 0.189 (n=174, z=+5.8)
- B. Mee / phase: middle third, centre, first, actor: observed 0.145, additive 0.083, population 0.075 (n=317, z=+4.0)
- B. Mee / phase: own third, left, first, actor: observed 0.252, additive 0.323, population 0.162 (n=516, z=-3.4)
- D. Sánchez / prev_event: own third, right, opp_pass, actor: observed 0.205, additive 0.314, population 0.167 (n=215, z=-3.5)
- D. Sánchez / prev_event: own third, left, opp_pass, actor: observed 0.134, additive 0.077, population 0.170 (n=217, z=+3.2)
- D. Sánchez / prev_event: own third, centre, own, actor: observed 0.254, additive 0.213, population 0.101 (n=914, z=+3.0)
- E. Dier / phase: own third, right, settled, actor: observed 0.206, additive 0.095, population 0.041 (n=267, z=+6.2)
- E. Dier / phase: middle third, right, settled, actor: observed 0.178, additive 0.145, population 0.110 (n=1421, z=+3.6)
- E. Dier / phase: own third, centre, settled, actor: observed 0.191, additive 0.131, population 0.047 (n=388, z=+3.5)
- A. Mawson / phase: own third, left, settled, actor: observed 0.614, additive 0.425, population 0.189 (n=293, z=+6.6)
- A. Mawson / phase: middle third, right, first, actor: observed 0.025, additive 0.005, population 0.103 (n=441, z=+6.0)
- A. Mawson / phase: own third, left, first, actor: observed 0.280, additive 0.381, population 0.163 (n=422, z=-4.3)
- L. Dunk / phase: own third, left, settled, actor: observed 0.682, additive 0.400, population 0.188 (n=283, z=+9.7)
- L. Dunk / phase: own third, left, first, actor: observed 0.263, additive 0.359, population 0.163 (n=558, z=-4.7)
- L. Dunk / phase: middle third, left, settled, actor: observed 0.220, additive 0.171, population 0.117 (n=850, z=+3.8)
- D. Sánchez / phase: final third, left, first, actor: observed 0.026, additive 0.003, population 0.065 (n=154, z=+5.6)
- D. Sánchez / phase: own third, centre, first, actor: observed 0.201, additive 0.296, population 0.149 (n=403, z=-4.2)
- D. Sánchez / phase: middle third, left, first, actor: observed 0.042, additive 0.016, population 0.102 (n=334, z=+3.9)

Team clustering of players' z_cal (eta squared = share of variance between teams; permutation p):

| variable | eta squared | p | teams with highest mean z_cal |
|---|---|---|---|
| phase | 0.092 | 0.1424 | Tottenham Hotspur +0.9, Chelsea +0.9, Huddersfield Town +0.6 |
| prev_event | 0.099 | 0.0960 | Tottenham Hotspur +0.4, Huddersfield Town +0.4, Burnley +0.3 |
| time_bin | 0.087 | 0.1934 | Tottenham Hotspur +0.2, Burnley +0.0, Leicester City +0.0 |
| score_state | 0.055 | 0.7391 | West Ham United +0.3, Leicester City +0.2, AFC Bournemouth +0.0 |
| venue | 0.077 | 0.2874 | Leicester City +0.1, Huddersfield Town +0.0, Brighton & Hove Albion +0.0 |
| opponent_class | 0.107 | 0.0475 | Brighton & Hove Albion +0.1, Swansea City +0.1, West Bromwich Albion +0.0 |

## Round-1 entities in round 2

| analysis | entity | phase | prev_event | time_bin | score_state | venue | opponent_class |
|---|---|---|---|---|---|---|---|
| player_what | H. Kane | +0.1 (+0.10) | +0.0 (-0.28) | +0.1 (-0.66) | +0.0 (-0.54) | +0.2 (-0.35) | +0.0 (-0.04) |
| player_what | B. Mee | +0.1 (+0.04) | +0.1 (+0.17) | +0.4 (-0.05) | +0.0 (-0.17) | +0.1 (-0.05) | +0.0 (+1.05) |
| player_what | N. Kanté | -0.0 (-0.02) | -0.1 (-0.05) | +0.1 (-0.11) | +0.0 (+0.10) | +0.1 (+0.02) | -0.4 (-0.09) |
| player_what | K. De Bruyne | +0.0 (+0.02) | +0.1 (+0.05) | +0.1 (-1.08) | -0.0 (+0.01) | +0.1 (-0.20) | +0.0 (-0.10) |
| player_what | Mohamed Salah | +0.1 (+0.33) | +0.0 (-0.12) | +0.1 (-0.48) | +0.0 (+0.39) | +0.1 (+0.33) | +0.1 (+0.06) |
| player_who | H. Kane | +0.0 (-0.05) | -0.1 (-0.19) | +0.4 (-0.36) | +0.4 (-1.14) | +0.1 (+0.20) | -0.9 (-0.04) |
| player_who | B. Mee | +3.9 (+2.46) | +2.2 (+1.06) | -1.3 (+0.07) | +0.1 (-0.09) | -0.1 (-0.11) | +0.1 (+0.72) |
| player_who | N. Kanté | +0.0 (-0.33) | +0.0 (+0.32) | -1.7 (-0.31) | -0.1 (+0.08) | -0.0 (-0.66) | -0.0 (+0.09) |
| player_who | K. De Bruyne | +1.3 (+0.77) | +0.2 (-0.07) | -0.2 (-0.07) | +0.1 (-0.07) | +0.0 (+0.02) | -0.0 (-0.15) |
| player_who | Mohamed Salah | +0.2 (+0.05) | +0.2 (-0.06) | +0.3 (+0.17) | -0.6 (-0.11) | +0.1 (+0.05) | +0.0 (-0.78) |
| team_what | Manchester United | +0.0 (-1.04) | -0.1 (-0.32) | +0.0 (-0.22) | +0.1 (+0.00) | -0.1 (-0.68) | +0.4 (+0.59) |
| team_what | Liverpool | +0.2 (+0.33) | +0.6 (+0.23) | +0.9 (+0.30) | +0.9 (+0.31) | +0.2 (-0.61) | -2.4 (-1.65) |
| team_what | Manchester City | -0.1 (+0.39) | -1.2 (+0.81) | -0.5 (+0.69) | +0.2 (+0.27) | -0.1 (-0.13) | -0.1 (-0.13) |
| team_what | Burnley | +1.0 (+0.73) | +2.5 (+0.39) | +0.0 (-0.64) | +1.6 (+0.27) | -0.8 (-0.70) | +0.1 (+0.38) |

Cells: z_cal (forward gain).

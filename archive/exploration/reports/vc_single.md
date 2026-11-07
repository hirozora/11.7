# Round 4 (development, England training matches): do entities differ in their scene responses?

Variance-component score test (see src/vc_dev.py). Population offset by zone x level (and fine role for player
and who specs); entity style per zone; one entity parameter per level, shared across zones. Null: entity labels
permuted between match units (200 permutations, null model refitted each time). Teams: across
team-matches. Players: [role] within fine role (player and team differences), [team_role] within team x fine role
(player differences beyond the team). tau from (T - mean T_perm) / sum V; typical OR = exp(tau).

| spec | rows | entities | levels | T | permutation mean (sd) | p | tau | typical OR |
|---|---|---|---|---|---|---|---|---|
| team_what|long_pass|time_bin | 303599 | 20 | 3 | 60.3 | 48.1 (11.8) | 0.159 | 0.024 | 1.02 |
| team_what|long_pass|score_state | 303599 | 20 | 3 | 173.5 | 136.0 (32.2) | 0.114 | 0.042 | 1.04 |
| team_what|long_pass|venue | 303599 | 20 | 2 | 53.9 | 108.1 (33.7) | 0.975 | 0.000 | 1.00 |
| team_what|long_pass|opponent_class | 303599 | 20 | 2 | 108.5 | 98.0 (33.0) | 0.333 | 0.022 | 1.02 |
| team_what|long_pass|opp_last_event | 303599 | 20 | 11 | 341.5 | 202.2 (20.7) | 0.005 | 0.081 | 1.08 |
| team_what|long_pass|possession_start | 303599 | 20 | 9 | 326.6 | 178.2 (21.0) | 0.005 | 0.083 | 1.09 |
| team_what|cross|time_bin | 303599 | 20 | 3 | 62.6 | 46.4 (11.3) | 0.085 | 0.047 | 1.05 |
| team_what|cross|score_state | 303599 | 20 | 3 | 64.3 | 57.3 (13.2) | 0.303 | 0.031 | 1.03 |
| team_what|cross|venue | 303599 | 20 | 2 | 30.2 | 42.4 (13.8) | 0.806 | 0.000 | 1.00 |
| team_what|cross|opponent_class | 303599 | 20 | 2 | 24.9 | 35.3 (9.9) | 0.856 | 0.000 | 1.00 |
| team_what|cross|opp_last_event | 303599 | 20 | 11 | 171.8 | 159.6 (16.1) | 0.214 | 0.041 | 1.04 |
| team_what|cross|possession_start | 303599 | 20 | 9 | 146.5 | 129.2 (15.0) | 0.104 | 0.048 | 1.05 |
| team_what|back_pass|time_bin | 303599 | 20 | 3 | 137.0 | 59.3 (12.8) | 0.005 | 0.045 | 1.05 |
| team_what|back_pass|score_state | 303599 | 20 | 3 | 171.5 | 93.3 (20.0) | 0.010 | 0.045 | 1.05 |
| team_what|back_pass|venue | 303599 | 20 | 2 | 49.7 | 59.7 (17.6) | 0.692 | 0.000 | 1.00 |
| team_what|back_pass|opponent_class | 303599 | 20 | 2 | 66.9 | 54.1 (16.1) | 0.214 | 0.018 | 1.02 |
| team_what|back_pass|opp_last_event | 303599 | 20 | 11 | 324.3 | 205.1 (19.6) | 0.005 | 0.055 | 1.06 |
| team_what|back_pass|possession_start | 303599 | 20 | 9 | 237.9 | 164.1 (18.4) | 0.005 | 0.044 | 1.04 |
| team_what|forward|time_bin | 303599 | 20 | 3 | 50.4 | 43.9 (10.0) | 0.229 | 0.011 | 1.01 |
| team_what|forward|score_state | 303599 | 20 | 3 | 90.3 | 87.2 (19.0) | 0.423 | 0.008 | 1.01 |
| team_what|forward|venue | 303599 | 20 | 2 | 35.9 | 63.8 (19.1) | 0.945 | 0.000 | 1.00 |
| team_what|forward|opponent_class | 303599 | 20 | 2 | 91.6 | 51.8 (16.4) | 0.015 | 0.028 | 1.03 |
| team_what|forward|opp_last_event | 303599 | 20 | 11 | 341.6 | 202.7 (20.0) | 0.005 | 0.053 | 1.05 |
| team_what|forward|possession_start | 303599 | 20 | 9 | 271.7 | 163.1 (17.7) | 0.005 | 0.047 | 1.05 |
| player_what[team_role_era_opp]|long_pass|time_bin | 302045 | 477 | 3 | 990.0 | 965.6 (35.3) | 0.234 | 0.035 | 1.04 |
| player_what[team_role_era_opp]|long_pass|score_state | 302045 | 477 | 3 | 1207.4 | 1296.6 (43.4) | 0.985 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|long_pass|venue | 302045 | 477 | 2 | 622.6 | 710.2 (39.8) | 1.000 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|long_pass|opponent_class | 302045 | 477 | 2 | 693.4 | 733.4 (36.3) | 0.871 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|long_pass|opp_last_event | 302045 | 477 | 11 | 2267.5 | 2223.4 (49.9) | 0.209 | 0.049 | 1.05 |
| player_what[team_role_era_opp]|long_pass|possession_start | 302045 | 477 | 9 | 2334.8 | 2431.7 (50.8) | 0.965 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|cross|time_bin | 302045 | 477 | 3 | 623.3 | 684.1 (29.1) | 0.980 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|cross|score_state | 302045 | 477 | 3 | 535.3 | 621.4 (27.3) | 0.990 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|cross|venue | 302045 | 477 | 2 | 349.0 | 415.7 (25.4) | 1.000 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|cross|opponent_class | 302045 | 477 | 2 | 355.2 | 414.2 (23.1) | 1.000 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|cross|opp_last_event | 302045 | 477 | 11 | 880.5 | 849.1 (30.0) | 0.164 | 0.070 | 1.07 |
| player_what[team_role_era_opp]|cross|possession_start | 302045 | 477 | 9 | 1163.7 | 1354.5 (38.4) | 1.000 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|back_pass|time_bin | 302045 | 477 | 3 | 1112.4 | 1054.5 (37.4) | 0.050 | 0.039 | 1.04 |
| player_what[team_role_era_opp]|back_pass|score_state | 302045 | 477 | 3 | 1085.9 | 1119.8 (39.6) | 0.791 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|back_pass|venue | 302045 | 477 | 2 | 583.7 | 611.6 (35.0) | 0.796 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|back_pass|opponent_class | 302045 | 477 | 2 | 549.4 | 601.7 (27.9) | 0.965 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|back_pass|opp_last_event | 302045 | 477 | 11 | 2783.9 | 2900.1 (60.8) | 0.980 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|back_pass|possession_start | 302045 | 477 | 9 | 2737.0 | 2851.1 (58.5) | 0.960 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|forward|time_bin | 302045 | 477 | 3 | 1005.1 | 957.5 (34.2) | 0.090 | 0.032 | 1.03 |
| player_what[team_role_era_opp]|forward|score_state | 302045 | 477 | 3 | 1056.7 | 1120.3 (38.3) | 0.945 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|forward|venue | 302045 | 477 | 2 | 535.0 | 610.2 (33.2) | 0.980 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|forward|opponent_class | 302045 | 477 | 2 | 548.9 | 606.6 (29.7) | 0.975 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|forward|opp_last_event | 302045 | 477 | 11 | 2982.1 | 3066.7 (58.8) | 0.945 | 0.000 | 1.00 |
| player_what[team_role_era_opp]|forward|possession_start | 302045 | 477 | 9 | 2975.5 | 3065.1 (61.5) | 0.930 | 0.000 | 1.00 |
| team_who|actor|time_bin | 3311250 | 20 | 3 | 1403.6 | 1198.1 (56.8) | 0.005 | 0.029 | 1.03 |
| team_who|actor|score_state | 3311250 | 20 | 3 | 1774.4 | 1856.7 (101.5) | 0.791 | 0.000 | 1.00 |
| team_who|actor|venue | 3311250 | 20 | 2 | 875.7 | 1307.3 (101.9) | 1.000 | 0.000 | 1.00 |
| team_who|actor|opponent_class | 3311250 | 20 | 2 | 1042.3 | 1077.8 (85.5) | 0.637 | 0.000 | 1.00 |
| team_who|actor|opp_last_event | 3311250 | 20 | 11 | 4413.2 | 3664.4 (83.9) | 0.005 | 0.055 | 1.06 |
| team_who|actor|possession_start | 3311250 | 20 | 9 | 4248.2 | 3383.1 (80.7) | 0.005 | 0.059 | 1.06 |
| player_who[team_role_era_opp]|actor|time_bin | 3311250 | 477 | 3 | 1514.8 | 1501.7 (51.4) | 0.428 | 0.007 | 1.01 |
| player_who[team_role_era_opp]|actor|score_state | 3311250 | 477 | 3 | 2031.5 | 2094.8 (74.3) | 0.821 | 0.000 | 1.00 |
| player_who[team_role_era_opp]|actor|venue | 3311250 | 477 | 2 | 909.3 | 1281.5 (85.8) | 1.000 | 0.000 | 1.00 |
| player_who[team_role_era_opp]|actor|opponent_class | 3311250 | 477 | 2 | 912.5 | 1078.7 (57.7) | 1.000 | 0.000 | 1.00 |
| player_who[team_role_era_opp]|actor|opp_last_event | 3311250 | 477 | 11 | 5779.9 | 4696.5 (71.0) | 0.005 | 0.067 | 1.07 |
| player_who[team_role_era_opp]|actor|possession_start | 3311250 | 477 | 9 | 5570.5 | 4596.8 (92.9) | 0.005 | 0.063 | 1.07 |

## Detection floor (planted effects vary between matches: match sd = 1.0 x tau; 20 replicates per tau)

Power at alpha = 0.01 (0.05). Base: real outcomes with one fixed label permutation.

| spec | OR 1.11 | OR 1.22 | OR 1.35 | OR 1.50 | OR 1.65 | OR 2.01 | MDE at 80% power, alpha 0.01 |
|---|---|---|---|---|---|---|---|
| team_what|long_pass|time_bin | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| team_who|actor|time_bin | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| player_who[team_role_era_opp]|actor|time_bin | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| player_what[team_role_era_opp]|long_pass|time_bin | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| team_what|long_pass|score_state | 0.90 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| team_who|actor|score_state | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| player_who[team_role_era_opp]|actor|score_state | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| player_what[team_role_era_opp]|long_pass|score_state | 0.70 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.22 |
| team_what|long_pass|venue | 0.80 (0.85) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| team_who|actor|venue | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| player_who[team_role_era_opp]|actor|venue | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| player_what[team_role_era_opp]|long_pass|venue | 0.80 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| team_what|long_pass|opponent_class | 0.55 (0.65) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.22 |
| team_who|actor|opponent_class | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| player_who[team_role_era_opp]|actor|opponent_class | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| player_what[team_role_era_opp]|long_pass|opponent_class | 0.70 (0.95) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.22 |
| team_what|long_pass|opp_last_event | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| team_who|actor|opp_last_event | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| player_who[team_role_era_opp]|actor|opp_last_event | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| player_what[team_role_era_opp]|long_pass|opp_last_event | 0.60 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.22 |
| team_what|long_pass|possession_start | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| team_who|actor|possession_start | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| player_who[team_role_era_opp]|actor|possession_start | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| player_what[team_role_era_opp]|long_pass|possession_start | 0.60 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.22 |

## Population: per-player involvement rate by fine role and steps since possession change

Share of the team's open-play on-ball actions performed by one on-pitch player of the role.

| fine role | opportunities | overall | steps 0 | steps 1 | steps 2 | steps 3-5 | steps 6+ | log(6+ / 0) |
|---|---|---|---|---|---|---|---|---|
| backmid/centre | 243222 | 11.6% | 12.3% | 12.3% | 11.0% | 11.1% | 11.4% | -0.08 |
| backmid/halfR | 70114 | 11.3% | 11.9% | 11.2% | 10.2% | 10.7% | 11.8% | -0.01 |
| backmid/halfL | 64723 | 10.9% | 11.7% | 11.3% | 10.1% | 10.6% | 10.5% | -0.11 |
| mid/centre | 214957 | 10.6% | 9.5% | 11.7% | 11.2% | 10.6% | 10.6% | +0.10 |
| front/wideL | 51652 | 10.4% | 6.9% | 10.8% | 11.6% | 12.3% | 10.7% | +0.44 |
| mid/halfR | 145193 | 10.4% | 8.6% | 11.4% | 10.5% | 10.6% | 10.8% | +0.22 |
| mid/halfL | 141196 | 10.3% | 8.7% | 10.7% | 10.8% | 11.0% | 10.6% | +0.20 |
| backmid/wideR | 173529 | 9.9% | 11.4% | 7.7% | 10.0% | 9.7% | 10.0% | -0.13 |
| mid/wideR | 117249 | 9.8% | 10.0% | 8.5% | 10.5% | 10.1% | 9.8% | -0.02 |
| mid/wideL | 134432 | 9.8% | 9.4% | 8.3% | 10.6% | 10.6% | 10.0% | +0.06 |
| deep/halfL | 273956 | 9.8% | 13.2% | 7.8% | 7.8% | 8.8% | 10.0% | -0.28 |
| backmid/wideL | 155673 | 9.8% | 11.3% | 7.9% | 10.1% | 9.6% | 9.7% | -0.16 |
| deep/halfR | 269114 | 9.6% | 13.0% | 7.6% | 7.7% | 8.6% | 10.0% | -0.27 |
| deep/wideR | 41285 | 9.6% | 12.3% | 7.2% | 8.7% | 8.7% | 9.8% | -0.22 |
| deep/wideL | 35275 | 9.6% | 12.8% | 7.8% | 8.8% | 8.8% | 8.8% | -0.37 |
| front/wideR | 33644 | 9.4% | 6.6% | 9.8% | 10.2% | 11.0% | 9.6% | +0.36 |
| deep/centre | 107271 | 9.2% | 13.0% | 7.5% | 7.4% | 7.9% | 8.9% | -0.39 |
| front/halfR | 190452 | 9.1% | 6.0% | 10.1% | 10.4% | 10.3% | 9.5% | +0.46 |
| front/halfL | 208884 | 8.9% | 5.7% | 9.9% | 9.9% | 10.2% | 9.2% | +0.48 |
| front/centre | 337257 | 7.7% | 4.8% | 9.5% | 9.3% | 8.4% | 7.6% | +0.47 |
| GK | 302172 | 3.0% | 2.9% | 5.9% | 3.2% | 2.4% | 1.8% | -0.45 |

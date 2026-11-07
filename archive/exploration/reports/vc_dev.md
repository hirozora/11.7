# Round 4 (development, England training matches): do entities differ in their scene responses?

Variance-component score test (see src/vc_dev.py). Population offset by zone x level (and fine role for player
and who specs); entity style per zone; one entity parameter per level, shared across zones. Null: entity labels
permuted between match units (200 permutations, null model refitted each time). Teams: across
team-matches. Players: [role] within fine role (player and team differences), [team_role] within team x fine role
(player differences beyond the team). tau from (T - mean T_perm) / sum V; typical OR = exp(tau).

| spec | rows | entities | levels | T | permutation mean (sd) | p | tau | typical OR |
|---|---|---|---|---|---|---|---|---|
| team_what|long_pass|steps_since_change | 303599 | 20 | 5 | 260.3 | 96.9 (16.2) | 0.005 | 0.088 | 1.09 |
| team_what|long_pass|trailing_late | 303599 | 20 | 2 | 45.3 | 41.0 (12.6) | 0.348 | 0.014 | 1.01 |
| team_what|cross|steps_since_change | 303599 | 20 | 5 | 139.5 | 85.9 (14.4) | 0.010 | 0.085 | 1.09 |
| team_what|cross|trailing_late | 303599 | 20 | 2 | 41.5 | 25.4 (7.5) | 0.025 | 0.047 | 1.05 |
| team_what|back_pass|steps_since_change | 303599 | 20 | 5 | 187.7 | 87.2 (13.4) | 0.005 | 0.051 | 1.05 |
| team_what|forward|steps_since_change | 303599 | 20 | 5 | 155.8 | 90.5 (14.8) | 0.005 | 0.037 | 1.04 |
| team_what|forward|high_regain_first | 303599 | 20 | 2 | 23.0 | 20.5 (6.4) | 0.313 | 0.007 | 1.01 |
| player_what[role]|long_pass|steps_since_change | 302045 | 477 | 5 | 2034.8 | 1708.0 (53.5) | 0.005 | 0.131 | 1.14 |
| player_what[team_role]|long_pass|steps_since_change | 302045 | 477 | 5 | 2034.8 | 2006.9 (51.5) | 0.299 | 0.038 | 1.04 |
| player_what[team_role_era]|long_pass|steps_since_change | 302045 | 477 | 5 | 2034.8 | 2002.0 (47.1) | 0.254 | 0.041 | 1.04 |
| player_what[team_role_era_opp]|long_pass|steps_since_change | 302045 | 477 | 5 | 2034.8 | 1995.4 (45.6) | 0.214 | 0.045 | 1.05 |
| player_what[role]|forward|steps_since_change | 302045 | 477 | 5 | 2000.5 | 1809.4 (57.6) | 0.010 | 0.065 | 1.07 |
| player_what[team_role]|forward|steps_since_change | 302045 | 477 | 5 | 2000.5 | 1981.5 (50.9) | 0.373 | 0.020 | 1.02 |
| player_what[team_role_era]|forward|steps_since_change | 302045 | 477 | 5 | 2000.5 | 1985.3 (44.1) | 0.383 | 0.018 | 1.02 |
| player_what[team_role_era_opp]|forward|steps_since_change | 302045 | 477 | 5 | 2000.5 | 1986.7 (44.1) | 0.403 | 0.017 | 1.02 |
| player_what[role]|back_pass|steps_since_change | 302045 | 477 | 5 | 2020.7 | 1727.5 (56.5) | 0.005 | 0.088 | 1.09 |
| player_what[team_role]|back_pass|steps_since_change | 302045 | 477 | 5 | 2020.7 | 1873.9 (52.1) | 0.005 | 0.062 | 1.06 |
| player_what[team_role_era]|back_pass|steps_since_change | 302045 | 477 | 5 | 2020.7 | 1876.1 (49.3) | 0.005 | 0.062 | 1.06 |
| player_what[team_role_era_opp]|back_pass|steps_since_change | 302045 | 477 | 5 | 2020.7 | 1890.8 (48.8) | 0.005 | 0.059 | 1.06 |
| team_who|actor|steps_since_change | 3311250 | 20 | 5 | 2541.9 | 1848.5 (61.2) | 0.005 | 0.053 | 1.05 |
| team_who|actor|phase | 3311250 | 20 | 3 | 1646.1 | 1052.5 (48.6) | 0.005 | 0.049 | 1.05 |
| player_who[role]|actor|steps_since_change | 3311250 | 477 | 5 | 3900.7 | 2207.9 (67.2) | 0.005 | 0.084 | 1.09 |
| player_who[team_role]|actor|steps_since_change | 3311250 | 477 | 5 | 3900.7 | 2692.8 (62.4) | 0.005 | 0.071 | 1.07 |
| player_who[team_role_era]|actor|steps_since_change | 3311250 | 477 | 5 | 3900.7 | 2744.4 (62.3) | 0.005 | 0.069 | 1.07 |
| player_who[team_role_era_opp]|actor|steps_since_change | 3311250 | 477 | 5 | 3900.7 | 2783.0 (62.8) | 0.005 | 0.068 | 1.07 |
| player_who[role]|actor|phase | 3311250 | 477 | 3 | 2673.6 | 1284.1 (55.9) | 0.005 | 0.076 | 1.08 |
| player_who[team_role]|actor|phase | 3311250 | 477 | 3 | 2673.6 | 1665.3 (52.6) | 0.005 | 0.065 | 1.07 |
| player_who[team_role_era]|actor|phase | 3311250 | 477 | 3 | 2673.6 | 1699.8 (51.3) | 0.005 | 0.064 | 1.07 |
| player_who[team_role_era_opp]|actor|phase | 3311250 | 477 | 3 | 2673.6 | 1724.1 (51.5) | 0.005 | 0.063 | 1.06 |

## Detection floor (planted effects vary between matches: match sd = 1.0 x tau; 20 replicates per tau)

Power at alpha = 0.01 (0.05). Base: real outcomes with one fixed label permutation.

| spec | OR 1.11 | OR 1.22 | OR 1.35 | OR 1.50 | OR 1.65 | OR 2.01 | MDE at 80% power, alpha 0.01 |
|---|---|---|---|---|---|---|---|
| team_what|long_pass|steps_since_change | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| team_what|long_pass|trailing_late | 0.60 (0.75) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.22 |
| team_what|forward|high_regain_first | 0.40 (0.50) | 0.95 (0.95) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.22 |
| team_who|actor|steps_since_change | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| player_who[team_role]|actor|steps_since_change | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.11 |
| player_what[team_role]|long_pass|steps_since_change | 0.70 (0.80) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | OR 1.22 |

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

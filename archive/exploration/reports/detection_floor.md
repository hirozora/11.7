# Detection floor of the exploration pipeline (semi-synthetic, training matches only)

Known entity-specific effects are planted into the real outcomes of sampled entities (20 teams; 60 players per player
analysis), only inside the target rows: single = trailing; pair = trailing and after the 75th minute. The odds of the
target outcome (who: the player performs the action; what: long_pass) are multiplied by omega within each zone.
Detection uses the round-2 pipeline and its calibration (z_cal = z / s, s from the real runs' z < 0; detected = z_cal > 3).
The planted effect is identical in every match, so these floors are optimistic for effects that vary between matches.

## Calibration from the real (unplanted) runs

| analysis | variable | s (z < 0) | false positives at z_cal > 3 | central centre | central scale | false positives (central) |
|---|---|---|---|---|---|---|
| player_what | score_state | 2.46 | 0.000 | +0.28 | 0.26 | 0.017 |
| player_what | score_time | 2.17 | 0.000 | +0.22 | 0.41 | 0.017 |
| player_who | score_state | 5.48 | 0.000 | +0.34 | 1.14 | 0.017 |
| player_who | score_time | 3.72 | 0.000 | +0.07 | 1.15 | 0.017 |
| team_what | score_state | 2.27 | 0.100 | +0.74 | 1.25 | 0.100 |
| team_what | score_time | 1.28 | 0.000 | +0.88 | 1.33 | 0.000 |

## Power by planted odds ratio

Power = share of sampled entities detected (z_cal > 3); in brackets the share also replicated forward; then the
power under central-matching calibration.

| analysis | plant | detected with | OR 1.25 | OR 1.5 | OR 2.0 | OR 3.0 |
|---|---|---|---|---|---|---|
| team_what | single | score_state | 0.10 (0.05); 0.15 | 0.10 (0.10); 0.15 | 0.50 (0.45); 0.60 | 0.95 (0.95); 1.00 |
| team_what | pair | score_state | 0.10 (0.05); 0.10 | 0.05 (0.00); 0.20 | 0.10 (0.05); 0.10 | 0.05 (0.00); 0.10 |
| team_what | pair | score_time | 0.00 (0.00); 0.00 | 0.05 (0.00); 0.05 | 0.20 (0.10); 0.10 | 0.45 (0.40); 0.35 |
| player_what | single | score_state | 0.00 (0.00); 0.02 | 0.00 (0.00); 0.02 | 0.00 (0.00); 0.05 | 0.02 (0.02); 0.12 |
| player_what | pair | score_state | 0.00 (0.00); 0.02 | 0.00 (0.00); 0.00 | 0.00 (0.00); 0.02 | 0.00 (0.00); 0.02 |
| player_what | pair | score_time | 0.00 (0.00); 0.02 | 0.00 (0.00); 0.02 | 0.00 (0.00); 0.02 | 0.02 (0.02); 0.03 |
| player_who | single | score_state | 0.00 (0.00); 0.07 | 0.02 (0.00); 0.13 | 0.13 (0.10); 0.47 | 0.57 (0.57); 0.85 |
| player_who | pair | score_state | 0.00 (0.00); 0.03 | 0.00 (0.00); 0.02 | 0.00 (0.00); 0.03 | 0.02 (0.02); 0.12 |
| player_who | pair | score_time | 0.00 (0.00); 0.02 | 0.00 (0.00); 0.02 | 0.00 (0.00); 0.08 | 0.08 (0.08); 0.27 |

## Minimum detectable odds ratio and what it means in probabilities

Base rates are medians over the sampled entities: share of the entity's rows in the target scene, and the target
outcome's rate there. 'Planted rate at MDE' converts the 80%-power MDE into a probability.

| analysis | plant | detected with | MDE (80%) | MDE (50%) | MDE (80%, central) | scene share | base rate | planted rate at MDE (80%) |
|---|---|---|---|---|---|---|---|---|
| team_what | single | score_state | 3.0 | 2.0 | 3.0 | 28.3% | 8.7% | 22.3% |
| team_what | pair | score_state | > 3.0 | > 3.0 | > 3.0 | 7.5% | 9.1% | - |
| team_what | pair | score_time | > 3.0 | > 3.0 | > 3.0 | 7.5% | 9.1% | - |
| player_what | single | score_state | > 3.0 | > 3.0 | > 3.0 | 28.2% | 4.5% | - |
| player_what | pair | score_state | > 3.0 | > 3.0 | > 3.0 | 6.5% | 6.0% | - |
| player_what | pair | score_time | > 3.0 | > 3.0 | > 3.0 | 6.5% | 6.0% | - |
| player_who | single | score_state | > 3.0 | 3.0 | 3.0 | 29.7% | 9.4% | - |
| player_who | pair | score_state | > 3.0 | > 3.0 | > 3.0 | 6.5% | 10.0% | - |
| player_who | pair | score_time | > 3.0 | > 3.0 | > 3.0 | 6.5% | 10.0% | - |


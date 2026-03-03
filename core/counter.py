"""
Counter-Tactic Engine v2 — Measurement-grounded tactical suggestions

Hardening changes vs v1:
  1. Every rule fires ONLY when measurable pitch metrics cross calibrated thresholds.
     No rule can emit a generic suggestion without binding to a specific metric value.

  2. Every TacticalCounter now carries:
     - supporting_metrics: dict of the exact metric values that triggered the rule
     - risk_tradeoffs: {'reward': ..., 'risk': ..., 'condition': ...}
     - confidence is derived from metric strength, not guessed.

  3. Rules are grouped by metric family:
     (A) Pressing line rules   — triggered by pressing_height, defensive_line_x
     (B) Shape rules           — triggered by width, compactness, depth
     (C) Zone density rules    — triggered by overload_zones counts
     (D) Structural rules      — triggered by line_structure counts (lines detected)
     (E) Transition rules      — triggered by combined pressing + zone metrics

  4. Confidence formula is explicit per rule (documented inline).

  5. ML extension hook: replace _confidence() per rule with a learned scorer
     that takes (opponent_metrics, own_metrics, historical_outcomes).

Pitch coordinate convention (post direction-normalisation):
  x = 0 (own goal) → x = 105 (opponent goal)  [attacking = +x]
  y = 0 (left touchline) → y = 68 (right touchline)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

PITCH_W = 105.0
PITCH_H = 68.0


# ─────────────────────────────────────────────
# RULE THRESHOLDS (calibrated to real football)
# ─────────────────────────────────────────────

# Pressing height: normalised to attacking direction (+x = forward)
PRESS_HIGH_THR   = 60.0   # m from own goal — above this is a high press
PRESS_LOW_THR    = 40.0   # m — below this is a deep/low block
PRESS_MID_THR    = 50.0   # m — medium block

# Width thresholds
WIDTH_NARROW_THR = 36.0   # m — below this = narrow
WIDTH_WIDE_THR   = 55.0   # m — above this = wide

# Compactness: mean inter-player distance
COMPACT_THR  = 20.0       # m — below this = compact block
STRETCHED_THR = 28.0      # m — above this = stretched

# Zone overload
ZONE_OVERLOAD_THR = 3     # players in a zone to flag overload
ZONE_WEAK_THR     = 1     # players in a zone to flag weakness

# Line structure
DEF_LINE_THIN = 3         # ≤ 3 defenders in back line = thin

# Depth
DEPTH_SHORT_THR = 35.0    # m — team compressed into short depth
DEPTH_LONG_THR  = 60.0    # m — team stretched over long depth


class CounterTacticEngine:

    def __init__(self, config: Dict):
        self.config = config
        self._rules = [
            self._rule_exploit_high_press,
            self._rule_exploit_deep_block,
            self._rule_attack_wide_corridors,
            self._rule_exploit_narrow_shape,
            self._rule_break_compact_block,
            self._rule_overload_weak_zone,
            self._rule_press_trap_stretched,
            self._rule_midfield_overload,
            self._rule_counter_attack_depth,
            self._rule_exploit_thin_backline,
            self._rule_high_line_runners,
        ]

    def generate(
        self,
        opponent: 'FormationSnapshot',
        own: 'FormationSnapshot',
    ) -> List['TacticalCounter']:
        from core.pipeline import TacticalCounter

        counters: List[TacticalCounter] = []
        for rule in self._rules:
            try:
                c = rule(opponent, own)
                if c is not None:
                    counters.append(c)
            except Exception as e:
                logger.debug(f"Rule {rule.__name__} error: {e}")

        # Deduplicate by mechanism, keep highest confidence
        best: Dict[str, TacticalCounter] = {}
        for c in counters:
            if c.mechanism not in best or c.confidence > best[c.mechanism].confidence:
                best[c.mechanism] = c

        return sorted(best.values(), key=lambda c: -c.confidence)[:5]

    # ─────────────────────────────────────────
    # RULE IMPLEMENTATIONS
    # ─────────────────────────────────────────

    def _rule_exploit_high_press(self, opp, own) -> Optional[object]:
        """
        Trigger: opponent pressing_height > PRESS_HIGH_THR.
        Space behind = PITCH_W - opp.pressing_height.
        Confidence scales linearly with space behind defence.
        """
        from core.pipeline import TacticalCounter
        ph = opp.pressing_height
        if ph < PRESS_HIGH_THR:
            return None

        space_behind = PITCH_W - ph
        conf = float(np.clip((ph - PRESS_HIGH_THR) / (PITCH_W - PRESS_HIGH_THR), 0.1, 0.95))

        return TacticalCounter(
            title="Exploit Space Behind High Press",
            reasoning=(
                f"Opponent's mean defensive line at {ph:.1f}m (threshold {PRESS_HIGH_THR}m), "
                f"leaving {space_behind:.1f}m of unprotected space. Diagonal balls behind "
                f"the fullbacks and runs from deep midfielders bypass the press structure."
            ),
            target_zone="channels_behind_defensive_line",
            mechanism="long_ball_over_press",
            confidence=conf,
            supporting_metrics={
                'opp_pressing_height_m': round(ph, 2),
                'space_behind_defense_m': round(space_behind, 2),
                'threshold_m': PRESS_HIGH_THR,
                'opp_defensive_line_x': round(opp.defensive_line_x, 2),
            },
            risk_tradeoffs={
                'reward': f"{space_behind:.0f}m of space to exploit; 1v1 with GK if line broken",
                'risk': "Ball loss in own half with opponent already forward",
                'condition': "Requires striker with pace and midfielder to cover transition",
            }
        )

    def _rule_exploit_deep_block(self, opp, own) -> Optional[object]:
        from core.pipeline import TacticalCounter
        ph = opp.pressing_height
        if ph > PRESS_LOW_THR:
            return None

        conf = float(np.clip((PRESS_LOW_THR - ph) / PRESS_LOW_THR, 0.1, 0.85))

        return TacticalCounter(
            title="Break Down Deep Defensive Block",
            reasoning=(
                f"Opponent defensive block at {ph:.1f}m (threshold {PRESS_LOW_THR}m). "
                f"Patient possession play draws them out; fullbacks overlap to create "
                f"crossing angles; set pieces in final third maximise opportunities."
            ),
            target_zone="final_third_wide_and_set_pieces",
            mechanism="patient_buildup_draw_out",
            confidence=conf,
            supporting_metrics={
                'opp_pressing_height_m': round(ph, 2),
                'opp_compactness_m': round(opp.compactness, 2),
                'threshold_m': PRESS_LOW_THR,
            },
            risk_tradeoffs={
                'reward': "Draw them out → space in behind on turn-over",
                'risk': "Low scoring probability if block stays compact",
                'condition': "Effective only if own team has technical quality in tight spaces",
            }
        )

    def _rule_attack_wide_corridors(self, opp, own) -> Optional[object]:
        from core.pipeline import TacticalCounter
        w = opp.width
        if w > WIDTH_NARROW_THR:
            return None

        unused_width = PITCH_H - w
        conf = float(np.clip((WIDTH_NARROW_THR - w) / WIDTH_NARROW_THR, 0.15, 0.90))

        return TacticalCounter(
            title="Attack Wide — Corridors Exposed",
            reasoning=(
                f"Opponent occupies {w:.1f}m of {PITCH_H:.0f}m pitch width "
                f"({w/PITCH_H*100:.0f}%), leaving {unused_width:.1f}m of lateral space. "
                f"Wingers holding width and overlapping fullbacks stretch the shape; "
                f"early crosses before they can recover."
            ),
            target_zone="wide_channels_both_flanks",
            mechanism="width_exploitation_overloads",
            confidence=conf,
            supporting_metrics={
                'opp_width_m': round(w, 2),
                'pitch_width_m': PITCH_H,
                'unused_width_m': round(unused_width, 2),
                'width_utilisation_pct': round(w / PITCH_H * 100, 1),
                'threshold_m': WIDTH_NARROW_THR,
            },
            risk_tradeoffs={
                'reward': f"{unused_width:.0f}m of lateral space → crossing opportunities",
                'risk': "Narrow opponents may be compact centrally — crosses may be blocked",
                'condition': "Need tall target forwards or cutback options in the box",
            }
        )

    def _rule_exploit_narrow_shape(self, opp, own) -> Optional[object]:
        from core.pipeline import TacticalCounter
        w = opp.width
        # Different angle: narrow + compact → switch play to exhaust them
        if w > 45.0 or opp.compactness > COMPACT_THR:
            return None

        conf = float(np.clip((45.0 - w) / 45.0 * 0.75, 0.1, 0.80))

        return TacticalCounter(
            title="Switch Play to Exhaust Narrow Block",
            reasoning=(
                f"Opponent is narrow ({w:.1f}m) and compact ({opp.compactness:.1f}m avg spacing). "
                f"Rapid side-to-side switches force continuous lateral movement — "
                f"after 3–4 switches a gap will appear in transition."
            ),
            target_zone="opposite_flank_after_switch",
            mechanism="switch_play_exhaustion",
            confidence=conf,
            supporting_metrics={
                'opp_width_m': round(w, 2),
                'opp_compactness_m': round(opp.compactness, 2),
            },
            risk_tradeoffs={
                'reward': "Lateral gaps open after 3-4 switches; fullback in 1v1",
                'risk': "Slow switches give them time to recover",
                'condition': "Requires quick one-touch passing quality and wide players",
            }
        )

    def _rule_break_compact_block(self, opp, own) -> Optional[object]:
        from core.pipeline import TacticalCounter
        c = opp.compactness
        if c > COMPACT_THR:
            return None

        conf = float(np.clip((COMPACT_THR - c) / COMPACT_THR, 0.10, 0.85))

        return TacticalCounter(
            title="Combination Play Through Compact Block",
            reasoning=(
                f"Mean inter-player distance {c:.1f}m (compact threshold {COMPACT_THR}m). "
                f"Third-man runs, blind-side overlaps, and quick give-and-gos exploit the "
                f"brief gaps created when defenders step to press."
            ),
            target_zone="half_spaces_between_lines",
            mechanism="third_man_combination_play",
            confidence=conf,
            supporting_metrics={
                'opp_compactness_m': round(c, 2),
                'compact_threshold_m': COMPACT_THR,
                'opp_depth_m': round(opp.depth, 2),
            },
            risk_tradeoffs={
                'reward': "Brief windows open when defenders step — 1v1 in tight space",
                'risk': "High dispossession risk; compact block recovers fast",
                'condition': "Requires players comfortable under pressure in tight areas",
            }
        )

    def _rule_overload_weak_zone(self, opp, own) -> Optional[object]:
        from core.pipeline import TacticalCounter
        zones = opp.overload_zones
        if not zones:
            return None

        # Find weakest zone opponent occupies (or absent zones)
        all_zones = [f"{x}_{y}"
                     for x in ['defensive','middle','attacking']
                     for y in ['left','centre','right']]
        # Zones opponent is absent from or weakest in
        zone_counts = {z: zones.get(z, 0) for z in all_zones}

        # Only look at forward half zones
        fwd_zones = {z: v for z, v in zone_counts.items()
                     if 'middle' in z or 'attacking' in z}
        if not fwd_zones:
            return None

        weakest_zone = min(fwd_zones, key=fwd_zones.get)
        weakest_count = fwd_zones[weakest_zone]
        strongest_count = max(fwd_zones.values())

        advantage = strongest_count - weakest_count
        if advantage < 2:
            return None

        conf = float(np.clip(advantage / 4.0, 0.1, 0.90))

        # Find our player count in that zone
        own_count = own.overload_zones.get(weakest_zone, 0)

        return TacticalCounter(
            title=f"Numerical Overload — {weakest_zone.replace('_',' ').title()}",
            reasoning=(
                f"Opponent has only {weakest_count} player(s) in {weakest_zone.replace('_',' ')} "
                f"vs {strongest_count} in their strongest zone (+{advantage} differential). "
                f"We have {own_count} in that zone — route play through this corridor."
            ),
            target_zone=weakest_zone,
            mechanism="zone_overload_routing",
            confidence=conf,
            supporting_metrics={
                'target_zone': weakest_zone,
                'opp_players_in_zone': weakest_count,
                'own_players_in_zone': own_count,
                'opp_max_zone_count': strongest_count,
                'zone_advantage': advantage,
            },
            risk_tradeoffs={
                'reward': f"+{advantage} player advantage in {weakest_zone.replace('_',' ')}",
                'risk': "Opponent may shift block to compensate once pattern is recognised",
                'condition': "Must commit quickly before they recover shape",
            }
        )

    def _rule_press_trap_stretched(self, opp, own) -> Optional[object]:
        from core.pipeline import TacticalCounter
        d = opp.depth
        ph = opp.pressing_height

        # Stretched vertically + pressing high = trap opportunity
        if d < DEPTH_LONG_THR or ph < PRESS_HIGH_THR:
            return None

        gap_between_lines = d - (PITCH_W - ph)
        if gap_between_lines < 10:
            return None

        conf = float(np.clip((d - DEPTH_LONG_THR) / (PITCH_W - DEPTH_LONG_THR), 0.1, 0.85))

        return TacticalCounter(
            title="Press Trap — Exploit Vertical Stretch",
            reasoning=(
                f"Opponent stretched {d:.1f}m vertically with defensive line at {ph:.1f}m. "
                f"Co-ordinated press triggers on their build-up — win ball high and attack "
                f"exposed space between separated defensive and midfield lines "
                f"({gap_between_lines:.1f}m gap)."
            ),
            target_zone="space_between_opp_lines",
            mechanism="coordinated_press_trigger",
            confidence=conf,
            supporting_metrics={
                'opp_depth_m': round(d, 2),
                'opp_pressing_height_m': round(ph, 2),
                'inter_line_gap_m': round(gap_between_lines, 2),
                'threshold_depth_m': DEPTH_LONG_THR,
            },
            risk_tradeoffs={
                'reward': f"{gap_between_lines:.0f}m between opponent lines if press wins ball",
                'risk': "Failed press leaves own team out of position",
                'condition': "All 4 attackers must press in unison; requires rehearsal",
            }
        )

    def _rule_midfield_overload(self, opp, own) -> Optional[object]:
        from core.pipeline import TacticalCounter
        own_mid = sum(v for k, v in own.overload_zones.items() if 'middle' in k)
        opp_mid = sum(v for k, v in opp.overload_zones.items() if 'middle' in k)

        advantage = own_mid - opp_mid
        if advantage < 2:
            return None

        conf = float(np.clip(advantage * 0.18, 0.1, 0.88))

        return TacticalCounter(
            title="Dominate Through Midfield Superiority",
            reasoning=(
                f"Own team has {own_mid} midfield players vs opponent's {opp_mid} "
                f"(+{advantage} advantage). Control tempo via short combinations; "
                f"use +{advantage} to progress ball and pin opponents defensively."
            ),
            target_zone="central_midfield",
            mechanism="midfield_numerical_control",
            confidence=conf,
            supporting_metrics={
                'own_midfield_count': own_mid,
                'opp_midfield_count': opp_mid,
                'numerical_advantage': advantage,
            },
            risk_tradeoffs={
                'reward': f"+{advantage} in midfield → possession dominance and tempo control",
                'risk': "Overloading midfield can leave attackers isolated",
                'condition': "Fullbacks must provide width to stop opponent from narrowing",
            }
        )

    def _rule_counter_attack_depth(self, opp, own) -> Optional[object]:
        from core.pipeline import TacticalCounter
        opp_fwd = sum(v for k, v in opp.overload_zones.items() if 'attacking' in k)
        opp_def = sum(v for k, v in opp.overload_zones.items() if 'defensive' in k)

        if opp_fwd < 3 or opp_fwd <= opp_def + 1:
            return None

        exposed = opp_fwd - opp_def
        conf = float(np.clip(exposed * 0.15, 0.1, 0.82))
        transition_space = PITCH_W - opp.pressing_height

        return TacticalCounter(
            title="Absorb and Counter — Exploit Forward Commitment",
            reasoning=(
                f"Opponent commits {opp_fwd} players forward vs {opp_def} in defence "
                f"({exposed} net forward commitment). When possession is regained, "
                f"{transition_space:.0f}m of space opens behind their line. "
                f"3-touch counter with pace runners."
            ),
            target_zone="space_behind_committed_forwards",
            mechanism="defensive_absorption_fast_counter",
            confidence=conf,
            supporting_metrics={
                'opp_attacking_zone_count': opp_fwd,
                'opp_defensive_zone_count': opp_def,
                'net_forward_commitment': exposed,
                'transition_space_m': round(transition_space, 2),
            },
            risk_tradeoffs={
                'reward': f"{transition_space:.0f}m in transition; {exposed} fewer defenders",
                'risk': "Absorbing pressure requires defensive discipline; deep concession risk",
                'condition': "Needs fast striker(s) to run in behind; structured low block",
            }
        )

    def _rule_exploit_thin_backline(self, opp, own) -> Optional[object]:
        from core.pipeline import TacticalCounter
        lines = opp.line_structure
        if not lines:
            return None

        # Defensive line is first group (sorted by x, ascending = deepest first)
        def_line = lines[0] if lines else []
        n_def = len(def_line)
        if n_def > DEF_LINE_THIN:
            return None

        own_fwd = sum(v for k, v in own.overload_zones.items() if 'attacking' in k)
        conf = float(np.clip((DEF_LINE_THIN - n_def + 1) / DEF_LINE_THIN, 0.1, 0.85))

        return TacticalCounter(
            title=f"Target Thin Back Line — {n_def} Defenders",
            reasoning=(
                f"Opponent back line has only {n_def} players detected vs our {own_fwd} "
                f"in the attacking zone. Direct balls into channels and runner runs "
                f"beyond the line overload the defence numerically."
            ),
            target_zone="channels_beyond_defensive_line",
            mechanism="overload_thin_backline",
            confidence=conf,
            supporting_metrics={
                'opp_backline_player_count': n_def,
                'own_attacking_zone_count': own_fwd,
                'threshold': DEF_LINE_THIN,
                'numerical_advantage': own_fwd - n_def,
            },
            risk_tradeoffs={
                'reward': f"{own_fwd} vs {n_def} — numerical superiority in final third",
                'risk': "Offside trap may be set; requires disciplined timing of runs",
                'condition': "Need attackers who can time runs and hold offside line",
            }
        )

    def _rule_high_line_runners(self, opp, own) -> Optional[object]:
        from core.pipeline import TacticalCounter
        dl = opp.defensive_line_x    # deepest defender x (after normalisation)

        # High defensive line = defenders far from own goal
        if dl < 25.0:
            return None

        space = dl                    # space between own goal and defensive line
        conf = float(np.clip((dl - 25.0) / (PITCH_W/2 - 25.0), 0.15, 0.88))

        return TacticalCounter(
            title="Runners Behind High Defensive Line",
            reasoning=(
                f"Opponent's deepest defender at x={dl:.1f}m — a very high defensive line. "
                f"Timed runs from deep midfielders burst into {dl:.0f}m of space "
                f"behind the line. One precise through ball creates clean 1v1 with GK."
            ),
            target_zone="space_behind_high_defensive_line",
            mechanism="timed_runners_behind_line",
            confidence=conf,
            supporting_metrics={
                'opp_defensive_line_x_m': round(dl, 2),
                'space_behind_line_m': round(dl, 2),
                'threshold_m': 25.0,
            },
            risk_tradeoffs={
                'reward': f"{dl:.0f}m of space; one ball creates clear chance",
                'risk': "Offside if run is too early; needs precise through-ball timing",
                'condition': "Midfielder with ability to play a lofted or low through ball",
            }
        )

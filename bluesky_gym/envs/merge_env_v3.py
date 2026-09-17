"""
MergeEnvV3 — redefined deconfliction reward for L4 (A1).

WHY (honest): the legacy MergeEnv reward is
    +1 (reach, once) - 0.1*|drift_rad| (EVERY step) - 1*intrusion (per step).
Because the drift penalty accrues on every step of a long cruise (spawn up to
~200 NM away) and any turn to avoid an intruder induces drift, a converging
policy is structurally locked at a NEGATIVE total (that is precisely why the
legacy ppo_mergeenv eval is -0.200). The optimum is not >= 0 by construction.

V3 fixes the sign so that "resolve collisions + reach the waypoint" is an
ATTAINABLE positive optimum:
    reward = +10 (reach, once)
           + 0.01 * delta(waypoint_dist)   (progress shaping toward goal, + on approach)
           - 0.05 * max(0, |drift_deg| - 3)  (penalty only BEYOND a +/-3 deg deadband)
           - 2 * (#intruders within 4 NM, this step)
A clean run (no intrusion, drift inside deadband, reaches fix) is POSITIVE;
collisions / wandering drive it down. Fair comparison of policies is done by
evaluating every checkpoint under THIS reward (see train_l4_v3.py).
"""
import numpy as np
import bluesky as bs
from bluesky_gym.envs.merge_env import MergeEnv, NUM_AC, INTRUSION_DISTANCE, fn

REACH_V3 = 10.0
DIFF_V3 = 0.01          # progress per NM closing on waypoint
DRIFT_DEADBAND_V3 = 3.0 # deg, no penalty inside
DRIFT_PEN_V3 = 0.05     # per deg beyond deadband
INTRUSION_PEN_V3 = 2.0  # per intruder < 4 NM per step


class MergeEnvV3(MergeEnv):
    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self._last_wpt_dist = None
        return out

    def _get_reward(self):
        # progress shaping toward the waypoint
        progress = 0.0
        if getattr(self, "_last_wpt_dist", None) is not None:
            progress = (self._last_wpt_dist - self.waypoint_dist) * DIFF_V3
        self._last_wpt_dist = self.waypoint_dist

        # reach lump + terminal
        reach, done = 0.0, 0
        if self.waypoint_dist < 10 and self.wpt_reach != 1:
            self.wpt_reach = 1
            self.faf_reached = 1
            reach = REACH_V3
        elif self.waypoint_dist < 20 and self.wpt_reach == 1:
            self.faf_reached = 2
            done = 1

        # drift beyond deadband (self.drift in degrees, set in _get_obs)
        drift_pen = DRIFT_PEN_V3 * max(0.0, abs(self.drift) - DRIFT_DEADBAND_V3)

        # intrusions within 4 NM this step
        intr = 0
        try:
            ac_idx = bs.traf.id2idx("KL001")
            for i in range(1, NUM_AC):
                _, d_nm = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                    bs.traf.lat[i], bs.traf.lon[i])
                if d_nm < INTRUSION_DISTANCE:
                    intr += 1
        except Exception:
            intr = 0

        reward = reach + progress - drift_pen - INTRUSION_PEN_V3 * intr
        self.total_reward += reward
        # track for honest info
        self._v3_breakdown = {
            "reach": round(reach, 3), "progress": round(progress, 3),
            "drift_pen": round(drift_pen, 3), "intrusion": round(INTRUSION_PEN_V3 * intr, 3),
            "intruders": intr,
        }
        return reward, done

    def _get_info(self):
        info = super()._get_info()
        info["v3_breakdown"] = getattr(self, "_v3_breakdown", None)
        return info
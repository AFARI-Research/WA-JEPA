#!/usr/bin/env python
"""Correct the heading reference inside upstream ``traj2control``, then run the closed loop.

## Why this is the entry every run uses

The upstream defect burns score in proportion to how much lateral jitter the plan carries, so
it is not a constant offset that cancels between methods: a planner that wobbles is punished
harder than one that does not, and a ranking measured through the broken controller is partly
a ranking of jitter rather than of driving.

The published UniAD / VAD / LTF figures were produced with the defect in place, so a
comparison against them inherits it. State that when quoting one.

## The defect and the correction

``sim/utils/sim_utils.py``::

    rot = np.arctan2(b - prev_b, a - prev_a)     # a = lateral (right), b = longitudinal

The iLQR state is ``[x, y, heading, v, steer]`` with dynamics ``x' = x + v*cos(theta)*dt``
and ``y' = y + v*sin(theta)*dt``, while ``plan_traj_stats[:, :2]`` is filled as
(forward, right). The heading therefore has to be ``arctan2(d_right, d_forward)``. With the
arguments swapped, a straight trajectory is handed a target heading of +90 degrees, and a
nearly straight one is worse: lateral values jittering around zero swing the target heading
between about -88 and +88 degrees, so iLQR receives a large heading error that changes sign
every step and saturates the steering rate at its limit of +-0.4.

Only the argument order on that one line changes. The wrap-around handling is left as is.

## Relation to upstream PR #57

Upstream received a pull request fixing the same defect
(https://github.com/hyzhou404/HUGSIM/pull/57, "Fix heading computation to ensure consistent
coordinate order with poses"). It changes the *unpacking* order rather than the argument
order::

    PR #57:  for i, (b, a) in enumerate(plan_traj):
                 rot = np.arctan2(b - prev_b, a - prev_a)
    here:    for i, (a, b) in enumerate(plan_traj):
                 rot = np.arctan2(a - prev_a, b - prev_b)

Both compute ``arctan2(d_right, d_forward)``, but swapping the variable names also changes
which column ``prev_a, prev_b = a, b`` binds at the end of the loop, so equivalence cannot
be established by reading. Comparing both implementations elementwise on straight,
purely lateral, all-zero, jittering and long random trajectories shows they agree exactly,
so this patch delivers the same fix PR #57 would.

## Why monkey-patching instead of editing HUGSIM

Editing the simulator would put the fix outside this repository, because HUGSIM lives
outside it -- a checkout that has silently diverged from upstream is then indistinguishable
from one that has not. ``closed_loop.py`` uses
``from ... import traj2control``, which binds the name at import time, so replacing the
module attribute *before* importing ``closed_loop`` is enough. Upstream files stay
byte-identical.

Usage -- the arguments are those of ``closed_loop.py`` and are forwarded untouched::

    cd "${HUGSIM_ROOT}" && python <repo>/close_loop/run_fixed_controller.py \\
        --scenario_path ... --base_path ... --camera_path ... --kinematic_path ... \\
        --ad ltf --ad_cuda 0
"""

import os
import runpy
import sys

import numpy as np

HUGSIM = os.environ.get("HUGSIM_ROOT") or os.getcwd()
sys.path.insert(0, HUGSIM)

import sim.utils.sim_utils as su  # noqa: E402
from sim.ilqr.lqr import plan2control  # noqa: E402


def traj2control_fixed(plan_traj, info):
    """Line-for-line the upstream function, with the two arctan2 arguments in order."""
    plan_traj_stats = np.zeros((plan_traj.shape[0] + 1, 5))
    plan_traj_stats[1:, :2] = plan_traj[:, [1, 0]]      # (forward, right)
    prev_a, prev_b = 0.0, 0.0
    for i, (a, b) in enumerate(plan_traj):              # a = right, b = forward
        rot = np.arctan2(a - prev_a, b - prev_b)        # the only change
        rot = np.where(rot > np.pi / 2, rot - np.pi, rot)
        rot = np.where(rot < -np.pi / 2, rot + np.pi, rot)
        plan_traj_stats[i + 1, 2] = rot
        prev_a, prev_b = a, b
    curr_stat = np.array([0.0, 0.0, 0.0, info["ego_velo"], info["ego_steer"]])
    return plan2control(plan_traj_stats, curr_stat)


su.traj2control = traj2control_fixed
print("[fixed-controller] replaced sim.utils.sim_utils.traj2control (this process only)",
      flush=True)

# closed_loop.py finds hugsim_env through `sys.path.append(os.getcwd())`, so it has to run
# from the simulator directory.
if os.path.realpath(os.getcwd()) != os.path.realpath(HUGSIM):
    print(f"!! run this from {HUGSIM}; closed_loop.py depends on the working directory",
          file=sys.stderr)
    raise SystemExit(2)

runpy.run_path(os.path.join(HUGSIM, "closed_loop.py"), run_name="__main__")

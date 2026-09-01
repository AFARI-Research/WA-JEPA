#!/usr/bin/env python
"""Two upstream defects that abort a scenario before it can be scored, patched in this
process only, then the closed loop is run.

Both are version-skew bugs in HUGSIM, not properties of any driving stack: one crashes in
``gymnasium.make()`` before the first frame is rendered, the other crashes mid-rollout when
a background vehicle reaches the end of its route. Without them, 16 of the 436 official
scenarios cannot produce an ``eval.json`` no matter what the planner does, and the
benchmark is scored over a denominator that quietly differs from the published one.

Each patch is a **strict no-op** wherever the unpatched code does not crash -- not "close
enough", but the same value at every index. The arguments for that are below, per defect.
Neither patch touches ``traj2control``: the controller question is orthogonal and is handled
by ``run_fixed_controller.py``, which this entry can delegate to.

## Defect 1 -- ``ConstantPlanner`` is handed a ``max_t`` it never implemented

Fifteen PandaSet ``medium_02`` scenarios declare::

    plan_list: [..., ConstantPlanner, {max_t: 25}]

``sim/utils/plan.py:55`` instantiates it as
``getattr(agent_controller, model.controller)(**model.controller_args)``, but
``sim/utils/agent_controller.py`` defines ``ConstantPlanner.__init__(self)`` with no
parameters, so the scenario dies with ``TypeError: ConstantPlanner.__init__() got an
unexpected keyword argument 'max_t'``. ``max_t`` appears in those fifteen scenario files
and nowhere in the simulator: the data package is newer than the code, and there is no
upstream revision implementing it to align with.

The patch accepts the argument and ignores it, which makes those fifteen scenarios use the
same argument-less constant-velocity background traffic as the other 421.

Why ignoring is safe rather than a guess: every plausible reading of ``max_t`` ("stop after
max_t", "despawn after max_t", "only plan up to max_t") agrees with "drive on forever" for
all ``t < max_t`` and can only diverge after it. The simulator step is 0.25 s
(``configs/sim/kinematic.yaml``), so ``max_t: 25`` is step 100. A rollout that ends before
step 100 is therefore byte-identical under every reading, and the ignoring is invisible.
That premise is a fact about the run, not an assumption, so it has to be checked after the
fact rather than asserted here -- ``close_loop/check_maxt_reached.py`` reads ``n_steps``
from every affected ``planner_stats.json`` and reports any scenario that reached step 100.
Such a scenario is not wrong, but its score does depend on the unknown semantics and must
be reported separately instead of being folded into the mean.

The patch replaces ``__init__`` on the existing class object instead of substituting a
subclass or wrapper, because ``sim/utils/plan.py:111`` dispatches on
``type(controller) is ConstantPlanner``. A subclass fails that identity test and falls
through to ``raise NotImplementedError``, which would turn a crash at setup into a crash
mid-rollout.

## Defect 2 -- IDM lead distance when one waypoint is left

``sim/utils/agent_controller.py``, inside ``IDM.update``::

    lookahead_path = path[nearest_index + 1:][:self.lookahead_path_length]      # n rows
    ...
    lookahead_lengths = torch.cumsum(torch.norm(
        lookahead_path[1:, :2] - lookahead_path[:-1, :2], dim=1), dim=0)        # n - 1
    lookahead_lengths = torch.cat([lookahead_lengths, lookahead_lengths[-1:]])  # back to n
    lead_distance = lookahead_lengths[indices_lead]

The ``cat`` exists to restore the row dropped by the pairwise difference, and it does that
for every ``n >= 2``. At ``n == 1`` the difference is empty, so ``lookahead_lengths`` is
empty, ``lookahead_lengths[-1:]`` is empty as well, and the concatenation stays empty while
``indices_lead`` still holds the index 0::

    File "sim/utils/plan.py", line 104, in plan_traj
    File "sim/utils/agent_controller.py", line 170, in update
    IndexError: index is out of bounds for dimension with size 0

``n == 1`` means one waypoint remains ahead of that background vehicle, and the crash needs
a neighbour inside ``lead_distance_threshold`` at the same moment. It is not periodic and
not the ego vehicle's own controller: which scenario hits it depends on where the ego
pushes the traffic, so it appears and disappears between planners. Because it aborts the
whole rollout, the score it destroys is the ego's.

The patch appends a copy of the final waypoint to ``path`` in exactly that state, which
makes the arc length ahead ``[0.0, 0.0]`` and the lead distance 0: the lead vehicle is on
the only waypoint left. That is the same filling the ``cat`` performs for longer paths.

Why it is a no-op everywhere else:

* It fires only when the lookahead slice has exactly one row. Any other length leaves
  ``path`` untouched, so those calls are the upstream calls.
* ``nearest_index = argmin(dists)`` does not move. The appended row duplicates the last
  one, ``argmin`` returns the first minimum, and the state where the last row *is* the
  minimum yields a lookahead slice of length 0, which this patch does not touch.
* ``target`` / ``is_end`` do not change. The scan ``range(nearest_index + 1, len(path))``
  gains one index whose coordinates equal the previous last row's, so it either breaks at
  the same earlier index or falls through to the same ``path[-1]`` value.
* The duplicated waypoint duplicates one column of ``is_lead``, so a lead vehicle can be
  counted twice. ``acceleration.min()`` closes that: the duplicate contributes an
  identical value, and the minimum of a set is unchanged by repeating an element.

A pad the slice cannot see is refused rather than applied silently: with
``lookahead_path_length == 1`` the slice stays one row long however long ``path`` becomes,
so the patch reports that instead of pretending to have fixed it. Upstream's default is
100 and no scenario file overrides it, so this is a guard, not a code path in use.

## Composing with the controller heading fix

Patched attributes live on different modules -- ``ConstantPlanner.__init__`` and
``IDM.update`` here, ``sim.utils.sim_utils.traj2control`` in ``run_fixed_controller.py`` --
and all of them are installed before ``closed_loop`` is imported, so the order is
irrelevant. This entry runs ``HUGSIM_COMPAT_INNER`` if it is set (normally the heading-fix
entry, which patches the controller and then runs ``closed_loop.py``) and ``closed_loop.py``
otherwise. The patch code is not copied between entries: each defect is fixed in one place
only.

Usage -- the arguments are those of ``closed_loop.py`` and are forwarded untouched::

    cd "${HUGSIM_ROOT}" && python <repo>/close_loop/run_upstream_compat.py \\
        --scenario_path ... --base_path ... --camera_path ... --kinematic_path ... \\
        --ad ltf --ad_cuda 0

``scripts/evaluation/run_hugsim_benchmark.sh`` selects this entry by default and wires
``HUGSIM_COMPAT_INNER`` to ``run_fixed_controller.py``. ``HUGSIM_COMPAT=0`` drops these two
patches, which is what reproduces a run made before they existed.
"""

import os
import runpy
import sys

import torch

HUGSIM = os.environ.get("HUGSIM_ROOT") or os.getcwd()
sys.path.insert(0, HUGSIM)

import sim.utils.agent_controller as ac  # noqa: E402

# ---------------------------------------------------------------- defect 1

_upstream_constant_init = ac.ConstantPlanner.__init__


def _constant_init_ignoring_extras(self, **kwargs):
    _upstream_constant_init(self)
    if kwargs:
        # Say it out loud. Swallowing the argument silently would produce a scenario that
        # runs and scores while nobody knows it is not the scenario the data package
        # describes.
        print(f"[upstream-compat] ConstantPlanner ignoring arguments upstream never "
              f"implemented: {kwargs}", flush=True)


ac.ConstantPlanner.__init__ = _constant_init_ignoring_extras

# ---------------------------------------------------------------- defect 2

_upstream_idm_update = ac.IDM.update


def _idm_update_padding_route_end(self, state, path, dt, neighbors):
    """Duplicate the final waypoint when the lookahead slice would hold exactly one row."""
    if path is not None and len(path) >= 2:
        nearest_index = int(torch.argmin(torch.norm(path[:, :2] - state[None, :2], dim=1)))
        lookahead_rows = min(len(path) - nearest_index - 1, self.lookahead_path_length)
        if lookahead_rows == 1:
            if self.lookahead_path_length >= 2:
                path = torch.cat([path, path[-1:]], dim=0)
            else:
                print("[upstream-compat] lookahead_path_length=1 keeps the slice one row "
                      "long whatever the path length; not padding, IDM.update will raise "
                      "as upstream does", flush=True)
    return _upstream_idm_update(self, state, path, dt, neighbors)


ac.IDM.update = _idm_update_padding_route_end

print("[upstream-compat] patched ConstantPlanner.__init__ (ignores max_t) and IDM.update "
      "(pads the route-end lookahead); traj2control untouched, this process only",
      flush=True)

# closed_loop.py finds hugsim_env through `sys.path.append(os.getcwd())`, so it has to run
# from the simulator directory -- and so does the inner entry, which ends up running it.
if os.path.realpath(os.getcwd()) != os.path.realpath(HUGSIM):
    print(f"!! run this from {HUGSIM}; closed_loop.py depends on the working directory",
          file=sys.stderr)
    raise SystemExit(2)

inner = os.environ.get("HUGSIM_COMPAT_INNER") or os.path.join(HUGSIM, "closed_loop.py")
print(f"[upstream-compat] delegating to {inner}", flush=True)
runpy.run_path(inner, run_name="__main__")

#!/usr/bin/env bash
# Run exactly one HUGSIM scenario. Invoked concurrently by run_hugsim_benchmark.sh through
# `xargs -P`.
#
# It is a separate script partly because xargs needs an entry point, and partly because it
# makes "one process per scenario" structurally impossible to violate: HUGSIM's
# sim/utils/plan.py keys per-agent state as `agent_<index>` and never clears it, so a second
# scenario in the same process fails with `KeyError: 'agent_N'` where N happens to be the
# previous scenario's agent count.
#
# Preconditions, all exported by run_hugsim_benchmark.sh (not re-derived here):
#   HUGSIM_ROOT SIM_PYTHON BASE_YAML CAMERA_YAML KINEMATIC_YAML OUT_ROOT LOG_DIR
#   optional: HUGSIM_NGPU (>1 enables slot claiming), GPU_LOCK_DIR, HUGSIM_PER_GPU,
#             SKIP_DONE, HUGSIM_ENTRY
#
# Usage: bash run_hugsim_one_scenario.sh <absolute path to scenario.yaml>
set -u

scen="${1:?a scenario yaml path is required}"
name="$(basename "${scen}" .yaml)"

# ---------------- claim one GPU slot atomically ----------------
# Not `hash(scene) % NGPU`: scenario runtimes differ by more than 10x (an early collision
# ends in ~80 s, a full 400-step rollout takes ~19 min), so a static assignment leaves GPUs
# idle at the end. Claiming slots on demand is work-pool semantics for free.
#
# mkdir rather than a lock file: mkdir is atomic on POSIX (it either creates or returns
# EEXIST), needs no flock, and does not care what filesystem the directory lives on.
NGPU="${HUGSIM_NGPU:-1}"
GPU=0
slot=""
if [ "${NGPU}" -gt 1 ]; then
  : "${GPU_LOCK_DIR:?GPU_LOCK_DIR is required when HUGSIM_NGPU > 1}"
  PER_GPU="${HUGSIM_PER_GPU:-1}"
  claimed=0
  while [ "${claimed}" -eq 0 ]; do
    g=0
    while [ "${g}" -lt "${NGPU}" ]; do
      k=1
      while [ "${k}" -le "${PER_GPU}" ]; do
        if mkdir "${GPU_LOCK_DIR}/g${g}_s${k}" 2>/dev/null; then
          GPU="${g}"
          slot="${GPU_LOCK_DIR}/g${g}_s${k}"
          claimed=1
          break
        fi
        k=$((k + 1))
      done
      [ "${claimed}" -eq 1 ] && break
      g=$((g + 1))
    done
    [ "${claimed}" -eq 0 ] && sleep 2
  done
  # Return the slot whether the run finishes normally or is killed, otherwise the effective
  # concurrency decays monotonically over a long run.
  trap 'rmdir "${slot}" 2>/dev/null' EXIT INT TERM
fi

# closed_loop.py composes the output directory as <output_dir><ad>/<scene_name>_<mode>. Both
# fields come from the scenario yaml; do not guess them from the file name, the numbering
# suffix does not always agree.
#
# The quotes have to be stripped. kitti360 and waymo scene names look like
# '0000_10320_10520' and the quotes in the yaml are mandatory -- YAML 1.1 reads `_` as a
# digit separator, so an unquoted value parses as an integer. But OmegaConf hands
# closed_loop.py the unquoted string while grep/sed here would carry the quote characters
# through, producing a directory name containing literal quotes. SKIP_DONE would then never
# recognise a finished scenario and the success check would never find eval.json. nuscenes
# names are unquoted, so this only ever breaks on kitti360 and waymo.
yaml_get() {
  grep "^$2:" "$1" | head -1 | sed -E "s/^$2:[[:space:]]*//; s/[[:space:]]*(#.*)?$//; s/^['\"](.*)['\"]$/\1/"
}
scene="$(yaml_get "${scen}" scene_name)"
mode="$(yaml_get "${scen}" mode)"
if [ -z "${scene}" ] || [ -z "${mode}" ]; then
  echo "[FAIL] ${name}: cannot read scene_name/mode from ${scen}"
  exit 1
fi
outdir="${OUT_ROOT}/${scene}_${mode}"
log="${LOG_DIR}/${name}.log"

# Resume support: a full split takes hours and being interrupted is normal. "Finished"
# requires both eval.json and planner_stats.json -- eval.json alone means scoring completed
# but the driving stack never wrote its statistics (killed process), and such a result is
# not trustworthy.
if [ "${SKIP_DONE:-1}" = "1" ] && [ -f "${outdir}/eval.json" ] && [ -f "${outdir}/planner_stats.json" ]; then
  echo "[skip] ${name} already finished"
  exit 0
fi

# Clear first. closed_loop.py only calls makedirs(exist_ok=True):
#   * a stale eval.json/output.txt would be read as this round's result
#   * stale obs_pipe/plan_pipe desynchronise the new FIFO handshake
rm -rf "${outdir}"

start="$(date +%s)"
# cd into the simulator: closed_loop.py relies on `sys.path.append(os.getcwd())` to import
# hugsim_env. Output is redirected, never piped -- `cmd | tail` reports tail's exit code and
# hides the real failure.
#
# Both places need the same GPU index, and missing either sends half the load to GPU 0:
#   * CUDA_VISIBLE_DEVICES controls closed_loop.py itself (3DGS rendering, hardcoded "cuda")
#   * --ad_cuda controls the driving stack: launch_ad.py passes it as $1 to the AD script,
#     which exports CUDA_VISIBLE_DEVICES from it, overriding whatever it inherited
# CUDA_VISIBLE_DEVICES is interpreted against the physical device list and a parent's
# restriction does not remap it, so passing the same absolute index to both is correct.
#
# HUGSIM_ENTRY defaults to upstream closed_loop.py, which is the only setting comparable to
# the published baselines. The controller ablation points it at close_loop/run_fixed_controller.py,
# which patches the heading bug and then runs closed_loop.py, so the command line is identical.
entry="${HUGSIM_ENTRY:-closed_loop.py}"
(
  cd "${HUGSIM_ROOT}" && CUDA_VISIBLE_DEVICES="${GPU}" "${SIM_PYTHON}" -u "${entry}" \
    --scenario_path "${scen}" \
    --base_path "${BASE_YAML}" \
    --camera_path "${CAMERA_YAML}" \
    --kinematic_path "${KINEMATIC_YAML}" \
    --ad ltf \
    --ad_cuda "${GPU}"
) > "${log}" 2>&1
rc=$?
elapsed=$(( $(date +%s) - start ))

# Success is judged by eval.json actually being on disk, not by the exit code:
# closed_loop.py catches the exception around create_gym_env and still returns 0.
if [ -f "${outdir}/eval.json" ]; then
  hd="$(grep -o '"hdscore":[^,}]*' "${outdir}/eval.json" | head -1)"
  term="$(grep -hoE 'Collision with (background|foreground)|^Complete|Far from preset trajectory' "${log}" | head -1)"
  echo "[ok  ] ${name}  ${elapsed}s  gpu${GPU}  ${hd}  termination=${term:-unknown}"
else
  echo "[FAIL] ${name}  ${elapsed}s  gpu${GPU}  rc=${rc}  see ${log} and ${outdir}/output.txt"
fi

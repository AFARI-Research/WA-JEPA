import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from close_loop.hugsim_planner import (
    DEFAULT_CAMERA_MAP,
    convert_absolute_to_relative_se2_array,
)
from close_loop.selfcheck import bare_planner, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "wa_jepa_hugsim.yaml"


class HugsimConfigTest(unittest.TestCase):
    """The HUGSIM preset must stay the same architecture as the reported NAVSIM preset.

    The adapter builds the model from this config, so config and checkpoint are always
    self-consistent and ``strict=True`` cannot detect "a different architecture". Only a
    comparison against the reported preset can.
    """

    def test_architecture_matches_the_reported_navsim_preset(self):
        hugsim = OmegaConf.load(CONFIG)
        epdms = OmegaConf.load(ROOT / "configs" / "wa_jepa_navsim_epdms.yaml")

        self.assertEqual(hugsim.model.target, epdms.model.target)
        for key in (
            "camera_names",
            "input_hw",
            "num_history_frames",
            "num_future_frames",
            "ego_status_dim",
            "trajectory_horizon",
            "flow_num_layers",
        ):
            self.assertEqual(
                OmegaConf.to_object(hugsim.model[key])
                if OmegaConf.is_config(hugsim.model[key])
                else hugsim.model[key],
                OmegaConf.to_object(epdms.model[key])
                if OmegaConf.is_config(epdms.model[key])
                else epdms.model[key],
                msg=f"model.{key} differs from the reported preset",
            )

    def test_pretrained_encoder_loading_is_disabled(self):
        cfg = OmegaConf.load(CONFIG)
        # Closed-loop inference restores a trained state dict; loading the V-JEPA
        # pretrained weights first would only waste minutes per scenario, times 436.
        self.assertFalse(cfg.model.require_pretrained)
        self.assertIn(cfg.model.get("vjepa2_ckpt"), (None, ""))

    def test_preset_has_no_site_specific_paths(self):
        text = CONFIG.read_text(encoding="utf-8")
        self.assertNotIn("/mnt/", text)
        self.assertNotIn("/data/", text)


class HugsimAdapterTest(unittest.TestCase):
    """Lock the six conversions between the simulator and the model.

    Each of them fails silently: the run still produces 436 scores, and a wrong camera
    slot or an inverted command is indistinguishable from a weaker model.
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(CONFIG)
        cls.planner = bare_planner(cls.cfg)

    def test_camera_slots_are_mapped_in_model_order(self):
        # The model indexes a camera embedding by position in model.camera_names, so the
        # order of hugsim_cams is part of the interface, not a detail.
        self.assertEqual(self.planner.model_cams, ["cam_l0", "cam_f0", "cam_r0", "cam_b0"])
        self.assertEqual(
            self.planner.hugsim_cams,
            ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK"],
        )
        self.assertEqual(
            [DEFAULT_CAMERA_MAP[c] for c in self.planner.model_cams],
            self.planner.hugsim_cams,
            msg="the config disagrees with the module default camera map",
        )

    def test_command_mapping_is_the_measured_permutation(self):
        # HUGSIM: 0 = right, 1 = left, 2 = straight. NAVSIM: 0 = left, 1 = straight,
        # 2 = right. Identity mapping would turn every right turn into a left turn.
        self.assertEqual(self.planner.command_map, [2, 0, 1])
        for raw, expected in ((0, 2), (1, 0), (2, 1)):
            st, nav, back = self.planner._ego_status(
                {"command": raw, "ego_velo": 4.0, "accelerate": -0.5}
            )
            self.assertEqual(back, raw)
            self.assertEqual(nav, expected)
            self.assertEqual(int(np.argmax(st[:4])), expected)
            self.assertEqual(float(st[4]), 4.0)   # vx
            self.assertEqual(float(st[5]), 0.0)   # vy, HUGSIM reports scalar speed only
            self.assertEqual(float(st[6]), -0.5)  # ax
            self.assertEqual(float(st[7]), 0.0)   # ay

    def test_out_of_range_command_raises(self):
        # It must raise so the client counts a failure, rather than steering somewhere.
        with self.assertRaises(RuntimeError):
            self.planner._ego_status({"command": 5, "ego_velo": 0.0, "accelerate": 0.0})

    def test_history_is_strided_by_two_simulator_steps(self):
        # HUGSIM steps at 0.25 s, the model's frames are 0.5 s apart. A stride of 1 shows
        # the model a world running at half speed.
        self.assertEqual(self.planner.history_stride, 2)
        self.assertEqual(self.planner.buf_len, 7)

        self.planner.buf = [None] * self.planner.buf_len
        self.assertEqual(self.planner._history_indices(), [0, 2, 4, 6])
        # During warm-up the oldest available frame repeats; the client counts those steps.
        self.planner.buf = [None]
        self.assertEqual(self.planner._history_indices(), [0, 0, 0, 0])
        self.planner.buf = [None] * 4
        self.assertEqual(self.planner._history_indices(), [0, 0, 1, 3])

    def test_ego_to_lidar_conversion(self):
        # closed_loop.py reads the plan as (x right, y forward) and applies
        # plan[:, [1, 0]] to get (forward, right) back.
        traj = np.array([[1.0, 0.5, 0.0], [2.0, 1.0, 0.0]])  # ego: x forward, y left
        lidar = np.stack([-traj[:, 1], traj[:, 0]], axis=1)
        np.testing.assert_allclose(lidar, [[-0.5, 1.0], [-1.0, 2.0]])
        # A left turn (ego y > 0) has to give lidar x < 0.
        self.assertLess(lidar[0, 0], 0.0)
        recovered = lidar[:, [1, 0]]
        np.testing.assert_allclose(recovered[:, 0], traj[:, 0])
        np.testing.assert_allclose(recovered[:, 1], -traj[:, 1])

    def test_world_to_ego_se2(self):
        origin = np.array([12.0, -3.0, 0.4])
        np.testing.assert_allclose(
            convert_absolute_to_relative_se2_array(origin, origin[None, :]), 0.0, atol=1e-12
        )
        # Facing +90 degrees, a point on the world +x axis is 5 m to the right, i.e. ego
        # y = -5. The opposite sign mirrors every trajectory.
        rel = convert_absolute_to_relative_se2_array(
            np.array([0.0, 0.0, np.pi / 2]), np.array([[5.0, 0.0, 0.0]])
        )
        np.testing.assert_allclose(rel[0], [0.0, -5.0, -np.pi / 2], atol=1e-9)
        # Headings must wrap: a raw difference of 350 degrees is really -10.
        rel = convert_absolute_to_relative_se2_array(
            np.array([0.0, 0.0, -np.pi + 0.05]), np.array([[0.0, 0.0, np.pi - 0.05]])
        )
        self.assertAlmostEqual(float(rel[0, 2]), -0.1, places=9)


class HugsimScriptWiringTest(unittest.TestCase):
    """The three shell layers have to agree, and nothing checks that at runtime.

    ``closed_loop.py`` appends ``--ad`` to ``base.output_dir``. If the generated base
    config and the directory the runner scans disagree, the run finishes with an empty
    results directory and exit code 0.
    """

    @classmethod
    def setUpClass(cls):
        d = ROOT / "scripts" / "evaluation"
        cls.bench = (d / "run_hugsim_benchmark.sh").read_text(encoding="utf-8")
        cls.one = (d / "run_hugsim_one_scenario.sh").read_text(encoding="utf-8")
        cls.ad = (d / "run_hugsim_ad.sh").read_text(encoding="utf-8")

    def test_output_prefix_and_results_root_agree_with_the_ad_slot(self):
        self.assertIn('OUT_PREFIX="${OUTPUT_DIR}/${DATASET}_"', self.bench)
        self.assertIn('OUT_ROOT="${OUT_PREFIX}ltf"', self.bench)
        self.assertIn("output_dir: ${OUT_PREFIX}", self.bench)
        self.assertIn("--ad ltf", self.one)

    def test_generated_base_config_points_at_the_ad_launcher(self):
        self.assertIn("ltf_path: ${REPO_ROOT}/scripts/evaluation/run_hugsim_ad.sh", self.bench)
        self.assertIn("realcar_path: ${HUGSIM_DATA_ROOT}/3DRealCar", self.bench)
        self.assertIn("model_base: ${HUGSIM_DATA_ROOT}/scenes/${DATASET}", self.bench)

    def test_ad_launcher_execs_the_client(self):
        # Without exec, launch_ad.py kills the shell and orphans python, which keeps
        # holding GPU memory until the whole run runs out.
        self.assertIn('exec "${WM_PYTHON}" -u "${REPO}/close_loop/hugsim_client.py"', self.ad)

    def test_checkpoint_has_no_default_anywhere(self):
        # A ${WM_CHECKPOINT:-<path>} fallback that survives three process layers would
        # silently score an unintended checkpoint while still producing every scenario.
        # Comments are excluded: the launcher explains this rule by quoting the pattern.
        code = [ln for ln in self.ad.splitlines() if not ln.lstrip().startswith("#")]
        self.assertIn(': "${WM_CHECKPOINT:?', self.ad)
        self.assertNotIn("WM_CHECKPOINT:-", "\n".join(code))
        self.assertNotIn("CHECKPOINT:-/", self.bench)

    def test_scripts_have_no_site_specific_paths(self):
        for name, text in (("benchmark", self.bench), ("one_scenario", self.one), ("ad", self.ad)):
            self.assertNotIn("/mnt/", text, msg=name)
            self.assertNotIn("/data/", text, msg=name)
        # The entry scripts take the simulator path from HUGSIM_ROOT, so a hardcoded
        # checkout would only surface on someone else's machine.
        for name in ("run_upstream_compat.py", "check_maxt_reached.py", "run_fixed_controller.py"):
            text = (ROOT / "close_loop" / name).read_text(encoding="utf-8")
            self.assertNotIn("/mnt/", text, msg=name)
            self.assertNotIn("/data/", text, msg=name)

    def test_compat_entry_wraps_the_heading_fix_entry(self):
        # The compatibility patches and the controller heading fix are independent concerns and
        # both are entry scripts. Wrapping rather than replacing keeps each patch in one
        # place; overwriting HUGSIM_ENTRY without carrying the old value into
        # HUGSIM_COMPAT_INNER would silently drop the heading fix from every run.
        self.assertIn('HUGSIM_ENTRY="${REPO_ROOT}/close_loop/run_fixed_controller.py"',
                      self.bench)
        self.assertIn('HUGSIM_COMPAT="${HUGSIM_COMPAT:-1}"', self.bench)
        self.assertIn('HUGSIM_COMPAT_INNER="${HUGSIM_ENTRY}"', self.bench)
        self.assertIn('HUGSIM_ENTRY="${REPO_ROOT}/close_loop/run_upstream_compat.py"', self.bench)
        self.assertIn('[[ -n "${HUGSIM_COMPAT_INNER}" ]] && export HUGSIM_COMPAT_INNER',
                      self.bench)
        self.assertIn('entry="${HUGSIM_ENTRY:-closed_loop.py}"', self.one)

    def test_compat_entry_patches_the_classes_in_place(self):
        # sim/utils/plan.py dispatches on `type(controller) is IDM` and
        # `type(controller) is ConstantPlanner`, so a subclass or wrapper class fails that
        # identity test and falls through to `raise NotImplementedError`: a crash at setup
        # would turn into a crash mid-rollout.
        text = (ROOT / "close_loop" / "run_upstream_compat.py").read_text(encoding="utf-8")
        self.assertIn("ac.ConstantPlanner.__init__ = ", text)
        self.assertIn("ac.IDM.update = ", text)
        body = text.split('"""', 2)[-1]
        self.assertNotIn("\nclass ", body)
        # Both patches must be installed before closed_loop is imported, which is what the
        # delegation at the end guarantees.
        self.assertLess(body.index("ac.IDM.update = "), body.index("runpy.run_path("))
        self.assertIn('os.environ.get("HUGSIM_COMPAT_INNER")', text)

    def test_default_concurrency_is_eight_gpus(self):
        self.assertIn('HUGSIM_NUM_GPUS="${HUGSIM_NUM_GPUS:-8}"', self.bench)
        self.assertIn('HUGSIM_PER_GPU="${HUGSIM_PER_GPU:-4}"', self.bench)

    def test_resume_requires_both_result_files(self):
        # eval.json alone means scoring finished but the driving stack never wrote its
        # statistics, which is exactly the result that must not be trusted or skipped.
        self.assertIn('-f "${outdir}/eval.json" ] && [ -f "${outdir}/planner_stats.json" ]',
                      self.one)


if __name__ == "__main__":
    unittest.main()

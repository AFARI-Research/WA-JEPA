"""Camera geometry and frame assembly for the HUGSIM videos.

These are the checks worth automating, because every one of them fails silently: a wrong
Euler convention, a wrong split's camera config or a trajectory read from the wrong key all
produce a video that looks like a trajectory overlay and is simply wrong.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from close_loop.make_video import (
    GROUND_Y_V,
    CameraModel,
    compare_cameras,
    dataset_from_path,
    entry_traj,
    euler_xyz_matrix,
    fov2focal,
    fov_to_radians,
    load_camera_yaml,
    pick_representative,
    resolve_camera,
    rt2pose,
)

ROOT = Path(__file__).resolve().parents[1]

# nuScenes' CAM_FRONT, from HUGSIM's configs/sim/nuscenes_camera.yaml.
NUSCENES_FRONT = {"H": 450, "W": 800, "cx": 413.294057390699, "cy": 234.99233131122904,
                  "fovx": 65.12158487224077, "fovy": 39.515728607991626}
NUSCENES_V2C_ROT = [89.50957761160001, -0.700717347395009, 89.4120791145]
NUSCENES_V2C_TRANS = [0.00536785836469495, 1.4801653630391234, -1.7346966604269407]


def front_camera(rect_rot=(0.0, 0.0, 0.0), rect_trans=(0.0, 0.3, 0.0), intrinsic=None):
    """A CAM_FRONT model built the way ``load_camera_yaml`` would build it."""
    intr = dict(intrinsic or NUSCENES_FRONT)
    intr["fovx"] = fov_to_radians(intr["fovx"])
    intr["fovy"] = fov_to_radians(intr["fovy"])
    cams = {"CAM_FRONT": {"intrinsic": intr,
                          "v2c": rt2pose(NUSCENES_V2C_ROT, NUSCENES_V2C_TRANS)}}
    return CameraModel.build(cams, rt2pose(rect_rot, rect_trans))


class EulerTest(unittest.TestCase):
    def test_xyz_is_the_intrinsic_composition_scipy_uses(self):
        # HUGSIM's rt2pose uses Rotation.from_euler('XYZ', ...), i.e. intrinsic, which
        # composes as Rx @ Ry @ Rz. The extrinsic reading (Rz @ Ry @ Rx) gives a different
        # matrix for the same angles and silently tilts every projection.
        a, b, c = math.radians(12.0), math.radians(-34.0), math.radians(56.0)
        rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
        rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
        np.testing.assert_allclose(euler_xyz_matrix([12.0, -34.0, 56.0]), rx @ ry @ rz, atol=1e-12)
        self.assertFalse(np.allclose(euler_xyz_matrix([12.0, -34.0, 56.0]), rz @ ry @ rx))

    def test_identity_and_orthonormality(self):
        np.testing.assert_allclose(euler_xyz_matrix([0, 0, 0]), np.eye(3), atol=1e-12)
        m = euler_xyz_matrix([5.0, 0.0, 0.0])
        np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(m)), 1.0, places=12)


class FovTest(unittest.TestCase):
    def test_degrees_and_radians_are_both_accepted(self):
        # The camera YAML stores degrees; info['cam_params'] (and so viz/camera.json) holds
        # radians, because load_camera_cfg converts in place. Both are read here.
        self.assertAlmostEqual(fov_to_radians(65.12158487224077), 1.1365860701375323, places=12)
        self.assertAlmostEqual(fov_to_radians(1.1365860701375323), 1.1365860701375323, places=12)

    def test_focal_matches_hugsims_create_cam(self):
        # create_cam derives fx from fovx and W, fy from fovy and H, independently. On
        # nuScenes they coincide; on Waymo they do not, so reusing fx as fy stretches the
        # overlay vertically by 7 per cent while raising nothing.
        self.assertAlmostEqual(fov2focal(fov_to_radians(65.12158487224077), 800), 626.4066,
                               places=3)
        self.assertAlmostEqual(fov2focal(fov_to_radians(39.515728607991626), 450), 626.4066,
                               places=3)
        self.assertAlmostEqual(fov2focal(fov_to_radians(58.12158487224077), 800), 719.8169,
                               places=3)
        self.assertAlmostEqual(fov2focal(fov_to_radians(32.51572860799163), 450), 771.5458,
                               places=3)


class ProjectionTest(unittest.TestCase):
    def test_cam_front_reduces_to_the_rectification(self):
        # _get_obs renders CAM_FRONT from c2front = v2front @ inv(v2front) @ cam_rect, so
        # the v2c terms cancel and only cam_rect is left. If they did not cancel, every
        # projection would carry the front camera's 1.73 m offset twice.
        cam = front_camera()
        np.testing.assert_allclose(cam.v2cam, np.linalg.inv(rt2pose([0, 0, 0], [0, 0.3, 0])),
                                   atol=1e-12)

    def test_a_point_straight_ahead_lands_on_the_principal_column(self):
        cam = front_camera()
        (uv,), (ok,) = cam.project_ground([(20.0, 0.0)])
        self.assertTrue(ok)
        self.assertAlmostEqual(uv[0], cam.cx, places=6)
        # The camera sits above the ground, so the ground projects below the horizon.
        self.assertGreater(uv[1], cam.cy)

    def test_the_ground_converges_to_the_horizon_with_distance(self):
        cam = front_camera()
        uv, ok = cam.project_ground([(10.0, 0.0), (30.0, 0.0), (60.0, 0.0)])
        self.assertTrue(ok.all())
        self.assertTrue(uv[0, 1] > uv[1, 1] > uv[2, 1] > cam.cy)

    def test_left_of_the_vehicle_is_left_in_the_image(self):
        cam = front_camera()
        uv, ok = cam.project_ground([(20.0, 3.0), (20.0, -3.0)])
        self.assertTrue(ok.all())
        # A sign error here mirrors every turn: the overlay curves the wrong way while
        # remaining a perfectly plausible trajectory.
        self.assertLess(uv[0, 0], cam.cx)
        self.assertGreater(uv[1, 0], cam.cx)

    def test_points_behind_the_camera_are_dropped(self):
        cam = front_camera()
        _, ok = cam.project_ground([(-5.0, 0.0), (0.1, 0.0), (5.0, 0.0)])
        self.assertFalse(ok[0])
        self.assertFalse(ok[1])
        self.assertTrue(ok[2])

    def test_camera_height_follows_cam_rect(self):
        # nuScenes and Waymo lift the camera 0.3 m off the ground plane, PandaSet 0.7 m.
        # A lower camera sees the same ground point further down the image.
        near = front_camera(rect_trans=(0.0, 0.7, 0.0))
        far = front_camera(rect_trans=(0.0, 0.3, 0.0))
        (a,), _ = near.project_ground([(15.0, 0.0)])
        (b,), _ = far.project_ground([(15.0, 0.0)])
        self.assertLess(a[1], b[1])
        # And the implied height is the one the config states.
        for cam, height in ((near, GROUND_Y_V - 0.7), (far, GROUND_Y_V - 0.3)):
            (uv,), _ = cam.project_ground([(GROUND_Y_V, 0.0)])
            self.assertAlmostEqual((uv[1] - cam.cy) * GROUND_Y_V / cam.fy, height, places=9)

    def test_kitti360_pitch_shifts_the_whole_image(self):
        # KITTI-360 rectifies with a 5 degree pitch and no translation. Rendering it with
        # nuScenes' values misplaces the ground by tens of pixels.
        kitti = front_camera(rect_rot=(5.0, 0.0, 0.0), rect_trans=(0.0, 0.0, 0.0))
        nusc = front_camera()
        (a,), _ = kitti.project_ground([(20.0, 0.0)])
        (b,), _ = nusc.project_ground([(20.0, 0.0)])
        self.assertGreater(abs(a[1] - b[1]), 20.0)


class CameraConfigTest(unittest.TestCase):
    def test_the_repository_does_not_hardcode_a_split(self):
        # The four camera configs differ; the renderer must read one, not embed one.
        text = (ROOT / "close_loop" / "make_video.py").read_text(encoding="utf-8")
        self.assertIn("load_camera_yaml", text)
        self.assertNotIn("/mnt/", text)
        self.assertNotIn("/data/", text)

    def test_mismatched_intrinsics_are_reported(self):
        wide = {"CAM_FRONT": {"intrinsic": dict(NUSCENES_FRONT, fovx=1.1365860701375323,
                                               fovy=0.6896795705339696),
                              "v2c": rt2pose(NUSCENES_V2C_ROT, NUSCENES_V2C_TRANS)}}
        narrow = {"CAM_FRONT": {"intrinsic": dict(NUSCENES_FRONT, fovx=1.0144, fovy=0.5676),
                                "v2c": rt2pose(NUSCENES_V2C_ROT, NUSCENES_V2C_TRANS)}}
        self.assertEqual(compare_cameras(wide, wide, "CAM_FRONT"), [])
        diffs = compare_cameras(wide, narrow, "CAM_FRONT")
        self.assertTrue(any("fovx" in d for d in diffs), diffs)

    def test_missing_camera_is_reported_rather_than_ignored(self):
        cams = {"CAM_FRONT": {"intrinsic": NUSCENES_FRONT, "v2c": np.eye(4)}}
        self.assertTrue(compare_cameras({}, cams, "CAM_FRONT"))
        self.assertTrue(compare_cameras(cams, {}, "CAM_FRONT"))

    def test_an_ambiguous_results_path_is_not_guessed(self):
        self.assertEqual(dataset_from_path(Path("/x/pandaset_ltf")), "pandaset")
        self.assertIsNone(dataset_from_path(Path("/x/nuscenes_vs_waymo_ltf")))
        self.assertIsNone(dataset_from_path(Path("/x/ltf")))

    def test_a_recorded_dataset_that_contradicts_the_request_stops_the_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            viz = Path(tmp) / "viz"
            viz.mkdir()
            (viz / "camera.json").write_text(json.dumps({"_dataset": "pandaset"}),
                                             encoding="utf-8")
            with self.assertRaises(SystemExit) as ctx:
                resolve_camera(Path(tmp), "nuscenes", None, None, log=lambda *_: None)
            self.assertIn("pandaset", str(ctx.exception))


class PlanLogTest(unittest.TestCase):
    def test_the_trajectory_comes_from_the_planner_record(self):
        # hugsim_client.py stores the ego-frame trajectory under `planner`; the top-level
        # `traj_lidar` is the same plan with forward and lateral swapped for the
        # simulator. Reading that one flips the overlay without raising anything.
        entry = {"traj_lidar": [[-0.5, 4.0]], "planner": {"traj_ego": [[4.0, 0.5, 0.0]]}}
        np.testing.assert_allclose(entry_traj(entry), [[4.0, 0.5]])

    def test_a_step_without_a_plan_yields_nothing(self):
        self.assertIsNone(entry_traj({"traj_lidar": [[0.0, 1.0]], "planner": None}))
        self.assertIsNone(entry_traj({"planner": {"traj_ego": []}}))
        self.assertIsNone(entry_traj({}))


class PickTest(unittest.TestCase):
    def test_each_tier_contributes_its_best_and_worst(self):
        # Deliberately not a highlight reel: showing only the good runs is dressing up the
        # result.
        summary = {"scenarios": [
            {"dir": "a", "tier": "easy_00", "hdscore": 0.9},
            {"dir": "b", "tier": "easy_00", "hdscore": 0.1},
            {"dir": "c", "tier": "easy_00", "hdscore": 0.5},
            {"dir": "d", "tier": "hard_00", "hdscore": 0.4},
            {"dir": "e", "tier": "hard_00", "hdscore": None},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.assertEqual(sorted(pick_representative(path, 2)), ["a", "b", "d"])
            self.assertEqual(pick_representative(path, 1), ["a", "d"])


class RealCameraConfigTest(unittest.TestCase):
    """Runs only where a HUGSIM checkout is available; skipped in CI."""

    def test_the_shipped_configs_parse_and_differ(self):
        import os

        root = os.environ.get("HUGSIM_ROOT", "")
        if not root or not (Path(root) / "configs" / "sim").is_dir():
            self.skipTest("HUGSIM_ROOT is not set to a checkout")
        heights, fovs = set(), set()
        for dataset in ("nuscenes", "kitti360", "waymo", "pandaset"):
            cams, rect = load_camera_yaml(
                Path(root) / "configs" / "sim" / f"{dataset}_camera.yaml")
            self.assertIn("CAM_FRONT", cams)
            heights.add(round(float(rect[1, 3]), 3))
            fovs.add(round(float(cams["CAM_FRONT"]["intrinsic"]["fovx"]), 6))
        # If these ever collapse to one value each, the per-split handling is no longer
        # load-bearing and this test should be the thing that says so.
        self.assertGreater(len(heights), 1)
        self.assertGreater(len(fovs), 1)


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


class WaJepaInferenceConfigTest(unittest.TestCase):
    def _load(self, name: str):
        return OmegaConf.load(ROOT / "configs" / name)

    def test_pdms_and_epdms_are_public_matching_presets(self):
        pdms = self._load("wa_jepa_navsim_pdms.yaml")
        epdms = self._load("wa_jepa_navsim_epdms.yaml")

        self.assertEqual(
            pdms.model.target,
            "models.multiview_causal_future_jepa:MultiViewCausalFutureMaskedJEPA",
        )
        self.assertEqual(pdms.model.camera_names, epdms.model.camera_names)
        self.assertEqual(pdms.data.camera_names, epdms.data.camera_names)
        self.assertEqual(pdms.model.num_history_frames, 4)
        self.assertEqual(pdms.model.num_future_frames, 8)
        self.assertTrue(pdms.model.enable_trajectory_head)
        self.assertEqual(epdms.model.flow_inference_seed, 1)

    def test_presets_do_not_contain_private_checkpoint_paths(self):
        for name in ("wa_jepa_navsim_pdms.yaml", "wa_jepa_navsim_epdms.yaml"):
            text = (ROOT / "configs" / name).read_text(encoding="utf-8")
            self.assertNotIn("/mnt/", text)
            self.assertNotIn("/data/", text)

    def test_navsim_versions_use_their_matching_scorers(self):
        pdms_script = (
            ROOT / "scripts" / "evaluation" / "run_navsim_pdms.sh"
        ).read_text(encoding="utf-8")
        epdms_script = (
            ROOT / "scripts" / "evaluation" / "run_navsim_epdms.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('navsim/planning/script/run_pdm_score.py"', pdms_script)
        self.assertNotIn("run_pdm_score_one_stage.py", pdms_script)
        self.assertIn("run_pdm_score_one_stage.py", epdms_script)


if __name__ == "__main__":
    unittest.main()

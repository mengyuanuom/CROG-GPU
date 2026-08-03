from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "drogoff",
    "drog",
    "crog",
    "lgd",
    "ggcnnclip",
    "grconvnetclip",
    "etrg",
}


class GraspToolSupportTest(unittest.TestCase):
    def test_all_requested_profiles_share_dataset_and_schedule(self):
        config_dir = ROOT / "config" / "grasp_tools"
        self.assertEqual({path.stem for path in config_dir.glob("*.yaml")}, MODELS)
        for model_name in sorted(MODELS):
            with self.subTest(model=model_name):
                cfg = yaml.safe_load(
                    (config_dir / f"{model_name}.yaml").read_text(encoding="utf-8")
                )
                self.assertEqual(cfg["DATA"]["dataset"], "GraspTool")
                self.assertEqual(
                    cfg["DATA"]["root_path"],
                    "./datasets/grasp-tools/aug_graspall_v2",
                )
                self.assertEqual(cfg["MODEL"]["architecture"], model_name)
                self.assertEqual(cfg["TRAIN"]["epochs"], 36)
                self.assertEqual(cfg["TRAIN"]["milestones"], [30])
                self.assertEqual(cfg["TRAIN"]["base_lr"], 0.0001)
                self.assertEqual(cfg["TRAIN"]["batch_size"], 32)
                self.assertEqual(cfg["TRAIN"]["batch_size_val"], 32)
                self.assertEqual(cfg["TRAIN"]["word_len"], 32)
                self.assertEqual(cfg["TEST"]["test_split"], "test")
                self.assertEqual(
                    cfg["TEST"]["evaluation_protocol"], "crog_legacy"
                )
                self.assertEqual(cfg["TEST"]["grasp_size_activation"], "auto")

    def test_drogoff_predicts_both_grasp_size_directions(self):
        cfg = yaml.safe_load(
            (ROOT / "config" / "grasp_tools" / "drogoff.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(cfg["TRAIN"]["predict_grasp_short_side"])
        self.assertEqual(cfg["TRAIN"]["short_side_loss_weight"], 1.0)
        self.assertTrue(cfg["TEST"]["use_offset_at_inference"])

    def test_builder_and_adapter_cover_schema_v21(self):
        builder = (ROOT / "utils" / "data_builder.py").read_text(encoding="utf-8")
        adapter = (ROOT / "utils" / "grasp_tool_dataset.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("return GraspToolDataset(", builder)
        self.assertIn('index_path = os.path.join(split_dir, "index.jsonl")', adapter)
        self.assertIn('query = queries[query_index]', adapter)
        self.assertIn('"short": torch.from_numpy', adapter)
        self.assertIn('grasp_masks["off"]', adapter)


if __name__ == "__main__":
    unittest.main()

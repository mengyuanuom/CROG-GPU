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
                accumulation = cfg["TRAIN"].get(
                    "gradient_accumulation_steps", 1
                )
                self.assertEqual(
                    cfg["TRAIN"]["batch_size"] * accumulation, 32
                )
                self.assertGreater(cfg["TRAIN"]["batch_size_val"], 0)
                self.assertEqual(cfg["TRAIN"]["word_len"], 32)
                self.assertEqual(cfg["TEST"]["test_split"], "test")
                self.assertEqual(
                    cfg["TEST"]["evaluation_protocol"], "crog_legacy"
                )
                self.assertEqual(cfg["TEST"]["grasp_size_activation"], "auto")

    def test_drogoff_skips_fixed_generator_short_side(self):
        cfg = yaml.safe_load(
            (ROOT / "config" / "grasp_tools" / "drogoff.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(cfg["TRAIN"].get("predict_grasp_short_side", False))
        self.assertNotIn("short_side_loss_weight", cfg["TRAIN"])
        self.assertTrue(cfg["TEST"]["use_offset_at_inference"])

    def test_builder_and_adapter_cover_schema_v21(self):
        builder = (ROOT / "utils" / "data_builder.py").read_text(encoding="utf-8")
        adapter = (ROOT / "utils" / "grasp_tool_dataset.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("return GraspToolDataset(", builder)
        self.assertIn('index_path = os.path.join(split_dir, "index.jsonl")', adapter)
        self.assertIn('query = queries[query_index]', adapter)
        self.assertIn("include_short=self.with_short_side", adapter)
        self.assertIn('if self.with_short_side:', adapter)
        self.assertIn('getattr(args, "predict_grasp_short_side", False)', builder)
        self.assertIn('grasp_masks["off"]', adapter)

    def test_auto_activation_matches_every_model_loss(self):
        clamp_models = (
            "model/crog.py",
            "model/drog.py",
            "model/toolrgs/lgd.py",
            "model/toolrgs/ggcnnclip.py",
            "model/toolrgs/grconvnetclip.py",
            "model/toolrgs/etrg/model.py",
        )
        for relative_path in clamp_models:
            with self.subTest(model=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn('grasp_size_loss_activation = "clamp"', source)
        drogoff = (ROOT / "model" / "drogoff.py").read_text(encoding="utf-8")
        self.assertIn('grasp_size_loss_activation = "sigmoid"', drogoff)

    def test_eight_npu_launcher_preserves_grasp_tool_root(self):
        launcher = (ROOT / "tools" / "train_gpu.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('DATASET_NAME="Grasp-Tools"', launcher)
        self.assertIn("datasets/grasp-tools/aug_graspall_v2", launcher)

    def test_drogoff_uses_amp_and_global_batch_32(self):
        cfg = yaml.safe_load(
            (ROOT / "config" / "grasp_tools" / "drogoff.yaml").read_text(
                encoding="utf-8"
            )
        )
        train = cfg["TRAIN"]
        self.assertTrue(train["amp"])
        self.assertFalse(train["find_unused_parameters"])
        self.assertEqual(train["batch_size"], 32)
        self.assertEqual(train["batch_size_val"], 32)
        self.assertNotIn("gradient_accumulation_steps", train)
        runner = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        self.assertIn(
            "find_unused_parameters=find_unused_parameters", runner
        )
        layers = (ROOT / "model" / "drog_layers.py").read_text(
            encoding="utf-8"
        )
        projector = layers.split("class MultiTaskProjector", 1)[1].split(
            "class OffsetMultiTaskProjector", 1
        )[0]
        self.assertIn("groups=batch_size,", projector)
        self.assertNotIn("groups=batch_size * branch_count", projector)
        self.assertIn("branch_features[:, branch_index].contiguous()", projector)


if __name__ == "__main__":
    unittest.main()

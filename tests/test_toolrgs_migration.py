from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "crogoff",
    "drog",
    "drogoff",
    "ggcnnclip",
    "grconvnetclip",
    "graspmamba",
    "lgd",
    "maplegrasp",
)
NAMESPACED_MODELS = (
    "crogoff",
    "ggcnnclip",
    "grconvnetclip",
    "graspmamba",
    "lgd",
    "maplegrasp",
)


class ToolRGSModelMigrationTest(unittest.TestCase):
    def test_all_configs_lock_the_crog_protocol(self):
        for model in MODELS:
            path = ROOT / "config" / "OCID-VLG" / f"{model}.yaml"
            source = path.read_text(encoding="utf-8-sig")
            with self.subTest(model=model):
                self.assertRegex(
                    source,
                    re.compile(
                        rf"^\s*architecture:\s*{model}\s*$",
                        re.MULTILINE,
                    ),
                )
                self.assertRegex(
                    source,
                    re.compile(
                        r"^\s*evaluation_protocol:\s*crog_legacy\s*$",
                        re.MULTILINE,
                    ),
                )
                self.assertRegex(
                    source,
                    re.compile(r"^\s*amp:\s*False\s*$", re.MULTILINE),
                )

    def test_original_crog_stays_the_default_builder(self):
        source = (ROOT / "model" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn('getattr(args, "architecture", "crog")', source)
        self.assertIn('"crog": build_crog', source)
        self.assertIn("from .toolrgs import build_toolrgs_model", source)

    def test_migrated_models_are_namespaced(self):
        for model in NAMESPACED_MODELS:
            with self.subTest(model=model):
                self.assertTrue(
                    (ROOT / "model" / "toolrgs" / f"{model}.py").is_file()
                )
        top_level = {path.name for path in (ROOT / "model").glob("*.py")}
        self.assertIn("drogoff.py", top_level)
        self.assertIn("drog.py", top_level)

    def test_crog_evaluation_contract_is_retained(self):
        source = (
            ROOT / "engine" / "crog_engine.py"
        ).read_text(encoding="utf-8")
        expected = (
            "flags=cv2.INTER_CUBIC",
            "ins_mask_pred = (ins_mask_pred > 0.35)",
            "num_grasps = [1,5]",
            "torch.sigmoid(grasp_qua_mask_preds)",
            "torch.sigmoid(grasp_wid_mask_preds)",
            "calculate_jacquard_index(grasp_preds, grasp_target)",
        )
        for token in expected:
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_only_offset_models_request_offset_targets(self):
        source = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        self.assertIn(
            'needs_offset = bool(getattr(model, "supports_offset", False))',
            source,
        )
        self.assertEqual(source.count("with_grasp_offset=needs_offset"), 1)

    def test_standalone_evaluator_uses_the_same_model_builder_and_npu_path(self):
        source = (ROOT / "test_crog.py").read_text(encoding="utf-8")
        self.assertIn("from model import build_model", source)
        self.assertIn("model, _ = build_model(args)", source)
        self.assertIn('args.evaluation_protocol = "crog_legacy"', source)
        self.assertNotIn(".cuda(", source)
        self.assertNotIn("torch.cuda", source)

    def test_offset_models_print_their_offset_loss(self):
        source = (
            ROOT / "engine" / "crog_engine.py"
        ).read_text(encoding="utf-8")
        self.assertIn("AverageMeter('Loss_off', ':2.4f')", source)
        self.assertIn('off_loss_metter.update(loss_dict["m_off"]', source)
        self.assertIn(
            'getattr(unwrapped_model, "supports_offset", False)',
            source,
        )


if __name__ == "__main__":
    unittest.main()

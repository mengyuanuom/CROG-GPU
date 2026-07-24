from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class OfficialCROGNPUConfigTest(unittest.TestCase):
    def test_official_training_hyperparameters_are_preserved(self):
        path = ROOT / "config" / "OCID-VLG" / "crog_multiple_r50.yaml"
        source = path.read_text(encoding="utf-8")
        expected_lines = (
            r"^\s*epochs:\s*50\s*$",
            r"^\s*milestones:\s*\[35\]\s*$",
            r"^\s*batch_size:\s*24\b",
            r"^\s*batch_size_val:\s*24\b",
            r"^\s*base_lr:\s*0\.0001\b",
            r"^\s*lr_multi:\s*0\.1\b",
            r"^\s*sync_bn:\s*True\s*$",
            r"^\s*amp:\s*True\s*$",
            r"^\s*pin_memory:\s*False\s*$",
            r"^\s*resume:\s*$",
            r"^\s*dist_backend:\s*['\"]hccl['\"]\s*$",
            r"^\s*dist_url:\s*env://\s*$",
        )
        for pattern in expected_lines:
            with self.subTest(pattern=pattern):
                self.assertRegex(source, re.compile(pattern, re.MULTILINE))

    def test_training_path_has_no_cuda_or_nccl_calls(self):
        paths = (
            ROOT / "train_crog.py",
            ROOT / "engine" / "crog_engine.py",
            ROOT / "utils" / "misc.py",
            ROOT / "utils" / "npu.py",
        )
        forbidden = (".cuda(", "torch.cuda", "'nccl'", '"nccl"')
        for path in paths:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                for token in forbidden:
                    self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()

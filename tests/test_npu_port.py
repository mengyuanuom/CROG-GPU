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
            r"^\s*amp:\s*False\b",
            r"^\s*pin_memory:\s*False\s*$",
            r"^\s*with_depth:\s*False\s*$",
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

    def test_clip_download_uses_the_official_hashed_url(self):
        source = (
            ROOT / "tools" / "download_clip_rn50.py"
        ).read_text(encoding="utf-8")
        digest = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
        self.assertIn("https://openaipublic.azureedge.net/clip/models/", source)
        self.assertGreaterEqual(source.count(digest), 2)

    def test_crog_dataloader_does_not_require_depth(self):
        dataset_source = (ROOT / "utils" / "dataset.py").read_text(encoding="utf-8")
        train_source = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        crog_dataset_source = dataset_source.split(
            "class OCIDVLGDataset", 1
        )[1].split("class OCIDGraspDataset", 1)[0]
        self.assertRegex(crog_dataset_source, r"with_depth\s*=\s*False")
        self.assertIn('if "depth" in batch[0]:', crog_dataset_source)
        self.assertNotIn(
            '"depth": torch.stack([torch.from_numpy(x["depth"]) for x in batch])',
            crog_dataset_source,
        )
        self.assertEqual(
            train_source.count(
                'with_depth=bool(getattr(args, "with_depth", False))'
            ),
            2,
        )

    def test_launcher_reads_amp_only_from_yaml(self):
        source = (
            ROOT / "tools" / "train_crog_8npu.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn('AMP=', source)
        self.assertNotIn('TRAIN.amp', source)

    def test_launcher_and_worker_bind_all_eight_npus(self):
        launcher = (
            ROOT / "tools" / "train_crog_8npu.sh"
        ).read_text(encoding="utf-8")
        trainer = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        self.assertIn("0,1,2,3,4,5,6,7", launcher)
        self.assertIn('NPROC_PER_NODE="${NPROC_PER_NODE:-8}"', launcher)
        self.assertIn('--nproc_per_node="${NPROC_PER_NODE}"', launcher)
        self.assertIn('os.environ.get("LOCAL_RANK", 0)', trainer)
        self.assertIn("set_device(local_rank)", trainer)
        self.assertIn('backend="hccl"', trainer)
        self.assertIn("DistributedDataParallel(", trainer)

    def test_fp32_path_does_not_construct_an_npu_grad_scaler(self):
        runtime = (ROOT / "utils" / "npu.py").read_text(encoding="utf-8")
        launcher = (
            ROOT / "tools" / "train_crog_8npu.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("class NoOpGradScaler:", runtime)
        self.assertIn("if not enabled:\n        return NoOpGradScaler()", runtime)
        self.assertNotIn("TRAIN.amp", launcher)


if __name__ == "__main__":
    unittest.main()

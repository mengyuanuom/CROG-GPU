from pathlib import Path
import unittest

from utils.grasp_tool_language import (
    CANONICAL_CATEGORY_NAMES,
    CATEGORY_DESCRIPTION_VARIANTS,
    COMMAND_TEMPLATES,
    category_prompt_for_epoch,
    category_prompt_pool,
)


ROOT = Path(__file__).resolve().parents[1]


class GraspToolLanguageCurriculumTest(unittest.TestCase):
    def test_every_category_has_a_32_prompt_cycle(self):
        self.assertEqual(len(CANONICAL_CATEGORY_NAMES), 22)
        self.assertEqual(len(COMMAND_TEMPLATES["train"]), 8)
        for category, variants in CATEGORY_DESCRIPTION_VARIANTS.items():
            self.assertEqual(len(variants), 4, category)
            pool = category_prompt_pool(category)
            self.assertEqual(len(pool), 32, category)
            self.assertEqual(len(set(pool)), 32, category)

    def test_epochs_are_randomly_ordered_unique_and_reproducible(self):
        prompts = [
            category_prompt_for_epoch("wrench", "scene-7:target-2", epoch)
            for epoch in range(1, 33)
        ]
        self.assertEqual(len(set(prompts[:24])), 24)
        self.assertEqual(len(set(prompts)), 32)
        self.assertNotEqual(prompts, list(category_prompt_pool("wrench")))
        self.assertEqual(
            prompts[19],
            category_prompt_for_epoch("wrench", "scene-7:target-2", 20),
        )
        other_target = [
            category_prompt_for_epoch("wrench", "scene-8:target-1", epoch)
            for epoch in range(1, 33)
        ]
        self.assertNotEqual(prompts, other_target)
        second_cycle = [
            category_prompt_for_epoch("wrench", "scene-7:target-2", epoch)
            for epoch in range(33, 65)
        ]
        self.assertEqual(len(set(second_cycle)), 32)
        self.assertNotEqual(prompts[-1], second_cycle[0])
        self.assertNotEqual(
            prompts,
            [
                category_prompt_for_epoch(
                    "wrench", "scene-7:target-2", epoch, seed=7
                )
                for epoch in range(1, 33)
            ],
        )

    def test_train_and_eval_templates_are_disjoint(self):
        self.assertFalse(
            set(COMMAND_TEMPLATES["train"]) & set(COMMAND_TEMPLATES["eval"])
        )

if __name__ == "__main__":
    unittest.main()

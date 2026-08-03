from collections import Counter
from pathlib import Path
import random
from types import SimpleNamespace

from tools.dataset_converters.grasp_tools.augment import (
    SourceObject,
    balanced_quotas,
    balanced_scene_sizes,
    plan_query_targets,
    plan_split_scenes,
)
from utils.grasp_tool_language import CANONICAL_CATEGORY_NAMES


def fake_sources():
    result = {}
    for category in CANONICAL_CATEGORY_NAMES:
        result[category] = [
            SourceObject(
                source_id=f"{category}:{index}",
                image_path=Path(f"{category}_{index}.jpg"),
                object_index=index,
                category_key=category,
                category_name=CANONICAL_CATEGORY_NAMES[category],
                mask=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
                grasps=(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),),
            )
            for index in range(3)
        ]
    return result


def delta(values):
    values = list(values)
    return max(values) - min(values)


def test_balanced_integer_quotas_and_scene_sizes():
    rng = random.Random(7)
    quotas = balanced_quotas(101, list("abcdef"), rng)
    assert sum(quotas.values()) == 101
    assert delta(quotas.values()) <= 1

    sizes = balanced_scene_sizes(800, 3, 5, rng)
    assert len(sizes) == 800
    assert min(sizes) == 3
    assert max(sizes) == 5
    assert sum(sizes) == 3200
    counts = Counter(sizes)
    assert delta(counts.values()) <= 1


def test_split_planner_balances_categories_sources_and_queries():
    sources = fake_sources()
    config = SimpleNamespace(
        objects_min=3,
        objects_max=5,
        same_category_probability=0.35,
        hard_negative_probability=0.30,
    )
    rng = random.Random(2025)
    scenes, source_usage = plan_split_scenes(66, sources, config, rng)

    placements = Counter(
        source.category_key for scene in scenes for source in scene
    )
    assert len(scenes) == 66
    assert sum(placements.values()) == 264
    assert delta(placements[category] for category in sources) <= 1

    for category, category_sources in sources.items():
        assert delta(
            source_usage.get(source.source_id, 0)
            for source in category_sources
        ) <= 1

    targets, target_quota = plan_query_targets(scenes, 6, rng)
    assert all(len(scene_targets) == 6 for scene_targets in targets)
    assert sum(target_quota.values()) == 396
    assert delta(target_quota.values()) <= 1
    for scene, scene_targets in zip(scenes, targets):
        assert all(0 <= target < len(scene) for target in scene_targets)
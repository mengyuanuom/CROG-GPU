# Grasp-Tools balanced dataset generation

The generator creates a compositional referring-grasp dataset from labeled
single-object cutouts and unrelated background images. It writes one image per
scene, stores several language queries in the paired JSON file, and expands
those queries through index.jsonl.

## Source layout

Put or link the source files under:

~~~text
datasets/grasp-tools/source/
├── graspall/
│   ├── 000000000001.jpg
│   ├── 000000000001.json
│   └── ...
└── backgrounds/
    ├── background_001.jpg
    └── ...
~~~

Every source JSON must contain an objects list. Each valid object needs a
canonical/recognized category, a polygon mask, and at least one grasp. The
generator checks that all 22 categories are present and reports skipped invalid
objects in metadata.json.

## Recommended generation

From the CROG-GPU root:

~~~bash
python tools/dataset_converters/grasp_tools/augment.py --overwrite
~~~

The defaults produce:

| Split | Images | Objects/image | Queries/image | Approx. query samples |
|---|---:|---:|---:|---:|
| train | 6000 | 3–5 (average 4) | 6 | 36000 |
| val | 800 | 3–5 (average 4) | 4 | 3200 |
| test | 1200 | 3–5 (average 4) | 4 | 4800 |
| total | 8000 | average 4 | — | 44000 |

Use a quick integration run before full generation:

~~~bash
python tools/dataset_converters/grasp_tools/augment.py   --smoke-test   --out-dir datasets/grasp-tools/smoke   --overwrite
~~~

## Balance guarantees

The complete plan for each split is created before rendering. Consequently:

- placements of the 22 categories differ by at most one;
- query-target counts of the 22 categories differ by at most one;
- source instances belonging to the same category are reused equally, with a
  maximum count difference of one;
- scene generation is atomic, so a failed placement retries the whole scene and
  cannot silently alter the planned counts;
- object counts 3, 4, and 5 are used nearly equally, with average 4 for the
  recommended split sizes.

The generator aborts if any guarantee is violated. Exact counts and deltas are
written to metadata.json.

## Language diversity

Each category has four safe surface forms: its canonical name plus aliases or
near-synonyms. Training has 22 command templates, giving 88 category-only
command/term combinations per category. The generator balances template and
term usage, and reserves four disjoint command templates for validation/test
when language-templates is heldout (the default). About one quarter of queries
use valid spatial or relational descriptions when the scene supports them; the
rest explicitly exercise the balanced category vocabulary.

The canonical category is always stored separately in each query, so changing
wording does not change the target label.
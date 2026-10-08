# Contributing

Keep algorithm code separate from simulator launch and experiment management.
Prefer a small API with explicit tensor/array shapes and device ownership.
Keep optional dependencies isolated: importing `robolearn` must not import Torch,
Warp, or a simulator.

Use a feature branch and focused commits. For changes to algorithm math, include
a numerical reference or a learning check that detects the intended failure.
Changes to device state or graph capture should compare eager and captured behavior.

```bash
uv sync --extra flashsac --extra warp
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python -m build
```

CUDA tests require an NVIDIA GPU. CI runs the FlashSAC and adapter CPU tests and package
checks; CUDA tests and direct MuJoCo Warp capture checks run locally.

Preserve upstream copyright and license notices when adapting code. Record the
upstream revision and adaptations in `ATTRIBUTION.md`, and cite original
algorithm authors in `CITATIONS.bib`. Follow semantic versioning, record user
visible changes in `CHANGELOG.md`, and build a wheel and source distribution
before a release. Releases are hosted on GitHub with tags named `v<version>`.

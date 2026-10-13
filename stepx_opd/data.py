"""Multi-source prompt sampling for source-routed distribution matching.

One training step draws its whole mini-batch from a single source, and the
source sequence is a weighted round robin over the configured sources. Every
data-parallel rank therefore agrees on which source, and so on which teacher,
the step belongs to, while the prompts themselves are sliced disjointly across
ranks. That agreement is what lets the step forward exactly one teacher without
any cross-rank negotiation.

The dataset classes for a plain prompt list and for GenEval metadata follow the
DiffusionOPD codebase, from which this release is derived.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Dict, List, Optional


def stable_source_hash(source: str) -> int:
    """Process-independent integer hash, used to seed the per-source shuffle."""
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def load_prompts(path: str, prompt_key: str = "prompt") -> List[Dict]:
    """Read one source file.

    A ``.jsonl`` file is parsed line by line and the whole object is kept as
    metadata, which is how GenEval ships its object/count/colour annotations; any
    other extension is read as one prompt per line.
    """
    items: List[Dict] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get(prompt_key)
                if not isinstance(prompt, str) or not prompt.strip():
                    continue
                items.append({"prompt": prompt.strip(), "metadata": obj})
    else:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                prompt = line.strip()
                if prompt:
                    items.append({"prompt": prompt, "metadata": {"prompt": prompt}})
    return items


class MultiSourcePromptSampler:
    """Deterministic, effectively infinite batch source indexed by training step.

    ``batch(step)`` returns the prompts for the calling rank at that step
    together with the source name. The source comes from the round-robin
    pattern, so it is a pure function of the step; the prompt offset advances by
    ``batch_size * world_size`` per cycle and is shifted by the rank, so the
    ranks never overlap and resuming at a step reproduces the same data.
    """

    def __init__(
        self,
        source_files: Dict[str, str],
        source_weights: Optional[Dict[str, float]] = None,
        batch_size: int = 8,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        prompt_key: str = "prompt",
    ):
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.sources = sorted(source_files)
        self.prompts: Dict[str, List[Dict]] = {}
        for source in self.sources:
            items = load_prompts(source_files[source], prompt_key=prompt_key)
            if not items:
                raise ValueError(
                    f"no prompts loaded for source {source!r} from "
                    f"{source_files[source]!r}"
                )
            rng = random.Random(int(seed) * 1000 + stable_source_hash(source) % 997)
            rng.shuffle(items)
            self.prompts[source] = items
        self.pattern = self._build_pattern(source_weights)

    def _build_pattern(self, weights: Optional[Dict[str, float]]) -> List[str]:
        weights = weights or {s: 1.0 for s in self.sources}
        active = [s for s in self.sources if float(weights.get(s, 0)) > 0]
        if not active:
            active = list(self.sources)
        min_w = min(float(weights[s]) for s in active)
        pattern: List[str] = []
        for source in active:
            pattern.extend([source] * max(1, round(float(weights[source]) / min_w)))
        return pattern

    def source_for_step(self, step: int) -> str:
        return self.pattern[int(step) % len(self.pattern)]

    def batch(self, step: int) -> Dict:
        step = int(step)
        source = self.source_for_step(step)
        pool = self.prompts[source]
        cycle = step // len(self.pattern)
        base = cycle * self.batch_size * self.world_size + self.rank * self.batch_size
        items = [pool[(base + j) % len(pool)] for j in range(self.batch_size)]
        return {
            "source": source,
            "prompts": [item["prompt"] for item in items],
            "metadata": [item["metadata"] for item in items],
        }

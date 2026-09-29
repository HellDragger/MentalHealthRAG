"""Near-duplicate chunk removal (MinHash + LSH banding, verified with exact Jaccard on word 5-shingles)."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np

from mhrag.types import Chunk

_WORD = re.compile(r"[a-z0-9']+")
_MERSENNE = (1 << 61) - 1


def shingles(text: str, k: int = 5) -> set[int]:
    words = _WORD.findall(text.lower())
    if len(words) < k:
        return {int.from_bytes(hashlib.blake2b(" ".join(words).encode(), digest_size=8).digest(), "little")}
    return {
        int.from_bytes(hashlib.blake2b(" ".join(words[i : i + k]).encode(), digest_size=8).digest(), "little")
        for i in range(len(words) - k + 1)
    }


def jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def near_duplicates(texts: list[str], threshold: float = 0.85, num_perm: int = 64, bands: int = 16, seed: int = 0):
    """Return a list of (i, j, jaccard) with i < j and jaccard >= threshold."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _MERSENNE, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, _MERSENNE, size=num_perm, dtype=np.uint64)
    sh = [shingles(t) for t in texts]
    rows = num_perm // bands
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for idx, s in enumerate(sh):
        arr = np.fromiter(s, dtype=np.uint64) % np.uint64(_MERSENNE)
        # (a*x + b) mod p with uint64 wrap-around is fine as a hash family for bucketing
        sig = ((np.outer(a, arr) + b[:, None]) % np.uint64(_MERSENNE)).min(axis=1)
        for band in range(bands):
            buckets[(band, sig[band * rows : (band + 1) * rows].tobytes())].append(idx)
    pairs: dict[tuple[int, int], float] = {}
    for members in buckets.values():
        if len(members) < 2:
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                i, j = members[x], members[y]
                if (i, j) in pairs:
                    continue
                jac = jaccard(sh[i], sh[j])
                if jac >= threshold:
                    pairs[(i, j)] = jac
    return [(i, j, v) for (i, j), v in sorted(pairs.items())]


# Preference when two near-duplicate chunks come from different source types: keep the richer/cited one.
_PRIORITY = {"mind_web": 0, "mind_booklet": 1, "faq": 2, "kb_fact": 3, "web_article": 4, "counselling": 5}


def drop_near_duplicates(chunks: list[Chunk], threshold: float = 0.85) -> tuple[list[Chunk], list[dict]]:
    pairs = near_duplicates([c.text for c in chunks], threshold)
    drop: set[int] = set()
    log = []
    for i, j, jac in pairs:
        if i in drop or j in drop:
            continue
        ci, cj = chunks[i], chunks[j]
        # keep the higher-priority source; tie -> keep the longer, then the earlier
        key_i = (_PRIORITY.get(ci.source_type, 9), -ci.n_tokens, i)
        key_j = (_PRIORITY.get(cj.source_type, 9), -cj.n_tokens, j)
        loser = j if key_i <= key_j else i
        keeper = i if loser == j else j
        drop.add(loser)
        log.append(
            {
                "dropped": chunks[loser].chunk_id,
                "dropped_source": chunks[loser].source_file,
                "kept": chunks[keeper].chunk_id,
                "kept_source": chunks[keeper].source_file,
                "jaccard": round(jac, 3),
            }
        )
    return [c for k, c in enumerate(chunks) if k not in drop], log

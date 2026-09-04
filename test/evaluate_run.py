#!/usr/bin/env python3
"""Score a FastqOptiFilter run against simulate_fastq.py ground truth.

Scoring is family-aware: for a true optical family of size k the ideal outcome
is that exactly k-1 of its members are removed, and it does not matter which
member is retained. Removals of reads that belong to no true optical family are
false positives, and they are broken out by the mechanism that actually
generated the read (independent cluster, library/PCR copy, or spatially
clustered poly-G artifact).

Cross-tile optical relations are scored separately, because a same-tile-only
spatial model cannot recover them at any FDR.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt", encoding="utf-8")


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--filtered-r1", required=True, type=Path)
    parser.add_argument("--label", default="run")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    with open_text(args.truth) as handle:
        truth = list(csv.DictReader(handle, delimiter="\t"))
    mechanism = {r["read_name"]: r["mechanism"] for r in truth}
    cross_tile = {r["read_name"]: r["cross_tile"] == "1" for r in truth}
    parent_of = {r["read_name"]: r["parent_read_name"] for r in truth}

    # True optical families: connected components of optical parent->child links
    # only. PCR/library copies are deliberately excluded; they are the nulls.
    union = UnionFind()
    for name, mech in mechanism.items():
        union.find(name)
        if mech == "optical" and parent_of[name] != "NA":
            union.union(parent_of[name], name)
    families: dict[str, list[str]] = defaultdict(list)
    for name in mechanism:
        families[union.find(name)].append(name)

    in_family = {n for members in families.values() if len(members) > 1 for n in members}
    removable = sum(len(m) - 1 for m in families.values() if len(m) > 1)
    # A family is cross-tile-only if none of its members share a tile: recovering
    # it requires a nearby-tile search.
    cross_families = [
        m
        for m in families.values()
        if len(m) > 1 and any(cross_tile.get(n) for n in m)
    ]
    cross_removable = sum(len(m) - 1 for m in cross_families)

    kept: set[str] = set()
    with gzip.open(args.filtered_r1, "rt") as handle:
        for index, line in enumerate(handle):
            if index % 4 == 0:
                kept.add(line[1:].split()[0])
    removed = set(mechanism) - kept

    correct = 0
    over_removed = 0
    cross_correct = 0
    for members in families.values():
        if len(members) < 2:
            continue
        hit = sum(1 for n in members if n in removed)
        correct += min(hit, len(members) - 1)
        over_removed += max(0, hit - (len(members) - 1))
        if any(cross_tile.get(n) for n in members):
            cross_correct += min(hit, len(members) - 1)

    false_positive = len(removed - in_family) + over_removed
    fp_by_mechanism: dict[str, int] = defaultdict(int)
    for name in removed - in_family:
        fp_by_mechanism[mechanism[name]] += 1
    # The poly-G blob really is a spatial cluster, so no test of spatial
    # exchangeability can call it null. Removing those reads is a failure of
    # mechanism attribution, not of the spatial null, and it is scored
    # separately from reads whose positions genuinely are exchangeable.
    spatially_null_fp = false_positive - fp_by_mechanism.get("polyg", 0)

    same_removable = removable - cross_removable
    same_correct = correct - cross_correct
    result = {
        "label": args.label,
        "input_read_pairs": len(mechanism),
        "removed_read_pairs": len(removed),
        "true_optical_removable": removable,
        "true_positives": correct,
        "false_positives": false_positive,
        "false_negatives": removable - correct,
        "sensitivity": correct / removable if removable else None,
        "empirical_read_level_fdr": (
            false_positive / len(removed) if removed else 0.0
        ),
        "spatially_null_fdr": (
            spatially_null_fp / len(removed) if removed else 0.0
        ),
        "spatially_null_false_positives": spatially_null_fp,
        "false_positives_by_true_mechanism": dict(sorted(fp_by_mechanism.items())),
        "cross_tile_removable": cross_removable,
        "cross_tile_recovered": cross_correct,
        "cross_tile_sensitivity": (
            cross_correct / cross_removable if cross_removable else None
        ),
        "same_tile_removable": same_removable,
        "same_tile_recovered": same_correct,
        "same_tile_sensitivity": (
            same_correct / same_removable if same_removable else None
        ),
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

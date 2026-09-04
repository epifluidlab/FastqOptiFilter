#!/usr/bin/env python3
"""Ground-truth simulator for FastqOptiFilter calibration testing.

Generates synchronized paired-end Illumina-style FASTQs together with a truth
table naming every duplicate relation and its mechanism. The simulator is
deliberately unkind to the statistical model:

* cluster density varies between tiles and across each tile, so a lane-pooled
  spatial null is miscalibrated on purpose;
* PCR/library duplicates land at independent positions -- true nulls that must
  survive filtering;
* optical duplicates land at short offsets, and a configurable share of them
  crosses a tile boundary into the physically adjacent tile;
* a spatially clustered poly-G artifact blob produces sequence-identical read
  pairs that are not optical duplicates of one another.

Tile identifiers follow the Illumina ``<surface><swath><tile>`` convention, so
tile 1207 is surface 1, swath 2, tile 7.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

BASES = np.array(list("ACGT"))
X_SPAN = 30000
Y_SPAN = 24000


def phred_to_char(q: np.ndarray) -> np.ndarray:
    return np.clip(q, 2, 41).astype(np.uint8) + 33


def sample_quality(
    rng: np.random.Generator,
    n: int,
    length: int,
    bad_tail_cycles: int = 0,
    bad_tail_quality: float = 10.0,
    flat_quality: float = 0.0,
) -> np.ndarray:
    """Position-dependent Phred qualities with a decaying tail.

    ``bad_tail_cycles`` collapses the last cycles to ``bad_tail_quality``, the
    way a real run degrades near the end of a read. Those cycles then carry
    real sequencing errors, which is what stops two copies of one molecule from
    sharing an exact seed there.
    """
    cycle = np.arange(length)
    mean_q = 37.0 - 11.0 * (cycle / max(length - 1, 1)) ** 2.4
    if flat_quality > 0:
        mean_q = np.full(length, float(flat_quality))
    if bad_tail_cycles > 0:
        start = max(length - bad_tail_cycles, 0)
        ramp = np.linspace(1.0, 0.0, length - start, endpoint=False)
        mean_q[start:] = bad_tail_quality + (mean_q[start:] - bad_tail_quality) * ramp
    noise = rng.normal(0.0, 2.0, size=(n, length))
    per_read = rng.normal(0.0, 1.6, size=(n, 1))
    return np.clip(mean_q[None, :] + noise + per_read, 2, 41).round().astype(np.int16)


def apply_errors(
    rng: np.random.Generator, bases: np.ndarray, qual: np.ndarray
) -> np.ndarray:
    error_p = 10.0 ** (-qual / 10.0)
    hit = rng.random(bases.shape) < error_p
    shift = rng.integers(1, 4, size=bases.shape)
    return np.where(hit, (bases + shift) % 4, bases)


def tile_id(surface: int, swath: int, tile: int) -> int:
    return surface * 1000 + swath * 100 + tile


def build_layout(
    n_surfaces: int, n_swaths: int, n_tiles: int
) -> list[tuple[int, int, int]]:
    return [
        (surface, swath, tile)
        for surface in range(1, n_surfaces + 1)
        for swath in range(1, n_swaths + 1)
        for tile in range(1, n_tiles + 1)
    ]


def sample_positions(
    rng: np.random.Generator, n: int, density_shape: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Inhomogeneous within-tile positions via independent Beta marginals."""
    ax, ay = density_shape
    x = rng.beta(ax, 1.35, size=n) * (X_SPAN - 2000) + 1000
    y = rng.beta(ay, 1.20, size=n) * (Y_SPAN - 2000) + 1000
    return x, y


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-prefix", required=True, type=Path)
    parser.add_argument("--reads", type=int, default=60000, help="Read pairs to emit")
    parser.add_argument("--read-length", type=int, default=90)
    parser.add_argument("--surfaces", type=int, default=1)
    parser.add_argument("--swaths", type=int, default=2)
    parser.add_argument("--tiles", type=int, default=6)
    parser.add_argument(
        "--pcr-duplicate-rate",
        type=float,
        default=0.06,
        help="Share of clusters that are library copies at independent positions",
    )
    parser.add_argument(
        "--optical-rate",
        type=float,
        default=0.04,
        help="Share of clusters that are optical children of a parent cluster",
    )
    parser.add_argument(
        "--optical-scale",
        type=float,
        default=45.0,
        help="Mean pixel offset of an optical child from its parent",
    )
    parser.add_argument(
        "--cross-tile-fraction",
        type=float,
        default=0.18,
        help="Share of optical children pushed across a tile boundary",
    )
    parser.add_argument(
        "--polyg-rate",
        type=float,
        default=0.004,
        help="Share of clusters that are spatially clustered poly-G artifacts",
    )
    parser.add_argument(
        "--shared-motif-rate",
        type=float,
        default=0.05,
        help=(
            "Share of clusters carrying a common primer-like motif. These reads "
            "collide in the seed index but are different molecules, so they "
            "supply the sequence-incompatible negative control."
        ),
    )
    parser.add_argument(
        "--bad-tail-cycles",
        type=int,
        default=0,
        help="Cycles at the end of each read collapsed to a low Phred score",
    )
    parser.add_argument(
        "--bad-tail-quality",
        type=float,
        default=10.0,
        help="Phred score the collapsed tail decays to",
    )
    parser.add_argument(
        "--flat-quality",
        type=float,
        default=0.0,
        help=(
            "Give every cycle this Phred score. Below about 13 no exact seed "
            "stays clean in both copies of a molecule, which is where exact "
            "seeding starts losing real duplicates"
        ),
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    rng = np.random.default_rng(args.seed)
    length = args.read_length
    layout = build_layout(args.surfaces, args.swaths, args.tiles)
    n_tiles_total = len(layout)

    # Tile-level density variation: no single lane-pooled null can be correct
    # for every tile at once.
    tile_weight = rng.gamma(2.2, 1.0, size=n_tiles_total)
    tile_weight /= tile_weight.sum()
    tile_shapes = [
        (float(rng.uniform(0.75, 2.4)), float(rng.uniform(0.75, 2.4)))
        for _ in range(n_tiles_total)
    ]

    n_parent = args.reads
    tile_choice = rng.choice(n_tiles_total, size=n_parent, p=tile_weight)

    tiles = np.empty(n_parent, dtype=np.int64)
    surfaces = np.empty(n_parent, dtype=np.int64)
    swaths = np.empty(n_parent, dtype=np.int64)
    tile_index = np.empty(n_parent, dtype=np.int64)
    xs = np.empty(n_parent, dtype=np.float64)
    ys = np.empty(n_parent, dtype=np.float64)
    for slot, (surface, swath, tile) in enumerate(layout):
        members = np.flatnonzero(tile_choice == slot)
        if not len(members):
            continue
        tiles[members] = tile_id(surface, swath, tile)
        surfaces[members] = surface
        swaths[members] = swath
        tile_index[members] = tile
        x, y = sample_positions(rng, len(members), tile_shapes[slot])
        xs[members] = x
        ys[members] = y

    # Template molecules. Most clusters carry a unique molecule.
    templates = rng.integers(0, 4, size=(n_parent, 2, length), dtype=np.int8)
    molecule = np.arange(n_parent, dtype=np.int64)
    origin = np.full(n_parent, -1, dtype=np.int64)
    mechanism = np.zeros(n_parent, dtype=np.int8)  # 0 unique, 1 pcr, 2 optical, 3 polyg

    # --- Library / PCR duplicates: same molecule, independent positions ------
    n_pcr = int(args.pcr_duplicate_rate * n_parent)
    if n_pcr:
        child = rng.choice(n_parent, size=n_pcr, replace=False)
        parent = rng.choice(np.setdiff1d(np.arange(n_parent), child), size=n_pcr)
        templates[child] = templates[parent]
        molecule[child] = molecule[parent]
        origin[child] = parent
        mechanism[child] = 1

    # --- Optical duplicates: same molecule, short spatial offset -------------
    available = np.flatnonzero(mechanism == 0)
    n_optical = min(int(args.optical_rate * n_parent), len(available) // 2)
    optical_child = rng.choice(available, size=n_optical, replace=False)
    optical_parent = rng.choice(np.setdiff1d(available, optical_child), size=n_optical)
    templates[optical_child] = templates[optical_parent]
    molecule[optical_child] = molecule[optical_parent]
    origin[optical_child] = optical_parent
    mechanism[optical_child] = 2

    angle = rng.uniform(0.0, 2 * np.pi, size=n_optical)
    radius = rng.exponential(args.optical_scale, size=n_optical)
    dx = radius * np.cos(angle)
    dy = radius * np.sin(angle)
    child_x = xs[optical_parent] + dx
    child_y = ys[optical_parent] + dy

    tiles[optical_child] = tiles[optical_parent]
    surfaces[optical_child] = surfaces[optical_parent]
    swaths[optical_child] = swaths[optical_parent]
    tile_index[optical_child] = tile_index[optical_parent]

    # A share of optical children crosses into the tile that is physically next
    # along y within the same swath and surface. Both members are moved to the
    # shared boundary so the pair really is spatially adjacent.
    cross = rng.random(n_optical) < args.cross_tile_fraction
    crossed = np.zeros(n_optical, dtype=bool)
    for pos in np.flatnonzero(cross):
        parent = optical_parent[pos]
        parent_tile = int(tile_index[parent])
        gap = float(rng.exponential(args.optical_scale))
        if parent_tile < args.tiles and rng.random() < 0.5:
            neighbour = parent_tile + 1
            ys[parent] = Y_SPAN - gap  # parent sits just below its top edge
            new_y = max(1.0, float(rng.exponential(args.optical_scale)))
        elif parent_tile > 1:
            neighbour = parent_tile - 1
            ys[parent] = gap  # parent sits just above its bottom edge
            new_y = Y_SPAN - max(1.0, float(rng.exponential(args.optical_scale)))
        else:
            continue
        child = optical_child[pos]
        tile_index[child] = neighbour
        tiles[child] = tile_id(int(surfaces[parent]), int(swaths[parent]), neighbour)
        child_x[pos] = xs[parent] + dx[pos]
        child_y[pos] = new_y
        crossed[pos] = True

    xs[optical_child] = np.clip(child_x, 1.0, X_SPAN)
    ys[optical_child] = np.clip(child_y, 1.0, Y_SPAN)

    # --- Poly-G artifact blob: identical sequence, spatially clustered ------
    remaining = np.flatnonzero(mechanism == 0)
    n_polyg = min(int(args.polyg_rate * n_parent), len(remaining))
    if n_polyg:
        polyg = rng.choice(remaining, size=n_polyg, replace=False)
        templates[polyg] = 2  # 'G'
        molecule[polyg] = -1
        mechanism[polyg] = 3
        blob_slot = int(rng.integers(0, n_tiles_total))
        surface, swath, tile = layout[blob_slot]
        tiles[polyg] = tile_id(surface, swath, tile)
        surfaces[polyg] = surface
        swaths[polyg] = swath
        tile_index[polyg] = tile
        centre_x = rng.uniform(4000, X_SPAN - 4000)
        centre_y = rng.uniform(4000, Y_SPAN - 4000)
        xs[polyg] = np.clip(rng.normal(centre_x, 1400, size=n_polyg), 1, X_SPAN)
        ys[polyg] = np.clip(rng.normal(centre_y, 1400, size=n_polyg), 1, Y_SPAN)

    # --- Shared primer-like motif: seed collisions between distinct molecules -
    n_motif = int(args.shared_motif_rate * n_parent)
    if n_motif:
        motif = rng.integers(0, 4, size=32, dtype=np.int8)
        carriers = rng.choice(n_parent, size=n_motif, replace=False)
        start = rng.integers(0, max(length - 32, 1), size=n_motif)
        for row, offset in zip(carriers, start):
            templates[row, 0, offset : offset + 32] = motif

    xi = np.rint(xs).astype(np.int64)
    yi = np.rint(ys).astype(np.int64)

    # Real flowcells never report two clusters at the same tile and coordinate,
    # and the truth table is keyed on the read name, so nudge any collision the
    # sampling produced until every read name is distinct.
    seen: set[tuple[int, int, int]] = set()
    for i in range(n_parent):
        key = (int(tiles[i]), int(xi[i]), int(yi[i]))
        while key in seen:
            xi[i] += 1
            key = (int(tiles[i]), int(xi[i]), int(yi[i]))
        seen.add(key)

    qual1 = sample_quality(
        rng, n_parent, length, args.bad_tail_cycles, args.bad_tail_quality,
        args.flat_quality,
    )
    qual2 = sample_quality(
        rng, n_parent, length, args.bad_tail_cycles, args.bad_tail_quality,
        args.flat_quality,
    )
    obs1 = apply_errors(rng, templates[:, 0, :].astype(np.int64), qual1)
    obs2 = apply_errors(rng, templates[:, 1, :].astype(np.int64), qual2)

    order = rng.permutation(n_parent)
    lane = "SIMFC:1:HXXXXDRXX:1"
    names = [f"{lane}:{tiles[i]}:{xi[i]}:{yi[i]}" for i in range(n_parent)]

    r1_path = Path(str(args.out_prefix) + "_R1.fastq.gz")
    r2_path = Path(str(args.out_prefix) + "_R2.fastq.gz")
    r1_path.parent.mkdir(parents=True, exist_ok=True)
    seq1 = ["".join(row) for row in BASES[obs1]]
    seq2 = ["".join(row) for row in BASES[obs2]]
    char1 = phred_to_char(qual1).tobytes().decode("ascii")
    char2 = phred_to_char(qual2).tobytes().decode("ascii")
    with gzip.open(r1_path, "wt") as h1, gzip.open(r2_path, "wt") as h2:
        for i in order:
            span = slice(i * length, (i + 1) * length)
            h1.write(f"@{names[i]} 1:N:0:1\n{seq1[i]}\n+\n{char1[span]}\n")
            h2.write(f"@{names[i]} 2:N:0:1\n{seq2[i]}\n+\n{char2[span]}\n")

    crossed_full = np.zeros(n_parent, dtype=bool)
    crossed_full[optical_child] = crossed
    truth_path = Path(str(args.out_prefix) + ".truth.tsv.gz")
    label = {0: "unique", 1: "pcr", 2: "optical", 3: "polyg"}
    with gzip.open(truth_path, "wt") as handle:
        handle.write(
            "emitted_rank\tread_name\tmolecule\tmechanism\tparent_read_name\tcross_tile\n"
        )
        for rank, i in enumerate(order):
            parent_name = names[origin[i]] if origin[i] >= 0 else "NA"
            handle.write(
                f"{rank + 1}\t{names[i]}\t{molecule[i]}\t{label[int(mechanism[i])]}\t"
                f"{parent_name}\t{int(crossed_full[i])}\n"
            )

    summary = {
        "read_pairs": int(n_parent),
        "read_length": length,
        "tiles": n_tiles_total,
        "layout": {
            "surfaces": args.surfaces,
            "swaths": args.swaths,
            "tiles_per_swath": args.tiles,
        },
        "pcr_duplicate_children": int(n_pcr),
        "optical_children": int(n_optical),
        "optical_children_crossing_tiles": int(crossed.sum()),
        "polyg_reads": int(n_polyg),
        "shared_motif_reads": int(n_motif),
        "bad_tail_cycles": args.bad_tail_cycles,
        "flat_quality": args.flat_quality,
        "seed": args.seed,
        "r1": str(r1_path),
        "r2": str(r2_path),
        "truth": str(truth_path),
    }
    Path(str(args.out_prefix) + ".truth.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

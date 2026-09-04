#!/usr/bin/env python3
"""FastqOptiFilter: quality-aware, FDR-controlled optical-duplicate filtering.

FastqOptiFilter operates directly on synchronized paired-end Illumina FASTQ files. It
does not impose a fixed spatial-distance cutoff. Instead it:

1. retrieves sequence-similar candidate pairs with exact full-pair hashes and
   multiple exact seeds (the seeds are a computational index, not a calling
   threshold);
2. evaluates whether differences between two read pairs are compatible with
   their Phred error probabilities and calculates a quality-aware sequence
   Bayes factor;
3. decodes the Illumina tile identifier into a physical grid and places every
   read in one continuous coordinate frame per lane and surface, so proximity
   duplicates split across a tile boundary are visible and neighbouring tiles
   are searched by default;
4. tests each read once against its nearest sequence-compatible partner,
   counting neighbouring clusters around that read so the test adapts to local
   cluster density;
5. controls FDR with Benjamini-Yekutieli, valid under the dependence that
   sharing reads between hypotheses creates, and also reports Benjamini-
   Hochberg and a two-groups local-FDR mixture; and
6. retains the highest-quality read pair from each significant spatial
   component and writes synchronized filtered FASTQs.

The statistical null assumes that, in the absence of optical/proximity
duplication, sequence-compatible molecules are exchangeable over cluster
positions within each lane and surface. That assumption is not taken on trust:
every run recomputes its p-values on a permutation of the position labels and
on candidate pairs the quality model rejected as different molecules, and
reports how far each departs from uniform. ``--qq-plot`` draws them.

Requirements: Python >=3.11, numpy, scipy >=1.17, matplotlib. If pigz is available,
FastqOptiFilter uses it for parallel compression of the two output FASTQs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit
from scipy.stats import chi2, poisson_binom


VERSION = "2.0.0"
BASES = b"ACGT"
BASE_LOOKUP = np.full(256, 4, dtype=np.uint8)
for _base_index, _base in enumerate(BASES):
    BASE_LOOKUP[_base] = _base_index
    BASE_LOOKUP[ord(chr(_base).lower())] = _base_index


@dataclass(frozen=True)
class FastqRecord:
    header: str
    sequence: bytes
    plus: str
    quality: bytes


@dataclass
class LoadedReads:
    names: list[str]
    lane_labels: list[str]
    lane_ids: np.ndarray
    tiles: np.ndarray
    x: np.ndarray
    y: np.ndarray
    sequence1: list[bytes]
    sequence2: list[bytes]
    quality1: list[bytes]
    quality2: list[bytes]
    unparsed_headers: int
    read1_length: int
    read2_length: int

    @property
    def count(self) -> int:
        return len(self.names)


@dataclass
class SequenceMatrices:
    bases: np.ndarray
    qualities: np.ndarray
    base_priors: np.ndarray
    quality_sums: np.ndarray


class RunLogger:
    def __init__(self, path: Path | None, interval: float) -> None:
        self.path = path
        self.interval = interval
        self._handle: TextIO | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("w", encoding="utf-8")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def log(self, stage: str, message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"{timestamp} stage={stage} {message}"
        print(line, file=sys.stderr, flush=True)
        if self._handle is not None:
            self._handle.write(line + "\n")
            self._handle.flush()

    def tracker(self, stage: str, total: float, unit: str) -> "ProgressTracker":
        return ProgressTracker(self, stage, total, unit)


class ProgressTracker:
    def __init__(
        self, logger: RunLogger, stage: str, total: float, unit: str
    ) -> None:
        self.logger = logger
        self.stage = stage
        self.total = max(float(total), 0.0)
        self.unit = unit
        self.start = time.monotonic()
        self.last_log = -math.inf
        self.logger.log(
            self.stage,
            f"status=start total={format_number(self.total)}_{self.unit} eta=estimating",
        )

    def update(self, completed: float, detail: str = "", force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_log < self.logger.interval:
            return
        elapsed = max(now - self.start, 1e-9)
        completed_value = max(0.0, min(float(completed), self.total))
        fraction = completed_value / self.total if self.total > 0 else 1.0
        rate = completed_value / elapsed
        remaining = max(self.total - completed_value, 0.0)
        eta = remaining / rate if rate > 0 else math.inf
        eta_text = (
            "estimating"
            if elapsed < 1.0 or (self.total > 0 and fraction < 0.01)
            else format_duration(eta)
        )
        suffix = f" {detail}" if detail else ""
        self.logger.log(
            self.stage,
            "status=running "
            f"completed={format_number(completed_value)}/{format_number(self.total)}_"
            f"{self.unit} progress={100.0 * fraction:.1f}% "
            f"rate={format_number(rate)}_{self.unit}/s eta={eta_text}"
            f" elapsed={format_duration(elapsed)}{suffix}",
        )
        self.last_log = now

    def finish(self, detail: str = "") -> None:
        elapsed = time.monotonic() - self.start
        suffix = f" {detail}" if detail else ""
        self.logger.log(
            self.stage,
            f"status=complete elapsed={format_duration(elapsed)} eta=00:00{suffix}",
        )


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "estimating"
    seconds_int = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@contextmanager
def open_input(path: Path) -> Iterator[TextIO]:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
            yield handle
    else:
        with path.open("r", encoding="ascii", newline="") as handle:
            yield handle


@contextmanager
def open_output(
    path: Path, gzip_level: int, pigz_threads: int = 1
) -> Iterator[TextIO]:
    if path.suffix.lower() != ".gz":
        with path.open("w", encoding="ascii", newline="") as handle:
            yield handle
        return

    pigz = shutil.which("pigz")
    if pigz is not None and pigz_threads > 1:
        with path.open("wb") as raw_output:
            process = subprocess.Popen(
                [pigz, "-c", f"-{gzip_level}", "-p", str(pigz_threads)],
                stdin=subprocess.PIPE,
                stdout=raw_output,
            )
            assert process.stdin is not None
            writer = io.TextIOWrapper(process.stdin, encoding="ascii", newline="")
            try:
                yield writer
            finally:
                writer.close()
                return_code = process.wait()
                if return_code != 0:
                    raise OSError(
                        f"pigz exited with status {return_code} while writing {path}"
                    )
        return

    with gzip.open(
        path, "wt", encoding="ascii", newline="", compresslevel=gzip_level
    ) as handle:
        yield handle


def compressed_position(handle: TextIO, path: Path) -> int:
    if path.suffix.lower() == ".gz":
        try:
            return int(handle.buffer.fileobj.tell())  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            return 0
    try:
        return int(handle.tell())
    except OSError:
        return 0


def read_fastq_record(handle: TextIO, path: Path, record_number: int) -> FastqRecord | None:
    header = handle.readline()
    if not header:
        return None
    sequence = handle.readline()
    plus = handle.readline()
    quality = handle.readline()
    if not sequence or not plus or not quality:
        raise ValueError(f"Truncated FASTQ record {record_number:,} in {path}")
    if not header.startswith("@"):
        raise ValueError(f"Record {record_number:,} in {path} lacks an '@' header")
    if not plus.startswith("+"):
        raise ValueError(f"Record {record_number:,} in {path} lacks a '+' line")
    seq = sequence.rstrip("\r\n").encode("ascii")
    qual = quality.rstrip("\r\n").encode("ascii")
    if len(seq) != len(qual):
        raise ValueError(
            f"Sequence and quality lengths differ in record {record_number:,} of {path}"
        )
    return FastqRecord(header, seq, plus, qual)


def normalized_read_name(header: str) -> str:
    token = header[1:].split(None, 1)[0]
    if token.endswith("/1") or token.endswith("/2"):
        token = token[:-2]
    return token


def parse_location(read_name: str) -> tuple[str, int, int, int]:
    fields = read_name.split(":")
    if len(fields) < 4:
        raise ValueError("fewer than four colon-separated fields")
    try:
        tile, x, y = map(int, fields[-3:])
    except ValueError as exc:
        raise ValueError("last three header fields are not integer tile/x/y") from exc
    lane_key = ":".join(fields[:-3])
    return lane_key, tile, x, y


@dataclass(frozen=True)
class TileLayout:
    """Physical placement of every tile inside a lane.

    Illumina encodes tile identity positionally. Four-digit identifiers are
    ``<surface><swath><tile>`` (MiSeq, HiSeq, NovaSeq); five-digit identifiers
    are ``<surface><swath><camera><tile>`` (NextSeq). Swaths sit side by side
    across the lane and tiles run along it, so decoding the identifier places
    every tile on an integer grid. ``column`` and ``row`` are dense 0-based
    ranks on that grid. Surfaces are separate physical planes and are never
    treated as neighbours.
    """

    convention: str
    surface: dict[int, int]
    column: dict[int, int]
    row: dict[int, int]

    @property
    def columns(self) -> int:
        return max(self.column.values(), default=-1) + 1

    @property
    def rows(self) -> int:
        return max(self.row.values(), default=-1) + 1


def parse_tile_layout(tile_ids: Iterable[int]) -> TileLayout:
    """Decode Illumina tile identifiers into a surface/column/row grid.

    Identifiers that do not follow a recognised convention fall back to a
    single column whose rows are the sorted tile identifiers. That fallback
    keeps every tile in one spatial group, so a nearby-tile search still works,
    but the resulting adjacency is nominal rather than physical.
    """

    unique = sorted({int(value) for value in tile_ids if int(value) >= 0})
    if not unique:
        return TileLayout("empty", {}, {}, {})

    texts = {tile: str(tile) for tile in unique}
    lengths = {len(text) for text in texts.values()}
    surface: dict[int, int] = {}
    raw_cell: dict[int, tuple[int, int]] = {}

    if lengths == {4} and all(text[0] in "12" for text in texts.values()):
        convention = "illumina_surface_swath_tile"
        for tile, text in texts.items():
            surface[tile] = int(text[0])
            raw_cell[tile] = (int(text[1]), int(text[2:]))
    elif lengths == {5} and all(text[0] in "12" for text in texts.values()):
        convention = "illumina_surface_swath_camera_tile"
        for tile, text in texts.items():
            surface[tile] = int(text[0])
            # The camera segments the lane along the same axis as the tile
            # index, so (camera, tile) ordering gives the physical row order.
            raw_cell[tile] = (int(text[1]), int(text[2]) * 1000 + int(text[3:]))
    else:
        convention = "unrecognised_single_column"
        for tile in unique:
            surface[tile] = 1
            raw_cell[tile] = (1, tile)

    column_ranks = {value: rank for rank, value in enumerate(sorted({c for c, _ in raw_cell.values()}))}
    row_ranks = {value: rank for rank, value in enumerate(sorted({r for _, r in raw_cell.values()}))}
    column = {tile: column_ranks[raw_cell[tile][0]] for tile in unique}
    row = {tile: row_ranks[raw_cell[tile][1]] for tile in unique}
    return TileLayout(convention, surface, column, row)


@dataclass
class FlowcellGeometry:
    """Reads placed in one continuous coordinate frame per lane and surface.

    Every read keeps its own tile, but its coordinates are translated into a
    lane-wide frame so that two clusters on either side of a tile boundary have
    a meaningful separation. The same frame is used for the observed distances
    and for the geometry null, so the choice of inter-tile gap changes power
    but never validity: a frame that packs tiles too tightly makes boundary
    pairs look close, and makes exactly the same boundary pairs look close in
    the null.
    """

    layout: TileLayout
    group_ids: np.ndarray
    group_labels: list[str]
    tiles: np.ndarray
    columns: np.ndarray
    rows: np.ndarray
    global_x: np.ndarray
    global_y: np.ndarray
    cell_x: int
    cell_y: int
    neighborhood: str
    max_grid_step: int

    def grid_step(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Chebyshev separation of two reads in units of whole tiles."""
        return np.maximum(
            np.abs(self.columns[left] - self.columns[right]),
            np.abs(self.rows[left] - self.rows[right]),
        )

    def pairs_are_testable(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Read pairs whose tiles are close enough to carry a spatial test."""
        same_group = (self.group_ids[left] == self.group_ids[right]) & (
            self.group_ids[left] >= 0
        )
        if self.neighborhood == "lane":
            return same_group
        return same_group & (self.grid_step(left, right) <= self.testable_grid_step)

    @property
    def testable_grid_step(self) -> int:
        return 0 if self.neighborhood == "same-tile" else self.max_grid_step


def build_geometry(
    reads: LoadedReads,
    neighborhood: str,
    tile_gap: int,
    max_grid_step: int,
    logger: RunLogger,
) -> FlowcellGeometry:
    layout = parse_tile_layout(reads.tiles.tolist())
    placed = reads.lane_ids >= 0
    if not np.any(placed):
        raise ValueError("No read has parseable lane/tile/X/Y coordinates")

    x_min = int(reads.x[placed].min())
    y_min = int(reads.y[placed].min())
    cell_x = int(reads.x[placed].max()) - x_min + 1 + max(tile_gap, 0)
    cell_y = int(reads.y[placed].max()) - y_min + 1 + max(tile_gap, 0)

    columns = np.asarray(
        [layout.column.get(int(tile), 0) for tile in reads.tiles], dtype=np.int64
    )
    rows = np.asarray(
        [layout.row.get(int(tile), 0) for tile in reads.tiles], dtype=np.int64
    )
    surfaces = np.asarray(
        [layout.surface.get(int(tile), 1) for tile in reads.tiles], dtype=np.int64
    )
    global_x = columns * cell_x + (reads.x.astype(np.int64) - x_min)
    global_y = rows * cell_y + (reads.y.astype(np.int64) - y_min)

    # Optical duplication never crosses between the two physical surfaces of a
    # flowcell, so a surface is its own spatial group and its own null.
    group_labels: list[str] = []
    group_lookup: dict[tuple[int, int], int] = {}
    group_ids = np.full(reads.count, -1, dtype=np.int32)
    for index in np.flatnonzero(placed):
        key = (int(reads.lane_ids[index]), int(surfaces[index]))
        if key not in group_lookup:
            group_lookup[key] = len(group_labels)
            group_labels.append(f"{reads.lane_labels[key[0]]}:surface{key[1]}")
        group_ids[index] = group_lookup[key]

    logger.log(
        "geometry",
        f"tile_convention={layout.convention} tiles={len(layout.column)} "
        f"grid={layout.columns}x{layout.rows} cell={cell_x}x{cell_y} "
        f"neighborhood={neighborhood} spatial_groups={len(group_labels)}",
    )
    return FlowcellGeometry(
        layout=layout,
        group_ids=group_ids,
        group_labels=group_labels,
        tiles=reads.tiles.astype(np.int64),
        columns=columns,
        rows=rows,
        global_x=global_x.astype(np.float64),
        global_y=global_y.astype(np.float64),
        cell_x=cell_x,
        cell_y=cell_y,
        neighborhood=neighborhood,
        max_grid_step=max_grid_step,
    )


def load_fastqs(
    r1_path: Path,
    r2_path: Path,
    unparsed_action: str,
    logger: RunLogger,
) -> LoadedReads:
    names: list[str] = []
    lane_labels: list[str] = []
    lane_to_id: dict[str, int] = {}
    lane_ids: list[int] = []
    tiles: list[int] = []
    x_values: list[int] = []
    y_values: list[int] = []
    sequence1: list[bytes] = []
    sequence2: list[bytes] = []
    quality1: list[bytes] = []
    quality2: list[bytes] = []
    unparsed = 0
    read1_length: int | None = None
    read2_length: int | None = None
    total_bytes = r1_path.stat().st_size + r2_path.stat().st_size
    tracker = logger.tracker("load_fastq", total_bytes, "compressed_bytes")

    with open_input(r1_path) as r1_handle, open_input(r2_path) as r2_handle:
        record_number = 0
        while True:
            record_number += 1
            r1 = read_fastq_record(r1_handle, r1_path, record_number)
            r2 = read_fastq_record(r2_handle, r2_path, record_number)
            if r1 is None and r2 is None:
                break
            if r1 is None or r2 is None:
                raise ValueError("R1 and R2 have different numbers of records")
            name1 = normalized_read_name(r1.header)
            name2 = normalized_read_name(r2.header)
            if name1 != name2:
                raise ValueError(
                    f"R1/R2 names differ at pair {record_number:,}: "
                    f"'{name1}' versus '{name2}'"
                )

            if read1_length is None:
                read1_length = len(r1.sequence)
                read2_length = len(r2.sequence)
            elif len(r1.sequence) != read1_length or len(r2.sequence) != read2_length:
                raise ValueError(
                    "FastqOptiFilter currently requires fixed-length raw FASTQs; found "
                    f"lengths {len(r1.sequence)}/{len(r2.sequence)} at pair "
                    f"{record_number:,}, expected {read1_length}/{read2_length}"
                )

            try:
                lane_label, tile, x, y = parse_location(name1)
                if lane_label not in lane_to_id:
                    lane_to_id[lane_label] = len(lane_labels)
                    lane_labels.append(lane_label)
                lane_id = lane_to_id[lane_label]
            except ValueError as exc:
                if unparsed_action == "error":
                    raise ValueError(
                        f"Cannot parse coordinates at pair {record_number:,} "
                        f"('{name1}'): {exc}"
                    ) from exc
                lane_id, tile, x, y = -1, -1, 0, 0
                unparsed += 1

            names.append(name1)
            lane_ids.append(lane_id)
            tiles.append(tile)
            x_values.append(x)
            y_values.append(y)
            sequence1.append(r1.sequence)
            sequence2.append(r2.sequence)
            quality1.append(r1.quality)
            quality2.append(r2.quality)

            completed = compressed_position(r1_handle, r1_path) + compressed_position(
                r2_handle, r2_path
            )
            tracker.update(completed, detail=f"pairs={len(names):,}")

    tracker.finish(detail=f"pairs={len(names):,} unparsed={unparsed:,}")
    if not names:
        raise ValueError("The FASTQ files contain no read pairs")
    assert read1_length is not None and read2_length is not None
    return LoadedReads(
        names=names,
        lane_labels=lane_labels,
        lane_ids=np.asarray(lane_ids, dtype=np.int32),
        tiles=np.asarray(tiles, dtype=np.int32),
        x=np.asarray(x_values, dtype=np.int32),
        y=np.asarray(y_values, dtype=np.int32),
        sequence1=sequence1,
        sequence2=sequence2,
        quality1=quality1,
        quality2=quality2,
        unparsed_headers=unparsed,
        read1_length=read1_length,
        read2_length=read2_length,
    )


def seed_offsets(read_length: int, seed_length: int) -> list[int]:
    if read_length < seed_length:
        return []
    offsets = list(range(0, read_length - seed_length + 1, seed_length))
    final_offset = read_length - seed_length
    if final_offset not in offsets:
        offsets.append(final_offset)
    return sorted(offsets)


def pair_key(left: int, right: int, read_count: int) -> int:
    if left > right:
        left, right = right, left
    return left * read_count + right


def decode_pair_keys(keys: set[int], read_count: int) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.fromiter(sorted(keys), dtype=np.int64, count=len(keys))
    return ordered // read_count, ordered % read_count


def add_all_pairs(
    members: list[int],
    candidates: set[int],
    read_count: int,
    max_candidates: int,
) -> None:
    for left_position in range(len(members) - 1):
        left = members[left_position]
        for right in members[left_position + 1 :]:
            candidates.add(pair_key(left, right, read_count))
        if len(candidates) > max_candidates:
            raise MemoryError(
                f"Candidate count exceeded --max-candidates={max_candidates:,}; "
                "increase the limit or pre-filter adapter/low-complexity reads"
            )


def choose_seed_length(
    matrices: SequenceMatrices,
    read_count: int,
    maximum_seed_length: int,
    adaptive: bool,
    logger: RunLogger,
) -> tuple[int, dict[str, object]]:
    """Pick a seed length the base qualities can actually support.

    Candidate retrieval is exact: two copies of one molecule are only found if
    some seed window is error-free in both of them. How often that happens is
    decided entirely by the base qualities, which is why the quality model
    belongs here and not only in the filter that follows. Reading Phred scores
    only after retrieval means a pair the quality model would happily accept --
    scattered errors, every one of them expected given the reported qualities
    -- was already discarded, and no downstream test can bring it back.

    From the reported qualities, the expected number of disagreeing cycles
    between two copies of the same molecule is

        E = sum over cycles of (e_L + e_R - (4/3) e_L e_R)

    Seeds are disjoint, so ``cycles / L`` of them tile the pair. Choosing
    ``L = cycles / (E + 1)`` keeps more windows than expected mismatches, which
    is the pigeonhole condition for at least one window to survive.

    The length is floored so that a seed still discriminates: with ``4**L``
    distinct keys and ``n`` reads, ``L = log4(n)`` leaves about one read per
    key, and anything shorter turns the index into noise. The requested length
    is the ceiling, so good data is never given shorter seeds than asked for.
    """
    error = np.power(10.0, -matrices.qualities.astype(np.float64) / 10.0)
    mean_error = np.clip(error.mean(axis=0), 0.0, 0.75)
    per_cycle_mismatch = 2.0 * mean_error - (4.0 / 3.0) * mean_error**2
    expected_mismatches = float(per_cycle_mismatch.sum())
    cycles = int(matrices.qualities.shape[1])
    discriminating = max(8, math.ceil(math.log2(max(read_count, 2)) / 2.0))
    supported = int(cycles / (expected_mismatches + 1.0))
    chosen = maximum_seed_length
    if adaptive:
        chosen = int(min(maximum_seed_length, max(discriminating, supported)))
    details = {
        "adaptive": adaptive,
        "requested_maximum": maximum_seed_length,
        "expected_mismatches_between_copies": expected_mismatches,
        "mean_reported_error_rate": float(mean_error.mean()),
        "length_supported_by_quality": supported,
        "length_needed_to_discriminate": discriminating,
        "selected": chosen,
    }
    logger.log(
        "seed_length",
        f"selected={chosen} requested_max={maximum_seed_length} "
        f"expected_mismatches_between_copies={expected_mismatches:.3f} "
        f"quality_supported={supported} discriminating_floor={discriminating}",
    )
    if adaptive and supported < discriminating:
        logger.log(
            "seed_length",
            "warning=base qualities imply more mismatches than any usable seed "
            "length can tolerate; candidate retrieval will miss real duplicates",
        )
    return chosen, details


def find_candidates(
    reads: LoadedReads,
    seed_length: int,
    max_seed_bucket: int,
    max_exact_family: int,
    max_candidates: int,
    logger: RunLogger,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    lane_indices: dict[int, list[int]] = defaultdict(list)
    for index, lane_id in enumerate(reads.lane_ids):
        if lane_id >= 0:
            lane_indices[int(lane_id)].append(index)

    offsets1 = seed_offsets(reads.read1_length, seed_length)
    offsets2 = seed_offsets(reads.read2_length, seed_length)
    total_steps = len(lane_indices) * (1 + len(offsets1) + len(offsets2))
    tracker = logger.tracker("candidate_search", total_steps, "index_passes")
    candidates: set[int] = set()
    skipped_seed_buckets = 0
    skipped_seed_members = 0
    exact_families = 0
    exact_pair_relations = 0
    completed_steps = 0

    for lane_id, indices in sorted(lane_indices.items()):
        exact_groups: dict[bytes, list[int]] = defaultdict(list)
        for index in indices:
            digest = hashlib.blake2b(digest_size=20)
            digest.update(reads.sequence1[index])
            digest.update(b"\x00")
            digest.update(reads.sequence2[index])
            exact_groups[digest.digest()].append(index)
        for members in exact_groups.values():
            if len(members) < 2:
                continue
            if len(members) > max_exact_family:
                raise ValueError(
                    f"An exact-sequence family contains {len(members):,} pairs, "
                    f"above --max-exact-family={max_exact_family:,}. This usually "
                    "indicates adapter/low-complexity artifacts."
                )
            exact_families += 1
            exact_pair_relations += len(members) * (len(members) - 1) // 2
            add_all_pairs(members, candidates, reads.count, max_candidates)
        completed_steps += 1
        tracker.update(
            completed_steps,
            detail=f"candidates={len(candidates):,} lane={reads.lane_labels[lane_id]}",
        )

        for mate_number, sequences, offsets in (
            (1, reads.sequence1, offsets1),
            (2, reads.sequence2, offsets2),
        ):
            for offset in offsets:
                buckets: dict[bytes, list[int]] = defaultdict(list)
                stop = offset + seed_length
                for index in indices:
                    buckets[sequences[index][offset:stop]].append(index)
                for members in buckets.values():
                    if len(members) < 2:
                        continue
                    if len(members) > max_seed_bucket:
                        skipped_seed_buckets += 1
                        skipped_seed_members += len(members)
                        continue
                    add_all_pairs(members, candidates, reads.count, max_candidates)
                completed_steps += 1
                tracker.update(
                    completed_steps,
                    detail=(
                        f"candidates={len(candidates):,} mate=R{mate_number} "
                        f"offset={offset}"
                    ),
                )

    tracker.finish(
        detail=(
            f"candidate_relations={len(candidates):,} "
            f"skipped_seed_buckets={skipped_seed_buckets:,}"
        )
    )
    edge_left, edge_right = decode_pair_keys(candidates, reads.count)
    details = {
        "seed_length": seed_length,
        "r1_seed_offsets": offsets1,
        "r2_seed_offsets": offsets2,
        "exact_sequence_families": exact_families,
        "exact_sequence_pair_relations": exact_pair_relations,
        "retrieved_candidate_pair_relations": len(candidates),
        "skipped_high_frequency_seed_buckets": skipped_seed_buckets,
        "members_in_skipped_seed_buckets_with_multiplicity": skipped_seed_members,
        "max_seed_bucket": max_seed_bucket,
        "max_exact_family": max_exact_family,
    }
    return edge_left, edge_right, details


def encode_matrices(reads: LoadedReads, logger: RunLogger) -> SequenceMatrices:
    tracker = logger.tracker("encode_sequences", 6, "steps")
    count = reads.count

    raw1 = np.frombuffer(b"".join(reads.sequence1), dtype=np.uint8).reshape(
        count, reads.read1_length
    )
    bases1 = BASE_LOOKUP[raw1]
    tracker.update(1, detail="R1_bases", force=True)
    raw2 = np.frombuffer(b"".join(reads.sequence2), dtype=np.uint8).reshape(
        count, reads.read2_length
    )
    bases2 = BASE_LOOKUP[raw2]
    tracker.update(2, detail="R2_bases", force=True)
    bases = np.concatenate((bases1, bases2), axis=1)
    del raw1, raw2, bases1, bases2

    raw_quality1 = np.frombuffer(b"".join(reads.quality1), dtype=np.uint8).reshape(
        count, reads.read1_length
    )
    quality1 = np.clip(raw_quality1.astype(np.int16) - 33, 0, 60).astype(np.uint8)
    tracker.update(3, detail="R1_qualities", force=True)
    raw_quality2 = np.frombuffer(b"".join(reads.quality2), dtype=np.uint8).reshape(
        count, reads.read2_length
    )
    quality2 = np.clip(raw_quality2.astype(np.int16) - 33, 0, 60).astype(np.uint8)
    tracker.update(4, detail="R2_qualities", force=True)
    qualities = np.concatenate((quality1, quality2), axis=1)
    quality_sums = qualities.sum(axis=1, dtype=np.int64)
    del raw_quality1, raw_quality2, quality1, quality2

    base_priors = np.empty((bases.shape[1], 4), dtype=np.float64)
    for cycle in range(bases.shape[1]):
        counts = np.bincount(bases[:, cycle], minlength=5)[:4].astype(np.float64)
        counts += 1.0  # Dirichlet(1,1,1,1) smoothing.
        base_priors[cycle] = counts / counts.sum()
    tracker.update(5, detail="empirical_cycle_base_priors", force=True)
    tracker.update(6, detail="matrices_ready", force=True)
    tracker.finish(
        detail=f"shape={bases.shape[0]:,}x{bases.shape[1]:,} bases"
    )
    return SequenceMatrices(
        bases=bases,
        qualities=qualities,
        base_priors=base_priors,
        quality_sums=quality_sums,
    )


def score_candidate_block(
    start: int,
    stop: int,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    matrices: SequenceMatrices,
) -> tuple[int, int, dict[str, np.ndarray]]:
    left = edge_left[start:stop]
    right = edge_right[start:stop]
    left_bases = matrices.bases[left]
    right_bases = matrices.bases[right]
    left_q = matrices.qualities[left]
    right_q = matrices.qualities[right]

    left_error = np.minimum(0.75, np.power(10.0, -left_q.astype(np.float64) / 10.0))
    right_error = np.minimum(
        0.75, np.power(10.0, -right_q.astype(np.float64) / 10.0)
    )
    valid = (left_bases < 4) & (right_bases < 4)
    mismatch_mask = valid & (left_bases != right_bases)
    mismatches = mismatch_mask.sum(axis=1, dtype=np.int32)
    valid_bases = valid.sum(axis=1, dtype=np.int32)

    same_call_probability = np.zeros(left_bases.shape, dtype=np.float64)
    left_marginal = np.zeros(left_bases.shape, dtype=np.float64)
    right_marginal = np.zeros(left_bases.shape, dtype=np.float64)
    for true_base in range(4):
        prior = matrices.base_priors[:, true_base][None, :]
        left_call = np.where(
            left_bases == true_base, 1.0 - left_error, left_error / 3.0
        )
        right_call = np.where(
            right_bases == true_base, 1.0 - right_error, right_error / 3.0
        )
        same_call_probability += prior * left_call * right_call
        left_marginal += prior * left_call
        right_marginal += prior * right_call

    independent_probability = left_marginal * right_marginal
    ratio = np.divide(
        same_call_probability,
        independent_probability,
        out=np.ones_like(same_call_probability),
        where=valid & (independent_probability > 0),
    )
    log10_bayes_factor = np.where(valid, np.log10(np.maximum(ratio, 1e-300)), 0.0).sum(
        axis=1
    )

    mismatch_probability = (
        left_error + right_error - (4.0 / 3.0) * left_error * right_error
    )
    mismatch_probability = np.where(
        valid, np.clip(mismatch_probability, 1e-300, 1.0), 0.0
    )
    expected_mismatches = mismatch_probability.sum(axis=1)
    # Exact Poisson-binomial upper tail: each cycle has its own mismatch
    # probability derived from the two Phred qualities.
    compatibility_p = poisson_binom.sf(mismatches - 1, mismatch_probability)
    compatibility_p = np.clip(compatibility_p, 1e-300, 1.0)
    mismatch_surprisal = np.where(
        mismatch_mask, -np.log10(mismatch_probability), 0.0
    ).sum(axis=1)

    return start, stop, {
        "mismatches": mismatches,
        "valid_bases": valid_bases,
        "expected_mismatches": expected_mismatches,
        "compatibility_p": compatibility_p,
        "log10_bayes_factor": log10_bayes_factor,
        "mismatch_surprisal": mismatch_surprisal,
    }


def score_candidates(
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    matrices: SequenceMatrices,
    threads: int,
    chunk_size: int,
    logger: RunLogger,
) -> dict[str, np.ndarray]:
    candidate_count = len(edge_left)
    outputs = {
        "mismatches": np.empty(candidate_count, dtype=np.int32),
        "valid_bases": np.empty(candidate_count, dtype=np.int32),
        "expected_mismatches": np.empty(candidate_count, dtype=np.float64),
        "compatibility_p": np.empty(candidate_count, dtype=np.float64),
        "log10_bayes_factor": np.empty(candidate_count, dtype=np.float64),
        "mismatch_surprisal": np.empty(candidate_count, dtype=np.float64),
    }
    tracker = logger.tracker("quality_model", candidate_count, "candidate_pairs")
    ranges = [
        (start, min(start + chunk_size, candidate_count))
        for start in range(0, candidate_count, chunk_size)
    ]
    completed = 0
    if threads == 1:
        iterator = (
            score_candidate_block(start, stop, edge_left, edge_right, matrices)
            for start, stop in ranges
        )
        for start, stop, block in iterator:
            for key, values in block.items():
                outputs[key][start:stop] = values
            completed += stop - start
            tracker.update(completed)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(
                    score_candidate_block,
                    start,
                    stop,
                    edge_left,
                    edge_right,
                    matrices,
                )
                for start, stop in ranges
            ]
            for future in concurrent.futures.as_completed(futures):
                start, stop, block = future.result()
                for key, values in block.items():
                    outputs[key][start:stop] = values
                completed += stop - start
                tracker.update(completed)
    tracker.finish(detail=f"candidate_pairs={candidate_count:,} threads={threads}")
    return outputs


def estimate_same_template_prior(
    log10_bayes_factor: np.ndarray,
    sequence_compatible: np.ndarray,
    total_possible_relations: int,
) -> tuple[float, np.ndarray, int]:
    if total_possible_relations <= 0:
        return 0.0, np.zeros_like(log10_bayes_factor), 0
    effective_log_bf = np.where(
        sequence_compatible,
        np.clip(log10_bayes_factor * math.log(10.0), -700.0, 700.0),
        -math.inf,
    )
    initial_count = max(int(sequence_compatible.sum()), 1)
    prior = min(0.1, initial_count / total_possible_relations)
    iterations = 0
    posterior = np.zeros_like(log10_bayes_factor)
    for iterations in range(1, 501):
        prior = min(max(prior, 1.0 / total_possible_relations), 1.0 - 1e-12)
        log_prior_odds = math.log(prior) - math.log1p(-prior)
        posterior = expit(log_prior_odds + effective_log_bf)
        updated = float(posterior.sum()) / total_possible_relations
        if abs(updated - prior) <= max(1e-14, prior * 1e-8):
            prior = updated
            break
        prior = updated
    return prior, posterior, iterations


def radius_grid(max_radius: float, resolution: int) -> np.ndarray:
    """Radii at which the geometry null is evaluated.

    Every integer radius up to 256 is kept, because optical duplication lives
    at that scale and the coordinates are integers. Larger radii are spaced
    logarithmically out to the largest separation a testable pair can have, so
    the tabulated null is a complete CDF rather than one truncated at the
    largest distance that happened to be observed. Lookups round an observed
    distance *up* to the next grid radius, which can only enlarge the reported
    p-value, so a coarse grid stays conservative rather than anti-conservative.
    """
    if not math.isfinite(max_radius) or max_radius < 0:
        return np.zeros(0, dtype=np.float64)
    dense = np.arange(0.0, min(max_radius, 256.0) + 1.0)
    if max_radius <= 256.0:
        grid = dense
    else:
        sparse = np.geomspace(257.0, max_radius, max(resolution, 16))
        grid = np.concatenate((dense, sparse, [max_radius]))
    return np.unique(np.round(grid, 6))


def maximum_testable_distance(
    geometry: FlowcellGeometry, group_id: int, p_norm: float
) -> float:
    """Largest separation two reads in a testable tile relation can have."""
    members = np.flatnonzero(geometry.group_ids == group_id)
    if len(members) < 2:
        return 0.0
    if geometry.neighborhood == "lane":
        span_x = float(np.ptp(geometry.global_x[members]))
        span_y = float(np.ptp(geometry.global_y[members]))
    else:
        step = geometry.testable_grid_step
        span_x = float((step + 1) * geometry.cell_x)
        span_y = float((step + 1) * geometry.cell_y)
    return max(span_x, span_y) if math.isinf(p_norm) else math.hypot(span_x, span_y)


def pair_counts_within(
    points_a: np.ndarray,
    points_b: np.ndarray | None,
    radii: np.ndarray,
    p_norm: float,
) -> np.ndarray:
    """Unordered pairs at distance <= r, within one point set or across two."""
    if len(radii) == 0 or len(points_a) == 0:
        return np.zeros(len(radii), dtype=np.float64)
    tree_a = cKDTree(points_a)
    if points_b is None:
        ordered_with_self = tree_a.count_neighbors(
            tree_a, radii, p=p_norm, cumulative=True
        ).astype(np.float64)
        return (ordered_with_self - len(points_a)) / 2.0
    if len(points_b) == 0:
        return np.zeros(len(radii), dtype=np.float64)
    # Distinct point sets: every unordered cross pair is counted exactly once.
    return tree_a.count_neighbors(
        cKDTree(points_b), radii, p=p_norm, cumulative=True
    ).astype(np.float64)


def pair_distances(
    geometry: FlowcellGeometry,
    left: np.ndarray,
    right: np.ndarray,
    p_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Global-frame distances for read pairs, and which pairs are testable."""
    testable = geometry.pairs_are_testable(left, right)
    distances = np.full(len(left), np.inf, dtype=np.float64)
    delta_x = np.abs(geometry.global_x[left] - geometry.global_x[right])
    delta_y = np.abs(geometry.global_y[left] - geometry.global_y[right])
    if math.isinf(p_norm):
        distances[testable] = np.maximum(delta_x[testable], delta_y[testable])
    else:
        distances[testable] = np.hypot(delta_x[testable], delta_y[testable])
    return distances, testable


def build_group_null(
    reads: LoadedReads,
    geometry: FlowcellGeometry,
    group_id: int,
    radii: np.ndarray,
    p_norm: float,
) -> list[tuple[int, int, np.ndarray, np.ndarray | None, np.ndarray, float]]:
    """Tile-level counting tasks whose sum is the exact group-wide null.

    Non-adjacent tile pairs are skipped only when their global bounding boxes
    are provably further apart than the largest radius, so the sum over the
    returned tasks equals the number of testable pairs within each radius.
    """
    member_indices = np.flatnonzero(geometry.group_ids == group_id)
    if not len(member_indices) or not len(radii):
        return []
    max_radius = float(radii[-1])
    tiles = geometry.tiles[member_indices]
    tile_values = sorted({int(value) for value in tiles})
    tile_points: dict[int, np.ndarray] = {}
    tile_box: dict[int, tuple[float, float, float, float]] = {}
    for tile in tile_values:
        selected = member_indices[tiles == tile]
        points = np.column_stack(
            (geometry.global_x[selected], geometry.global_y[selected])
        )
        tile_points[tile] = points
        tile_box[tile] = (
            float(points[:, 0].min()),
            float(points[:, 0].max()),
            float(points[:, 1].min()),
            float(points[:, 1].max()),
        )

    tasks: list[tuple[int, int, np.ndarray, np.ndarray | None, np.ndarray, float]] = []
    column = geometry.layout.column
    row = geometry.layout.row
    step_limit = geometry.testable_grid_step
    for position, tile in enumerate(tile_values):
        tasks.append((group_id, tile, tile_points[tile], None, radii, p_norm))
        for other in tile_values[position + 1 :]:
            if geometry.neighborhood != "lane":
                step = max(
                    abs(column.get(tile, 0) - column.get(other, 0)),
                    abs(row.get(tile, 0) - row.get(other, 0)),
                )
                if step > step_limit:
                    continue
            a = tile_box[tile]
            b = tile_box[other]
            gap_x = max(0.0, max(a[0] - b[1], b[0] - a[1]))
            gap_y = max(0.0, max(a[2] - b[3], b[2] - a[3]))
            separation = (
                max(gap_x, gap_y) if math.isinf(p_norm) else math.hypot(gap_x, gap_y)
            )
            if separation > max_radius:
                continue
            tasks.append(
                (group_id, tile, tile_points[tile], tile_points[other], radii, p_norm)
            )
    return tasks


def testable_relation_count(
    reads: LoadedReads, geometry: FlowcellGeometry, group_id: int
) -> int:
    """Number of read pairs in a group whose tiles permit a spatial test."""
    member_indices = np.flatnonzero(geometry.group_ids == group_id)
    tiles = geometry.tiles[member_indices]
    counts = Counter(int(value) for value in tiles)
    total = sum(count * (count - 1) // 2 for count in counts.values())
    if geometry.neighborhood == "same-tile":
        return total
    tile_values = sorted(counts)
    column = geometry.layout.column
    row = geometry.layout.row
    step_limit = geometry.testable_grid_step
    for position, tile in enumerate(tile_values):
        for other in tile_values[position + 1 :]:
            if geometry.neighborhood != "lane":
                step = max(
                    abs(column.get(tile, 0) - column.get(other, 0)),
                    abs(row.get(tile, 0) - row.get(other, 0)),
                )
                if step > step_limit:
                    continue
            total += counts[tile] * counts[other]
    return total


def spatial_null_lookup(
    model: dict[str, object], distances: np.ndarray
) -> np.ndarray:
    """Null CDF at each distance, rounding up to the next evaluated radius."""
    radii = np.asarray(model["radii"], dtype=np.float64)
    null_cdf = np.asarray(model["null_cdf"], dtype=np.float64)
    result = np.ones(len(distances), dtype=np.float64)
    if not len(radii):
        return result
    finite = np.isfinite(distances)
    if not np.any(finite):
        return result
    positions = np.searchsorted(radii, distances[finite], side="left")
    positions = np.clip(positions, 0, len(radii) - 1)
    result[finite] = null_cdf[positions]
    return result


def calibrate_spatial_pvalues(
    reads: LoadedReads,
    geometry: FlowcellGeometry,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    eligible_indices: np.ndarray,
    metric: str,
    null_resolution: int,
    threads: int,
    logger: RunLogger,
    tabulate_null: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, object]]]:
    """Spatial p-values from the observed flowcell geometry.

    The p-value of a pair at distance ``d`` is the probability that a random
    unordered pair of clusters from the same lane and surface both sits in a
    testable tile relation and is no further apart than ``d``. Only the
    definition of "testable" changes with ``--tile-neighborhood``; the observed
    statistic and the null are always built in the same coordinate frame, so
    the p-value stays calibrated whichever neighbourhood is chosen.

    Counting neighbours over every tile relation at every radius is the most
    expensive stage in the run, and the read-level test does not need it.
    ``tabulate_null`` skips it, still returning the distances, the testable
    mask and the occupancy counts that the report summarises, but leaving the
    tabulated null empty and every edge p-value at one.
    """
    tested_left = edge_left[eligible_indices]
    tested_right = edge_right[eligible_indices]
    tested_count = len(eligible_indices)
    p_norm = math.inf if metric == "chebyshev" else 2.0

    distances, testable = pair_distances(geometry, tested_left, tested_right, p_norm)
    spatial_p = np.ones(tested_count, dtype=np.float64)

    group_ids = sorted({int(value) for value in geometry.group_ids if value >= 0})
    group_models: dict[int, dict[str, object]] = {}
    tasks: list[tuple[int, int, np.ndarray, np.ndarray | None, np.ndarray, float]] = []
    for group_id in group_ids:
        member_count = int(np.count_nonzero(geometry.group_ids == group_id))
        group_mask = geometry.group_ids[tested_left] == group_id
        group_testable = group_mask & testable
        radii = (
            radius_grid(
                maximum_testable_distance(geometry, group_id, p_norm), null_resolution
            )
            if tabulate_null
            else np.zeros(0, dtype=np.float64)
        )
        group_models[group_id] = {
            "radii": radii,
            "null_counts": np.zeros(len(radii), dtype=np.float64),
            "read_pairs": member_count,
            "total_random_pair_relations": member_count * (member_count - 1) // 2,
            "testable_random_pair_relations": testable_relation_count(
                reads, geometry, group_id
            ),
            "tested_hypotheses": int(group_mask.sum()),
            "testable_tested_hypotheses": int(group_testable.sum()),
        }
        if tabulate_null:
            tasks.extend(build_group_null(reads, geometry, group_id, radii, p_norm))

    tracker = logger.tracker("spatial_null", max(len(tasks), 1), "tile_relations")
    completed = 0
    if threads == 1:
        for group_id, tile, points_a, points_b, radii, norm in tasks:
            group_models[group_id]["null_counts"] += pair_counts_within(
                points_a, points_b, radii, norm
            )
            completed += 1
            tracker.update(completed, detail=f"group={group_id} tile={tile}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            future_metadata = {
                executor.submit(pair_counts_within, points_a, points_b, radii, norm): (
                    group_id,
                    tile,
                )
                for group_id, tile, points_a, points_b, radii, norm in tasks
            }
            for future in concurrent.futures.as_completed(future_metadata):
                group_id, tile = future_metadata[future]
                group_models[group_id]["null_counts"] += future.result()
                completed += 1
                tracker.update(completed, detail=f"group={group_id} tile={tile}")

    for group_id, model in group_models.items():
        total_pairs = int(model["total_random_pair_relations"])
        counts = np.asarray(model["null_counts"], dtype=np.float64)
        model["null_cdf"] = (counts + 1.0) / (total_pairs + 1.0)
        model["testable_probability"] = (
            int(model["testable_random_pair_relations"]) / total_pairs
            if total_pairs
            else 0.0
        )
        positions = np.flatnonzero(
            (geometry.group_ids[tested_left] == group_id) & testable
        )
        if len(positions):
            spatial_p[positions] = spatial_null_lookup(model, distances[positions])

    tracker.finish(
        detail=(
            f"tested_hypotheses={tested_count:,} "
            f"testable={int(testable.sum()):,} tile_relations={len(tasks):,}"
        )
    )
    return spatial_p, distances, testable, group_models


@dataclass
class ReadLevelTest:
    """One spatial hypothesis per read instead of one per candidate edge."""

    read_index: np.ndarray
    partner_index: np.ndarray
    nearest_distance: np.ndarray
    partner_count: np.ndarray
    neighbours_within: np.ndarray
    pvalue: np.ndarray
    supporting_edge: np.ndarray


def nearest_partner_summary(
    read_count: int,
    left: np.ndarray,
    right: np.ndarray,
    distances: np.ndarray,
    testable: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per read: how many candidate partners it has and which is closest."""
    both_reads = np.concatenate((left, right))
    both_partners = np.concatenate((right, left))
    both_edges = np.concatenate((np.arange(len(left)), np.arange(len(left))))
    both_distances = np.concatenate((distances, distances))
    both_testable = np.concatenate((testable, testable))

    # Every sequence-compatible partner counts towards the multiplicity, even
    # one on a tile too far away to test: under the null that partner could
    # equally well have landed nearby, and conditioning on the ones that did
    # would understate how many chances the read had.
    partner_count = np.bincount(both_reads, minlength=read_count)

    nearest_distance = np.full(read_count, np.inf, dtype=np.float64)
    usable = np.flatnonzero(both_testable & np.isfinite(both_distances))
    np.minimum.at(nearest_distance, both_reads[usable], both_distances[usable])

    partner_index = np.full(read_count, -1, dtype=np.int64)
    supporting_edge = np.full(read_count, -1, dtype=np.int64)
    if len(usable):
        # Sorting by (read, distance) puts each read's closest partner first.
        order = usable[np.lexsort((both_distances[usable], both_reads[usable]))]
        sorted_reads = both_reads[order]
        first = np.concatenate(([True], sorted_reads[1:] != sorted_reads[:-1]))
        winners = order[first]
        partner_index[both_reads[winners]] = both_partners[winners]
        supporting_edge[both_reads[winners]] = both_edges[winners]
    return partner_count, nearest_distance, partner_index, supporting_edge, both_reads


def exact_nearest_partner_pvalue(
    neighbours: np.ndarray, partners: np.ndarray, population: np.ndarray
) -> np.ndarray:
    """P(at least one of m exchangeable partners lands this close), exactly.

    Given the read's own position and the observed cluster pattern, the null
    places its ``m`` sequence-compatible partners uniformly among the other
    ``M`` reads of the lane and surface, without replacement. If ``K`` of those
    reads sit within the observed nearest-partner distance, the probability
    that none of the ``m`` partners falls that close is ``C(M-K, m) / C(M, m)``.

    Counting neighbours around the read itself, rather than reading a
    lane-averaged curve, makes the test adapt to local cluster density: the
    same separation is unremarkable in a crowded patch of the tile and
    surprising in a sparse one.

    The ratio is accumulated term by term as ``sum(log1p(-K / (M - j)))``
    rather than from four log-gamma terms. On a full NovaSeq lane surface,
    ``lgamma(M + 1)`` is of order 1e9 while the quantity being extracted from
    it is of order 1e-8, so the log-gamma form loses the entire answer to
    cancellation: at M = 1e8 it returns exactly zero, which would be reported
    as an infinitely significant duplicate. The term-by-term sum costs one
    element per read-partner incidence, which is the size of the candidate
    edge list that already exists.
    """
    neighbours = np.asarray(neighbours, dtype=np.float64)
    partners = np.asarray(partners, dtype=np.float64)
    population = np.asarray(population, dtype=np.float64)
    result = np.ones(len(neighbours), dtype=np.float64)
    feasible = (
        (partners > 0)
        & (population > 0)
        & (population - neighbours - partners >= 0)
        & np.isfinite(neighbours)
    )
    if not np.any(feasible):
        return result

    counts = partners[feasible].astype(np.int64)
    owner = np.repeat(np.arange(len(counts)), counts)
    # Position of each factor inside its own read's product.
    step = np.arange(len(owner), dtype=np.int64) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    denominator = population[feasible][owner] - step
    ratio = np.clip(neighbours[feasible][owner] / denominator, 0.0, 1.0)
    log_none = np.bincount(
        owner, weights=np.log1p(-ratio), minlength=len(counts)
    )
    result[feasible] = -np.expm1(np.minimum(log_none, 0.0))
    return np.clip(result, 0.0, 1.0)


def neighbour_block_of_read(geometry: FlowcellGeometry) -> np.ndarray:
    """The population a read's partners could have landed in.

    Its own tile when only same-tile relations are testable, otherwise its
    whole lane and surface. Densely ranked rather than arithmetically combined:
    a tile identifier is whatever integer the read header carried, so
    ``group * constant + tile`` can alias one group onto the next.
    """
    if geometry.neighborhood != "same-tile":
        return geometry.group_ids.astype(np.int64)
    return np.unique(
        np.column_stack((geometry.group_ids.astype(np.int64), geometry.tiles)),
        axis=0,
        return_inverse=True,
    )[1].reshape(-1)


def build_neighbour_trees(
    geometry: FlowcellGeometry, logger: RunLogger
) -> dict[int, cKDTree]:
    """One k-d tree per block, reusable across permutations.

    Permuting reads over positions moves each read's coordinates and its tile
    together, so the set of positions inside any block is exactly what it was.
    The trees are therefore identical run to run and only the query points and
    radii change, which is what makes a permutation null affordable.
    """
    block_of_read = neighbour_block_of_read(geometry)
    trees: dict[int, cKDTree] = {}
    for key in np.unique(block_of_read):
        members = np.flatnonzero(block_of_read == key)
        if len(members) < 2:
            continue
        trees[int(key)] = cKDTree(
            np.column_stack((geometry.global_x[members], geometry.global_y[members]))
        )
    logger.log("neighbour_index", f"blocks={len(trees):,}")
    return trees


def count_spatial_neighbours(
    geometry: FlowcellGeometry,
    read_index: np.ndarray,
    radii: np.ndarray,
    metric: str,
    logger: RunLogger,
    trees: dict[int, cKDTree] | None = None,
    quiet: bool = False,
) -> np.ndarray:
    """Reads within each given radius of each given read, excluding itself.

    For the ``same-tile`` neighbourhood the count is taken inside the read's
    own tile. Otherwise it is taken over the whole lane and surface in the
    global frame, which is exact whenever the radius is smaller than one tile
    cell -- the only reads it could wrongly include sit at least two cells away
    -- and conservative beyond that, because an over-count can only enlarge the
    p-value.
    """
    p_norm = math.inf if metric == "chebyshev" else 2.0
    counts = np.zeros(len(read_index), dtype=np.float64)
    if not len(read_index):
        return counts

    block_of_read = neighbour_block_of_read(geometry)
    query_blocks = block_of_read[read_index]
    order = np.argsort(query_blocks, kind="stable")
    boundaries = np.flatnonzero(
        np.concatenate(([True], query_blocks[order][1:] != query_blocks[order][:-1]))
    )
    block_starts = np.append(boundaries, len(order))

    tracker = (
        None
        if quiet
        else logger.tracker("neighbour_counts", len(boundaries), "spatial_blocks")
    )
    done = 0
    for block_number in range(len(boundaries)):
        positions = order[block_starts[block_number] : block_starts[block_number + 1]]
        key = int(query_blocks[positions[0]])
        if trees is not None:
            tree = trees.get(key)
            if tree is None:
                done += 1
                continue
        else:
            members = np.flatnonzero(block_of_read == key)
            if len(members) < 2:
                done += 1
                if tracker is not None:
                    tracker.update(done)
                continue
            tree = cKDTree(
                np.column_stack(
                    (geometry.global_x[members], geometry.global_y[members])
                )
            )
        query_points = np.column_stack(
            (
                geometry.global_x[read_index[positions]],
                geometry.global_y[read_index[positions]],
            )
        )
        found = tree.query_ball_point(
            query_points, radii[positions], p=p_norm, return_length=True
        )
        counts[positions] = np.maximum(np.asarray(found, dtype=np.float64) - 1.0, 0.0)
        done += 1
        if tracker is not None:
            tracker.update(done, detail=f"block={key} reads={len(positions):,}")
    if tracker is not None:
        tracker.finish(detail=f"reads_tested={len(read_index):,}")
    return counts


def read_level_spatial_test(
    reads: LoadedReads,
    geometry: FlowcellGeometry,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    distances: np.ndarray,
    testable: np.ndarray,
    metric: str,
    logger: RunLogger,
    trees: dict[int, cKDTree] | None = None,
    quiet: bool = False,
) -> ReadLevelTest:
    """Test each read once: is its closest look-alike closer than chance?

    Testing every candidate edge lets a single family of ``k`` sequence-
    identical reads raise ``k(k-1)/2`` hypotheses over only ``k`` reads. One
    low-complexity cluster then supplies most of the hypotheses in the run, and
    it dominates both the multiple-testing correction and any estimate of how
    many hypotheses are null. Reads are also what actually get removed, so
    testing reads makes the controlled error rate the one that matters.
    """
    (
        partner_count,
        nearest_distance,
        partner_index,
        supporting_edge,
        _,
    ) = nearest_partner_summary(
        reads.count, edge_left, edge_right, distances, testable
    )
    # Every read with a sequence-compatible partner is a hypothesis, including
    # one whose partners all landed on tiles too far apart to compare. Keeping
    # only the reads that happen to have a testable partner would condition the
    # test on its own outcome and make the p-values anti-conservative, because
    # exactly the reads that would have scored one are the ones dropped.
    tested = np.flatnonzero(partner_count > 0)
    if not len(tested):
        empty_int = np.zeros(0, dtype=np.int64)
        empty_float = np.zeros(0, dtype=np.float64)
        return ReadLevelTest(
            empty_int, empty_int, empty_float, empty_int, empty_float, empty_float, empty_int
        )

    finite = np.isfinite(nearest_distance[tested])
    neighbours = np.full(len(tested), np.inf, dtype=np.float64)
    if np.any(finite):
        neighbours[finite] = count_spatial_neighbours(
            geometry,
            tested[finite],
            nearest_distance[tested][finite],
            metric,
            logger,
            trees=trees,
            quiet=quiet,
        )
    group_sizes = {
        group: int(np.count_nonzero(geometry.group_ids == group))
        for group in {int(v) for v in geometry.group_ids[tested]}
    }
    population = np.asarray(
        [group_sizes[int(geometry.group_ids[index])] - 1 for index in tested],
        dtype=np.float64,
    )
    pvalue = exact_nearest_partner_pvalue(
        neighbours, partner_count[tested].astype(np.float64), population
    )
    observed = nearest_distance[tested][finite]
    if quiet:
        return ReadLevelTest(
            read_index=tested,
            partner_index=partner_index[tested],
            nearest_distance=nearest_distance[tested],
            partner_count=partner_count[tested],
            neighbours_within=neighbours,
            pvalue=pvalue,
            supporting_edge=supporting_edge[tested],
        )
    logger.log(
        "read_test",
        f"reads_tested={len(tested):,} with_testable_partner={int(finite.sum()):,} "
        f"median_nearest_distance="
        f"{float(np.median(observed)) if len(observed) else float('nan'):.6g} "
        f"median_partner_count={float(np.median(partner_count[tested])):.6g}",
    )
    return ReadLevelTest(
        read_index=tested,
        partner_index=partner_index[tested],
        nearest_distance=nearest_distance[tested],
        partner_count=partner_count[tested],
        neighbours_within=neighbours,
        pvalue=pvalue,
        supporting_edge=supporting_edge[tested],
    )


def shuffle_positions(geometry: FlowcellGeometry, seed: int) -> FlowcellGeometry:
    """Reassign reads to the observed cluster positions within each group.

    Permuting which read sits at which position leaves the point pattern, and
    therefore the geometry null itself, completely unchanged while destroying
    every real spatial relation. p-values recomputed on the permuted geometry
    are uniform exactly when the null is modelled correctly, which is what the
    QQ plot checks.
    """
    rng = np.random.default_rng(seed)
    permuted_x = geometry.global_x.copy()
    permuted_y = geometry.global_y.copy()
    permuted_columns = geometry.columns.copy()
    permuted_rows = geometry.rows.copy()
    # The tile travels with the position. Leaving it behind would let the
    # neighbour count be taken over one tile while testability is decided on
    # another, which turns the control into noise instead of a check.
    permuted_tiles = geometry.tiles.copy()
    for group_id in {int(value) for value in geometry.group_ids if value >= 0}:
        members = np.flatnonzero(geometry.group_ids == group_id)
        shuffled = rng.permutation(members)
        permuted_x[members] = geometry.global_x[shuffled]
        permuted_y[members] = geometry.global_y[shuffled]
        permuted_columns[members] = geometry.columns[shuffled]
        permuted_rows[members] = geometry.rows[shuffled]
        permuted_tiles[members] = geometry.tiles[shuffled]
    return FlowcellGeometry(
        layout=geometry.layout,
        group_ids=geometry.group_ids,
        group_labels=geometry.group_labels,
        tiles=permuted_tiles,
        columns=permuted_columns,
        rows=permuted_rows,
        global_x=permuted_x,
        global_y=permuted_y,
        cell_x=geometry.cell_x,
        cell_y=geometry.cell_y,
        neighborhood=geometry.neighborhood,
        max_grid_step=geometry.max_grid_step,
    )


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    count = len(pvalues)
    if count == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(pvalues, kind="stable")
    ordered = pvalues[order]
    ranks = np.arange(1, count + 1, dtype=np.float64)
    adjusted = ordered * count / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty(count, dtype=np.float64)
    result[order] = adjusted
    return result


def sequence_weighted_bh(
    spatial_p: np.ndarray,
    same_template_posterior: np.ndarray,
    compatibility_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # The square root prevents one very small compatibility p-value from making
    # a weight numerically zero while retaining a smooth quality penalty.
    raw_weights = same_template_posterior * np.sqrt(compatibility_p)
    raw_weights = np.maximum(raw_weights, 1e-12)
    weights = raw_weights / raw_weights.mean()
    weighted_p = np.minimum(1.0, spatial_p / weights)
    return benjamini_hochberg(weighted_p), weights


def duplicate_decomposition(
    observed_distances: np.ndarray,
    permuted_distances: list[np.ndarray],
    read_count: int,
) -> dict[str, object]:
    """Split duplicated reads into a proximity component and a library component.

    A library or PCR copy lands anywhere on the flowcell, so at any distance
    the permuted replicates say how many reads would have a look-alike that
    close by chance alone. The difference is the proximity component. Counting
    it at a series of distances gives a curve that rises while real proximity
    duplication is still being accumulated and then falls once the chance
    background starts to dominate, so its peak estimates how many reads are
    involved in proximity duplication and the distance at which it saturates
    estimates the length scale of the mechanism.

    That length scale is a property of the run, not a constant of the platform:
    on a patterned flowcell with heavy pad-hopping the curve is still climbing
    at several thousand pixels, while a clean non-patterned run saturates
    inside a hundred. Reading it off the data is what removes the need to pick
    a distance threshold per instrument, and getting that constant wrong is
    exactly how library duplicates get removed as if they were optical.
    """
    cuts = np.array(
        [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000], dtype=float
    )

    def below(values: np.ndarray) -> np.ndarray:
        finite = values[np.isfinite(values)]
        if not len(finite):
            return np.zeros(len(cuts), dtype=np.float64)
        ordered = np.sort(finite)
        return np.searchsorted(ordered, cuts, side="right").astype(np.float64)

    observed = below(observed_distances)
    if not permuted_distances:
        return {"permutations": 0}
    replicates = np.array([below(values) for values in permuted_distances])
    null = replicates.mean(axis=0)
    spread = replicates.std(axis=0)
    excess = observed - null
    peak = int(np.argmax(excess))
    best = float(excess[peak])
    # An excess within the permutation noise is not evidence of anything, and
    # the distance at which such a curve happens to peak is meaningless. Only
    # report a length scale once the peak clears the scatter of the replicates.
    noise = float(spread[peak]) if len(replicates) > 1 else 0.0
    detected = best > max(3.0 * noise, 1.0)
    saturation = None
    if detected:
        saturation = next(
            (float(cuts[i]) for i in range(len(cuts)) if excess[i] >= 0.95 * best),
            float(cuts[peak]),
        )
    return {
        "permutations": len(permuted_distances),
        "proximity_duplicate_reads": max(best, 0.0) if detected else 0.0,
        "proximity_duplicate_fraction": (
            max(best, 0.0) / max(read_count, 1) if detected else 0.0
        ),
        "proximity_duplication_detected": bool(detected),
        "permutation_noise_at_peak": noise,
        "chance_positioned_reads": float(observed[-1] - max(best, 0.0)),
        "chance_positioned_fraction": (
            float(observed[-1] - max(best, 0.0)) / max(read_count, 1)
        ),
        "chance_positioned_note": (
            "tested reads whose nearest look-alike sits at a distance the "
            "permuted flowcell reproduces. An upper bound on the library "
            "duplicate load, not an estimate of it: this count also contains "
            "sequence-similar reads that are not duplicates of anything."
        ),
        "saturation_distance_px": saturation,
        "peak_distance_px": float(cuts[peak]) if detected else None,
        "curve": [
            {
                "distance_px": float(cuts[i]),
                "observed_reads": int(observed[i]),
                "expected_if_positions_random": float(null[i]),
                "excess_reads": float(excess[i]),
            }
            for i in range(len(cuts))
        ],
        "interpretation": (
            "excess_reads peaks at the number of reads involved in proximity "
            "duplication; saturation_distance_px is the length scale of the "
            "mechanism in this run, read from the data rather than assumed"
        ),
    }


def permutation_fdr_qvalues(
    observed: np.ndarray, permuted: list[np.ndarray]
) -> tuple[np.ndarray, dict[str, object]]:
    """FDR read straight off a permuted flowcell, with no analytic correction.

    Benjamini-Hochberg and Benjamini-Yekutieli convert a per-hypothesis p-value
    into an error rate by *assuming* something about how the hypotheses depend
    on each other -- positive regression dependence for BH, nothing at all for
    BY, at the price of a factor of about ``log(m)``. Neither assumption has to
    be made if the whole flowcell is reshuffled: each permutation reproduces
    the real dependence, because the same reads, the same candidate relations
    and the same cluster pattern are all still there, only the pairing of read
    to position is broken.

    At threshold ``t``, the average number of reads a permuted run calls is a
    direct estimate of how many of the real calls are false, so

        FDR(t) = mean permuted calls at t / observed calls at t

    and the q-value of a read is the smallest such estimate over every
    threshold at least as large as its own p-value. This is still multiplicity
    control -- testing many reads at once does not make the false positives go
    away -- but it is measured rather than bounded, so it avoids the ``log(m)``
    that BY has to pay for its guarantee.
    """
    count = len(observed)
    if count == 0 or not permuted:
        return np.ones(count, dtype=np.float64), {"permutations": len(permuted)}

    order = np.argsort(observed, kind="stable")
    thresholds = observed[order]
    observed_calls = np.searchsorted(thresholds, thresholds, side="right").astype(
        np.float64
    )
    expected_false = np.zeros(count, dtype=np.float64)
    for values in permuted:
        expected_false += np.searchsorted(np.sort(values), thresholds, side="right")
    expected_false /= len(permuted)

    fdr = np.minimum(1.0, expected_false / np.maximum(observed_calls, 1.0))
    # A q-value is the best (smallest) FDR available at this threshold or any
    # looser one, so the sequence is made non-decreasing from the left.
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    result = np.empty(count, dtype=np.float64)
    result[order] = fdr
    diagnostics = {
        "permutations": len(permuted),
        "hypotheses": count,
        "mean_permuted_calls_at_observed_median_p": float(
            expected_false[count // 2]
        ),
    }
    return result, diagnostics


def benjamini_yekutieli(pvalues: np.ndarray) -> np.ndarray:
    """BH inflated by the harmonic number, valid under arbitrary dependence.

    Candidate edges are strongly dependent: a family of k sequence-identical
    read pairs contributes k(k-1)/2 edges over only k reads, and every edge
    incident on one read shares that read's position. BH needs positive
    regression dependence, which that structure does not guarantee, so BY is
    offered as the assumption-free alternative.
    """
    count = len(pvalues)
    if count == 0:
        return np.empty(0, dtype=np.float64)
    harmonic = float(np.sum(1.0 / np.arange(1, count + 1)))
    return np.minimum(1.0, benjamini_hochberg(pvalues) * harmonic)


def least_concave_majorant_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Slopes of the least concave majorant of the points (x, y).

    Used as the Grenander estimator: the slopes of the least concave majorant
    of the empirical CDF form the non-increasing maximum-likelihood density.
    """
    hull: list[int] = []
    for index in range(len(x)):
        while len(hull) >= 2:
            first, second = hull[-2], hull[-1]
            left = (y[second] - y[first]) * (x[index] - x[first])
            right = (y[index] - y[first]) * (x[second] - x[first])
            if left <= right:  # the middle vertex is below the chord
                hull.pop()
            else:
                break
        hull.append(index)
    slopes = np.empty(len(x) - 1, dtype=np.float64)
    for position in range(len(hull) - 1):
        start, stop = hull[position], hull[position + 1]
        width = x[stop] - x[start]
        slope = (y[stop] - y[start]) / width if width > 0 else 0.0
        slopes[start:stop] = slope
    return slopes


def conditional_spatial_pvalues(
    spatial_p: np.ndarray,
    testable: np.ndarray,
    group_ids: np.ndarray,
    group_models: dict[int, dict[str, object]],
) -> np.ndarray:
    """Rescale spatial p-values to be uniform given a testable tile relation.

    The raw p-value is ``P(testable relation and distance <= d)``, so it can
    never exceed ``P(testable relation)`` and it collapses to exactly one for
    every pair on tiles too far apart to test. Dividing by that probability
    gives ``P(distance <= d | testable relation)``, which is uniform on the
    unit interval under the null. Non-testable pairs keep a value of one.
    """
    result = np.ones(len(spatial_p), dtype=np.float64)
    for group_id, model in group_models.items():
        probability = float(model.get("testable_probability", 0.0))
        if probability <= 0.0:
            continue
        positions = np.flatnonzero((group_ids == group_id) & testable)
        if len(positions):
            result[positions] = np.clip(spatial_p[positions] / probability, 0.0, 1.0)
    return result


def local_false_discovery_rate(
    pvalues: np.ndarray, null_proportion_lambda: float
) -> tuple[np.ndarray, dict[str, float]]:
    """Two-groups local FDR for spatial p-values.

    The mixture is ``f(p) = pi0 + (1 - pi0) * f1(p)`` on the unit interval,
    where ``f1`` is an unknown decreasing density. The decreasing shape is the
    only assumption about the alternative, and it is exactly what optical
    duplication implies: a real proximity duplicate cannot make a pair *less*
    close than chance. ``f`` is estimated by the Grenander estimator and the
    local FDR is ``pi0 / f(p)``.

    This is the piece a pure tail test is missing. Benjamini-Hochberg rejects
    whenever a p-value is small relative to the number of hypotheses, so with
    enough candidate edges it will call pairs that are thousands of pixels
    apart -- distances at which no optical mechanism operates -- purely because
    they are marginally closer than a random pair of clusters. The local FDR
    depends on the density ratio at the observed p-value rather than on the
    number of hypotheses, so such pairs keep a local FDR near one.
    """
    count = len(pvalues)
    if count == 0:
        return np.empty(0, dtype=np.float64), {"pi0": 1.0, "grenander_density_at_1": 1.0}

    # A hypothesis with no testable spatial relation scores exactly one and is
    # certainly null. Those are kept in the fit rather than rescaled away: the
    # p-value's support depends on how many candidate partners a read has, so
    # there is no single scale that would map the non-atom part back onto the
    # unit interval, and pooling them anyway inflates the apparent signal.
    # Leaving the atom in makes the estimated null proportion conservative,
    # which is the right direction for it to err in.
    clipped = np.clip(pvalues, 0.0, 1.0)
    atom = float(np.count_nonzero(clipped >= 1.0)) / count
    order = np.argsort(clipped, kind="stable")
    ordered = clipped[order]
    # Empirical CDF as a step function on [0, 1]. The spatial p-value is a
    # lookup into a tabulated null, so the number of distinct values is small
    # and the hull below runs over that short list rather than every hypothesis.
    grid = np.concatenate(([0.0], ordered, [1.0]))
    cdf = np.concatenate(([0.0], np.arange(1, count + 1) / count, [1.0]))
    unique_grid = np.unique(grid)
    unique_cdf = cdf[np.searchsorted(grid, unique_grid, side="right") - 1]
    slopes = least_concave_majorant_slopes(unique_grid, unique_cdf)
    density_at = np.searchsorted(unique_grid, ordered, side="right") - 1
    density_at = np.clip(density_at, 0, len(slopes) - 1)
    density = np.maximum(slopes[density_at], 1e-12)

    # Storey's estimator from the flat upper part of the distribution, floored
    # by the Grenander density at one, which is itself a lower bound on pi0.
    lam = min(max(null_proportion_lambda, 0.05), 0.95)
    above = float(np.count_nonzero(clipped > lam))
    storey = min(1.0, above / ((1.0 - lam) * count))
    grenander_at_one = float(slopes[-1]) if len(slopes) else 1.0
    pi0 = float(min(1.0, max(storey, grenander_at_one, 1e-6)))

    ordered_lfdr = np.minimum(1.0, pi0 / density)
    # The local FDR must not increase as the p-value falls.
    ordered_lfdr = np.maximum.accumulate(ordered_lfdr)
    result = np.empty(count, dtype=np.float64)
    result[order] = ordered_lfdr
    result[clipped >= 1.0] = 1.0
    diagnostics = {
        "pi0": pi0,
        "storey_pi0": storey,
        "grenander_density_at_1": grenander_at_one,
        "lambda": lam,
        "atom_at_one": atom,
    }
    return result, diagnostics


def local_fdr_qvalues(local_fdr: np.ndarray) -> np.ndarray:
    """Bayes-FDR q-values: the running mean local FDR of the rejected set.

    Rejecting every hypothesis whose q-value is at most the target keeps the
    expected proportion of false discoveries among rejections at or below the
    target, which is the same guarantee Benjamini-Hochberg offers but obtained
    from the fitted mixture instead of from the p-value ranks alone.

    The rejected set is a prefix of the local-FDR ordering, so a long run of
    near-zero local FDRs can hold the running mean under the target past the
    point where the local FDR reaches one. A hypothesis whose posterior
    probability of being null is one carries no evidence at all and must never
    be rejected on the strength of other hypotheses, so those are pinned at a
    q-value of one.
    """
    count = len(local_fdr)
    if count == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(local_fdr, kind="stable")
    ordered = local_fdr[order]
    running = np.cumsum(ordered) / np.arange(1, count + 1)
    running = np.minimum.accumulate(running[::-1])[::-1]
    result = np.empty(count, dtype=np.float64)
    result[order] = np.minimum(running, 1.0)
    return np.where(local_fdr >= 1.0, 1.0, result)


def uniform_qq_points(pvalues: np.ndarray, maximum_points: int = 40_000):
    """Expected/observed quantile pairs for a uniform QQ plot, on -log10 axes."""
    values = np.clip(np.asarray(pvalues, dtype=np.float64), 1e-300, 1.0)
    count = len(values)
    if count == 0:
        return np.empty(0), np.empty(0)
    ordered = np.sort(values)
    expected = (np.arange(1, count + 1) - 0.5) / count
    if count > maximum_points:
        # Keep the whole extreme tail and thin only the uninformative bulk.
        head = np.arange(min(count, maximum_points // 2))
        rest = np.unique(
            np.geomspace(
                len(head) + 1, count, maximum_points - len(head)
            ).astype(np.int64)
            - 1
        )
        keep = np.unique(np.concatenate((head, rest)))
        ordered, expected = ordered[keep], expected[keep]
    return -np.log10(expected), -np.log10(ordered)


def calibration_statistics(pvalues: np.ndarray) -> dict[str, object]:
    """How far a set of p-values is from uniform, scored for a discrete null.

    A valid p-value only has to satisfy ``P(p <= alpha) <= alpha``. These
    p-values are discrete and, unless every tile relation is testable, carry a
    genuine atom at one: a read whose only look-alike landed on a tile too far
    away to compare cannot score anything else. A two-sided
    Kolmogorov-Smirnov distance or a median-based inflation factor treats that
    atom as a failure even though it is exactly correct, so the headline
    numbers here are one-sided.

    ``max_excess_over_uniform`` is the largest amount by which the observed
    rejection rate exceeds its nominal level; at most zero means the p-values
    are valid everywhere. ``tail_inflation`` is the median observed/expected
    ratio over the alphas that actually drive FDR: one means calibrated, above
    one means the null is too optimistic. ``lambda_gc`` is the familiar
    genomic-control factor, reported only when the atom is small enough for a
    median-based statistic to mean anything.
    """
    values = np.clip(np.asarray(pvalues, dtype=np.float64), 0.0, 1.0)
    count = len(values)
    if count == 0:
        return {"count": 0}
    atom = float(np.count_nonzero(values >= 1.0)) / count
    median = float(np.median(values))
    lambda_gc = (
        float(chi2.isf(median, 1) / chi2.isf(0.5, 1))
        if 0.0 < median < 1.0 and atom < 0.25
        else None
    )
    ordered = np.sort(values)
    empirical = np.arange(1, count + 1) / count
    max_excess = float(np.max(empirical - ordered))
    max_deficit = float(np.max(ordered - np.arange(count) / count))
    tail: dict[str, object] = {}
    ratios: list[float] = []
    for alpha in (1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5):
        hits = int(np.count_nonzero(values <= alpha))
        observed = hits / count
        tail[f"{alpha:g}"] = {
            "observed": observed,
            "observed_over_expected": observed / alpha,
            "hits": hits,
        }
        if hits >= 10:
            ratios.append(observed / alpha)
    return {
        "count": count,
        "atom_at_one": atom,
        "median_p": median,
        "lambda_gc": lambda_gc,
        "tail_inflation": float(np.median(ratios)) if ratios else None,
        "max_excess_over_uniform": max_excess,
        "max_deficit_below_uniform": max_deficit,
        "two_sided_kolmogorov_smirnov_d": float(max(max_excess, max_deficit)),
        "tail_observed_over_expected": tail,
    }


def build_null_controls(
    reads: LoadedReads,
    geometry: FlowcellGeometry,
    group_models: dict[int, dict[str, object]],
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    eligible_indices: np.ndarray,
    scores: dict[str, np.ndarray],
    distances: np.ndarray,
    testable: np.ndarray,
    conditional_p: np.ndarray,
    analysis_pvalues: np.ndarray,
    inference_unit: str,
    metric: str,
    seed: int,
    logger: RunLogger,
) -> dict[str, dict[str, object]]:
    """Negative-control p-value sets for checking that the null is right.

    Three sets are produced, all on the same scale as the p-values that drive
    the analysis, so the QQ plot checks the quantity actually used.

    ``permutation`` reassigns reads to the observed cluster positions within a
    lane and surface. That leaves the point pattern, and therefore the null
    itself, untouched while destroying every real spatial relation, so these
    p-values must be uniform if the geometry null is computed correctly. This
    checks the machinery.

    ``sequence_incompatible`` uses candidate edges that the quality model
    rejected: real read pairs, retrieved because they share a seed, whose
    mismatches are far too many for one template. They cannot be duplicates of
    each other, so their spatial relation is null -- but they are real reads,
    so any spatial structure that within-lane exchangeability misses (dark
    patches, bubbles, low-complexity wells) shows up here and not in the
    permutation. This checks the assumption.

    ``analysis`` is the analysed set itself, which should depart from uniform
    in the extreme tail when proximity duplicates are present.
    """
    controls: dict[str, dict[str, object]] = {}
    tested_left = edge_left[eligible_indices]
    tested_right = edge_right[eligible_indices]
    p_norm = math.inf if metric == "chebyshev" else 2.0

    def summarise(name: str, description: str, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        controls[name] = {
            "description": description,
            "pvalues": values,
            "statistics": calibration_statistics(values),
        }

    shuffled_geometry = shuffle_positions(geometry, seed)
    permuted_distances, permuted_testable = pair_distances(
        shuffled_geometry, tested_left, tested_right, p_norm
    )
    if inference_unit == "read":
        permuted_test = read_level_spatial_test(
            reads,
            shuffled_geometry,
            tested_left,
            tested_right,
            permuted_distances,
            permuted_testable,
            metric,
            logger,
        )
        summarise(
            "permutation",
            "reads reassigned to observed cluster positions within lane and surface",
            permuted_test.pvalue,
        )
    else:
        permuted_p = np.ones(len(tested_left), dtype=np.float64)
        for group_id, model in group_models.items():
            positions = np.flatnonzero(
                (geometry.group_ids[tested_left] == group_id) & permuted_testable
            )
            if len(positions):
                permuted_p[positions] = spatial_null_lookup(
                    model, permuted_distances[positions]
                )
        permuted_conditional = conditional_spatial_pvalues(
            permuted_p, permuted_testable, geometry.group_ids[tested_left], group_models
        )
        summarise(
            "permutation",
            "reads reassigned to observed cluster positions within lane and surface",
            permuted_conditional,
        )

    incompatible = np.flatnonzero(scores["log10_bayes_factor"] <= 0.0)
    if len(incompatible):
        control_left = edge_left[incompatible]
        control_right = edge_right[incompatible]
        control_distances, control_testable = pair_distances(
            geometry, control_left, control_right, p_norm
        )
        description = "seed-retrieved candidate edges rejected by the quality model"
        if inference_unit == "read":
            control_test = read_level_spatial_test(
                reads,
                geometry,
                control_left,
                control_right,
                control_distances,
                control_testable,
                metric,
                logger,
            )
            summarise("sequence_incompatible", description, control_test.pvalue)
        else:
            control_p = np.ones(len(incompatible), dtype=np.float64)
            for group_id, model in group_models.items():
                positions = np.flatnonzero(
                    (geometry.group_ids[control_left] == group_id) & control_testable
                )
                if len(positions):
                    control_p[positions] = spatial_null_lookup(
                        model, control_distances[positions]
                    )
            control_conditional = conditional_spatial_pvalues(
                control_p,
                control_testable,
                geometry.group_ids[control_left],
                group_models,
            )
            summarise("sequence_incompatible", description, control_conditional)

    summarise(
        "analysis",
        (
            "one hypothesis per read carried into FDR control"
            if inference_unit == "read"
            else "sequence-compatible candidate edges carried into FDR control"
        ),
        analysis_pvalues,
    )

    for name, control in controls.items():
        statistics = control["statistics"]
        logger.log(
            "null_check",
            f"control={name} n={statistics.get('count', 0):,} "
            f"atom_at_1={statistics.get('atom_at_one')} "
            f"tail_inflation={statistics.get('tail_inflation')} "
            f"max_excess={statistics.get('max_excess_over_uniform')}",
        )
    return controls


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, value: int) -> int:
        parent = int(self.parent[value])
        while parent != value:
            grandparent = int(self.parent[parent])
            self.parent[value] = grandparent
            value = parent
            parent = grandparent
        return value

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def component_filter_decisions(
    read_count: int,
    tested_left: np.ndarray,
    tested_right: np.ndarray,
    selected_q: np.ndarray,
    quality_sums: np.ndarray,
    target_fdr: float,
) -> tuple[set[int], dict[int, dict[str, object]], dict[str, object]]:
    # A hypothesis whose partner is missing names no relation to act on, so it
    # can never justify removing a read however small its q-value came out.
    significant_positions = np.flatnonzero(
        (selected_q <= target_fdr) & (tested_left >= 0) & (tested_right >= 0)
    )
    union_find = UnionFind(read_count)
    best_incident: dict[int, int] = {}
    component_edges: list[int] = []

    for edge_position in significant_positions[np.argsort(selected_q[significant_positions])]:
        left = int(tested_left[edge_position])
        right = int(tested_right[edge_position])
        if union_find.union(left, right):
            component_edges.append(int(edge_position))
        for node in (left, right):
            previous = best_incident.get(node)
            if previous is None or selected_q[edge_position] < selected_q[previous]:
                best_incident[node] = int(edge_position)

    components: dict[int, list[int]] = defaultdict(list)
    involved_nodes = sorted(
        set(int(tested_left[pos]) for pos in significant_positions)
        | set(int(tested_right[pos]) for pos in significant_positions)
    )
    for node in involved_nodes:
        components[union_find.find(node)].append(node)

    component_edge_positions: dict[int, list[int]] = defaultdict(list)
    for edge_position in significant_positions:
        root = union_find.find(int(tested_left[edge_position]))
        component_edge_positions[root].append(int(edge_position))

    removed: set[int] = set()
    removal_details: dict[int, dict[str, object]] = {}
    size_histogram: Counter[int] = Counter()
    sorted_components = sorted(components.values(), key=lambda values: min(values))
    for component_number, members in enumerate(sorted_components, start=1):
        size_histogram[len(members)] += 1
        representative = max(
            members, key=lambda index: (int(quality_sums[index]), -index)
        )
        root = union_find.find(members[0])
        edge_positions = component_edge_positions[root]
        component_min_q = min(float(selected_q[pos]) for pos in edge_positions)
        for member in members:
            if member == representative:
                continue
            removed.add(member)
            support_position = best_incident[member]
            support_left = int(tested_left[support_position])
            support_right = int(tested_right[support_position])
            support_other = support_right if support_left == member else support_left
            removal_details[member] = {
                "representative": representative,
                "component_id": component_number,
                "component_size": len(members),
                "component_edge_count": len(edge_positions),
                "component_min_q": component_min_q,
                "support_edge_position": support_position,
                "support_other": support_other,
            }

    stats = {
        "significant_candidate_edges": int(len(significant_positions)),
        "optical_components": int(len(sorted_components)),
        "component_size_histogram": {
            str(size): count for size, count in sorted(size_histogram.items())
        },
        "filtered_read_pairs": int(len(removed)),
        "forest_edges_used": int(len(component_edges)),
    }
    return removed, removal_details, stats


def filtered_count_at_threshold(
    read_count: int,
    tested_left: np.ndarray,
    tested_right: np.ndarray,
    qvalues: np.ndarray,
    threshold: float,
) -> tuple[int, int, int]:
    significant = np.flatnonzero(
        (qvalues <= threshold) & (tested_left >= 0) & (tested_right >= 0)
    )
    if len(significant) == 0:
        return 0, 0, 0
    union_find = UnionFind(read_count)
    nodes: set[int] = set()
    for position in significant:
        left = int(tested_left[position])
        right = int(tested_right[position])
        union_find.union(left, right)
        nodes.add(left)
        nodes.add(right)
    component_count = len({union_find.find(node) for node in nodes})
    return len(significant), component_count, len(nodes) - component_count


def write_fastq_record(handle: TextIO, record: FastqRecord) -> None:
    handle.write(record.header)
    handle.write(record.sequence.decode("ascii") + "\n")
    handle.write(record.plus)
    handle.write(record.quality.decode("ascii") + "\n")


def write_filtered_fastqs(
    r1_path: Path,
    r2_path: Path,
    output_r1: Path,
    output_r2: Path,
    removed: set[int],
    total_pairs: int,
    gzip_level: int,
    threads: int,
    logger: RunLogger,
) -> tuple[int, int]:
    pigz_threads = (
        max(1, threads // 2) if shutil.which("pigz") is not None and threads > 1 else 1
    )
    tracker = logger.tracker("write_fastq", total_pairs, "read_pairs")
    kept = 0
    removed_count = 0
    with open_input(r1_path) as r1_handle, open_input(r2_path) as r2_handle, open_output(
        output_r1, gzip_level, pigz_threads
    ) as output1, open_output(output_r2, gzip_level, pigz_threads) as output2:
        for record_index in range(total_pairs):
            record_number = record_index + 1
            r1 = read_fastq_record(r1_handle, r1_path, record_number)
            r2 = read_fastq_record(r2_handle, r2_path, record_number)
            if r1 is None or r2 is None:
                raise RuntimeError("Unexpected FASTQ truncation during output pass")
            if normalized_read_name(r1.header) != normalized_read_name(r2.header):
                raise RuntimeError(
                    f"R1/R2 desynchronization during output pass at {record_number:,}"
                )
            if record_index in removed:
                removed_count += 1
            else:
                write_fastq_record(output1, r1)
                write_fastq_record(output2, r2)
                kept += 1
            tracker.update(record_number, detail=f"kept={kept:,} removed={removed_count:,}")
        if read_fastq_record(r1_handle, r1_path, total_pairs + 1) is not None:
            raise RuntimeError("R1 contains extra records during output pass")
        if read_fastq_record(r2_handle, r2_path, total_pairs + 1) is not None:
            raise RuntimeError("R2 contains extra records during output pass")
    tracker.finish(
        detail=(
            f"kept={kept:,} removed={removed_count:,} "
            f"compression={'pigz' if pigz_threads > 1 else 'gzip'}"
        )
    )
    return kept, removed_count


def write_candidate_audit(
    path: Path,
    reads: LoadedReads,
    geometry: FlowcellGeometry,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    eligible_indices: np.ndarray,
    scores: dict[str, np.ndarray],
    same_template_posterior: np.ndarray,
    spatial_p: np.ndarray,
    conditional_p: np.ndarray,
    distances: np.ndarray,
    testable: np.ndarray,
    local_fdr: np.ndarray,
    qvalue_sets: dict[str, np.ndarray],
    weights: np.ndarray,
    selected_q: np.ndarray,
    target_fdr: float,
) -> None:
    tested_left = edge_left[eligible_indices]
    tested_right = edge_right[eligible_indices]
    grid_step = geometry.grid_step(tested_left, tested_right)
    method_columns = sorted(qvalue_sets)
    columns = [
        "candidate_edge_id",
        "read_pair_1_index",
        "read_pair_1_name",
        "read_pair_2_index",
        "read_pair_2_name",
        "lane",
        "spatial_group",
        "tile_1",
        "x_1",
        "y_1",
        "tile_2",
        "x_2",
        "y_2",
        "same_tile",
        "tile_grid_step",
        "testable_relation",
        "spatial_distance",
        "observed_mismatches",
        "expected_mismatches_from_phred",
        "quality_weighted_mismatch_surprisal",
        "sequence_compatibility_p",
        "sequence_log10_bayes_factor",
        "same_template_posterior",
        "spatial_p_value",
        "conditional_spatial_p_value",
        "local_fdr",
        "sequence_weight",
        *[f"{name}_q_value" for name in method_columns],
        "selected_q_value",
        "called_optical_at_target_fdr",
    ]
    with open_output(path, gzip_level=6) as handle:
        handle.write("\t".join(columns) + "\n")
        for position, candidate_index in enumerate(eligible_indices):
            left = int(tested_left[position])
            right = int(tested_right[position])
            lane_id = int(reads.lane_ids[left])
            group_id = int(geometry.group_ids[left])
            finite = bool(testable[position]) and math.isfinite(distances[position])
            values = [
                position + 1,
                left + 1,
                reads.names[left],
                right + 1,
                reads.names[right],
                reads.lane_labels[lane_id] if lane_id >= 0 else "NA",
                geometry.group_labels[group_id] if group_id >= 0 else "NA",
                int(reads.tiles[left]),
                int(reads.x[left]),
                int(reads.y[left]),
                int(reads.tiles[right]),
                int(reads.x[right]),
                int(reads.y[right]),
                int(reads.tiles[left] == reads.tiles[right]),
                int(grid_step[position]),
                int(testable[position]),
                f"{distances[position]:.6g}" if finite else "NA",
                int(scores["mismatches"][candidate_index]),
                f"{scores['expected_mismatches'][candidate_index]:.8g}",
                f"{scores['mismatch_surprisal'][candidate_index]:.8g}",
                f"{scores['compatibility_p'][candidate_index]:.8g}",
                f"{scores['log10_bayes_factor'][candidate_index]:.8g}",
                f"{same_template_posterior[candidate_index]:.8g}",
                f"{spatial_p[position]:.8g}",
                f"{conditional_p[position]:.8g}",
                f"{local_fdr[position]:.8g}",
                f"{weights[position]:.8g}",
                *[f"{qvalue_sets[name][position]:.8g}" for name in method_columns],
                f"{selected_q[position]:.8g}",
                int(selected_q[position] <= target_fdr),
            ]
            handle.write("\t".join(map(str, values)) + "\n")


def write_removed_audit(
    path: Path,
    reads: LoadedReads,
    removal_details: dict[int, dict[str, object]],
    read_test: "ReadLevelTest | None",
    tested_left: np.ndarray,
    tested_right: np.ndarray,
    eligible_indices: np.ndarray,
    scores: dict[str, np.ndarray],
    unit_p: np.ndarray,
    distances: np.ndarray,
    local_fdr: np.ndarray,
    selected_q: np.ndarray,
    unit_left: np.ndarray,
    unit_right: np.ndarray,
) -> None:
    """Per removed read pair: what it was removed for and how strong the call was.

    ``support_spatial_p`` is the p-value of the hypothesis that justified the
    removal, so with the read unit it is the read's nearest-partner p-value and
    with the edge unit it is that relation's p-value. It is never the tabulated
    edge null, which the read-level path does not compute at all.
    """
    columns = [
        "removed_pair_index",
        "removed_read_name",
        "representative_pair_index",
        "representative_read_name",
        "component_id",
        "component_size",
        "component_edge_count",
        "component_min_q",
        "supporting_pair_index",
        "supporting_read_name",
        "support_distance",
        "support_candidate_partners",
        "support_neighbours_within_distance",
        "support_mismatches",
        "support_expected_mismatches",
        "support_sequence_compatibility_p",
        "support_spatial_p",
        "support_local_fdr",
        "support_q",
    ]
    with open_output(path, gzip_level=6) as handle:
        handle.write("\t".join(columns) + "\n")
        for removed_index in sorted(removal_details):
            details = removal_details[removed_index]
            representative = int(details["representative"])
            position = int(details["support_edge_position"])
            support_other = int(details["support_other"])
            if read_test is not None:
                edge_position = int(read_test.supporting_edge[position])
                support_distance = float(read_test.nearest_distance[position])
                partners: object = int(read_test.partner_count[position])
                neighbours: object = f"{read_test.neighbours_within[position]:.6g}"
            else:
                edge_position = position
                support_distance = float(distances[position])
                partners = "NA"
                neighbours = "NA"
            candidate_index = int(eligible_indices[edge_position])
            values = [
                removed_index + 1,
                reads.names[removed_index],
                representative + 1,
                reads.names[representative],
                details["component_id"],
                details["component_size"],
                details["component_edge_count"],
                f"{float(details['component_min_q']):.8g}",
                support_other + 1,
                reads.names[support_other],
                (
                    f"{support_distance:.6g}"
                    if math.isfinite(support_distance)
                    else "NA"
                ),
                partners,
                neighbours,
                int(scores["mismatches"][candidate_index]),
                f"{scores['expected_mismatches'][candidate_index]:.8g}",
                f"{scores['compatibility_p'][candidate_index]:.8g}",
                f"{unit_p[position]:.8g}",
                f"{local_fdr[position]:.8g}",
                f"{selected_q[position]:.8g}",
            ]
            handle.write("\t".join(map(str, values)) + "\n")


def write_read_audit(
    path: Path,
    reads: LoadedReads,
    geometry: FlowcellGeometry,
    read_test: ReadLevelTest,
    local_fdr: np.ndarray,
    qvalue_sets: dict[str, np.ndarray],
    selected_q: np.ndarray,
    removed: set[int],
    target_fdr: float,
) -> None:
    """Per tested read: the nearest look-alike, the local density, the call."""
    method_columns = sorted(qvalue_sets)
    columns = [
        "read_pair_index",
        "read_name",
        "lane",
        "spatial_group",
        "tile",
        "x",
        "y",
        "nearest_partner_index",
        "nearest_partner_name",
        "nearest_partner_tile",
        "nearest_partner_distance",
        "cross_tile",
        "candidate_partners",
        "neighbours_within_distance",
        "group_read_pairs",
        "spatial_p_value",
        "local_fdr",
        *[f"{name}_q_value" for name in method_columns],
        "selected_q_value",
        "called_optical_at_target_fdr",
        "removed",
    ]
    group_sizes = {
        group: int(np.count_nonzero(geometry.group_ids == group))
        for group in {int(value) for value in geometry.group_ids if value >= 0}
    }
    with open_output(path, gzip_level=6) as handle:
        handle.write("\t".join(columns) + "\n")
        for position, index in enumerate(read_test.read_index):
            index = int(index)
            partner = int(read_test.partner_index[position])
            lane_id = int(reads.lane_ids[index])
            group_id = int(geometry.group_ids[index])
            values = [
                index + 1,
                reads.names[index],
                reads.lane_labels[lane_id] if lane_id >= 0 else "NA",
                geometry.group_labels[group_id] if group_id >= 0 else "NA",
                int(reads.tiles[index]),
                int(reads.x[index]),
                int(reads.y[index]),
                partner + 1,
                reads.names[partner] if partner >= 0 else "NA",
                int(reads.tiles[partner]) if partner >= 0 else "NA",
                f"{read_test.nearest_distance[position]:.6g}",
                int(partner >= 0 and reads.tiles[index] != reads.tiles[partner]),
                int(read_test.partner_count[position]),
                f"{read_test.neighbours_within[position]:.6g}",
                group_sizes.get(group_id, 0),
                f"{read_test.pvalue[position]:.8g}",
                f"{local_fdr[position]:.8g}",
                *[f"{qvalue_sets[name][position]:.8g}" for name in method_columns],
                f"{selected_q[position]:.8g}",
                int(selected_q[position] <= target_fdr),
                int(index in removed),
            ]
            handle.write("\t".join(map(str, values)) + "\n")


def weighted_null_curve(
    lane_models: dict[int, dict[str, object]], grid: np.ndarray
) -> np.ndarray:
    """Return the geometry-null CDF averaged over tested hypotheses by lane."""
    numerator = np.zeros(len(grid), dtype=np.float64)
    denominator = 0
    for model in lane_models.values():
        tested = int(model["tested_hypotheses"])
        if tested == 0:
            continue
        radii = np.asarray(model["radii"], dtype=np.float64)
        null_cdf = np.asarray(model["null_cdf"], dtype=np.float64)
        positions = np.searchsorted(radii, grid, side="right") - 1
        curve = np.zeros(len(grid), dtype=np.float64)
        valid = positions >= 0
        curve[valid] = null_cdf[positions[valid]]
        numerator += tested * curve
        denominator += tested
    if denominator == 0:
        return numerator
    return numerator / denominator


def save_figure_atomically(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp.png")
    try:
        figure.savefig(temporary_path, dpi=180, format="png")
    finally:
        plt.close(figure)
    # A complete PNG terminates with this fixed IEND chunk. Write atomically so
    # downstream workflows never observe a partially rendered diagnostic.
    png_iend = b"\x00\x00\x00\x00IEND\xaeB\x60\x82"
    with temporary_path.open("rb") as handle:
        handle.seek(-len(png_iend), os.SEEK_END)
        if handle.read() != png_iend:
            raise OSError(f"Plot did not finish writing: {temporary_path}")
    os.replace(temporary_path, path)


def make_qq_plot(
    path: Path,
    controls: dict[str, dict[str, object]],
    local_fdr: np.ndarray,
    analysis_p: np.ndarray,
    distances: np.ndarray,
    target_fdr: float,
    called: np.ndarray,
    inference_unit: str,
) -> None:
    """Uniform QQ plots for the null controls plus the diagnostics behind them."""
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.6), constrained_layout=True)
    unit_label = "reads" if inference_unit == "read" else "edges"
    palette = {
        "permutation": ("#0B7285", "Permutation null (position labels shuffled)"),
        "sequence_incompatible": (
            "#B08900",
            "Sequence-incompatible partners (real-data null)",
        ),
        "analysis": ("#C1121F", f"Analysed {unit_label} (null + any real duplicates)"),
    }

    ax = axes[0, 0]
    limit = 1.0
    for name in ("permutation", "sequence_incompatible", "analysis"):
        control = controls.get(name)
        if control is None or not len(control["pvalues"]):
            continue
        colour, label = palette[name]
        expected, observed = uniform_qq_points(np.asarray(control["pvalues"]))
        if not len(expected):
            continue
        limit = max(limit, float(expected.max()), 1.0)
        statistics = control["statistics"]
        inflation = statistics.get("tail_inflation")
        suffix = f" (tail inflation {inflation:.2f})" if inflation is not None else ""
        ax.plot(expected, observed, ".", ms=3.0, color=colour, label=label + suffix)
    ax.plot([0, limit], [0, limit], color="#495057", lw=1.2, ls="--", label="uniform null")
    ax.set_xlabel("Expected -log10(p) under a uniform null")
    ax.set_ylabel("Observed -log10(p)")
    ax.set_title("A. QQ plot of the spatial null")
    ax.legend(frameon=True, framealpha=0.85, fontsize=8.0, loc="lower right")

    ax = axes[0, 1]
    for name in ("permutation", "sequence_incompatible", "analysis"):
        control = controls.get(name)
        if control is None or not len(control["pvalues"]):
            continue
        colour, label = palette[name]
        values = np.asarray(control["pvalues"])
        alphas = np.geomspace(max(1.0 / max(len(values), 2), 1e-6), 1.0, 60)
        ratio = np.asarray(
            [np.count_nonzero(values <= alpha) / len(values) / alpha for alpha in alphas]
        )
        ax.plot(alphas, ratio, lw=1.8, color=colour, label=label)
    ax.axhline(1.0, color="#495057", lw=1.2, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nominal alpha")
    ax.set_ylabel("Observed / expected rejection rate")
    ax.set_title("B. Tail calibration (1.0 means the null is exact)")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1, 0]
    control = controls.get("permutation")
    if control is not None and len(control["pvalues"]):
        ax.hist(
            np.asarray(control["pvalues"]),
            bins=50,
            range=(0, 1),
            color="#0B7285",
            alpha=0.55,
            density=True,
            label="permutation null",
        )
    if len(analysis_p):
        ax.hist(
            analysis_p,
            bins=50,
            range=(0, 1),
            color="#C1121F",
            alpha=0.45,
            density=True,
            label=f"analysed {unit_label}",
        )
    ax.axhline(1.0, color="#495057", lw=1.2, ls="--", label="uniform density")
    ax.set_xlabel("Spatial p-value")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.set_title("C. Null uniformity and the alternative it must separate")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1, 1]
    finite = np.isfinite(distances)
    if np.any(finite):
        colours = np.where(called[finite], "#C1121F", "#457B9D")
        ax.scatter(
            np.maximum(distances[finite], 0.5),
            np.clip(local_fdr[finite], 1e-6, 1.0),
            c=colours,
            s=8,
            alpha=0.45,
            linewidths=0,
        )
        ax.axhline(target_fdr, color="black", ls="--", lw=1.1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Nearest-partner distance in the global lane frame (pixels)")
        ax.set_ylabel("Local FDR")
    else:
        ax.text(0.5, 0.5, "No testable candidate pairs", ha="center", va="center")
    ax.set_title(f"D. Local FDR against distance for analysed {unit_label}")

    figure.suptitle(
        "FastqOptiFilter null-model calibration", fontsize=16, weight="bold"
    )
    save_figure_atomically(figure, path)


def make_diagnostic_plot(
    path: Path,
    reads: LoadedReads,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    eligible_indices: np.ndarray,
    scores: dict[str, np.ndarray],
    distances: np.ndarray,
    testable: np.ndarray,
    group_models: dict[int, dict[str, object]],
    bh_q: np.ndarray,
    weighted_q: np.ndarray,
    selected_q: np.ndarray,
    fdr_method: str,
    target_fdr: float,
    kept_count: int,
    removed_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tested_count = len(eligible_indices)
    called = selected_q <= target_fdr
    finite_distances = distances[testable]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)

    ax = axes[0, 0]
    if len(finite_distances):
        positive = finite_distances[finite_distances > 0]
        minimum = float(positive.min()) if len(positive) else 1.0
        maximum = max(float(finite_distances.max()), minimum)
        grid = np.unique(
            np.concatenate(
                (
                    np.asarray([0.0]),
                    np.geomspace(max(1.0, minimum), max(1.0, maximum), 180),
                )
            )
        )
        observed = np.asarray(
            [np.count_nonzero(testable & (distances <= value)) / tested_count for value in grid]
        )
        expected = weighted_null_curve(group_models, grid)
        ax.plot(grid, observed, color="#006D77", lw=2.2, label="sequence-compatible pairs")
        ax.plot(grid, expected, color="#E29578", lw=2.2, label="random-position null")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(left=0)
        ax.set_yscale("log")
        ax.set_xlabel("Spatial distance in the global lane frame")
        ax.set_ylabel("Cumulative fraction across all tested pairs")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No testable candidate pairs", ha="center", va="center")
    ax.set_title("A. Empirical spatial enrichment")

    ax = axes[0, 1]
    same_positions = np.flatnonzero(testable)
    if len(same_positions):
        plot_positions = same_positions
        if len(plot_positions) > 100_000:
            # Deterministic thinning for plotting only; inference always uses all edges.
            plot_positions = plot_positions[
                np.linspace(0, len(plot_positions) - 1, 100_000, dtype=np.int64)
            ]
        colors = np.where(called[plot_positions], "#D1495B", "#457B9D")
        ax.scatter(
            distances[plot_positions],
            -np.log10(np.maximum(selected_q[plot_positions], 1e-300)),
            c=colors,
            s=10,
            alpha=0.48,
            linewidths=0,
        )
        ax.axhline(-math.log10(target_fdr), color="black", ls="--", lw=1.1)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(left=0)
        ax.set_xlabel("Spatial distance in the global lane frame")
        ax.set_ylabel(f"-log10({fdr_method} q-value)")
    else:
        ax.text(0.5, 0.5, "No testable candidate pairs", ha="center", va="center")
    ax.set_title("B. FDR calls emerge from the data")

    ax = axes[1, 0]
    candidate_mismatches = scores["mismatches"][eligible_indices]
    bins = np.arange(0, 7)
    labels = ["0", "1", "2", "3", "4", "5", ">=6"]
    all_counts = np.asarray(
        [np.count_nonzero(candidate_mismatches == value) for value in range(6)]
        + [np.count_nonzero(candidate_mismatches >= 6)]
    )
    called_counts = np.asarray(
        [np.count_nonzero(candidate_mismatches[called] == value) for value in range(6)]
        + [np.count_nonzero(candidate_mismatches[called] >= 6)]
    )
    width = 0.38
    ax.bar(bins - width / 2, all_counts, width, color="#457B9D", label="all tested")
    ax.bar(bins + width / 2, called_counts, width, color="#D1495B", label="called")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(bins, labels)
    ax.set_xlabel("Observed R1+R2 mismatches")
    ax.set_ylabel("Candidate edges")
    ax.legend(frameon=False)
    ax.set_title("C. Quality-aware sequence compatibility")

    ax = axes[1, 1]
    values = [reads.count, kept_count, removed_count, int(called.sum())]
    labels = ["Input pairs", "Retained pairs", "Removed pairs", "Significant edges"]
    colors = ["#264653", "#2A9D8F", "#E76F51", "#F4A261"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(f"D. Filtering at {100 * target_fdr:g}% target FDR")

    fig.suptitle("FastqOptiFilter statistical diagnostics", fontsize=16, weight="bold")
    save_figure_atomically(fig, path)


def serializable_group_models(
    reads: LoadedReads,
    geometry: FlowcellGeometry,
    group_models: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    """Per lane-and-surface summary of the geometry null.

    ``null_cdf_at_maximum_radius`` should reproduce
    ``testable_random_pair_probability`` up to the add-one smoothing: the two
    are computed by completely different routes (counting neighbours with a
    k-d tree versus multiplying tile occupancies), so agreement is a direct
    check that the tabulated null really is a complete CDF over the testable
    relations.
    """
    summaries: list[dict[str, object]] = []
    for group_id, model in sorted(group_models.items()):
        member_indices = np.flatnonzero(geometry.group_ids == group_id)
        tile_counts = Counter(int(value) for value in reads.tiles[member_indices])
        total_pairs = int(model["total_random_pair_relations"])
        testable_pairs = int(model["testable_random_pair_relations"])
        testable_probability = testable_pairs / total_pairs if total_pairs else 0.0
        tested_hypotheses = int(model["tested_hypotheses"])
        observed_testable = int(model["testable_tested_hypotheses"])
        expected_testable = tested_hypotheses * testable_probability
        radii = np.asarray(model["radii"], dtype=np.float64)
        null_cdf = np.asarray(model["null_cdf"], dtype=np.float64)
        same_tile_pairs = sum(count * (count - 1) // 2 for count in tile_counts.values())
        summaries.append(
            {
                "group_id": group_id,
                "group_label": geometry.group_labels[group_id],
                "read_pairs": int(len(member_indices)),
                "tiles": len(tile_counts),
                "tile_read_pair_counts": {
                    str(key): value for key, value in sorted(tile_counts.items())
                },
                "total_random_pair_relations": total_pairs,
                "testable_random_pair_relations": testable_pairs,
                "testable_random_pair_probability": testable_probability,
                "same_tile_random_pair_probability": (
                    same_tile_pairs / total_pairs if total_pairs else 0.0
                ),
                "tested_hypotheses": tested_hypotheses,
                "testable_tested_hypotheses": observed_testable,
                "untestable_tested_hypotheses": tested_hypotheses - observed_testable,
                "expected_testable_hypotheses_if_all_null": expected_testable,
                "testable_enrichment_fold": (
                    observed_testable / expected_testable if expected_testable else None
                ),
                "maximum_null_radius": float(radii[-1]) if len(radii) else None,
                "null_cdf_at_maximum_radius": (
                    float(null_cdf[-1]) if len(null_cdf) else None
                ),
                "null_radii_evaluated": int(len(radii)),
            }
        )
    return summaries


def build_sensitivity_table(
    reads: LoadedReads,
    tested_left: np.ndarray,
    tested_right: np.ndarray,
    qvalue_sets: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in (0.001, 0.005, 0.01, 0.05, 0.1):
        row: dict[str, object] = {"fdr": threshold}
        for method, qvalues in qvalue_sets.items():
            edges, components, removed = filtered_count_at_threshold(
                reads.count, tested_left, tested_right, qvalues, threshold
            )
            row[method] = {
                "significant_edges": edges,
                "components": components,
                "read_pairs_removed": removed,
            }
        rows.append(row)
    return rows


def estimate_read_level_fdr(
    removal_details: dict[int, dict[str, object]],
    local_fdr: np.ndarray,
) -> dict[str, object]:
    """Estimated share of removed read pairs that are not proximity duplicates.

    Reads are removed per connected component, so even a read-level FDR target
    is not exactly the error rate among removals. Each removed read is
    supported by one hypothesis, and that hypothesis's local FDR is the
    posterior probability that it is null. Summing those posteriors over the
    removed reads estimates how many removals are spurious.
    """
    if not removal_details:
        return {
            "removed_read_pairs": 0,
            "expected_false_removals": 0.0,
            "estimated_read_level_fdr": 0.0,
        }
    supports = np.asarray(
        [int(details["support_edge_position"]) for details in removal_details.values()],
        dtype=np.int64,
    )
    supports = np.clip(supports, 0, max(len(local_fdr) - 1, 0))
    expected_false = float(np.sum(local_fdr[supports]))
    return {
        "removed_read_pairs": len(removal_details),
        "expected_false_removals": expected_false,
        "estimated_read_level_fdr": expected_false / len(removal_details),
        "method": "sum of the supporting hypothesis local FDR over removed read pairs",
    }


def write_markdown_report(path: Path, report: dict[str, object]) -> None:
    counts = report["counts"]  # type: ignore[assignment]
    filtering = report["filtering"]  # type: ignore[assignment]
    configuration = report["configuration"]  # type: ignore[assignment]
    quality_model = report["quality_model"]  # type: ignore[assignment]
    spatial_model = report["spatial_model"]  # type: ignore[assignment]
    geometry = report["flowcell_geometry"]  # type: ignore[assignment]
    calibration = report["null_calibration"]  # type: ignore[assignment]
    multiple_testing = report["multiple_testing"]  # type: ignore[assignment]
    decomposition = report.get("duplicate_decomposition", {})  # type: ignore[assignment]
    sensitivity = report["fdr_sensitivity"]  # type: ignore[assignment]
    read_level = multiple_testing["read_level_fdr_estimate"]

    lines = [
        "# FastqOptiFilter report",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Filtering result",
        "",
        f"- Input paired reads: **{counts['input_read_pairs']:,}**",
        f"- Retained paired reads: **{filtering['retained_read_pairs']:,}**",
        f"- Removed paired reads: **{filtering['removed_read_pairs']:,} "
        f"({100 * filtering['removed_fraction']:.3f}%)**",
        f"- Decision rule: **{multiple_testing['primary_method']} q <= "
        f"{configuration['target_fdr']:.4g}**"
        + (
            f" (chosen automatically: {multiple_testing['auto_selection_reason']})"
            if multiple_testing.get("auto_selection_reason")
            else ""
        ),
        f"- Hypothesis unit: **{configuration['inference_unit']}**",
        f"- Significant hypotheses: **{filtering['significant_candidate_edges']:,}**",
        f"- Optical/proximity components: **{filtering['optical_components']:,}**",
        f"- Estimated read-level FDR of the removals: "
        f"**{read_level['estimated_read_level_fdr']:.4g}** "
        f"({read_level['expected_false_removals']:.1f} expected false removals)",
        "",
        "## Flowcell geometry",
        "",
        f"- Tile identifier convention: **{geometry['tile_identifier_convention']}**",
        f"- Decoded tile grid: **{geometry['tile_grid_columns']} swath columns x "
        f"{geometry['tile_grid_rows']} tile rows**",
        f"- Tile neighbourhood searched: **{geometry['neighborhood']}**",
        f"- Spatial groups (lane and surface): **{len(geometry['spatial_groups'])}**",
        f"- Testable tested hypotheses: **{spatial_model['testable_tested_hypotheses']:,}** "
        f"of {counts['tested_candidate_edges']:,}",
        f"- Called edges that cross a tile boundary: "
        f"**{spatial_model['cross_tile_called_edges']:,}**",
        "",
        "## Duplicate decomposition",
        "",
        "How many reads are duplicated *because of where they sit*, measured by",
        "reshuffling the flowcell. The saturation distance is the length scale of the",
        "mechanism in this run, read from the data rather than assumed per platform.",
        "",
        (
            f"- Reads involved in proximity duplication: "
            f"**{decomposition['proximity_duplicate_reads']:,.0f}** "
            f"({100 * decomposition['proximity_duplicate_fraction']:.3f}% of the run)"
            if decomposition.get("permutations")
            else "- Not computed: --permutations was zero."
        ),
        (
            (
                f"- Length scale (excess saturates at): "
                f"**{decomposition['saturation_distance_px']:g} px**"
                if decomposition.get("proximity_duplication_detected")
                else "- Length scale: **not determined** (no proximity excess "
                "above permutation noise)"
            )
            if decomposition.get("permutations")
            else ""
        ),
        (
            f"- Reads whose look-alike sits at a chance distance: "
            f"**{decomposition['chance_positioned_reads']:,.0f}** "
            f"(upper bound on the library-duplicate load)"
            if decomposition.get("permutations")
            else ""
        ),
        "",
        "## Null calibration",
        "",
        "A spatial p-value is only meaningful if it is uniform when no proximity",
        "duplication is present. Each control below is a set of hypotheses for which",
        "that has to hold.",
        "",
        "| Control | Hypotheses | Atom at p=1 | Tail inflation | Max excess over uniform | O/E at 0.01 |",
        "|:---|---:|---:|---:|---:|---:|",
    ]

    def _format(value: object, spec: str = ".4g") -> str:
        if value is None:
            return "NA"
        return format(value, spec)

    for name, control in calibration["controls"].items():
        if not control.get("count"):
            continue
        tail = control.get("tail_observed_over_expected", {}).get("0.01", {})
        lines.append(
            f"| {name} | {control['count']:,} | "
            f"{_format(control.get('atom_at_one'), '.3f')} | "
            f"{_format(control.get('tail_inflation'), '.3f')} | "
            f"{_format(control.get('max_excess_over_uniform'), '.4f')} | "
            f"{_format(tail.get('observed_over_expected'), '.3f')} |"
        )
    lines.extend(
        [
            "",
            "A valid p-value only has to satisfy P(p <= alpha) <= alpha, so the column",
            "to read is the max excess over uniform: at or below zero means valid.",
            "An atom at p = 1 is expected whenever some look-alike partners sit on tiles",
            "too far apart to compare, and is not a defect.",
            "",
            "`permutation` shuffles which read sits at which observed cluster position,",
            "so it must be uniform if the geometry null is computed correctly.",
            "`sequence_incompatible` uses real read pairs that the quality model rejected",
            "as different molecules, so it must be uniform if within-lane exchangeability",
            "actually holds in this run. `analysis` is the analysed set itself and is",
            "expected to depart from uniform in the extreme tail when proximity",
            "duplicates are present.",
            "",
            "## Model summary",
            "",
            f"- Retrieved candidate pair relations: **{counts['retrieved_candidate_pair_relations']:,}**",
            f"- Sequence-compatible hypotheses tested spatially: **{counts['tested_candidate_edges']:,}**",
            f"- Probability a random pair sits in a testable tile relation: "
            f"**{spatial_model['testable_random_pair_probability']:.6g}**",
            f"- Estimated null fraction among tested edges (pi0): "
            f"**{multiple_testing['mixture']['pi0']:.4g}**",
            (
                "- Optional sequence-compatibility p screen: **disabled**"
                if configuration["sequence_min_p"] == 0
                else f"- Sequence compatibility threshold: **p >= {configuration['sequence_min_p']:.2g}**"
            ),
            f"- Estimated same-template prior among all within-lane pair relations: "
            f"**{quality_model['estimated_same_template_prior']:.6g}**",
            "- Spatial p-values use the complete observed lane/tile/X/Y geometry. "
            "No fixed distance threshold is used.",
            "",
            "## FDR sensitivity",
            "",
            "| FDR | permutation removed | BY removed | BH removed | local-FDR removed |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sensitivity:
        permutation_removed = (
            f"{row['permutation']['read_pairs_removed']:,}"
            if "permutation" in row
            else "NA"
        )
        lines.append(
            f"| {row['fdr']:.3g} | {permutation_removed} | "
            f"{row['by']['read_pairs_removed']:,} | "
            f"{row['bh']['read_pairs_removed']:,} | "
            f"{row['local_fdr']['read_pairs_removed']:,} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "With the default read unit, each read is tested once: given its own",
            "position and how many sequence-compatible partners it has anywhere in the",
            "lane and surface, how surprising is it that the closest of them landed as",
            "close as it did? Neighbours are counted around the read itself, so the",
            "test adapts to local cluster density: the same separation is unremarkable",
            "in a crowded patch and surprising in a sparse one. Testing reads rather",
            "than candidate relations also stops one family of k identical reads from",
            "raising k(k-1)/2 hypotheses and dominating the correction, and it makes",
            "the controlled error rate the read-level one that matters, since reads",
            "are what get removed.",
            "",
            "The default Benjamini-Yekutieli correction is valid under arbitrary",
            "dependence. That matters here because hypotheses share reads and share",
            "the observed cluster pattern, which is not a dependence structure",
            "Benjamini-Hochberg is guaranteed against. BH and the local-FDR mixture",
            "are reported alongside it; the local FDR is also the per-hypothesis",
            "posterior used for the read-level FDR estimate above.",
            "",
            "Read removal is component-based, so the nominal FDR is still not an exact",
            "read-level error rate. Valid inference assumes sequence-compatible",
            "non-optical copies are exchangeable over observed cluster coordinates",
            "within a lane and surface, which the null-calibration table tests",
            "directly. UMIs, if available, remain preferable for molecular-complexity",
            "estimation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="FastqOptiFilter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Quality-aware, FDR-controlled optical/proximity duplicate filtering "
            "for synchronized paired-end Illumina FASTQs. No fixed spatial-distance "
            "cutoff is used."
        ),
    )
    parser.add_argument("--r1", required=True, type=Path, help="Input R1 FASTQ[.gz]")
    parser.add_argument("--r2", required=True, type=Path, help="Input R2 FASTQ[.gz]")
    parser.add_argument("--output-r1", required=True, type=Path, help="Filtered R1 FASTQ[.gz]")
    parser.add_argument("--output-r2", required=True, type=Path, help="Filtered R2 FASTQ[.gz]")
    parser.add_argument("--report-json", required=True, type=Path, help="Machine-readable JSON report")
    parser.add_argument("--report-md", type=Path, help="Human-readable Markdown report")
    parser.add_argument("--candidate-audit", type=Path, help="Per-tested-edge TSV[.gz]")
    parser.add_argument("--removed-audit", type=Path, help="Per-removed-pair TSV[.gz]")
    parser.add_argument(
        "--read-audit",
        type=Path,
        help="Per-tested-read TSV[.gz]; written when --inference-unit is read",
    )
    parser.add_argument("--diagnostic-plot", type=Path, help="Diagnostic PNG")
    parser.add_argument(
        "--qq-plot",
        type=Path,
        help="QQ/calibration PNG comparing the null controls with a uniform null",
    )
    parser.add_argument("--log", type=Path, help="Timestamped progress/ETA log")
    parser.add_argument("--fdr", type=float, default=0.01, help="Target edge FDR")
    parser.add_argument(
        "--fdr-method",
        choices=("auto", "permutation", "by", "local-fdr", "bh", "weighted-bh"),
        default="auto",
        help=(
            "Primary multiple-testing method. auto picks local-fdr when the "
            "run is duplicate-dominated and permutation otherwise, because a "
            "tail test is underpowered once most hypotheses are true; "
            "permutation reshuffles the whole "
            "flowcell and measures how many calls a null run makes, so it needs "
            "no assumption about dependence and pays no analytic penalty; by is "
            "Benjamini-Yekutieli, valid under arbitrary dependence but costs a "
            "factor of about log(m); bh assumes positive regression dependence; "
            "local-fdr fits a two-groups mixture and reports a per-hypothesis "
            "posterior; weighted-bh uses sequence-only weights, edge unit only"
        ),
    )
    parser.add_argument(
        "--inference-unit",
        choices=("read", "edge"),
        default="read",
        help=(
            "What a hypothesis is. read tests each read once against its "
            "nearest sequence-compatible partner, so a family of k identical "
            "reads raises k hypotheses rather than k(k-1)/2, and the "
            "controlled error rate is the read-level one. edge reproduces the "
            "original per-candidate-relation test"
        ),
    )
    parser.add_argument(
        "--tile-neighborhood",
        choices=("same-tile", "adjacent", "lane"),
        default="adjacent",
        help=(
            "Which tile relations may carry a spatial test. adjacent also "
            "searches the tiles touching a read's own tile on the decoded "
            "flowcell grid; lane compares every tile on the same surface and "
            "is much slower"
        ),
    )
    parser.add_argument(
        "--tile-gap",
        type=int,
        default=0,
        help=(
            "Pixels of dead space assumed between neighbouring tiles when "
            "placing them in one lane-wide frame. Affects power only: the "
            "observed distances and the null both use this frame"
        ),
    )
    parser.add_argument(
        "--max-grid-step",
        type=int,
        default=1,
        help="Tile-grid rings searched when --tile-neighborhood is adjacent",
    )
    parser.add_argument(
        "--null-resolution",
        type=int,
        default=384,
        help=(
            "Log-spaced radii above 256 pixels at which the geometry null is "
            "tabulated; every integer radius at or below 256 is always used"
        ),
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=20,
        help=(
            "Reshuffled replicates of the flowcell used to measure the FDR "
            "directly. Zero disables them and the permutation method with them"
        ),
    )
    parser.add_argument(
        "--spatial-test",
        dest="spatial_test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require spatial evidence before removing a read. With "
            "--no-spatial-test every sequence-compatible candidate is treated "
            "as a duplicate, which deduplicates by sequence alone and removes "
            "library copies and repeated independent molecules along with "
            "optical ones"
        ),
    )
    parser.add_argument(
        "--min-log10-bayes-factor",
        type=float,
        default=0.0,
        help=(
            "Sequence evidence a candidate needs to be tested at all. Lower it "
            "to keep weaker matches; set it very negative with "
            "--no-spatial-test to remove every retrieved candidate"
        ),
    )
    parser.add_argument(
        "--auto-pi0-threshold",
        type=float,
        default=0.5,
        help=(
            "With --fdr-method auto, use local-fdr when the estimated null "
            "proportion falls below this and a tail test would therefore be "
            "leaving most of the real duplicates behind"
        ),
    )
    parser.add_argument(
        "--pi0-lambda",
        type=float,
        default=0.5,
        help="Storey tuning point for the null proportion used by local-fdr",
    )
    parser.add_argument(
        "--null-check-seed",
        type=int,
        default=20260903,
        help="Seed for the permutation null used by the calibration check",
    )
    parser.add_argument(
        "--sequence-min-p",
        type=float,
        default=0.0,
        help=(
            "Minimum Phred-error compatibility p-value for spatial testing; "
            "zero disables this optional extra screen"
        ),
    )
    parser.add_argument(
        "--seed-length",
        type=int,
        default=20,
        help=(
            "Maximum exact indexing seed length. The reported base qualities "
            "can select a shorter one, because a seed only retrieves a "
            "duplicate when it is error-free in both copies"
        ),
    )
    parser.add_argument(
        "--adaptive-seed",
        dest="adaptive_seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Shorten the seed when the base qualities imply more mismatches "
            "between two copies of a molecule than the seed count can tolerate"
        ),
    )
    parser.add_argument(
        "--max-seed-bucket",
        type=int,
        default=500,
        help="Skip larger non-informative seed buckets",
    )
    parser.add_argument(
        "--max-exact-family",
        type=int,
        default=5000,
        help="Reject exact families larger than this as likely artifacts",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=5_000_000,
        help="Safety limit for retrieved candidate relations",
    )
    parser.add_argument(
        "--spatial-metric",
        choices=("chebyshev", "euclidean"),
        default="chebyshev",
        help="Distance metric calibrated against empirical geometry",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Worker threads; 0 uses all detected CPU cores",
    )
    parser.add_argument(
        "--score-chunk",
        type=int,
        default=1000,
        help="Candidate relations per parallel quality-scoring task",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="Seconds between progress/ETA messages",
    )
    parser.add_argument("--gzip-level", type=int, default=6, choices=range(1, 10))
    parser.add_argument(
        "--unparsed",
        choices=("error", "keep"),
        default="error",
        help="Action for headers without lane/tile/X/Y coordinates",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    return parser


def validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if not (0.0 < args.fdr < 1.0):
        parser.error("--fdr must be between 0 and 1")
    if not (0.0 <= args.sequence_min_p <= 1.0):
        parser.error("--sequence-min-p must be between 0 and 1")
    if args.seed_length < 8:
        parser.error("--seed-length must be at least 8")
    if args.max_seed_bucket < 2 or args.max_exact_family < 2:
        parser.error("candidate family limits must be at least 2")
    if args.max_candidates < 1 or args.score_chunk < 1:
        parser.error("candidate/chunk limits must be positive")
    if args.tile_gap < 0:
        parser.error("--tile-gap must not be negative")
    if args.max_grid_step < 1:
        parser.error("--max-grid-step must be at least 1")
    if args.null_resolution < 16:
        parser.error("--null-resolution must be at least 16")
    if not (0.0 < args.pi0_lambda < 1.0):
        parser.error("--pi0-lambda must be strictly between 0 and 1")
    if args.permutations < 0:
        parser.error("--permutations must not be negative")
    if args.fdr_method == "permutation" and args.permutations < 1:
        parser.error("--fdr-method permutation needs --permutations of at least 1")
    if not (0.0 < args.auto_pi0_threshold < 1.0):
        parser.error("--auto-pi0-threshold must be strictly between 0 and 1")
    if args.fdr_method == "weighted-bh" and args.inference_unit != "edge":
        parser.error(
            "--fdr-method weighted-bh weights individual candidate relations by "
            "their sequence evidence, so it needs --inference-unit edge"
        )
    if args.read_audit is not None and args.inference_unit != "read":
        parser.error("--read-audit requires --inference-unit read")
    if args.progress_interval <= 0:
        parser.error("--progress-interval must be positive")
    threads = os.cpu_count() or 1 if args.threads == 0 else args.threads
    if threads < 1:
        parser.error("--threads must be 0 or a positive integer")
    args.threads = threads

    input_paths = [args.r1.resolve(), args.r2.resolve()]
    for path in input_paths:
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")
    output_paths = [
        value.resolve()
        for value in (
            args.output_r1,
            args.output_r2,
            args.report_json,
            args.report_md,
            args.candidate_audit,
            args.removed_audit,
            args.read_audit,
            args.diagnostic_plot,
            args.qq_plot,
            args.log,
        )
        if value is not None
    ]
    if len(set(output_paths)) != len(output_paths):
        parser.error("all output paths must be distinct")
    if set(input_paths) & set(output_paths):
        parser.error("an output path cannot overwrite an input FASTQ")
    if not args.force:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            parser.error("output already exists (use --force): " + ", ".join(map(str, existing)))
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    return threads


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    threads = validate_arguments(args, parser)
    logger = RunLogger(args.log, args.progress_interval)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc)
    logger.log(
        "run",
        f"status=start tool=FastqOptiFilter version={VERSION} threads={threads} "
        f"fdr_method={args.fdr_method} target_fdr={args.fdr:g}",
    )
    try:
        reads = load_fastqs(args.r1, args.r2, args.unparsed, logger)
        logger.log(
            "input",
            f"read_pairs={reads.count:,} read_lengths={reads.read1_length}+{reads.read2_length} "
            f"lanes={len(reads.lane_labels)} unparsed={reads.unparsed_headers:,}",
        )
        geometry = build_geometry(
            reads,
            args.tile_neighborhood,
            args.tile_gap,
            args.max_grid_step,
            logger,
        )
        matrices = encode_matrices(reads, logger)
        seed_length, seed_details = choose_seed_length(
            matrices,
            reads.count,
            args.seed_length,
            args.adaptive_seed,
            logger,
        )
        edge_left, edge_right, candidate_details = find_candidates(
            reads,
            seed_length,
            args.max_seed_bucket,
            args.max_exact_family,
            args.max_candidates,
            logger,
        )
        candidate_details["seed_length_selection"] = seed_details
        scores = score_candidates(
            edge_left,
            edge_right,
            matrices,
            threads,
            args.score_chunk,
            logger,
        )

        sequence_compatible = (
            (scores["compatibility_p"] >= args.sequence_min_p)
            & (scores["log10_bayes_factor"] > args.min_log10_bayes_factor)
        )
        eligible_indices = np.flatnonzero(sequence_compatible)
        if len(eligible_indices) == 0:
            logger.log("quality_model", "eligible_candidate_pairs=0; no reads will be removed")

        total_possible_relations = 0
        for lane_id in set(int(value) for value in reads.lane_ids if value >= 0):
            lane_count = int(np.count_nonzero(reads.lane_ids == lane_id))
            total_possible_relations += lane_count * (lane_count - 1) // 2
        prior, posterior, em_iterations = estimate_same_template_prior(
            scores["log10_bayes_factor"], sequence_compatible, total_possible_relations
        )
        logger.log(
            "quality_model",
            f"eligible_candidate_pairs={len(eligible_indices):,} "
            f"estimated_same_template_prior={prior:.8g} em_iterations={em_iterations}",
        )

        # The tabulated geometry null drives the edge-level p-values. The
        # read-level test does not use it, so it is only built when the
        # inference or an edge-level output actually needs it.
        needs_edge_null = (
            args.inference_unit == "edge"
            or args.candidate_audit is not None
            or args.diagnostic_plot is not None
        )
        spatial_p, distances, testable, group_models = calibrate_spatial_pvalues(
            reads,
            geometry,
            edge_left,
            edge_right,
            eligible_indices,
            args.spatial_metric,
            args.null_resolution,
            threads,
            logger,
            tabulate_null=needs_edge_null,
        )
        tested_left = edge_left[eligible_indices]
        tested_right = edge_right[eligible_indices]
        conditional_p = conditional_spatial_pvalues(
            spatial_p, testable, geometry.group_ids[tested_left], group_models
        )
        posterior_tested = posterior[eligible_indices]
        compatibility_tested = scores["compatibility_p"][eligible_indices]
        weighted_q, weights = sequence_weighted_bh(
            spatial_p, posterior_tested, compatibility_tested
        )

        read_test: ReadLevelTest | None = None
        neighbour_trees: dict[int, cKDTree] | None = None
        if args.inference_unit == "read":
            neighbour_trees = build_neighbour_trees(geometry, logger)
            read_test = read_level_spatial_test(
                reads,
                geometry,
                tested_left,
                tested_right,
                distances,
                testable,
                args.spatial_metric,
                logger,
                trees=neighbour_trees,
            )
            unit_p = read_test.pvalue
            unit_left = read_test.read_index
            unit_right = read_test.partner_index
        else:
            # A pair on tiles too far apart to compare carries no spatial
            # evidence, so it stays at a p-value of one.
            unit_p = conditional_p
            unit_left = tested_left
            unit_right = tested_right

        null_controls = build_null_controls(
            reads,
            geometry,
            group_models,
            edge_left,
            edge_right,
            eligible_indices,
            scores,
            distances,
            testable,
            conditional_p,
            unit_p,
            args.inference_unit,
            args.spatial_metric,
            args.null_check_seed,
            logger,
        )

        permuted_replicates: list[np.ndarray] = []
        permuted_distances: list[np.ndarray] = []
        permutation_diagnostics: dict[str, object] = {"permutations": 0}
        if args.permutations > 0:
            p_norm = math.inf if args.spatial_metric == "chebyshev" else 2.0
            tracker = logger.tracker(
                "permutation_null", args.permutations, "permutations"
            )
            for replicate in range(args.permutations):
                shuffled = shuffle_positions(
                    geometry, args.null_check_seed + 1000 * (replicate + 1)
                )
                shuffled_distances, shuffled_testable = pair_distances(
                    shuffled, tested_left, tested_right, p_norm
                )
                if args.inference_unit == "read":
                    replicate_test = read_level_spatial_test(
                        reads,
                        shuffled,
                        tested_left,
                        tested_right,
                        shuffled_distances,
                        shuffled_testable,
                        args.spatial_metric,
                        logger,
                        trees=neighbour_trees,
                        quiet=True,
                    )
                    permuted_replicates.append(replicate_test.pvalue)
                    permuted_distances.append(replicate_test.nearest_distance)
                else:
                    replicate_p = np.ones(len(tested_left), dtype=np.float64)
                    for group_id, model in group_models.items():
                        positions = np.flatnonzero(
                            (geometry.group_ids[tested_left] == group_id)
                            & shuffled_testable
                        )
                        if len(positions):
                            replicate_p[positions] = spatial_null_lookup(
                                model, shuffled_distances[positions]
                            )
                    permuted_distances.append(shuffled_distances)
                    permuted_replicates.append(
                        conditional_spatial_pvalues(
                            replicate_p,
                            shuffled_testable,
                            geometry.group_ids[tested_left],
                            group_models,
                        )
                    )
                tracker.update(replicate + 1)
            tracker.finish(detail=f"replicates={len(permuted_replicates)}")

        local_fdr, mixture_diagnostics = local_false_discovery_rate(
            unit_p, args.pi0_lambda
        )
        # A read or pair with no testable spatial relation carries no evidence.
        local_fdr = np.where(unit_p >= 1.0, 1.0, local_fdr)
        qvalue_sets = {
            "local_fdr": local_fdr_qvalues(local_fdr),
            "bh": benjamini_hochberg(unit_p),
            "by": benjamini_yekutieli(unit_p),
        }
        if permuted_replicates:
            qvalue_sets["permutation"], permutation_diagnostics = (
                permutation_fdr_qvalues(unit_p, permuted_replicates)
            )
        if args.inference_unit == "edge":
            qvalue_sets["weighted_bh"] = weighted_q
        # A hypothesis whose look-alikes all landed on tiles too far apart to
        # compare, or that names no partner at all, carries no spatial evidence
        # and must not be rejected on the strength of the rest of the run.
        no_evidence = (unit_p >= 1.0) | (unit_right < 0) | (unit_left < 0)
        for values in qvalue_sets.values():
            values[no_evidence] = 1.0
        local_fdr[no_evidence] = 1.0
        method_name = args.fdr_method
        auto_reason = None
        if method_name == "auto":
            # A tail test asks whether a p-value is extreme against a uniform
            # null. Once most hypotheses are genuinely non-null that question
            # has the wrong answer: a real duplicate a long way from its twin
            # is unremarkable on its own, and only the fitted mixture knows
            # that almost everything around it is also a duplicate.
            dominated = mixture_diagnostics["pi0"] < args.auto_pi0_threshold
            method_name = "local-fdr" if dominated else "permutation"
            if method_name == "permutation" and not permuted_replicates:
                method_name = "by"
            auto_reason = (
                f"pi0={mixture_diagnostics['pi0']:.4g} "
                f"{'<' if dominated else '>='} {args.auto_pi0_threshold:g}"
            )
            logger.log(
                "fdr",
                f"auto_selected={method_name} reason={auto_reason}",
            )
        method_key = method_name.replace("-", "_")
        selected_q = qvalue_sets[method_key]
        if not args.spatial_test:
            # Sequence evidence alone decides. Every candidate relation that
            # survived the quality model is collapsed, so this removes library
            # copies and independent molecules that happen to share an insert
            # along with the proximity duplicates.
            selected_q = np.zeros(len(unit_p), dtype=np.float64)
            if args.inference_unit == "read":
                selected_q[unit_right < 0] = 1.0
        logger.log(
            "fdr",
            f"unit={args.inference_unit} method={method_name} "
            f"hypotheses={len(unit_p):,} pi0={mixture_diagnostics['pi0']:.6g} "
            f"storey_pi0={mixture_diagnostics['storey_pi0']:.6g}",
        )
        removed, removal_details, filtering_stats = component_filter_decisions(
            reads.count,
            unit_left,
            unit_right,
            selected_q,
            matrices.quality_sums,
            args.fdr,
        )
        decomposition = duplicate_decomposition(
            read_test.nearest_distance if read_test is not None else distances,
            permuted_distances,
            reads.count,
        )
        if decomposition.get("permutations"):
            logger.log(
                "decomposition",
                f"proximity_duplicate_reads={decomposition['proximity_duplicate_reads']:,.0f} "
                f"({100 * decomposition['proximity_duplicate_fraction']:.3f}%) "
                f"chance_positioned_reads={decomposition['chance_positioned_reads']:,.0f} "
                f"saturation_distance_px={decomposition['saturation_distance_px']}",
            )
        read_level_fdr = estimate_read_level_fdr(removal_details, local_fdr)
        # The candidate audit stays edge-level whatever the inference unit, so
        # that every retrieved relation remains inspectable. Its q-values are
        # therefore always computed on the edge p-values.
        if needs_edge_null:
            edge_local_fdr, _ = local_false_discovery_rate(
                conditional_p, args.pi0_lambda
            )
            edge_qvalue_sets = {
                "local_fdr": local_fdr_qvalues(edge_local_fdr),
                "bh": benjamini_hochberg(spatial_p),
                "by": benjamini_yekutieli(spatial_p),
                "weighted_bh": weighted_q,
            }
            edge_selected_q = edge_qvalue_sets[
                method_key if method_key in edge_qvalue_sets else "by"
            ]
        logger.log(
            "fdr",
            f"tested_edges={len(eligible_indices):,} testable_edges={int(testable.sum()):,} "
            f"significant_edges={filtering_stats['significant_candidate_edges']:,} "
            f"components={filtering_stats['optical_components']:,} "
            f"pairs_marked_for_removal={len(removed):,} "
            f"estimated_read_level_fdr={read_level_fdr['estimated_read_level_fdr']:.4g}",
        )

        kept_count, removed_count = write_filtered_fastqs(
            args.r1,
            args.r2,
            args.output_r1,
            args.output_r2,
            removed,
            reads.count,
            args.gzip_level,
            threads,
            logger,
        )
        if removed_count != len(removed):
            raise RuntimeError("Internal removed-pair count mismatch")

        if args.candidate_audit is not None:
            logger.log("audit", f"status=start file={args.candidate_audit}")
            write_candidate_audit(
                args.candidate_audit,
                reads,
                geometry,
                edge_left,
                edge_right,
                eligible_indices,
                scores,
                posterior,
                spatial_p,
                conditional_p,
                distances,
                testable,
                edge_local_fdr,
                edge_qvalue_sets,
                weights,
                edge_selected_q,
                args.fdr,
            )
            logger.log("audit", f"status=complete tested_edges={len(eligible_indices):,}")
        if args.removed_audit is not None:
            write_removed_audit(
                args.removed_audit,
                reads,
                removal_details,
                read_test,
                tested_left,
                tested_right,
                eligible_indices,
                scores,
                unit_p,
                distances,
                local_fdr,
                selected_q,
                unit_left,
                unit_right,
            )
        if args.read_audit is not None and read_test is not None:
            write_read_audit(
                args.read_audit,
                reads,
                geometry,
                read_test,
                local_fdr,
                qvalue_sets,
                selected_q,
                removed,
                args.fdr,
            )

        sensitivity = build_sensitivity_table(
            reads, unit_left, unit_right, qvalue_sets
        )
        group_summaries = serializable_group_models(reads, geometry, group_models)
        called = selected_q <= args.fdr
        if read_test is not None:
            called_distances = read_test.nearest_distance[called]
        else:
            called_distances = distances[called & testable & np.isfinite(distances)]
        testable_random_probability = 0.0
        total_testable_relations = sum(
            int(model["testable_random_pair_relations"]) for model in group_models.values()
        )
        total_group_relations = sum(
            int(model["total_random_pair_relations"]) for model in group_models.values()
        )
        if total_group_relations:
            testable_random_probability = total_testable_relations / total_group_relations
        cross_tile_called = int(
            np.count_nonzero(
                called & (reads.tiles[unit_left] != reads.tiles[unit_right])
            )
        )

        elapsed = time.monotonic() - started
        filtering_stats.update(
            {
                "retained_read_pairs": kept_count,
                "removed_read_pairs": removed_count,
                "removed_fraction": removed_count / reads.count,
                "maximum_called_distance_data_derived": (
                    float(called_distances.max()) if len(called_distances) else None
                ),
                "minimum_called_q": float(selected_q[called].min()) if called.any() else None,
                "maximum_called_q": float(selected_q[called].max()) if called.any() else None,
            }
        )
        report: dict[str, object] = {
            "tool": "FastqOptiFilter",
            "version": VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "command": [sys.executable, *sys.argv],
            "inputs": {
                "r1": str(args.r1.resolve()),
                "r2": str(args.r2.resolve()),
                "r1_bytes": args.r1.stat().st_size,
                "r2_bytes": args.r2.stat().st_size,
            },
            "outputs": {
                "r1": str(args.output_r1.resolve()),
                "r2": str(args.output_r2.resolve()),
                "candidate_audit": str(args.candidate_audit.resolve()) if args.candidate_audit else None,
                "removed_audit": str(args.removed_audit.resolve()) if args.removed_audit else None,
                "read_audit": str(args.read_audit.resolve()) if args.read_audit else None,
                "diagnostic_plot": str(args.diagnostic_plot.resolve()) if args.diagnostic_plot else None,
                "qq_plot": str(args.qq_plot.resolve()) if args.qq_plot else None,
                "log": str(args.log.resolve()) if args.log else None,
            },
            "configuration": {
                "target_fdr": args.fdr,
                "fdr_method": args.fdr_method,
                "inference_unit": args.inference_unit,
                "spatial_test": args.spatial_test,
                "permutations": args.permutations,
                "min_log10_bayes_factor": args.min_log10_bayes_factor,
                "adaptive_seed": args.adaptive_seed,
                "sequence_min_p": args.sequence_min_p,
                "spatial_metric": args.spatial_metric,
                "tile_neighborhood": args.tile_neighborhood,
                "tile_gap": args.tile_gap,
                "max_grid_step": args.max_grid_step,
                "null_resolution": args.null_resolution,
                "pi0_lambda": args.pi0_lambda,
                "threads": threads,
                "gzip_level": args.gzip_level,
                "unparsed_header_action": args.unparsed,
            },
            "counts": {
                "input_read_pairs": reads.count,
                "read1_length": reads.read1_length,
                "read2_length": reads.read2_length,
                "lanes": len(reads.lane_labels),
                "unparsed_headers": reads.unparsed_headers,
                "total_possible_within_lane_pair_relations": total_possible_relations,
                "retrieved_candidate_pair_relations": len(edge_left),
                "tested_candidate_edges": len(eligible_indices),
            },
            "candidate_retrieval": candidate_details,
            "quality_model": {
                "model": "cycle-specific empirical-base-prior latent-template likelihood with Phred error probabilities",
                "compatibility_tail": "exact Poisson-binomial upper tail for the observed mismatch count",
                "estimated_same_template_prior": prior,
                "em_iterations": em_iterations,
                "eligible_log10_bayes_factor_rule": "> 0",
                "median_eligible_log10_bayes_factor": (
                    float(np.median(scores["log10_bayes_factor"][eligible_indices]))
                    if len(eligible_indices)
                    else None
                ),
                "median_eligible_compatibility_p": (
                    float(np.median(compatibility_tested)) if len(eligible_indices) else None
                ),
            },
            "flowcell_geometry": {
                "tile_identifier_convention": geometry.layout.convention,
                "tile_grid_columns": geometry.layout.columns,
                "tile_grid_rows": geometry.layout.rows,
                "tile_cell_pixels": {"x": geometry.cell_x, "y": geometry.cell_y},
                "spatial_groups": geometry.group_labels,
                "neighborhood": geometry.neighborhood,
                "frame_note": (
                    "Tiles are placed in one lane-wide frame per surface. Observed "
                    "distances and the null are built in the same frame, so the "
                    "assumed inter-tile gap changes power but not validity."
                ),
            },
            "spatial_model": {
                "null": "random unordered pair of observed cluster positions within one lane and surface",
                "p_value_event": "testable tile relation and distance less than or equal to observed",
                "fixed_distance_cutoff_used": False,
                "metric": args.spatial_metric,
                "testable_tested_hypotheses": int(testable.sum()),
                "untestable_tested_hypotheses": int((~testable).sum()),
                "testable_random_pair_probability": testable_random_probability,
                "cross_tile_called_edges": cross_tile_called,
                "groups": group_summaries,
            },
            "duplicate_decomposition": decomposition,
            "null_calibration": {
                "purpose": (
                    "A spatial p-value is only meaningful if it is uniform when no "
                    "proximity duplication is present. Each control below is a set "
                    "of hypotheses for which that has to hold."
                ),
                "controls": {
                    name: {
                        "description": control["description"],
                        **control["statistics"],
                    }
                    for name, control in null_controls.items()
                },
                "interpretation": (
                    "max_excess_over_uniform at or below zero means the p-values are "
                    "valid. A permutation control near zero means the geometry null "
                    "is computed correctly. A sequence-incompatible control that is "
                    "clearly inflated means real reads carry spatial structure that "
                    "within-lane exchangeability does not capture, so q-values are "
                    "optimistic by roughly that factor and a stricter target FDR, or "
                    "upstream removal of the low-complexity reads driving it, is "
                    "warranted."
                ),
            },
            "multiple_testing": {
                "primary_method": method_name,
                "requested_method": args.fdr_method,
                "auto_selection_reason": auto_reason,
                "local_fdr": (
                    "Two-groups mixture on the conditional spatial p-value with a "
                    "monotone alternative estimated by the Grenander estimator; "
                    "q-values are the running mean local FDR of the rejected set"
                ),
                "bh": "Benjamini-Hochberg over all sequence-compatible candidate edges",
                "by": "Benjamini-Yekutieli, valid under the dependence induced by shared reads",
                "sequence_weighted_bh": "BH on spatial p divided by normalized sequence-only weight",
                "weight_formula": "posterior_same_template * sqrt(sequence_compatibility_p), normalized to mean 1",
                "mixture": mixture_diagnostics,
                "permutation": {
                    "method": (
                        "expected calls from reshuffled flowcells divided by "
                        "observed calls, at every threshold"
                    ),
                    **permutation_diagnostics,
                },
                "scope": "candidate-edge FDR; component-based read-removal FDR can differ",
                "read_level_fdr_estimate": read_level_fdr,
            },
            "filtering": filtering_stats,
            "fdr_sensitivity": sensitivity,
            "limitations": [
                "Candidate retrieval requires at least one exact seed or exact full paired sequence.",
                "The null assumes sequence-compatible non-optical copies are exchangeable over observed positions within one lane and surface.",
                "Base quality values are assumed approximately calibrated and conditionally independent by cycle.",
                "The exact Poisson-binomial compatibility test treats per-cycle sequencing errors as conditionally independent.",
                "Cross-tile adjacency is decoded from the tile identifier; an unrecognised convention falls back to a single nominal column.",
                "The local-FDR mixture needs enough real duplicates to estimate the alternative; on a library with almost none it reverts to calling nothing.",
                "Without UMIs, independently generated biological molecules with the same insert cannot always be distinguished.",
            ],
        }

        if args.diagnostic_plot is not None:
            logger.log("plot", f"status=start file={args.diagnostic_plot}")
            make_diagnostic_plot(
                args.diagnostic_plot,
                reads,
                edge_left,
                edge_right,
                eligible_indices,
                scores,
                distances,
                testable,
                group_models,
                edge_qvalue_sets["bh"],
                weighted_q,
                edge_selected_q,
                args.fdr_method,
                args.fdr,
                kept_count,
                removed_count,
            )
            logger.log("plot", "status=complete")

        if args.qq_plot is not None:
            logger.log("plot", f"status=start file={args.qq_plot}")
            make_qq_plot(
                args.qq_plot,
                null_controls,
                local_fdr,
                unit_p,
                (
                    read_test.nearest_distance
                    if read_test is not None
                    else distances
                ),
                args.fdr,
                called,
                args.inference_unit,
            )
            logger.log("plot", "status=complete")

        args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if args.report_md is not None:
            write_markdown_report(args.report_md, report)
        logger.log(
            "run",
            f"status=complete elapsed={format_duration(time.monotonic() - started)} "
            f"retained={kept_count:,} removed={removed_count:,} eta=00:00",
        )
        return 0
    except Exception as exc:
        logger.log("run", f"status=failed error={type(exc).__name__}:{exc}")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())

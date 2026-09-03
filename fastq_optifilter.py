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
3. calculates an empirical spatial p-value from the complete observed
   run/lane/tile geometry, including the chance that two random clusters fall
   on different tiles;
4. controls candidate-edge FDR with Benjamini-Hochberg (BH), optionally using
   sequence-evidence weights that are independent of spatial position under
   the null; and
5. retains the highest-quality read pair from each significant spatial
   component and writes synchronized filtered FASTQs.

The statistical null assumes that, in the absence of optical/proximity
duplication, sequence-compatible molecules are exchangeable over cluster
positions within each run/lane. The output report includes both unweighted BH
q-values and sequence-weighted BH q-values so the assumption can be audited.

Requirements: Python >=3.10, numpy, scipy >=1.17, matplotlib. If pigz is available,
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
from typing import Iterator, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit
from scipy.stats import poisson_binom


VERSION = "1.0.0"
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


def tile_null_counts(
    points: np.ndarray, radii: np.ndarray, p_norm: float
) -> np.ndarray:
    if len(points) < 2 or len(radii) == 0:
        return np.zeros(len(radii), dtype=np.float64)
    tree = cKDTree(points)
    ordered_with_self = tree.count_neighbors(
        tree, radii, p=p_norm, cumulative=True
    ).astype(np.float64)
    return (ordered_with_self - len(points)) / 2.0


def calibrate_spatial_pvalues(
    reads: LoadedReads,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    eligible_indices: np.ndarray,
    metric: str,
    threads: int,
    logger: RunLogger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, object]]]:
    tested_left = edge_left[eligible_indices]
    tested_right = edge_right[eligible_indices]
    tested_count = len(eligible_indices)
    spatial_p = np.ones(tested_count, dtype=np.float64)
    distances = np.full(tested_count, np.nan, dtype=np.float64)
    same_tile = (
        (reads.lane_ids[tested_left] == reads.lane_ids[tested_right])
        & (reads.tiles[tested_left] == reads.tiles[tested_right])
        & (reads.lane_ids[tested_left] >= 0)
    )
    delta_x = np.abs(reads.x[tested_left] - reads.x[tested_right]).astype(np.float64)
    delta_y = np.abs(reads.y[tested_left] - reads.y[tested_right]).astype(np.float64)
    if metric == "chebyshev":
        distances[same_tile] = np.maximum(delta_x[same_tile], delta_y[same_tile])
        p_norm = math.inf
    else:
        distances[same_tile] = np.hypot(delta_x[same_tile], delta_y[same_tile])
        p_norm = 2.0

    lane_ids = sorted(set(int(value) for value in reads.lane_ids if value >= 0))
    tile_tasks: list[tuple[int, int, np.ndarray, np.ndarray, float]] = []
    lane_models: dict[int, dict[str, object]] = {}
    for lane_id in lane_ids:
        lane_read_indices = np.flatnonzero(reads.lane_ids == lane_id)
        lane_test_mask = reads.lane_ids[tested_left] == lane_id
        lane_same_mask = lane_test_mask & same_tile
        radii = np.unique(distances[lane_same_mask])
        total_pairs = len(lane_read_indices) * (len(lane_read_indices) - 1) // 2
        lane_models[lane_id] = {
            "radii": radii,
            "null_counts": np.zeros(len(radii), dtype=np.float64),
            "total_random_pair_relations": total_pairs,
            "tested_hypotheses": int(lane_test_mask.sum()),
            "same_tile_tested_hypotheses": int(lane_same_mask.sum()),
        }
        for tile in sorted(set(int(value) for value in reads.tiles[lane_read_indices])):
            tile_indices = lane_read_indices[reads.tiles[lane_read_indices] == tile]
            points = np.column_stack((reads.x[tile_indices], reads.y[tile_indices])).astype(
                np.float64
            )
            tile_tasks.append((lane_id, tile, points, radii, p_norm))

    tracker = logger.tracker("spatial_null", len(tile_tasks), "tile_models")
    completed = 0
    if threads == 1:
        results = (
            (lane_id, tile, tile_null_counts(points, radii, p_norm))
            for lane_id, tile, points, radii, p_norm in tile_tasks
        )
        for lane_id, tile, counts in results:
            lane_models[lane_id]["null_counts"] += counts  # type: ignore[operator]
            completed += 1
            tracker.update(completed, detail=f"lane={lane_id} tile={tile}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            future_metadata = {
                executor.submit(tile_null_counts, points, radii, norm): (lane_id, tile)
                for lane_id, tile, points, radii, norm in tile_tasks
            }
            for future in concurrent.futures.as_completed(future_metadata):
                lane_id, tile = future_metadata[future]
                lane_models[lane_id]["null_counts"] += future.result()  # type: ignore[operator]
                completed += 1
                tracker.update(completed, detail=f"lane={lane_id} tile={tile}")

    for lane_id, model in lane_models.items():
        radii = model["radii"]
        null_counts = model["null_counts"]
        total_pairs = int(model["total_random_pair_relations"])
        null_cdf = (null_counts + 1.0) / (total_pairs + 1.0)  # type: ignore[operator]
        model["null_cdf"] = null_cdf
        lane_edge_positions = np.flatnonzero(
            (reads.lane_ids[tested_left] == lane_id) & same_tile
        )
        if len(lane_edge_positions):
            lookup_positions = np.searchsorted(
                radii, distances[lane_edge_positions], side="left"
            )
            spatial_p[lane_edge_positions] = null_cdf[lookup_positions]

    tracker.finish(detail=f"tested_hypotheses={tested_count:,}")
    return spatial_p, distances, same_tile, lane_models


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
    significant_positions = np.flatnonzero(selected_q <= target_fdr)
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
    significant = np.flatnonzero(qvalues <= threshold)
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
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    eligible_indices: np.ndarray,
    scores: dict[str, np.ndarray],
    same_template_posterior: np.ndarray,
    spatial_p: np.ndarray,
    distances: np.ndarray,
    same_tile: np.ndarray,
    bh_q: np.ndarray,
    weighted_q: np.ndarray,
    weights: np.ndarray,
    selected_q: np.ndarray,
    target_fdr: float,
) -> None:
    tested_left = edge_left[eligible_indices]
    tested_right = edge_right[eligible_indices]
    columns = [
        "candidate_edge_id",
        "read_pair_1_index",
        "read_pair_1_name",
        "read_pair_2_index",
        "read_pair_2_name",
        "lane",
        "tile_1",
        "x_1",
        "y_1",
        "tile_2",
        "x_2",
        "y_2",
        "same_tile",
        "spatial_distance",
        "observed_mismatches",
        "expected_mismatches_from_phred",
        "quality_weighted_mismatch_surprisal",
        "sequence_compatibility_p",
        "sequence_log10_bayes_factor",
        "same_template_posterior",
        "spatial_p_value",
        "bh_q_value",
        "sequence_weight",
        "weighted_bh_q_value",
        "selected_q_value",
        "called_optical_at_target_fdr",
    ]
    with open_output(path, gzip_level=6) as handle:
        handle.write("\t".join(columns) + "\n")
        for position, candidate_index in enumerate(eligible_indices):
            left = int(tested_left[position])
            right = int(tested_right[position])
            lane_id = int(reads.lane_ids[left])
            values = [
                position + 1,
                left + 1,
                reads.names[left],
                right + 1,
                reads.names[right],
                reads.lane_labels[lane_id],
                int(reads.tiles[left]),
                int(reads.x[left]),
                int(reads.y[left]),
                int(reads.tiles[right]),
                int(reads.x[right]),
                int(reads.y[right]),
                int(same_tile[position]),
                "NA" if not same_tile[position] else f"{distances[position]:.6g}",
                int(scores["mismatches"][candidate_index]),
                f"{scores['expected_mismatches'][candidate_index]:.8g}",
                f"{scores['mismatch_surprisal'][candidate_index]:.8g}",
                f"{scores['compatibility_p'][candidate_index]:.8g}",
                f"{scores['log10_bayes_factor'][candidate_index]:.8g}",
                f"{same_template_posterior[candidate_index]:.8g}",
                f"{spatial_p[position]:.8g}",
                f"{bh_q[position]:.8g}",
                f"{weights[position]:.8g}",
                f"{weighted_q[position]:.8g}",
                f"{selected_q[position]:.8g}",
                int(selected_q[position] <= target_fdr),
            ]
            handle.write("\t".join(map(str, values)) + "\n")


def write_removed_audit(
    path: Path,
    reads: LoadedReads,
    removal_details: dict[int, dict[str, object]],
    tested_left: np.ndarray,
    tested_right: np.ndarray,
    eligible_indices: np.ndarray,
    scores: dict[str, np.ndarray],
    spatial_p: np.ndarray,
    distances: np.ndarray,
    selected_q: np.ndarray,
) -> None:
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
        "support_mismatches",
        "support_expected_mismatches",
        "support_sequence_compatibility_p",
        "support_spatial_p",
        "support_q",
    ]
    with open_output(path, gzip_level=6) as handle:
        handle.write("\t".join(columns) + "\n")
        for removed_index in sorted(removal_details):
            details = removal_details[removed_index]
            representative = int(details["representative"])
            support_position = int(details["support_edge_position"])
            support_other = int(details["support_other"])
            candidate_index = int(eligible_indices[support_position])
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
                f"{distances[support_position]:.6g}",
                int(scores["mismatches"][candidate_index]),
                f"{scores['expected_mismatches'][candidate_index]:.8g}",
                f"{scores['compatibility_p'][candidate_index]:.8g}",
                f"{spatial_p[support_position]:.8g}",
                f"{selected_q[support_position]:.8g}",
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


def make_diagnostic_plot(
    path: Path,
    reads: LoadedReads,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    eligible_indices: np.ndarray,
    scores: dict[str, np.ndarray],
    distances: np.ndarray,
    same_tile: np.ndarray,
    lane_models: dict[int, dict[str, object]],
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
    finite_distances = distances[same_tile]

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
            [np.count_nonzero(same_tile & (distances <= value)) / tested_count for value in grid]
        )
        expected = weighted_null_curve(lane_models, grid)
        ax.plot(grid, observed, color="#006D77", lw=2.2, label="sequence-compatible pairs")
        ax.plot(grid, expected, color="#E29578", lw=2.2, label="random-position null")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(left=0)
        ax.set_yscale("log")
        ax.set_xlabel("Same-tile spatial distance")
        ax.set_ylabel("Cumulative fraction across all tested pairs")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No same-tile candidate pairs", ha="center", va="center")
    ax.set_title("A. Empirical spatial enrichment")

    ax = axes[0, 1]
    same_positions = np.flatnonzero(same_tile)
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
        ax.set_xlabel("Same-tile spatial distance")
        ax.set_ylabel(f"-log10({fdr_method} q-value)")
    else:
        ax.text(0.5, 0.5, "No same-tile candidate pairs", ha="center", va="center")
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
    ax.set_title(f"D. Filtering at {100 * target_fdr:g}% edge FDR")

    fig.suptitle("FastqOptiFilter statistical diagnostics", fontsize=16, weight="bold")
    temporary_path = path.with_name(path.name + ".tmp.png")
    try:
        fig.savefig(temporary_path, dpi=180, format="png")
    finally:
        plt.close(fig)
    # A complete PNG terminates with this fixed IEND chunk. Write atomically so
    # downstream workflows never observe a partially rendered diagnostic.
    png_iend = b"\x00\x00\x00\x00IEND\xaeB\x60\x82"
    with temporary_path.open("rb") as handle:
        handle.seek(-len(png_iend), os.SEEK_END)
        if handle.read() != png_iend:
            raise OSError(f"Diagnostic plot did not finish writing: {temporary_path}")
    os.replace(temporary_path, path)


def serializable_lane_models(
    reads: LoadedReads, lane_models: dict[int, dict[str, object]]
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for lane_id, model in sorted(lane_models.items()):
        lane_indices = np.flatnonzero(reads.lane_ids == lane_id)
        tile_counts = Counter(int(value) for value in reads.tiles[lane_indices])
        total_pairs = int(model["total_random_pair_relations"])
        same_tile_pairs = sum(count * (count - 1) // 2 for count in tile_counts.values())
        random_same_tile_probability = same_tile_pairs / total_pairs if total_pairs else 0.0
        tested_hypotheses = int(model["tested_hypotheses"])
        observed_same_tile = int(model["same_tile_tested_hypotheses"])
        observed_cross_tile = tested_hypotheses - observed_same_tile
        estimated_null_edges = (
            min(
                float(tested_hypotheses),
                observed_cross_tile / (1.0 - random_same_tile_probability),
            )
            if random_same_tile_probability < 1.0
            else 0.0
        )
        estimated_spatial_excess = max(0.0, tested_hypotheses - estimated_null_edges)
        radii = np.asarray(model["radii"], dtype=np.float64)
        null_cdf = np.asarray(model["null_cdf"], dtype=np.float64)
        summaries.append(
            {
                "lane_id": lane_id,
                "lane_label": reads.lane_labels[lane_id],
                "read_pairs": int(len(lane_indices)),
                "tiles": len(tile_counts),
                "tile_read_pair_counts": {str(key): value for key, value in sorted(tile_counts.items())},
                "total_random_pair_relations": total_pairs,
                "random_same_tile_probability": random_same_tile_probability,
                "tested_hypotheses": tested_hypotheses,
                "same_tile_tested_hypotheses": observed_same_tile,
                "cross_tile_tested_hypotheses": observed_cross_tile,
                "expected_same_tile_hypotheses_if_all_null": (
                    tested_hypotheses * random_same_tile_probability
                ),
                "same_tile_enrichment_fold": (
                    observed_same_tile
                    / (tested_hypotheses * random_same_tile_probability)
                    if tested_hypotheses and random_same_tile_probability
                    else None
                ),
                "estimated_null_edges_from_cross_tile_count": estimated_null_edges,
                "estimated_spatial_excess_edges": estimated_spatial_excess,
                "maximum_tested_same_tile_distance": (
                    float(radii[-1]) if len(radii) else None
                ),
                "null_cdf_at_maximum_tested_distance": (
                    float(null_cdf[-1]) if len(null_cdf) else None
                ),
            }
        )
    return summaries


def build_sensitivity_table(
    reads: LoadedReads,
    tested_left: np.ndarray,
    tested_right: np.ndarray,
    bh_q: np.ndarray,
    weighted_q: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in (0.001, 0.005, 0.01, 0.05, 0.1):
        bh_edges, bh_components, bh_removed = filtered_count_at_threshold(
            reads.count, tested_left, tested_right, bh_q, threshold
        )
        weighted_edges, weighted_components, weighted_removed = filtered_count_at_threshold(
            reads.count, tested_left, tested_right, weighted_q, threshold
        )
        rows.append(
            {
                "fdr": threshold,
                "bh": {
                    "significant_edges": bh_edges,
                    "components": bh_components,
                    "read_pairs_removed": bh_removed,
                },
                "sequence_weighted_bh": {
                    "significant_edges": weighted_edges,
                    "components": weighted_components,
                    "read_pairs_removed": weighted_removed,
                },
            }
        )
    return rows


def write_markdown_report(path: Path, report: dict[str, object]) -> None:
    counts = report["counts"]  # type: ignore[assignment]
    filtering = report["filtering"]  # type: ignore[assignment]
    configuration = report["configuration"]  # type: ignore[assignment]
    quality_model = report["quality_model"]  # type: ignore[assignment]
    spatial_model = report["spatial_model"]  # type: ignore[assignment]
    sensitivity = report["fdr_sensitivity"]  # type: ignore[assignment]

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
        f"- Decision rule: **{configuration['fdr_method']} q <= "
        f"{configuration['target_fdr']:.4g}**",
        f"- Significant candidate edges: **{filtering['significant_candidate_edges']:,}**",
        f"- Optical/proximity components: **{filtering['optical_components']:,}**",
        "",
        "## Model summary",
        "",
        f"- Retrieved candidate pair relations: **{counts['retrieved_candidate_pair_relations']:,}**",
        f"- Sequence-compatible hypotheses tested spatially: **{counts['tested_candidate_edges']:,}**",
        f"- Same-tile tested hypotheses: **{spatial_model['same_tile_tested_hypotheses']:,}**",
        f"- Estimated total spatial-excess candidate edges: "
        f"**{spatial_model['estimated_spatial_excess_edges']:,.1f}** "
        "(point estimate from cross-tile relations)",
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
        "| FDR | BH removed | Weighted-BH removed | BH edges | Weighted-BH edges |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in sensitivity:
        lines.append(
            f"| {row['fdr']:.3g} | {row['bh']['read_pairs_removed']:,} | "
            f"{row['sequence_weighted_bh']['read_pairs_removed']:,} | "
            f"{row['bh']['significant_edges']:,} | "
            f"{row['sequence_weighted_bh']['significant_edges']:,} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "The spatial p-value is the conditional null probability that two "
            "random cluster positions from the same lane fall on the same tile "
            "and are at least as close as the observed pair. A q-value is an "
            "FDR-adjusted p-value; it is not the posterior probability that a "
            "specific edge is optical.",
            "",
            "FDR is controlled over tested candidate edges. Read removal is "
            "component-based, so the nominal edge FDR is not an exact read-level "
            "FDR. Valid inference assumes sequence-compatible non-optical copies "
            "are exchangeable over observed cluster coordinates within each lane. "
            "UMIs, if available, remain preferable for molecular-complexity estimation.",
            "",
            "The sequence-weighted analysis assumes sequence evidence is independent "
            "of spatial position under the null. Unweighted BH is the conservative "
            "default and the weighted result is included as sensitivity analysis.",
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
    parser.add_argument("--diagnostic-plot", type=Path, help="Diagnostic PNG")
    parser.add_argument("--log", type=Path, help="Timestamped progress/ETA log")
    parser.add_argument("--fdr", type=float, default=0.01, help="Target edge FDR")
    parser.add_argument(
        "--fdr-method",
        choices=("bh", "weighted-bh"),
        default="bh",
        help="Primary multiple-testing method; weighted-BH uses sequence-only weights",
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
    parser.add_argument("--seed-length", type=int, default=20, help="Exact indexing seed length")
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
            args.diagnostic_plot,
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
        edge_left, edge_right, candidate_details = find_candidates(
            reads,
            args.seed_length,
            args.max_seed_bucket,
            args.max_exact_family,
            args.max_candidates,
            logger,
        )
        matrices = encode_matrices(reads, logger)
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
            & (scores["log10_bayes_factor"] > 0.0)
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

        spatial_p, distances, same_tile, lane_models = calibrate_spatial_pvalues(
            reads,
            edge_left,
            edge_right,
            eligible_indices,
            args.spatial_metric,
            threads,
            logger,
        )
        bh_q = benjamini_hochberg(spatial_p)
        posterior_tested = posterior[eligible_indices]
        compatibility_tested = scores["compatibility_p"][eligible_indices]
        weighted_q, weights = sequence_weighted_bh(
            spatial_p, posterior_tested, compatibility_tested
        )
        selected_q = bh_q if args.fdr_method == "bh" else weighted_q
        tested_left = edge_left[eligible_indices]
        tested_right = edge_right[eligible_indices]
        removed, removal_details, filtering_stats = component_filter_decisions(
            reads.count,
            tested_left,
            tested_right,
            selected_q,
            matrices.quality_sums,
            args.fdr,
        )
        logger.log(
            "fdr",
            f"tested_edges={len(eligible_indices):,} same_tile_edges={int(same_tile.sum()):,} "
            f"significant_edges={filtering_stats['significant_candidate_edges']:,} "
            f"components={filtering_stats['optical_components']:,} "
            f"pairs_marked_for_removal={len(removed):,}",
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
                edge_left,
                edge_right,
                eligible_indices,
                scores,
                posterior,
                spatial_p,
                distances,
                same_tile,
                bh_q,
                weighted_q,
                weights,
                selected_q,
                args.fdr,
            )
            logger.log("audit", f"status=complete tested_edges={len(eligible_indices):,}")
        if args.removed_audit is not None:
            write_removed_audit(
                args.removed_audit,
                reads,
                removal_details,
                tested_left,
                tested_right,
                eligible_indices,
                scores,
                spatial_p,
                distances,
                selected_q,
            )

        sensitivity = build_sensitivity_table(
            reads, tested_left, tested_right, bh_q, weighted_q
        )
        lane_summaries = serializable_lane_models(reads, lane_models)
        estimated_null_edges = sum(
            float(lane["estimated_null_edges_from_cross_tile_count"])
            for lane in lane_summaries
        )
        estimated_spatial_excess_edges = sum(
            float(lane["estimated_spatial_excess_edges"])
            for lane in lane_summaries
        )
        called = selected_q <= args.fdr
        called_distances = distances[called & same_tile]
        all_same_tile_random_probability = 0.0
        if total_possible_relations:
            same_tile_random_pairs = 0
            for lane_id in set(int(value) for value in reads.lane_ids if value >= 0):
                lane_mask = reads.lane_ids == lane_id
                for tile in set(int(value) for value in reads.tiles[lane_mask]):
                    tile_count = int(np.count_nonzero(lane_mask & (reads.tiles == tile)))
                    same_tile_random_pairs += tile_count * (tile_count - 1) // 2
            all_same_tile_random_probability = same_tile_random_pairs / total_possible_relations

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
                "diagnostic_plot": str(args.diagnostic_plot.resolve()) if args.diagnostic_plot else None,
                "log": str(args.log.resolve()) if args.log else None,
            },
            "configuration": {
                "target_fdr": args.fdr,
                "fdr_method": args.fdr_method,
                "sequence_min_p": args.sequence_min_p,
                "spatial_metric": args.spatial_metric,
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
            "spatial_model": {
                "null": "random unordered pair of observed cluster positions within lane",
                "p_value_event": "same tile and distance less than or equal to observed",
                "fixed_distance_cutoff_used": False,
                "metric": args.spatial_metric,
                "same_tile_tested_hypotheses": int(same_tile.sum()),
                "random_same_tile_probability_across_lanes": all_same_tile_random_probability,
                "estimated_null_edges_from_cross_tile_count": estimated_null_edges,
                "estimated_spatial_excess_edges": estimated_spatial_excess_edges,
                "estimated_spatial_excess_fraction": (
                    estimated_spatial_excess_edges / len(eligible_indices)
                    if len(eligible_indices)
                    else 0.0
                ),
                "spatial_excess_note": (
                    "Point estimate from cross-tile relations; it estimates the total "
                    "spatial component but does not identify every contributing edge."
                ),
                "lanes": lane_summaries,
            },
            "multiple_testing": {
                "primary_method": args.fdr_method,
                "bh": "Benjamini-Hochberg over all sequence-compatible candidate edges",
                "sequence_weighted_bh": "BH on spatial p divided by normalized sequence-only weight",
                "weight_formula": "posterior_same_template * sqrt(sequence_compatibility_p), normalized to mean 1",
                "scope": "candidate-edge FDR; component-based read-removal FDR can differ",
            },
            "filtering": filtering_stats,
            "fdr_sensitivity": sensitivity,
            "limitations": [
                "Candidate retrieval requires at least one exact seed or exact full paired sequence.",
                "The null assumes sequence-compatible non-optical copies are exchangeable over observed positions within lane.",
                "Base quality values are assumed approximately calibrated and conditionally independent by cycle.",
                "The exact Poisson-binomial compatibility test treats per-cycle sequencing errors as conditionally independent.",
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
                same_tile,
                lane_models,
                bh_q,
                weighted_q,
                selected_q,
                args.fdr_method,
                args.fdr,
                kept_count,
                removed_count,
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

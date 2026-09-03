#!/usr/bin/env python3
"""Filter high-confidence optical/proximity duplicates from paired FASTQ files.

A read pair is considered part of the same candidate optical-duplicate cluster
when all of the following are true:

  1. Its complete R1 and R2 sequences are identical (SHA-256 signature).
  2. Its Illumina header places it on the same run/lane/tile.
  3. It is spatially connected to another member with
     max(abs(delta_x), abs(delta_y)) <= --distance.

One pair per spatial component is retained, choosing the pair with the largest
sum of FASTQ Phred qualities (then the earliest pair on ties). Non-spatial
sequence duplicates are deliberately retained. The input is read twice and a
temporary SQLite database is used, so memory use does not scale with the total
number of input pairs.

This is a conservative FASTQ-level filter. It can miss optical duplicates whose
copies contain different sequencing errors. For definitive library-complexity
estimation, align the filtered reads and rerun a duplicate-marking tool; also
retain the unfiltered data and report that FASTQ-level optical filtering was
performed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import itertools
import json
import multiprocessing
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO


@dataclass(frozen=True)
class FastqRecord:
    header: str
    sequence: str
    plus: str
    quality: str


@dataclass(frozen=True)
class Candidate:
    pair_index: int
    x: int
    y: int
    quality_score: int
    read_name: str


@contextmanager
def open_text(
    path: Path,
    mode: str,
    gzip_level: int = 6,
    pigz_threads: int = 1,
) -> Iterator[TextIO]:
    """Open text, optionally using pigz for parallel output compression."""
    if path.suffix.lower() != ".gz":
        with path.open(mode, newline="") as handle:
            yield handle
        return

    pigz = shutil.which("pigz")
    if "w" in mode and pigz_threads > 1 and pigz:
        with path.open("wb") as raw_output:
            process = subprocess.Popen(
                [
                    pigz,
                    "-c",
                    f"-{gzip_level}",
                    "-p",
                    str(pigz_threads),
                ],
                stdin=subprocess.PIPE,
                stdout=raw_output,
            )
            assert process.stdin is not None
            text_input = io.TextIOWrapper(
                process.stdin, encoding="ascii", newline=""
            )
            try:
                yield text_input
            finally:
                text_input.close()
                return_code = process.wait()
                if return_code != 0:
                    raise OSError(
                        f"pigz failed with exit code {return_code} while writing {path}"
                    )
        return

    if "w" in mode:
        with gzip.open(
            path, mode, compresslevel=gzip_level, newline="", encoding="ascii"
        ) as handle:
            yield handle
    else:
        with gzip.open(path, mode, newline="", encoding="ascii") as handle:
            yield handle


def fastq_records(handle: TextIO, source: Path) -> Iterator[FastqRecord]:
    record_number = 0
    while True:
        header = handle.readline()
        if not header:
            return
        record_number += 1
        sequence = handle.readline()
        plus = handle.readline()
        quality = handle.readline()
        if not sequence or not plus or not quality:
            raise ValueError(
                f"Truncated FASTQ record {record_number:,} in {source}"
            )
        if not header.startswith("@"):
            raise ValueError(
                f"FASTQ record {record_number:,} in {source} has no '@' header"
            )
        if not plus.startswith("+"):
            raise ValueError(
                f"FASTQ record {record_number:,} in {source} has no '+' line"
            )
        seq = sequence.rstrip("\r\n")
        qual = quality.rstrip("\r\n")
        if len(seq) != len(qual):
            raise ValueError(
                f"Sequence/quality lengths differ at record {record_number:,} "
                f"in {source}: {len(seq)} != {len(qual)}"
            )
        yield FastqRecord(header, seq, plus, qual)


def normalized_read_name(header: str) -> str:
    token = header[1:].split(None, 1)[0]
    if token.endswith("/1") or token.endswith("/2"):
        token = token[:-2]
    return token


def parse_illumina_location(read_name: str) -> tuple[str, int, int]:
    """Return (run/lane/tile key, x, y) using the last three header fields."""
    fields = read_name.split(":")
    if len(fields) < 4:
        raise ValueError("fewer than four colon-separated fields")
    try:
        tile = int(fields[-3])
        x = int(fields[-2])
        y = int(fields[-1])
    except ValueError as exc:
        raise ValueError("last three fields are not integer tile/x/y values") from exc
    prefix = ":".join(fields[:-3])
    return f"{prefix}:{tile}", x, y


def paired_sequence_signature(r1_sequence: str, r2_sequence: str) -> bytes:
    digest = hashlib.sha256()
    r1 = r1_sequence.encode("ascii")
    r2 = r2_sequence.encode("ascii")
    digest.update(len(r1).to_bytes(4, byteorder="big", signed=False))
    digest.update(r1)
    digest.update(r2)
    return digest.digest()


def phred_sum(quality1: str, quality2: str) -> int:
    return sum(ord(value) - 33 for value in quality1) + sum(
        ord(value) - 33 for value in quality2
    )


def paired_records(
    r1_path: Path, r2_path: Path
) -> Iterator[tuple[int, FastqRecord, FastqRecord, str]]:
    with open_text(r1_path, "rt") as r1_handle, open_text(
        r2_path, "rt"
    ) as r2_handle:
        r1_iter = fastq_records(r1_handle, r1_path)
        r2_iter = fastq_records(r2_handle, r2_path)
        sentinel = object()
        for pair_index, (r1, r2) in enumerate(
            itertools.zip_longest(r1_iter, r2_iter, fillvalue=sentinel), start=1
        ):
            if r1 is sentinel or r2 is sentinel:
                raise ValueError("R1 and R2 contain different numbers of records")
            assert isinstance(r1, FastqRecord) and isinstance(r2, FastqRecord)
            name1 = normalized_read_name(r1.header)
            name2 = normalized_read_name(r2.header)
            if name1 != name2:
                raise ValueError(
                    f"R1/R2 names differ at pair {pair_index:,}: "
                    f"'{name1}' != '{name2}'"
                )
            yield pair_index, r1, r2, name1


def index_chunk(
    payload: tuple[list[tuple[int, str, str, str, str, str]], str]
) -> tuple[list[tuple[object, ...]], int]:
    """CPU-worker function for parsing, hashing, and quality scoring."""
    chunk, unparsed_action = payload
    rows: list[tuple[object, ...]] = []
    unparsed_pairs = 0
    for pair_index, read_name, r1_sequence, r2_sequence, r1_quality, r2_quality in chunk:
        try:
            location, x, y = parse_illumina_location(read_name)
        except ValueError as exc:
            if unparsed_action == "error":
                raise ValueError(
                    f"Cannot parse Illumina coordinates at pair {pair_index:,} "
                    f"('{read_name}'): {exc}"
                ) from exc
            location, x, y = None, None, None
            unparsed_pairs += 1
        rows.append(
            (
                pair_index,
                paired_sequence_signature(r1_sequence, r2_sequence),
                location,
                x,
                y,
                phred_sum(r1_quality, r2_quality),
                read_name,
            )
        )
    return rows, unparsed_pairs


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def spatial_components(
    candidates: list[Candidate], distance: int
) -> list[list[Candidate]]:
    if len(candidates) < 2:
        return [candidates]

    cell_size = max(1, distance)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    union_find = UnionFind(len(candidates))

    for current_index, current in enumerate(candidates):
        cell_x = current.x // cell_size
        cell_y = current.y // cell_size
        for neighbor_x in range(cell_x - 1, cell_x + 2):
            for neighbor_y in range(cell_y - 1, cell_y + 2):
                for other_index in grid.get((neighbor_x, neighbor_y), ()):
                    other = candidates[other_index]
                    if (
                        abs(current.x - other.x) <= distance
                        and abs(current.y - other.y) <= distance
                    ):
                        union_find.union(current_index, other_index)
        grid[(cell_x, cell_y)].append(current_index)

    components: dict[int, list[Candidate]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        components[union_find.find(index)].append(candidate)
    return list(components.values())


def initialize_database(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        """
        CREATE TABLE reads (
            pair_index INTEGER PRIMARY KEY,
            signature BLOB NOT NULL,
            location TEXT,
            x INTEGER,
            y INTEGER,
            quality_score INTEGER NOT NULL,
            read_name TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE removed (
            pair_index INTEGER PRIMARY KEY,
            representative_index INTEGER NOT NULL,
            representative_name TEXT NOT NULL,
            location TEXT NOT NULL,
            x INTEGER NOT NULL,
            y INTEGER NOT NULL,
            representative_x INTEGER NOT NULL,
            representative_y INTEGER NOT NULL,
            component_size INTEGER NOT NULL
        )
        """
    )
    return connection


def index_fastqs(
    connection: sqlite3.Connection,
    r1_path: Path,
    r2_path: Path,
    unparsed_action: str,
    workers: int,
    chunk_pairs: int,
) -> tuple[int, int]:
    total_pairs = 0
    unparsed_pairs = 0
    insert_sql = (
        "INSERT INTO reads "
        "(pair_index, signature, location, x, y, quality_score, read_name) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)"
    )

    def work_chunks() -> Iterator[
        tuple[list[tuple[int, str, str, str, str, str]], str]
    ]:
        nonlocal total_pairs
        chunk: list[tuple[int, str, str, str, str, str]] = []
        for pair_index, r1, r2, read_name in paired_records(r1_path, r2_path):
            total_pairs = pair_index
            chunk.append(
                (
                    pair_index,
                    read_name,
                    r1.sequence,
                    r2.sequence,
                    r1.quality,
                    r2.quality,
                )
            )
            if len(chunk) >= chunk_pairs:
                yield chunk, unparsed_action
                chunk = []
        if chunk:
            yield chunk, unparsed_action

    def save_result(result: tuple[list[tuple[object, ...]], int]) -> None:
        nonlocal unparsed_pairs
        rows, chunk_unparsed = result
        connection.executemany(insert_sql, rows)
        unparsed_pairs += chunk_unparsed

    if workers == 1:
        for payload in work_chunks():
            save_result(index_chunk(payload))
    else:
        # "spawn" prevents worker processes from inheriting the open SQLite
        # connection and makes behavior consistent across Linux/macOS/Windows.
        multiprocessing_context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing_context
        ) as executor:
            pending: deque[concurrent.futures.Future] = deque()
            for payload in work_chunks():
                pending.append(executor.submit(index_chunk, payload))
                # Bound queued sequence data to keep RAM usage predictable.
                if len(pending) >= workers * 2:
                    save_result(pending.popleft().result())
            while pending:
                save_result(pending.popleft().result())

    connection.commit()
    connection.execute(
        "CREATE INDEX reads_signature_location "
        "ON reads(signature, location, pair_index)"
    )
    connection.commit()
    return total_pairs, unparsed_pairs


def find_optical_candidates(
    connection: sqlite3.Connection, distance: int
) -> dict[str, object]:
    query = """
        SELECT r.signature, r.location, r.pair_index, r.x, r.y,
               r.quality_score, r.read_name
        FROM reads AS r
        JOIN (
            SELECT signature
            FROM reads
            GROUP BY signature
            HAVING COUNT(*) > 1
        ) AS duplicated
          ON r.signature = duplicated.signature
        ORDER BY r.signature, r.location, r.pair_index
    """

    duplicate_family_count = 0
    duplicate_family_members = 0
    exact_sequence_duplicate_excess = 0
    optical_component_count = 0
    optical_removed_count = 0
    component_sizes: Counter[int] = Counter()
    insert_rows: list[tuple[object, ...]] = []

    rows = connection.execute(query)
    for _, signature_group in itertools.groupby(rows, key=lambda row: row[0]):
        duplicate_family_count += 1
        family_members = 0
        for location, location_group in itertools.groupby(
            signature_group, key=lambda row: row[1]
        ):
            location_rows = list(location_group)
            family_members += len(location_rows)
            if location is None or len(location_rows) < 2:
                continue
            candidates = [
                Candidate(
                    pair_index=row[2],
                    x=row[3],
                    y=row[4],
                    quality_score=row[5],
                    read_name=row[6],
                )
                for row in location_rows
            ]
            for component in spatial_components(candidates, distance):
                if len(component) < 2:
                    continue
                optical_component_count += 1
                component_sizes[len(component)] += 1
                representative = max(
                    component,
                    key=lambda candidate: (
                        candidate.quality_score,
                        -candidate.pair_index,
                    ),
                )
                for candidate in component:
                    if candidate.pair_index == representative.pair_index:
                        continue
                    insert_rows.append(
                        (
                            candidate.pair_index,
                            representative.pair_index,
                            representative.read_name,
                            location,
                            candidate.x,
                            candidate.y,
                            representative.x,
                            representative.y,
                            len(component),
                        )
                    )
                    optical_removed_count += 1
                    if len(insert_rows) >= 25_000:
                        connection.executemany(
                            "INSERT INTO removed VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            insert_rows,
                        )
                        insert_rows.clear()
        duplicate_family_members += family_members
        exact_sequence_duplicate_excess += family_members - 1

    if insert_rows:
        connection.executemany(
            "INSERT INTO removed VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            insert_rows,
        )
    connection.commit()

    return {
        "exact_sequence_duplicate_families": duplicate_family_count,
        "pairs_in_exact_sequence_duplicate_families": duplicate_family_members,
        "exact_sequence_duplicate_excess_pairs": exact_sequence_duplicate_excess,
        "candidate_optical_components": optical_component_count,
        "candidate_optical_pairs_removed": optical_removed_count,
        "optical_component_size_histogram": {
            str(size): count for size, count in sorted(component_sizes.items())
        },
    }


def write_filtered_fastqs(
    connection: sqlite3.Connection,
    r1_path: Path,
    r2_path: Path,
    output_r1: Path,
    output_r2: Path,
    gzip_level: int,
    pigz_threads_per_output: int,
) -> tuple[int, int]:
    removed_cursor = connection.execute(
        "SELECT pair_index FROM removed ORDER BY pair_index"
    )
    next_removed_row = removed_cursor.fetchone()
    next_removed = next_removed_row[0] if next_removed_row else None
    kept_count = 0
    removed_count = 0

    with open_text(
        output_r1, "wt", gzip_level, pigz_threads_per_output
    ) as output_r1_handle, open_text(
        output_r2, "wt", gzip_level, pigz_threads_per_output
    ) as output_r2_handle:
        for pair_index, r1, r2, _ in paired_records(r1_path, r2_path):
            if pair_index == next_removed:
                removed_count += 1
                next_removed_row = removed_cursor.fetchone()
                next_removed = next_removed_row[0] if next_removed_row else None
                continue
            kept_count += 1
            output_r1_handle.write(r1.header)
            output_r1_handle.write(r1.sequence + "\n")
            output_r1_handle.write(r1.plus)
            output_r1_handle.write(r1.quality + "\n")
            output_r2_handle.write(r2.header)
            output_r2_handle.write(r2.sequence + "\n")
            output_r2_handle.write(r2.plus)
            output_r2_handle.write(r2.quality + "\n")

    if next_removed is not None:
        raise RuntimeError("Internal error: not all removal indices were encountered")
    return kept_count, removed_count


def write_audit(connection: sqlite3.Connection, audit_path: Path) -> None:
    query = """
        SELECT removed.pair_index,
               reads.read_name,
               removed.representative_index,
               removed.representative_name,
               removed.location,
               removed.x,
               removed.y,
               removed.representative_x,
               removed.representative_y,
               removed.component_size
        FROM removed
        JOIN reads ON reads.pair_index = removed.pair_index
        ORDER BY removed.pair_index
    """
    with open_text(audit_path, "wt") as handle:
        handle.write(
            "removed_pair_index\tremoved_read_name\trepresentative_pair_index\t"
            "representative_read_name\trun_lane_tile\tremoved_x\tremoved_y\t"
            "representative_x\trepresentative_y\tdelta_x\tdelta_y\t"
            "chebyshev_distance_to_representative\tcomponent_size\n"
        )
        for row in connection.execute(query):
            (
                pair_index,
                read_name,
                representative_index,
                representative_name,
                location,
                x,
                y,
                representative_x,
                representative_y,
                component_size,
            ) = row
            delta_x = abs(x - representative_x)
            delta_y = abs(y - representative_y)
            handle.write(
                "\t".join(
                    str(value)
                    for value in (
                        pair_index,
                        read_name,
                        representative_index,
                        representative_name,
                        location,
                        x,
                        y,
                        representative_x,
                        representative_y,
                        delta_x,
                        delta_y,
                        max(delta_x, delta_y),
                        component_size,
                    )
                )
                + "\n"
            )


def ensure_writable_outputs(paths: list[Path], force: bool) -> None:
    for path in paths:
        if path.exists() and not force:
            raise FileExistsError(
                f"Output already exists: {path}. Use --force to overwrite it."
            )
        path.parent.mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove high-confidence spatial optical duplicates from synchronized "
            "paired-end FASTQs while retaining one high-quality representative."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--r1", required=True, type=Path, help="Input R1 FASTQ[.gz]")
    parser.add_argument("--r2", required=True, type=Path, help="Input R2 FASTQ[.gz]")
    parser.add_argument(
        "--output-r1", required=True, type=Path, help="Filtered R1 FASTQ[.gz]"
    )
    parser.add_argument(
        "--output-r2", required=True, type=Path, help="Filtered R2 FASTQ[.gz]"
    )
    parser.add_argument(
        "--report", required=True, type=Path, help="JSON summary report"
    )
    parser.add_argument(
        "--audit",
        type=Path,
        help="Optional TSV[.gz] listing each removed pair and its representative",
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=2500,
        help=(
            "Maximum absolute difference in both X and Y for a spatial link; "
            "2500 is Picard's common patterned-flow-cell setting"
        ),
    )
    parser.add_argument(
        "--unparsed",
        choices=("error", "keep"),
        default="error",
        help="What to do with headers whose tile/X/Y fields cannot be parsed",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="Directory for the temporary SQLite database",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help=(
            "CPU workers for FASTQ parsing/hashing; if pigz is installed, the "
            "same CPU budget is divided across the two output compressors. Use "
            "0 to detect the available CPU count"
        ),
    )
    parser.add_argument(
        "--chunk-pairs",
        type=int,
        default=20_000,
        help="Read pairs sent to each parallel worker task",
    )
    parser.add_argument(
        "--gzip-level",
        type=int,
        choices=range(1, 10),
        default=6,
        metavar="1-9",
        help="Compression level for gzipped outputs",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output files"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.distance < 0:
        raise ValueError("--distance must be >= 0")
    if args.threads < 0:
        raise ValueError("--threads must be >= 0")
    workers = args.threads or (os.cpu_count() or 1)
    if args.chunk_pairs < 1:
        raise ValueError("--chunk-pairs must be >= 1")
    if args.r1.resolve() == args.r2.resolve():
        raise ValueError("--r1 and --r2 must be different files")
    for input_path in (args.r1, args.r2):
        if not input_path.is_file():
            raise FileNotFoundError(f"Input does not exist: {input_path}")

    output_paths = [args.output_r1, args.output_r2, args.report]
    if args.audit:
        output_paths.append(args.audit)
    resolved_inputs = {args.r1.resolve(), args.r2.resolve()}
    resolved_outputs = [path.resolve() for path in output_paths]
    if any(path in resolved_inputs for path in resolved_outputs):
        raise ValueError("Output paths must not overwrite either input FASTQ")
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("Every output path must be distinct")
    ensure_writable_outputs(output_paths, args.force)

    pigz_available = shutil.which("pigz") is not None
    pigz_threads_per_output = (
        max(1, workers // 2) if pigz_available and workers > 1 else 1
    )

    temp_parent = args.temp_dir.resolve() if args.temp_dir else None
    if temp_parent:
        temp_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="optical_fastq_", dir=temp_parent
    ) as temporary_directory:
        database_path = Path(temporary_directory) / "optical_candidates.sqlite3"
        connection = initialize_database(database_path)
        try:
            total_pairs, unparsed_pairs = index_fastqs(
                connection,
                args.r1,
                args.r2,
                args.unparsed,
                workers,
                args.chunk_pairs,
            )
            detection = find_optical_candidates(connection, args.distance)
            kept_count, removed_count = write_filtered_fastqs(
                connection,
                args.r1,
                args.r2,
                args.output_r1,
                args.output_r2,
                args.gzip_level,
                pigz_threads_per_output,
            )
            if args.audit:
                write_audit(connection, args.audit)
        finally:
            connection.close()

    detected_removed = int(detection["candidate_optical_pairs_removed"])
    if removed_count != detected_removed or kept_count + removed_count != total_pairs:
        raise RuntimeError(
            "Internal count mismatch after filtering: "
            f"total={total_pairs}, kept={kept_count}, removed={removed_count}, "
            f"detected_removed={detected_removed}"
        )

    exact_duplicate_excess = int(detection["exact_sequence_duplicate_excess_pairs"])
    remaining_exact_duplicate_excess = exact_duplicate_excess - removed_count
    report = {
        "script_version": "1.0.0",
        "method": "exact paired-sequence plus spatial proximity",
        "classification_scope": "high-confidence FASTQ-level candidates",
        "input_r1": args.r1.name,
        "input_r2": args.r2.name,
        "output_r1": args.output_r1.name,
        "output_r2": args.output_r2.name,
        "spatial_metric": "max(abs(delta_x), abs(delta_y))",
        "coordinate_distance_threshold": args.distance,
        "paired_sequence_signature": "SHA-256 over complete R1 and R2 sequences",
        "representative_selection": "highest summed Phred quality; earliest on tie",
        "parallel_cpu_workers": workers,
        "parallel_gzip_program": "pigz" if pigz_threads_per_output > 1 else None,
        "pigz_threads_per_output": (
            pigz_threads_per_output if pigz_threads_per_output > 1 else None
        ),
        "total_input_pairs": total_pairs,
        "headers_without_parsed_coordinates": unparsed_pairs,
        **detection,
        "raw_exact_sequence_duplicate_fraction": (
            exact_duplicate_excess / total_pairs if total_pairs else 0.0
        ),
        "candidate_optical_fraction_of_exact_sequence_duplicate_excess": (
            removed_count / exact_duplicate_excess if exact_duplicate_excess else 0.0
        ),
        "remaining_exact_sequence_duplicate_excess_pairs": (
            remaining_exact_duplicate_excess
        ),
        "remaining_exact_sequence_duplicate_fraction_among_retained": (
            remaining_exact_duplicate_excess / kept_count if kept_count else 0.0
        ),
        "retained_pairs": kept_count,
        "retained_fraction": kept_count / total_pairs if total_pairs else 0.0,
        "candidate_optical_fraction_removed": (
            removed_count / total_pairs if total_pairs else 0.0
        ),
        "notes": [
            "R1 and R2 remained synchronized; a pair was removed from both files.",
            "Non-spatial exact-sequence duplicates were retained.",
            "Connected spatial components were used, so a component member may "
            "be more than the threshold from the selected representative while "
            "remaining linked through another member.",
            "Sequence-divergent copies caused by sequencing errors are not detected.",
            "Re-align the filtered FASTQs and run duplicate metrics for the adjusted "
            "library-complexity analysis; preserve the original FASTQs.",
        ],
    }
    with args.report.open("w", encoding="utf-8") as report_handle:
        json.dump(report, report_handle, indent=2, sort_keys=True)
        report_handle.write("\n")

    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

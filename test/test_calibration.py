#!/usr/bin/env python3
"""End-to-end calibration and accuracy checks for FastqOptiFilter.

Four simulated datasets are generated, filtered and scored:

1. a pure null with library/PCR duplicates but no proximity duplication at
   all, where nothing may be removed and the analysed p-values must be uniform;
2. a mixed dataset with same-tile and cross-tile optical duplicates plus a
   spatially clustered poly-G artifact, where sensitivity and read-level FDR
   are scored against ground truth;
3. a low-quality dataset where a fixed seed length loses real duplicates
   before the quality model can see them, confirming that seeding from the
   reported qualities recovers them;
4. the same mixed dataset filtered with the pre-existing same-tile edge-level
   configuration, to confirm the cross-tile search is what recovers the
   cross-tile duplicates rather than a change in the significance threshold.

Run directly. Exit status is zero when every assertion holds.

    python3 test/test_calibration.py --work-dir /tmp/fastqoptifilter-test
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FILTER = REPO / "fastq_optifilter.py"
SIMULATE = HERE / "simulate_fastq.py"
EVALUATE = HERE / "evaluate_run.py"


class CheckFailed(Exception):
    pass


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise CheckFailed(
            f"command failed: {' '.join(command)}\n{result.stdout}\n{result.stderr}"
        )


def simulate(prefix: Path, **options: object) -> None:
    command = [sys.executable, str(SIMULATE), "--out-prefix", str(prefix)]
    for key, value in options.items():
        command += [f"--{key.replace('_', '-')}", str(value)]
    run(command)


def filter_reads(prefix: Path, out_dir: Path, extra: list[str]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.json"
    run(
        [
            sys.executable,
            str(FILTER),
            "--r1", f"{prefix}_R1.fastq.gz",
            "--r2", f"{prefix}_R2.fastq.gz",
            "--output-r1", str(out_dir / "out_R1.fastq.gz"),
            "--output-r2", str(out_dir / "out_R2.fastq.gz"),
            "--report-json", str(report),
            "--threads", "4",
            "--force",
            *extra,
        ]
    )
    return json.loads(report.read_text())


def score(prefix: Path, out_dir: Path, label: str) -> dict:
    result = out_dir / "score.json"
    run(
        [
            sys.executable,
            str(EVALUATE),
            "--truth", f"{prefix}.truth.tsv.gz",
            "--filtered-r1", str(out_dir / "out_R1.fastq.gz"),
            "--label", label,
            "--json-out", str(result),
        ]
    )
    return json.loads(result.read_text())


def retrieval_recall(prefix: Path, candidate_audit: Path) -> float:
    """Share of true optical relations that survive to the tested set."""
    wanted = set()
    with gzip.open(f"{prefix}.truth.tsv.gz", "rt") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["mechanism"] == "optical" and row["parent_read_name"] != "NA":
                wanted.add(frozenset((row["read_name"], row["parent_read_name"])))
    found = set()
    with gzip.open(candidate_audit, "rt") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = frozenset((row["read_pair_1_name"], row["read_pair_2_name"]))
            if key in wanted:
                found.add(key)
    return len(found) / len(wanted) if wanted else 0.0


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes = 0

    def check(self, description: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passes += 1
            print(f"  PASS  {description}" + (f"  [{detail}]" if detail else ""))
        else:
            self.failures.append(f"{description}  [{detail}]")
            print(f"  FAIL  {description}" + (f"  [{detail}]" if detail else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--reads", type=int, default=40000)
    parser.add_argument("--keep", action="store_true", help="Keep the work directory")
    args = parser.parse_args(argv)

    temporary = None
    if args.work_dir is None:
        temporary = tempfile.TemporaryDirectory()
        work = Path(temporary.name)
    else:
        work = args.work_dir
        work.mkdir(parents=True, exist_ok=True)

    checker = Checker()

    print("\n[1/4] Pure null: library duplicates only, no proximity duplication")
    null_prefix = work / "null"
    simulate(
        null_prefix,
        reads=args.reads,
        seed=101,
        optical_rate=0,
        polyg_rate=0,
        pcr_duplicate_rate=0.15,
    )
    report = filter_reads(null_prefix, work / "null_run", [])
    controls = report["null_calibration"]["controls"]
    analysis = controls["analysis"]
    checker.check(
        "nothing is removed when there is nothing to remove",
        report["filtering"]["removed_read_pairs"] == 0,
        f"removed={report['filtering']['removed_read_pairs']}",
    )
    checker.check(
        "analysed p-values are valid (no excess over uniform)",
        analysis["max_excess_over_uniform"] <= 0.02,
        f"max_excess={analysis['max_excess_over_uniform']:.4f}",
    )
    checker.check(
        "analysed p-values are not inflated in the tail",
        0.7 <= analysis["tail_inflation"] <= 1.35,
        f"tail_inflation={analysis['tail_inflation']:.3f}",
    )
    checker.check(
        "the mixture finds no alternative",
        report["multiple_testing"]["mixture"]["pi0"] >= 0.95,
        f"pi0={report['multiple_testing']['mixture']['pi0']:.4f}",
    )
    checker.check(
        "permutation control is calibrated",
        controls["permutation"]["max_excess_over_uniform"] <= 0.02,
        f"max_excess={controls['permutation']['max_excess_over_uniform']:.4f}",
    )

    print("\n[2/4] Mixed: same-tile and cross-tile optical duplicates, defaults")
    mixed_prefix = work / "mixed"
    simulate(mixed_prefix, reads=args.reads, seed=17)
    report = filter_reads(mixed_prefix, work / "mixed_run", [])
    result = score(mixed_prefix, work / "mixed_run", "defaults")
    controls = report["null_calibration"]["controls"]
    checker.check(
        "sensitivity is at least 95%",
        result["sensitivity"] >= 0.95,
        f"sensitivity={result['sensitivity']:.3f}",
    )
    checker.check(
        "cross-tile optical duplicates are recovered",
        result["cross_tile_sensitivity"] >= 0.85,
        f"cross_tile_sensitivity={result['cross_tile_sensitivity']:.3f}",
    )
    checker.check(
        "read-level FDR stays below 2%",
        result["empirical_read_level_fdr"] <= 0.02,
        f"fdr={result['empirical_read_level_fdr']:.4f}",
    )
    checker.check(
        "permutation control is still calibrated with signal present",
        controls["permutation"]["max_excess_over_uniform"] <= 0.03,
        f"max_excess={controls['permutation']['max_excess_over_uniform']:.4f}",
    )
    checker.check(
        "the analysed set departs from the null",
        controls["analysis"]["tail_inflation"] >= 2.0,
        f"tail_inflation={controls['analysis']['tail_inflation']:.2f}",
    )

    print("\n[3/4] Low-quality reads: seed length must follow the qualities")
    noisy_prefix = work / "noisy"
    simulate(noisy_prefix, reads=args.reads, seed=17, flat_quality=13)
    fixed = filter_reads(
        noisy_prefix,
        work / "noisy_fixed",
        ["--no-adaptive-seed", "--candidate-audit", str(work / "noisy_fixed" / "c.tsv.gz")],
    )
    adaptive = filter_reads(
        noisy_prefix,
        work / "noisy_adaptive",
        ["--candidate-audit", str(work / "noisy_adaptive" / "c.tsv.gz")],
    )
    fixed_recall = retrieval_recall(
        noisy_prefix, work / "noisy_fixed" / "c.tsv.gz"
    )
    adaptive_recall = retrieval_recall(
        noisy_prefix, work / "noisy_adaptive" / "c.tsv.gz"
    )
    selected = adaptive["candidate_retrieval"]["seed_length_selection"]["selected"]
    checker.check(
        "poor qualities select a shorter seed",
        selected < fixed["candidate_retrieval"]["seed_length_selection"]["selected"],
        f"{fixed['candidate_retrieval']['seed_length_selection']['selected']} -> {selected}",
    )
    checker.check(
        "adaptive seeding recovers duplicates a fixed seed loses",
        adaptive_recall - fixed_recall >= 0.2,
        f"retrieval recall {fixed_recall:.3f} -> {adaptive_recall:.3f}",
    )
    checker.check(
        "adaptive retrieval recall is high in absolute terms",
        adaptive_recall >= 0.95,
        f"recall={adaptive_recall:.3f}",
    )

    print("\n[4/4] Same-tile edge-level configuration on the same data")
    legacy = filter_reads(
        mixed_prefix,
        work / "legacy_run",
        ["--tile-neighborhood", "same-tile", "--inference-unit", "edge",
         "--fdr-method", "bh"],
    )
    legacy_result = score(mixed_prefix, work / "legacy_run", "same-tile edge BH")
    checker.check(
        "a same-tile search cannot recover cross-tile duplicates",
        legacy_result["cross_tile_sensitivity"] <= 0.10,
        f"cross_tile_sensitivity={legacy_result['cross_tile_sensitivity']:.3f}",
    )
    checker.check(
        "the nearby-tile search is what recovers them",
        result["cross_tile_sensitivity"] - legacy_result["cross_tile_sensitivity"] >= 0.5,
        f"{legacy_result['cross_tile_sensitivity']:.3f} -> "
        f"{result['cross_tile_sensitivity']:.3f}",
    )
    checker.check(
        "the read-level default is not simply removing more reads",
        result["removed_read_pairs"] <= 1.3 * legacy_result["removed_read_pairs"],
        f"{legacy_result['removed_read_pairs']} -> {result['removed_read_pairs']}",
    )
    assert legacy["configuration"]["tile_neighborhood"] == "same-tile"

    print(f"\n{checker.passes} passed, {len(checker.failures)} failed")
    for failure in checker.failures:
        print(f"  FAILED: {failure}")
    if args.keep or args.work_dir is not None:
        print(f"Work directory: {work}")
    if temporary is not None and not args.keep:
        temporary.cleanup()
    return 1 if checker.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

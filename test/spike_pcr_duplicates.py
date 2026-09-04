#!/usr/bin/env python3
"""Spike library/PCR duplicates into a real run, at independent positions.

Takes real FASTQs and overwrites a chosen share of read pairs with a copy of
another read pair's sequence, leaving the recipient's own header -- and
therefore its own cluster position -- untouched. The copies are re-sequenced:
errors are re-drawn from the recipient's own quality string, so the two members
of a pair differ exactly as two independent reads of one molecule would.

The result is a run with a known library-duplicate rate, no added optical
duplicates, and a completely real flowcell geometry and quality profile. Any
spiked read a proximity filter removes is a false positive, because its
position was drawn from the run's own clusters independently of its sequence.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

BASES = "ACGT"
INDEX = {b: i for i, b in enumerate(BASES)}


def read_fastq(path: Path) -> tuple[list[str], list[str], list[str]]:
    headers, seqs, quals = [], [], []
    with gzip.open(path, "rt") as handle:
        while True:
            head = handle.readline()
            if not head:
                break
            seqs.append(handle.readline().rstrip("\n"))
            handle.readline()
            quals.append(handle.readline().rstrip("\n"))
            headers.append(head.rstrip("\n"))
    return headers, seqs, quals


def resequence(rng: np.random.Generator, template: str, quality: str) -> str:
    """Re-read a template under the error rates a quality string implies."""
    length = min(len(template), len(quality))
    bases = np.array([INDEX.get(c, 0) for c in template[:length]], dtype=np.int8)
    phred = np.frombuffer(quality[:length].encode(), dtype=np.uint8).astype(np.int16) - 33
    error = np.power(10.0, -phred / 10.0)
    hit = rng.random(length) < error
    if hit.any():
        bases = np.where(hit, (bases + rng.integers(1, 4, size=length)) % 4, bases)
    out = "".join(BASES[b] for b in bases)
    return out + template[length:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1", required=True, type=Path)
    parser.add_argument("--r2", required=True, type=Path)
    parser.add_argument("--out-prefix", required=True, type=Path)
    parser.add_argument(
        "--pcr-rate",
        type=float,
        default=0.05,
        help="Share of read pairs replaced by a copy of another pair",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    h1, s1, q1 = read_fastq(args.r1)
    h2, s2, q2 = read_fastq(args.r2)
    if len(h1) != len(h2):
        raise SystemExit("R1 and R2 differ in length")
    count = len(h1)

    n_spike = int(args.pcr_rate * count)
    recipients = rng.choice(count, size=n_spike, replace=False)
    pool = np.setdiff1d(np.arange(count), recipients)
    donors = rng.choice(pool, size=n_spike, replace=True)

    spiked: dict[str, str] = {}
    for recipient, donor in zip(recipients, donors):
        s1[recipient] = resequence(rng, s1[donor], q1[recipient])
        s2[recipient] = resequence(rng, s2[donor], q2[recipient])
        spiked[h1[recipient][1:].split()[0]] = h1[donor][1:].split()[0]

    out1 = Path(str(args.out_prefix) + "_R1.fastq.gz")
    out2 = Path(str(args.out_prefix) + "_R2.fastq.gz")
    out1.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out1, "wt") as a, gzip.open(out2, "wt") as b:
        for i in range(count):
            a.write(f"{h1[i]}\n{s1[i]}\n+\n{q1[i]}\n")
            b.write(f"{h2[i]}\n{s2[i]}\n+\n{q2[i]}\n")

    truth = {
        "read_pairs": count,
        "pcr_rate": args.pcr_rate,
        "spiked_pcr_duplicates": n_spike,
        "spiked": spiked,
        "seed": args.seed,
        "note": (
            "Each spiked read keeps its own cluster position, which was chosen "
            "independently of its sequence, so no spiked read is a proximity "
            "duplicate. Removing one is a false positive."
        ),
    }
    Path(str(args.out_prefix) + ".spike.json").write_text(
        json.dumps(truth), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in truth.items() if k != "spiked"}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

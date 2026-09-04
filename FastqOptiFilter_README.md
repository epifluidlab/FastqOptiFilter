# FastqOptiFilter

FastqOptiFilter is a quality-aware, FDR-controlled filter for optical/proximity
duplicates in synchronized paired-end Illumina FASTQ files. It uses Illumina
lane/tile/X/Y coordinates from read names and does **not** require or apply a
fixed spatial-distance threshold.

## What the model does

1. Reads the base qualities first and chooses a seed length they can support,
   then finds sequence-similar candidate read-pair relations with an exact
   paired hash plus several exact seeds. Seeds are only an index; they are
   never used as the final duplicate rule.
2. For every candidate, converts Phred scores to per-cycle error probabilities
   and compares a common-template model with an independent-template model.
   It also reports an exact Poisson-binomial mismatch-compatibility p-value.
3. Decodes the Illumina tile identifier into a physical grid and places every
   read in one continuous coordinate frame per lane and surface, so a pair
   split across a tile boundary has a real separation. Neighbouring tiles are
   searched by default.
4. Tests each read once: given its own position and how many sequence-
   compatible partners it has anywhere in the lane and surface, how surprising
   is it that the closest of them landed as close as it did? Neighbours are
   counted around the read itself, so the test adapts to local cluster density.
5. Measures the FDR by reshuffling the whole flowcell rather than assuming a
   dependence structure. Benjamini-Yekutieli, Benjamini-Hochberg and a
   two-groups local-FDR mixture are reported alongside it.
6. Joins significant relations into components, retains the read pair with the
   largest total R1+R2 quality score, and writes synchronized filtered FASTQs.
7. Checks its own null against two negative controls and writes a QQ plot.

## The null, and how to check it

A spatial p-value is only meaningful if it is uniform when no proximity
duplication is present. Every run scores that directly and writes the numbers
into the report, and `--qq-plot` draws them:

| Control | What it is | What it tests |
|:---|:---|:---|
| `permutation` | reads reassigned to the observed cluster positions within a lane and surface | that the geometry null is *computed* correctly. The point pattern, and so the null itself, is untouched; only the real spatial relations are destroyed |
| `sequence_incompatible` | seed-retrieved candidate pairs that the quality model rejected as different molecules | that within-lane exchangeability actually *holds* in this run. These are real reads, so structure the assumption misses — dark patches, bubbles, low-complexity wells — shows up here and not in the permutation |
| `analysis` | the hypotheses that are actually filtered | should depart from uniform in the extreme tail when proximity duplicates are present |

The column to read is `max_excess_over_uniform`, the largest amount by which
the observed rejection rate exceeds its nominal level. At or below zero means
the p-values are valid. `tail_inflation` near 1 means calibrated; clearly above
1 on `sequence_incompatible` means q-values are optimistic by roughly that
factor, and either a stricter `--fdr` or upstream removal of the low-complexity
reads driving it is warranted.

`sequence_incompatible` has one failure mode worth knowing about. It assumes
the quality model only rejects pairs that are genuinely different molecules,
but a real proximity duplicate whose two copies diverge badly — adapter
read-through in one of them, a dead cycle, a low-quality patch — is also
rejected, and it *is* spatially adjacent. Such pairs push this control's
extreme tail up even when exchangeability holds perfectly. Read a departure
confined to the far tail as ambiguous; read a departure that starts near the
diagonal's middle as a genuine violation of exchangeability. `permutation` is
unaffected either way, so a `permutation` control that is itself off means the
geometry or the frame is wrong, not the assumption.

An atom at `p = 1` is expected and is not a defect: a read whose only look-alike
landed on a tile too far away to compare cannot score anything else. That is
also why the calibration statistics are one-sided rather than a two-sided
Kolmogorov-Smirnov distance, which would score a correct discrete null as a
failure.

## Why the read is the unit of inference

Testing every candidate *relation* lets one family of `k` sequence-identical
reads raise `k(k-1)/2` hypotheses over only `k` reads. A single low-complexity
cluster then supplies most of the hypotheses in a run and dominates both the
multiple-testing correction and any estimate of how many hypotheses are null.
Reads are also what actually get removed, so testing reads makes the controlled
error rate the one that matters. `--inference-unit edge` restores the
per-relation test.

Counting neighbours around each read, rather than reading a lane-averaged
curve, is what makes the test adapt to local density. The same separation is
unremarkable in a crowded patch of a tile and surprising in a sparse one; a
lane-pooled null cannot tell those apart, and on a dense artifact cluster it
will call pairs thousands of pixels apart, at distances where no optical
mechanism operates.

## Seed length is chosen from the base qualities

Retrieval is exact: two copies of one molecule are only found if some seed
window is error-free in **both** of them, and how often that happens is decided
entirely by the base qualities. Fixing the seed length in advance therefore
throws away real duplicates before the quality model ever sees them — pairs
whose every mismatch is exactly what their Phred scores predict.

From the reported qualities the expected number of disagreeing cycles between
two copies is `E = Σ (e_L + e_R − (4/3)·e_L·e_R)`. Seeds are disjoint, so
`cycles / L` of them tile the pair, and `L = cycles / (E + 1)` keeps more
windows than expected mismatches — the pigeonhole condition for one window to
survive. The length is floored at `log4(n)` so a seed still discriminates, and
`--seed-length` is the ceiling, so good data is never given shorter seeds than
asked for. Measured on 90 bp simulated reads with uniform quality:

| flat Phred | per-base error | fixed 20 bp seeds | adaptive |
|---:|---:|---:|---:|
| 20 | 1.0% | 99.3% | 99.3% (seed 20) |
| 16 | 2.5% | 95.0% | 98.8% (seed 15) |
| 13 | 5.0% | **61.6%** | **99.4%** (seed 8) |
| 11 | 7.9% | **25.8%** | **98.3%** (seed 8) |

Shorter seeds cost candidates, not accuracy: at Phred 13 retrieval grows from
30k to 335k relations but the tested set stays at ~16.8k, because the quality
model discards the extra collisions. `--no-adaptive-seed` restores the fixed
length.

## Choosing `--fdr-method`

The default is `auto`, and the reason is that **no single method is right for
both signal regimes**. The report's `pi0` says which regime a run is in.

A tail test — BH, BY, permutation — asks whether a p-value is extreme against a
uniform null. That question has the wrong answer once most hypotheses are
genuinely non-null: a real duplicate a long way from its twin is unremarkable
on its own, and only a fitted mixture knows that nearly everything around it is
also a duplicate. On a real cfWGS run with an abnormal proximity-duplicate load
(`pi0 = 0.038`, i.e. 96% of tested reads truly duplicated), the difference is
not subtle:

| method | reads removed | sensitivity | specificity |
|:---|---:|---:|---:|
| `permutation` at 1% FDR | 798 | **12.3%** | 99.96% |
| `local-fdr` at 1% FDR | 6,537 | **99.7%** | 99.66% |

On a sparse-signal simulation (`pi0 = 0.66`) the ordering reverses and the tail
test is the better-calibrated choice. `auto` therefore selects `local-fdr` when
`pi0` falls below `--auto-pi0-threshold` (default 0.5) and `permutation`
otherwise, and both the log and the report state which was chosen and why.

`permutation` is worth understanding on its own terms. BH and BY convert a
per-read p-value into an error rate by *assuming* how the hypotheses depend on
each other — positive regression dependence for BH, nothing at all for BY at a
price of about `log(m)`. Neither assumption is needed if the flowcell itself is
reshuffled: each replicate keeps the same reads, the same candidate relations
and the same cluster pattern, and only breaks the pairing of read to position.
The FDR at a threshold is then the average number of calls a reshuffled run
makes divided by the number of real calls.

This still *is* multiplicity control. Testing many reads at once does not make
false positives go away — 100,000 reads at p ≤ 0.01 still yield ~1,000 by
chance. What changes is that the correction is measured rather than bounded,
so it does not pay BY's `log(m)`. Across three simulated seeds, against the
null it actually tests:

| method | sensitivity | FDR vs 1% target | cost |
|:---|---:|---:|:---|
| `permutation` | 98.2–98.6% | **0.19–0.91%** | 20 reshuffles, ~2 s |
| `by` | 98.0–98.6% | 0.00–0.06% | ~15× conservative |
| `bh` | 98.2–98.8% | 0.37–0.91% | assumes PRDS |

`by` remains the choice when you want to under-remove by construction. It is
also, incidentally, the only one that suppresses a dense low-complexity
cluster — but it does that through blanket conservatism, not because it
recognises the artifact. No test of spatial exchangeability can call a real
spatial cluster null; that is what the `sequence_incompatible` control is for.

## Validation on a real run

`test_data/` holds a shallow cfWGS MiSeq-scale run (90,927 pairs, 18 tiles, one
lane, 151 bp, 3-bin qualities) with an abnormally high proximity-duplicate
load. It is a useful worked example because the geometry identifies the
mechanism without needing a simulator.

Duplicate pairs sort by tile relation as follows, against what independent
placement would give:

| tile grid step | observed | expected if positions independent | ratio |
|---:|---:|---:|---:|
| 0 (same tile) | 3,590 | 312 | 11.5× |
| 1 (adjacent) | 1,974 | 532 | 3.7× |
| ≥2 (distant) | **2** | 3,763 | 0.0005× |

A library/PCR duplicate lands anywhere in the lane, so ~3,763 duplicate pairs
should sit on non-adjacent tiles. Two do. Essentially every duplicate in this
run is a proximity duplicate, and the residual PCR rate is near zero. The same
table independently confirms the `<surface><swath><tile>` decode: the nine most
enriched tile pairs are exactly `X01`–`X02` for every swath.

Scored against a sequence-only ground truth (reads differing at ≤4 of 302
bases, an allowance calibrated from pairs one well apart, of which 96.7% differ
at ≤4 — and using no coordinate, so it is independent of the spatial test):

| | value |
|:---|---:|
| true duplicate load | 6,255 reads (6.88%) |
| removed at default settings | 6,537 |
| **sensitivity** | **99.71%** |
| **specificity** | **99.66%** |
| duplicate rate before / after | 6.21% → **0.175%** |

Note the coordinates are quantised to a 10-pixel lattice — a patterned
flowcell — so 12.8% of same-tile duplicate pairs sit in a directly adjacent
well, against 0.003% expected by chance.

## Does it eat library duplicates?

The obvious worry about a filter that removed 7% of a run is that on a normal
library it will strip real PCR duplicates and destroy the complexity estimate.
Two checks say it does not.

**A clean non-patterned run.** `test_data/SCA1-cfWGS-1_*` is a MiSeq run
(124,257 pairs, 2 tiles, continuous coordinates) with 16 exact-duplicate
groups. The duplicate distances are cleanly bimodal: 34 reads have a
look-alike within 100 px and then *nothing* until 500 px. The filter removes
17 reads (0.014%), all at 12–35 px. Of the 16 duplicate groups, the 15 that sit
far apart or on different tiles — genuine library duplicates — are **all
kept**; only the one 35 px group is called optical. A Picard-style 100 px rule
flags exactly the same 34 reads, but a 2500 px rule (the patterned setting)
would flag 172, a five-fold over-call. Reading the scale off the data is what
removes that choice.

**Spiked library duplicates.** `test/spike_pcr_duplicates.py` overwrites a
share of a real run with copies of other read pairs, keeping each recipient's
own cluster position and re-drawing errors from its own quality string. The
geometry, qualities and sequences stay real; the library-duplicate rate becomes
known and no optical duplicates are added, so every spiked read the filter
removes is a false positive.

| run | spiked library duplicates | removed | false positives |
|:---|---:|---:|---:|
| MiSeq + 1% | 1,242 | 17 | **0** |
| MiSeq + 5% | 6,212 | 0 | **0** |
| MiSeq + 10% | 12,425 | 11 | **0** |
| MiSeq + 20% | 24,851 | 0 | **0** |
| patterned lattice + 5% | 4,219 | 0 | **0** |
| MiSeq + 5%, forcing `local-fdr` | 6,212 | 0 | **0** |
| MiSeq + 5%, `--no-spatial-test` | 6,212 | 7,927 | **3,207 (51.6%)** |

The last row is the control: without the spatial test, half the spiked library
duplicates are removed. With it, none are — and that holds even when
`local-fdr`, the method that removed 7% of the patterned run, is forced. It is
self-limiting: with `pi0 = 1` the mixture finds no alternative and every local
FDR is one. The aggressive behaviour on the patterned run came from that run's
data, not from the method.

The `duplicate_decomposition` block in the report is what separates the two
cases without being told which is which:

| | patterned run | MiSeq run | MiSeq + 5% PCR |
|:---|---:|---:|---:|
| reads in proximity duplication | 10,901 (12.0%) | 34 (0.027%) | 26 |
| length scale | 5000 px | 50 px | 50 px |
| `pi0` | 0.038 | 1.000 | 1.000 |
| method chosen | local-fdr | permutation | permutation |

The proximity estimate barely moves when 5% library duplicates are added, which
is the point: it measures the spatial component only. When the excess does not
clear the permutation scatter the length scale is reported as not determined
rather than guessed.

`local-fdr` is the most informative per hypothesis — a posterior probability
rather than an adjusted p-value, and what the read-level FDR estimate is built
from — but it needs the alternative to be visible in the data. Where many
hypotheses have no testable relation, the resulting atom at `p = 1` makes its
null-proportion estimate conservative, up to `pi0 = 1`. That is the safe
direction, but it pairs best with `--tile-neighborhood lane`, which leaves no
atom. The local FDR is written to the audits whichever method decides.

## Removing candidates without a spatial test

`--no-spatial-test` collapses every sequence-compatible candidate, ignoring
geometry. This is **sequence-level deduplication, not optical filtering**: it
removes library/PCR copies and independent molecules that happen to share an
insert along with the proximity duplicates. Lowering
`--min-log10-bayes-factor` far enough additionally keeps every relation the
seed index retrieved, including collisions between genuinely different
molecules. On the simulated benchmark, scored against optical truth:

| mode | tested relations | removed | sensitivity | non-optical removals |
|:---|---:|---:|---:|---:|
| default | 16,744 | 1,579 | 98.6% | 0.5% |
| `--no-spatial-test` | 16,744 | 2,695 | 99.6% | 37.6% |
| plus `--min-log10-bayes-factor -1e12` | 37,811 | 3,745 | 99.8% | 54.9% |

The last two columns are the point: these modes are doing a different job, and
the "non-optical" share is mostly library duplicates they are meant to remove.
Use them as an upper bound or when sequence-level deduplication is what you
want, not as a more sensitive optical filter.

## Install

```bash
python3 -m pip install -r fastq_optifilter_requirements.txt
```

`pigz` is optional. When installed, it is used for parallel output compression;
quality scoring and spatial-null construction use `--threads` regardless.

## Example

```bash
python3 fastq_optifilter.py \
  --r1 sample_R1.fastq.gz \
  --r2 sample_R2.fastq.gz \
  --output-r1 sample.fastqoptifilter.R1.fastq.gz \
  --output-r2 sample.fastqoptifilter.R2.fastq.gz \
  --report-json sample.fastqoptifilter.report.json \
  --report-md sample.fastqoptifilter.report.md \
  --read-audit sample.fastqoptifilter.reads.tsv.gz \
  --removed-audit sample.fastqoptifilter.removed.tsv.gz \
  --qq-plot sample.fastqoptifilter.qq.png \
  --diagnostic-plot sample.fastqoptifilter.diagnostics.png \
  --log sample.fastqoptifilter.run.log \
  --threads 8
```

Use `--threads 0` to use every detected CPU core. Progress messages contain a
stage, throughput, elapsed time, and estimated remaining time, and are written
to both stderr and `--log`.

For a less conservative sensitivity run, change `--fdr`, for example
`--fdr 0.05`. Do not select the FDR after examining which individual reads are
removed; set it as an analysis policy.

## Tile neighbourhood

`--tile-neighborhood` chooses which tile relations may carry a spatial test.

- `adjacent` (default) also searches the tiles touching a read's own tile on
  the decoded flowcell grid.
- `same-tile` restricts to a read's own tile.
- `lane` compares every tile on the same surface. It removes the atom at
  `p = 1` entirely, at the cost of a much more expensive edge-level null.

Tile identifiers are decoded as Illumina's `<surface><swath><tile>` (MiSeq,
HiSeq, NovaSeq) or `<surface><swath><camera><tile>` (NextSeq). An unrecognised
convention falls back to a single nominal column, which still works but whose
adjacency is nominal rather than physical; the report names the convention that
was used. Surfaces are separate physical planes and are never neighbours.

`--tile-gap` sets the dead space assumed between neighbouring tiles. It affects
power only, never validity: the observed distances and the null are built in the
same frame, so a frame that packs tiles too tightly makes boundary pairs look
close and makes exactly the same boundary pairs look close in the null.

## Outputs

- Filtered R1 and R2 FASTQs remain synchronized.
- The JSON report contains configuration, decoded flowcell geometry, the
  null-calibration table, model estimates, FDR sensitivity, and filtering
  counts.
- The Markdown report is a concise human-readable summary.
- The read audit gives, per tested read, its nearest sequence-compatible
  partner, that partner's distance and tile, how many candidate partners the
  read had, how many clusters lie within that distance, the spatial p-value,
  the local FDR, every method's q-value, and whether the read was removed.
- The candidate audit is the same information per candidate relation.
- The removed audit identifies the retained representative and supporting
  relation for every removed read pair.
- The QQ plot compares each null control against a uniform null and plots local
  FDR against separation.
- The diagnostic plot compares the observed spatial distribution with the
  geometry-derived null.

## Tests

`test/simulate_fastq.py` generates paired FASTQs with a ground-truth table,
including same-tile and cross-tile optical duplicates, library duplicates at
independent positions, tile-to-tile and within-tile density variation, and a
spatially clustered poly-G artifact. `test/evaluate_run.py` scores a run against
that truth. `test/test_calibration.py` runs both and asserts that the null is
calibrated, that nothing is removed from a pure-null dataset, and that the
nearby-tile search is what recovers cross-tile duplicates:

```bash
python3 test/test_calibration.py
```

## Important scope limits

FastqOptiFilter estimates proximity duplication from raw reads. It cannot
perfectly distinguish a true library/PCR copy from an independently generated
biological molecule with the same insert, especially for non-randomly
fragmented cfDNA. Nor can it distinguish optical duplication from any other
mechanism that puts sequence-identical reads close together, such as a
low-complexity or poly-G cluster; the `sequence_incompatible` control is what
tells you how much of that is present. UMI-aware analysis remains preferable
when UMIs are present.

Read removal is component-based, so the nominal FDR is still not an exact
read-level error rate; the report carries an estimate that sums the local FDR
of the relation supporting each removal. For final library-complexity
estimates, remap the filtered FASTQs and rerun the same duplicate-marking
pipeline used for the unfiltered data, while retaining the FastqOptiFilter
sensitivity table as uncertainty around the correction.

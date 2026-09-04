# FastqOptiFilter

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22307671.svg)](https://doi.org/10.5281/zenodo.22307671)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg)](https://www.python.org/)

Quality-aware, FDR-controlled removal of optical and proximity duplicates from
synchronized paired-end Illumina FASTQ files — **before alignment**, using the
lane/tile/X/Y coordinates already present in the read names.

**The point is library complexity.** Complexity and library-size estimators —
Picard `EstimateLibraryComplexity`, `preseq`, and every yield extrapolation
built on them — read the duplicate rate as evidence that you are resampling a
finite pool of distinct molecules. Optical duplicates are not resampling. They
are one molecule counted twice by the instrument, and every one of them left in
the data drags the complexity estimate down and the predicted future yield with
it. On the patterned-flowcell run benchmarked below, that mistake understates
library size by **37-fold**, and it is the whole reason this tool exists.

FastqOptiFilter applies **no fixed spatial-distance cutoff**. The length scale
of duplication is estimated from the run itself, so the same command works on a
patterned flowcell where duplicates spread over thousands of pixels and on a
non-patterned MiSeq where they sit within tens of pixels, without being told
which is which.

Every run also reports whether its own statistical null is calibrated, using
two negative controls built from the data.

---

## Contents

- [Why this exists](#why-this-exists)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Where this fits in a pipeline](#where-this-fits-in-a-pipeline)
- [Outputs](#outputs)
- [Options](#options)
- [Large runs and memory](#large-runs-and-memory)
- [Benchmarks](#benchmarks)
- [How the model works](#how-the-model-works)
- [Scope and limitations](#scope-and-limitations)
- [Tests](#tests)
- [Citation](#citation)
- [Authors](#authors)
- [Contact](#contact)
- [License](#license)

---

## Why this exists

### Optical duplicates corrupt library complexity

Library complexity estimation rests on one assumption: that when you see the
same molecule twice, you sampled the same molecule twice. From the duplicate
rate, the standard Lander–Waterman relation

```
distinct molecules recovered = N · (1 − e^(−n/N))
```

is inverted to estimate `N`, the number of distinct molecules in the library.
That estimate then drives everything downstream — whether a sample has enough
complexity to call low-frequency variants, how deep to sequence next, whether
to rebuild the library at all.

An optical duplicate breaks the assumption. It is not a second sampling of a
molecule from the library; it is one molecule that the instrument read twice,
because a cluster was split during image analysis or, on a patterned flowcell,
because the molecule seeded a neighbouring well. Counting it as a library
duplicate inflates the apparent duplicate rate, and because the inversion above
is steeply non-linear at low duplicate rates, a small absolute error becomes a
large error in `N`.

The patterned-flowcell run benchmarked below shows the size of it. Its raw
duplicate rate is 6.21%, but essentially all of that is proximity duplication —
of the ~3,763 duplicate pairs that should have landed on non-adjacent tiles if
they were library duplicates, **two** did. After filtering, the true library
duplicate rate is 0.175%.

| | apparent duplicate rate | estimated library size | predicted useful fraction at 10× depth |
|:---|---:|---:|---:|
| optical duplicates left in | 6.21% | **702,000** molecules | 56.1% |
| after FastqOptiFilter | 0.175% | **25,900,000** molecules | 98.3% |

A **37-fold** underestimate of complexity, and a forecast that 44% of further
sequencing would be wasted when in fact almost none of it would be. For cfDNA
and other low-input applications, where complexity is the limiting factor for
detecting rare variants, that is the difference between using a sample and
discarding it.

### The problem is growing, not shrinking

Illumina has moved from non-patterned flowcells with randomly placed clusters —
the legacy MiSeq, HiSeq 2500, NextSeq 500/550 — to patterned flowcells with
ordered nanowells and ExAmp chemistry, including the iSeq series, NextSeq
1000/2000, NovaSeq, and the current MiSeq i100 generation that replaces the
legacy MiSeq. On a patterned flowcell a library molecule can seed a
*neighbouring* well rather than only the one it landed in, so proximity
duplicates are both far more common and spread far further.

The two real runs benchmarked here bracket exactly that transition, and the
tool measures the difference without being told about it:

| | legacy non-patterned MiSeq | patterned flowcell |
|:---|---:|---:|
| reads in proximity duplication | 0.027% | **12.0%** |
| length scale of the mechanism | 50 px | **5000 px** |

Roughly 440× the rate and 100× the reach. Anyone whose duplicate-handling was
tuned on non-patterned data is now applying it to a regime it was never
calibrated for.

### Why a fixed pixel threshold does not solve it

The conventional fix flags a duplicate pair as optical when the two clusters
fall within a fixed pixel distance — 100 px for non-patterned flowcells,
2500 px for patterned ones, by convention. That constant has to be chosen per
instrument, and choosing it wrong destroys genuine library duplicates: on the
clean MiSeq run benchmarked below, the patterned setting flags **172** reads
where the correct answer is **34**.

FastqOptiFilter replaces the constant with a null built from the run's own
cluster pattern, estimates the length scale from the data, and controls a false
discovery rate rather than applying a threshold. Working on FASTQ rather than
BAM also means the correction is available before alignment, and the same
coordinates are used whether or not a read ever maps.

---

## Requirements

### Prerequisites

| | |
|:---|:---|
| **Python** | ≥ 3.11 (developed and tested on 3.14) — the floor comes from SciPy, see below |
| **OS** | Linux or macOS |
| **Memory** | roughly 4 GB per million read pairs |
| **Input** | synchronized paired-end FASTQ (`.fastq` or `.fastq.gz`), fixed read length, Illumina-style read names |

### Python packages

```
numpy>=1.26
scipy>=1.17
matplotlib>=3.7
```

> **`scipy>=1.17` is a hard requirement.** FastqOptiFilter uses
> `scipy.stats.poisson_binom`, which does not exist in earlier releases. A
> too-old SciPy fails at import with a confusing `ImportError`, not a version
> message. SciPy 1.17 was never released for Python 3.10, which is what sets
> the Python floor at 3.11 — the FastqOptiFilter source itself is 3.10-clean.

### Optional

- **`pigz`** — used automatically for parallel output compression when present
  on `PATH`. Everything else is threaded via `--threads` regardless.

### Input requirements

Read names must carry coordinates in the standard Illumina layout, where the
last three colon-separated fields are tile, X and Y:

```
@<instrument>:<run>:<flowcell>:<lane>:<tile>:<x>:<y>
 └────────────── lane key ──────────────┘ └── position ──┘
```

Tile identifiers are decoded as `<surface><swath><tile>` (MiSeq, HiSeq,
NovaSeq) or `<surface><swath><camera><tile>` (NextSeq). An unrecognised
convention still runs, but tile adjacency becomes nominal rather than physical;
the report always names the convention that was used. Reads whose names cannot
be parsed abort the run unless `--unparsed keep` is given.

---

## Installation

Everything below installs the same tool. Pick one.

> **Python must be 3.11 or newer.** FastqOptiFilter uses
> `scipy.stats.poisson_binom`, which does not exist before SciPy 1.17, and no
> SciPy 1.17 was ever built for Python 3.10 — so on 3.10 the install fails to
> resolve rather than producing a working environment. Many systems still
> default to 3.9 or 3.10. If you pin dependencies by hand, note that an older
> SciPy fails at *import* with a confusing `ImportError` rather than a version
> message.

### Option A — pip (recommended)

```bash
pip install fastqoptifilter
```

This installs the `fastqoptifilter` command and pulls NumPy, SciPy and
Matplotlib automatically. Into an isolated environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install fastqoptifilter
```

Or with [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install fastqoptifilter
```

Or with [`pipx`](https://pipx.pypa.io/), which keeps the tool in its own
environment while putting the command on your `PATH`:

```bash
pipx install fastqoptifilter
```

### Option B — conda / mamba

```bash
conda create -n fastqoptifilter -c conda-forge -c bioconda fastqoptifilter
conda activate fastqoptifilter
```

`-c conda-forge` matters: SciPy 1.17+ comes from conda-forge, not from the
`defaults` channel. If your conda is old enough that the solve is slow, `mamba`
is a drop-in replacement.

To build the conda package yourself from a clone, a recipe is included:

```bash
conda install -c conda-forge conda-build
conda build conda-recipe
conda install -c conda-forge --use-local fastqoptifilter
```

### Option C — from source

No installation step; the tool is a single self-contained module.

```bash
git clone https://github.com/epifluidlab/FastqOptiFilter.git
cd FastqOptiFilter
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r fastq_optifilter_requirements.txt
python3 fastq_optifilter.py --help
```

Or install the clone as an editable package, which also gives you the
`fastqoptifilter` command:

```bash
pip install -e .
```

### Verify the installation

```bash
python3 -c "from scipy.stats import poisson_binom; print('scipy OK')"
fastqoptifilter --help          # or: python3 fastq_optifilter.py --help
```

### Run the test suite (optional, ~5 minutes)

Requires a clone, since the tests are not shipped in the package. This
simulates data with known ground truth and asserts that the null is calibrated,
that nothing is removed from a pure-null dataset, and that the nearby-tile
search is what recovers cross-tile duplicates.

```bash
python3 test/test_calibration.py
```

Expect `16 passed, 0 failed`.

---

## Quick start

Minimal:

```bash
python3 fastq_optifilter.py \
  --r1 sample_R1.fastq.gz \
  --r2 sample_R2.fastq.gz \
  --output-r1 sample.filtered.R1.fastq.gz \
  --output-r2 sample.filtered.R2.fastq.gz \
  --report-json sample.report.json \
  --threads 8
```

Recommended — adds the human-readable report, the per-read audit and the
calibration plot, which are what let you check the result rather than trust it:

```bash
python3 fastq_optifilter.py \
  --r1 sample_R1.fastq.gz \
  --r2 sample_R2.fastq.gz \
  --output-r1 sample.filtered.R1.fastq.gz \
  --output-r2 sample.filtered.R2.fastq.gz \
  --report-json sample.report.json \
  --report-md   sample.report.md \
  --read-audit  sample.reads.tsv.gz \
  --removed-audit sample.removed.tsv.gz \
  --qq-plot     sample.qq.png \
  --log         sample.run.log \
  --threads 8
```

`--threads 0` uses every detected core. Progress lines carry a stage,
throughput, elapsed time and ETA, and go to both stderr and `--log`.

### Reading the result

Three numbers in `sample.report.md` tell you most of what happened:

| field | meaning |
|:---|:---|
| **Reads involved in proximity duplication** | how much of the run is spatially duplicated |
| **Length scale** | the distance at which that excess saturates — the mechanism's reach in this run |
| **Max excess over uniform** (null calibration) | at or below zero means the p-values are valid |

If the `permutation` control shows a max excess well above zero, the geometry
is being computed wrongly and the result should not be trusted. If the
`sequence_incompatible` control is inflated, real reads carry spatial structure
that within-lane exchangeability does not capture, and q-values are optimistic
by roughly that factor.

---

## Where this fits in a pipeline

**Run FastqOptiFilter first, on raw FASTQ, before adapter or quality trimming.**

```
raw FASTQ
   ↓
FastqOptiFilter          ← here
   ↓
adapter / quality trimming     (cutadapt, fastp, Trimmomatic …)
   ↓
alignment
   ↓
MarkDuplicates / UMI collapsing
   ↓
library complexity estimate    (EstimateLibraryComplexity, preseq)
```

### Why before trimming

**It will not run on trimmed reads.** FastqOptiFilter requires fixed-length
input and aborts otherwise:

```
ValueError: FastqOptiFilter currently requires fixed-length raw FASTQs;
found lengths 151/151 at pair 2, expected 134/134
```

Trimming the MiSeq run benchmarked here produces 111 distinct read lengths, so
it fails immediately. This is an implementation limit rather than a statistical
one — `encode_matrices` builds a fixed *n* × cycles matrix, and variable lengths
would need padding plus a validity mask — but as it stands it is a blocker.

**Nothing is gained by trimming first.** Cluster coordinates are untouched by
trimming, so the spatial model sees exactly the same geometry either way.

**Adapter read-through does not fool the sequence filter.** The obvious worry is
that two different short fragments both reading into the adapter will look like
duplicates. They do not: the Bayes factor sums over all 302 cycles, so a shared
adapter tail is overwhelmed by a mismatching insert. On the patterned run
benchmarked here, 625,545 candidate relations were retrieved and only 40,125
survived the sequence filter.

**Less data reaches every downstream step.** On a run like the patterned one,
you trim and align around 7% fewer reads.

**Seed length is calibrated from the raw quality profile.** Quality trimming
removes exactly the low-quality bases that calibration reads, pushing the seed
length back toward the fixed default and losing the recall shown in
[benchmark 5](#5-retrieval-under-poor-base-quality).

For completeness: trimming does *not* damage duplicate detection, provided the
trimmer is error-tolerant. On the MiSeq run, an exact-match trimmer loses 60% of
duplicate pairs (20 → 8) because the adapter is found at different offsets in
two copies of one molecule, while a trimmer allowing a single mismatch loses
none (20 → 20). That is an argument about trimmer quality, not about ordering.

### One case for pre-filtering

Adapter-dimers and very short inserts form large low-complexity families that
bloat the candidate set. In the patterned run, a single 259-read
adapter-dimer/poly-G family generated **83% of all candidate relations**. Those
reads were handled correctly — they received zero calls — but they cost runtime
and inflate the hypothesis count.

If a library is heavy in adapter-dimers, discard those reads first. That is a
length or complexity filter, not adapter trimming: drop reads whose adapter
begins before roughly cycle 20, which leaves every surviving read at full
length and keeps the fixed-length requirement satisfied.

### Downstream

The filtered FASTQs are ordinary synchronized FASTQs and need no special
handling. Removing proximity duplicates before `MarkDuplicates` is what makes
the resulting duplicate rate mean what complexity estimators assume it means —
see [Why this exists](#optical-duplicates-corrupt-library-complexity).

If UMIs are present, prefer UMI-aware collapsing for molecular counting;
FastqOptiFilter still helps by removing instrument-generated copies that share
a UMI, but it is not a substitute.

---

## Outputs

| output | flag | contents |
|:---|:---|:---|
| Filtered FASTQs | `--output-r1/--output-r2` | remain synchronized and in input order |
| JSON report | `--report-json` | configuration, decoded geometry, duplicate decomposition, null calibration, FDR sensitivity, counts |
| Markdown report | `--report-md` | the same, readable |
| Read audit | `--read-audit` | per tested read: nearest look-alike, its distance and tile, candidate partner count, local neighbour count, p-value, local FDR, every method's q-value, removed or not |
| Candidate audit | `--candidate-audit` | the same per candidate relation |
| Removed audit | `--removed-audit` | per removed read: which read was retained instead, and the evidence |
| QQ plot | `--qq-plot` | null controls against a uniform null, plus local FDR against distance |
| Diagnostic plot | `--diagnostic-plot` | observed spatial distribution against the geometry null |

---

## Options

Defaults are chosen so that `--r1/--r2/--output-*/--report-json` alone gives a
sensible run. Everything below is optional.

### Core

| option | default | what it does |
|:---|:---|:---|
| `--fdr` | `0.01` | target false discovery rate |
| `--fdr-method` | `auto` | `auto`, `permutation`, `by`, `local-fdr`, `bh`, `weighted-bh` — see below |
| `--threads` | `1` | worker threads; `0` uses all cores |
| `--force` | off | overwrite existing outputs |

### Spatial model

| option | default | what it does |
|:---|:---|:---|
| `--tile-neighborhood` | `adjacent` | which tile relations may carry a spatial test: `same-tile`, `adjacent` (also the tiles touching a read's own), or `lane` (every tile on the surface; slower, but leaves no atom at `p = 1`) |
| `--tile-gap` | `0` | pixels of dead space assumed between neighbouring tiles. Affects power only, never validity — observed distances and the null use the same frame |
| `--max-grid-step` | `1` | tile-grid rings searched when the neighbourhood is `adjacent` |
| `--spatial-metric` | `chebyshev` | `chebyshev` or `euclidean` |
| `--inference-unit` | `read` | `read` tests each read once against its nearest look-alike; `edge` reproduces the older per-relation test |
| `--permutations` | `20` | reshuffled replicates used to measure the FDR and the duplicate decomposition; `0` disables both |

### Candidate retrieval

| option | default | what it does |
|:---|:---|:---|
| `--seed-length` | `20` | **maximum** exact seed length; the qualities may select shorter |
| `--adaptive-seed` | on | let the base qualities shorten the seed. `--no-adaptive-seed` pins it |
| `--min-log10-bayes-factor` | `0.0` | sequence evidence a candidate needs to be tested at all |
| `--sequence-min-p` | `0.0` | optional extra Poisson-binomial compatibility screen; `0` disables |
| `--max-seed-bucket` | `500` | skip seed buckets larger than this; the main lever for a run that will not fit — see [Large runs and memory](#large-runs-and-memory) |
| `--min-seed-entropy` | `0.0` | skip low-entropy seeds (homopolymer, poly-G); `0` disables |
| `--max-exact-family` | `5000` | abort on larger exact families (adapter/low-complexity artifacts) |
| `--max-candidates` | `5000000` | safety limit on retrieved relations |

### Advanced

| option | default | what it does |
|:---|:---|:---|
| `--spatial-test` | on | `--no-spatial-test` removes every sequence-compatible candidate, ignoring geometry. **This is sequence-level deduplication, not optical filtering** |
| `--auto-pi0-threshold` | `0.5` | with `--fdr-method auto`, use `local-fdr` below this estimated null proportion |
| `--pi0-lambda` | `0.5` | Storey tuning point for the null proportion |
| `--null-resolution` | `384` | log-spaced radii above 256 px at which the geometry null is tabulated |
| `--null-check-seed` | `20260903` | seed for the permutation controls |
| `--unparsed` | `error` | `keep` tolerates read names without coordinates |
| `--gzip-level` | `6` | output compression level |
| `--score-chunk` | `1000` | candidate relations per parallel scoring task |
| `--progress-interval` | `10.0` | seconds between progress messages |

### Choosing `--fdr-method`

No single method suits every run, which is why the default is `auto`.

A tail test — `bh`, `by`, `permutation` — asks whether a p-value is extreme
against a uniform null. That question has the wrong answer once most hypotheses
are genuinely non-null: a real duplicate far from its twin is unremarkable on
its own, and only a fitted mixture knows that nearly everything around it is
also a duplicate. `auto` selects `local-fdr` when the estimated null proportion
`pi0` falls below `--auto-pi0-threshold`, and `permutation` otherwise. Both the
log and the report state which was chosen and why.

| method | assumption | use when |
|:---|:---|:---|
| `auto` | — | **default**; picks between the two below |
| `permutation` | none — reshuffles the flowcell and measures the FDR directly | ordinary runs, sparse duplication |
| `local-fdr` | alternative is present and estimable | duplicate-dominated runs (`pi0` small) |
| `by` | none (Benjamini-Yekutieli) | you want to under-remove by construction; costs ~`log(m)` |
| `bh` | positive regression dependence | comparison against conventional pipelines |
| `weighted-bh` | sequence evidence independent of position | sensitivity analysis, `--inference-unit edge` only |

---

## Large runs and memory

Candidate retrieval, not the statistical model, is what limits scale. A seed
bucket holding `k` reads contributes `k(k-1)/2` candidate pairs, so cost grows
with the *square* of the largest bucket rather than with read count. One bucket
of 500 reads is already 124,750 pairs.

If a run stops with

```
MemoryError: Candidate count exceeded --max-candidates=...
```

raising `--max-candidates` is usually the wrong response. The message names the
bucket responsible and whether it came from the seed index or the full-pair
hash; read that first, because a bucket of thousands of reads sharing one exact
20-mer is adapter, poly-G or another low-complexity sequence, not real
duplication. Genuine duplicate families hold a handful of reads.

`--max-seed-bucket` is the effective lever. Measured on a simulated 1M-read run
at 151 bp:

| `--max-seed-bucket` | candidates | tested edges | reads removed | peak RSS |
|---:|---:|---:|---:|---:|
| 500 (default) | 11,822,602 | 4,789,652 | 41,038 | 6.6 GB |
| 100 | 4,969,428 | 4,789,652 | **41,038** | 5.6 GB |
| 25 | 4,837,272 | 4,789,652 | **41,038** | 4.9 GB |

Candidates fall by 59% while the tested set and the calls are unchanged: the
extra pairs were all discarded by the sequence model anyway.

**It is not free, though.** Excluding those reads also removes them from the
multiple-testing correction, which loosens the threshold for everything else.
On the benchmark simulation, which contains a 259-read poly-G family:

| setting | removed | sensitivity | read-level FDR |
|:---|---:|---:|---:|
| default (bucket 500, `--fdr 0.01`) | 1,579 | 98.6% | 0.89% |
| bucket 100, `--fdr 0.01` | 1,601 | 98.8% | **2.06%** |
| bucket 100, `--fdr 0.005` | 1,568 | 98.6% | **0.26%** |

So pair a lower bucket cap with a stricter target:

```bash
--max-seed-bucket 100 --fdr 0.005
```

That reproduces the default result while cutting retrieval several-fold. Other
levers, in rough order of usefulness:

| lever | effect |
|:---|:---|
| `--max-seed-bucket 100` | 3–10× fewer candidates; pair with a stricter `--fdr` |
| `--seed-length 32` | longer seeds collide far less; costs sensitivity on low-quality reads |
| `--min-seed-entropy 1.0` | drops homopolymer seeds; same correction caveat as above |
| `--max-exact-family` | governs the full-pair hash path, which `--max-seed-bucket` does not touch |
| remove adapter-dimers upstream | a length filter, not trimming — see [Where this fits in a pipeline](#one-case-for-pre-filtering) |

Candidate keys are held as sorted `int64` rather than in a Python set, which
costs about 8 bytes per pair instead of roughly 60. Budget on that basis when
choosing `--max-candidates`, and remember the per-candidate scoring arrays are
allocated on top.

---

## Benchmarks

Three independent evaluations: a simulation with full ground truth, two real
runs of opposite character, and library duplicates spiked into real data at
known rates.

### 1. Simulation with ground truth

`test/simulate_fastq.py` generates paired FASTQs with a truth table, including
same-tile and cross-tile optical duplicates, library duplicates at independent
positions, tile-to-tile and within-tile density variation, and a spatially
clustered poly-G artifact. Scoring is family-aware: for a true optical family
of size *k*, exactly *k−1* members should be removed, and it does not matter
which is retained. 40,000 read pairs, three seeds.

| configuration | removed | TP | FN | FP | sensitivity | FDR | cross-tile recall |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **FastqOptiFilter** (seed 17) | 1,579 | 1,565 | 22 | 8 | **98.6%** | **0.51%** | **95.2%** |
| **FastqOptiFilter** (seed 33) | 1,642 | 1,561 | 28 | 15 | **98.2%** | **0.91%** | **94.6%** |
| **FastqOptiFilter** (seed 71) | 1,569 | 1,563 | 24 | 3 | **98.5%** | **0.19%** | **95.3%** |
| same-tile / per-relation / BH (seed 17) | 1,473 | 1,351 | 236 | 0 | 85.1% | 0.00% | 0.9% |
| same-tile / per-relation / BH (seed 33) | 1,434 | 1,318 | 271 | 0 | 82.9% | 0.00% | 0.0% |
| same-tile / per-relation / BH (seed 71) | 1,412 | 1,303 | 284 | 0 | 82.1% | 0.00% | 0.7% |

The measured FDR (0.19–0.91%) sits just under the 1% target. Cross-tile
duplicates are unreachable without a nearby-tile search.

**Pure-null control.** On a dataset with library duplicates but no proximity
duplication at all, the tool estimates `pi0 = 1.000` and removes **0** reads.

### 2. Real run with heavy optical duplication (patterned flowcell)

90,927 read pairs, 18 tiles, 151 bp, coordinates on a 10-pixel lattice.

The geometry identifies the mechanism without a simulator. Library duplicates
land anywhere in the lane, so their tile relation should follow chance:

| tile grid step | observed pairs | expected if positions independent | ratio |
|---:|---:|---:|---:|
| 0 (same tile) | 3,590 | 312 | **11.5×** |
| 1 (adjacent) | 1,974 | 532 | **3.7×** |
| ≥ 2 (distant) | **2** | 3,763 | **0.0005×** |

Around 3,763 duplicate pairs should have landed on non-adjacent tiles. **Two
did.** Essentially every duplicate in this run is a proximity duplicate. The
same table independently confirms the tile decode — the nine most enriched tile
pairs are exactly `X01`–`X02` for every swath.

Scored against a sequence-only ground truth (reads differing at ≤ 4 of 302
bases, an allowance calibrated from pairs one well apart, of which 96.7% differ
at ≤ 4 — and using no coordinate, so it is independent of the spatial test):

| | |
|:---|---:|
| true duplicate load | 6,255 reads (6.88%) |
| removed | 6,537 |
| true positives | 6,237 |
| false negatives | 18 |
| false positives | 283 |
| **sensitivity** | **99.71%** |
| **specificity** | **99.66%** |
| duplicate rate before → after | 6.21% → **0.175%** |

The 283 false positives are an upper bound: the truth caps at 4 mismatches, and
3.3% of certain-optical pairs exceed that.

The effect on complexity estimation, inverting `distinct = N(1 − e^(−n/N))` at
n = 90,927 reads:

| | duplicate rate | estimated library size |
|:---|---:|---:|
| unfiltered | 6.21% | 702,000 |
| filtered | 0.175% | **25,900,000** |

Extrapolating to ten times the depth, the unfiltered estimate predicts 510,000
distinct molecules from 909,270 reads (56.1% useful); the filtered estimate
predicts 894,000 (98.3%). The unfiltered numbers would justify abandoning a
library that is in fact close to ideal.

A 259-read adapter-dimer/poly-G family, contributing 83% of all candidate
relations, received **zero** calls — the multiplicity term and the
density-adaptive neighbour count reject it.

### 3. Clean run with almost no optical duplication (non-patterned MiSeq)

124,257 read pairs, 2 tiles, continuous coordinates, 16 exact-duplicate groups
(0.016%). This is the case where over-removal would destroy library complexity.

Duplicate distances are cleanly bimodal — 34 reads have a look-alike within
100 px, then **nothing** until 500 px.

| | |
|:---|---:|
| reads removed | **17 (0.014%)** |
| distance range of removals | 12–35 px |
| distant / cross-tile duplicate groups (real library duplicates) | 15 of 16 |
| **of those, removed** | **0** |

A Picard-style 100 px rule flags exactly the same 34 reads. The *patterned*
2500 px setting would flag **172** — a five-fold over-call. Estimating the
scale from the data is what removes that choice.

### 4. Simulated library duplicates spiked into real runs

`test/spike_pcr_duplicates.py` overwrites a share of a real run with copies of
other read pairs, keeping each recipient's own cluster position and re-drawing
errors from its own quality string. Geometry, qualities and sequences stay
real; the library-duplicate rate becomes known and no optical duplicates are
added, so **every spiked read the filter removes is a false positive**.

| run | spiked library duplicates | removed | **wrongly removed** |
|:---|---:|---:|---:|
| MiSeq + 1% | 1,242 | 17 | **0** |
| MiSeq + 5% | 6,212 | 0 | **0** |
| MiSeq + 10% | 12,425 | 11 | **0** |
| MiSeq + 20% | 24,851 | 0 | **0** |
| patterned lattice + 5% | 4,219 | 0 | **0** |
| MiSeq + 5%, forcing `local-fdr` | 6,212 | 0 | **0** |
| MiSeq + 5%, `--no-spatial-test` | 6,212 | 7,927 | **3,207 (51.6%)** |

Zero false positives up to a 20% library-duplicate rate. The last row is the
control: without the spatial test, half the spiked duplicates are removed.

Forcing `local-fdr` — the method that removed 7% of the patterned run — still
removes **zero** here. It is self-limiting: with `pi0 = 1` the mixture finds no
alternative and every local FDR is one. The aggressive behaviour on the
patterned run came from that run's data, not from the method.

### 5. Retrieval under poor base quality

Retrieval is exact, so a duplicate is only found when some seed window is
error-free in **both** copies — which the base qualities decide. Fixing the
seed length in advance discards real duplicates before the quality model can
see them. Measured on 90 bp simulated reads at uniform quality:

| flat Phred | per-base error | fixed 20 bp seed | adaptive | seed chosen |
|---:|---:|---:|---:|---:|
| 20 | 1.0% | 99.3% | 99.3% | 20 |
| 16 | 2.5% | 95.0% | 98.8% | 15 |
| 13 | 5.0% | **61.6%** | **99.4%** | 8 |
| 11 | 7.9% | **25.8%** | **98.3%** | 8 |

Shorter seeds cost candidates, not accuracy: at Phred 13 retrieval grows from
30k to 335k relations while the tested set stays at ~16.8k, because the quality
model discards the extra collisions.

### 6. Performance

| | |
|:---|---:|
| 40,000 read pairs, 4 threads | ~6 s |
| 400,000 read pairs, 8 threads | **107 s**, 1.7 GB peak RSS |

Measured on an Apple M-series laptop, including 20 permutation replicates, both
plots and all audit files.

---

## How the model works

**1 — Candidate retrieval.** Within each lane, reads are indexed by an exact
paired hash and by exact seeds at several offsets in both mates. The seed
length is chosen from the base qualities, as above. This stage is an index, not
a calling rule: recall matters, precision does not, because a spurious
candidate is discarded next.

**2 — Sequence filter.** For each candidate, Phred scores become per-cycle
error probabilities and a Bayes factor compares "one shared template" with "two
independent templates", marginalising over the true base with the empirical
per-cycle base composition as prior. An exact Poisson-binomial tail gives the
probability of seeing at least this many mismatches under a shared template.
This stage uses **only sequence and quality, never position**, which is what
makes the spatial test that follows a genuine test.

**3 — Geometry.** The tile identifier is decoded into a physical grid and every
read is placed in one lane-wide frame per surface, so a pair split across a
tile boundary has a real separation. Surfaces are never neighbours.

**4 — The spatial test.** Each read is tested once. Given its own position, and
given that it has *m* sequence-compatible look-alikes anywhere in the lane and
surface, how surprising is it that the closest of them landed as close as it
did? With *M* other clusters in the lane and *K* of them within the observed
distance,

```
p = 1 − C(M−K, m) / C(M, m)          (for m = 1, simply K / M)
```

Counting neighbours **around the read itself** is what makes the test adapt to
local density: the same separation is unremarkable in a crowded patch and
surprising in a sparse one. Testing reads rather than relations stops a family
of *k* identical reads from raising *k(k−1)/2* hypotheses over *k* reads.

**5 — Decision.** FDR control by the method above, then significant reads and
their nearest partners are joined into components and the highest total
R1+R2 quality member of each is retained.

**6 — Self-check.** Two negative controls are recomputed every run: a
permutation of the position labels, which tests whether the geometry null is
computed correctly, and quality-model-rejected read pairs, which tests whether
within-lane exchangeability holds on real reads.

---

## Scope and limitations

- FastqOptiFilter cannot distinguish optical duplication from **any other
  mechanism that puts sequence-identical reads close together**, such as a
  low-complexity or poly-G cluster. The `sequence_incompatible` control tells
  you how much of that a run contains.
- Without UMIs it cannot perfectly separate a library copy from an
  independently generated molecule with the same insert, which matters most for
  non-randomly fragmented cfDNA.
- Read removal is component-based, so the nominal FDR is not an exact
  read-level error rate; the report carries an estimate that sums the local FDR
  of the relation supporting each removal.
- Candidate retrieval needs at least one exact seed or exact full paired
  sequence match.
- Base qualities are assumed approximately calibrated and conditionally
  independent by cycle.
- Cross-tile adjacency is decoded from the tile identifier; an unrecognised
  convention falls back to a single nominal column.
- Fixed-length reads only.
- For final library-complexity estimates, remap the filtered FASTQs and rerun
  the same duplicate-marking pipeline used on the unfiltered data, retaining
  the FDR sensitivity table as uncertainty around the correction.

---

## Tests

```bash
# full calibration and accuracy suite (~5 min)
python3 test/test_calibration.py

# generate simulated data with ground truth
python3 test/simulate_fastq.py --out-prefix sim --reads 40000 --seed 17

# score a run against that truth
python3 test/evaluate_run.py --truth sim.truth.tsv.gz --filtered-r1 out_R1.fastq.gz

# spike library duplicates into a real run at a known rate
python3 test/spike_pcr_duplicates.py --r1 real_R1.fastq.gz --r2 real_R2.fastq.gz \
    --out-prefix spiked --pcr-rate 0.05
```

---

## Citation

FastqOptiFilter is archived on Zenodo and has a DOI. If you use it, please
cite it:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22307671.svg)](https://doi.org/10.5281/zenodo.22307671)

> Liu, Y. *FastqOptiFilter: quality-aware, FDR-controlled optical and proximity
> duplicate filtering for paired-end Illumina FASTQ files.*
> https://doi.org/10.5281/zenodo.22307671

```bibtex
@software{fastqoptifilter,
  author  = {Liu, Yaping},
  title   = {{FastqOptiFilter: quality-aware, FDR-controlled optical and
             proximity duplicate filtering for paired-end Illumina FASTQ files}},
  year    = {2026},
  doi     = {10.5281/zenodo.22307671},
  url     = {https://github.com/epifluidlab/FastqOptiFilter},
  version = {0.1.1},
  license = {MIT}
}
```

The DOI above always resolves to the latest release. Zenodo also mints a
version-specific DOI for each release; use that one if you need to pin an exact
version for reproducibility. Machine-readable metadata lives in
[CITATION.cff](CITATION.cff), which is what GitHub's "Cite this repository"
button reads.

---

## Authors

**Yaping Liu** — design, direction and scientific review.

**Claude Code (Opus 5, Anthropic)** — co-author. Contributed to the design of
the spatial statistical model, the implementation, and the benchmarking and
calibration work described above, in collaboration with and under the review of
the author. Contributions are recorded per commit as `Co-Authored-By` trailers
in the git history.

Responsibility for the correctness and the scientific claims of this software
rests with the human author.

---

## Contact

Questions, bug reports and feature requests are welcome.

**Yaping Liu** — <lyping1986@gmail.com>

Please open an issue on
[GitHub](https://github.com/epifluidlab/FastqOptiFilter/issues) for anything
reproducible, and include the JSON report where possible — it contains the
geometry decode and the null-calibration table, which is usually enough to
diagnose a problem.

---

## License

Released under the **MIT License**. See [LICENSE](LICENSE) for the full text.

```
MIT License

Copyright (c) 2026 epifluidlab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

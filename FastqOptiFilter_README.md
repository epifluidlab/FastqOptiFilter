# FastqOptiFilter

FastqOptiFilter is a quality-aware, FDR-controlled filter for optical/proximity
duplicates in synchronized paired-end Illumina FASTQ files. It uses Illumina
lane/tile/X/Y coordinates from read names and does **not** require or apply a
fixed spatial-distance threshold.

## What the model does

1. Finds sequence-similar candidate read-pair relations with an exact paired
   hash plus several exact seeds. Seeds are only an index; they are never used
   as the final duplicate rule.
2. For every candidate, converts Phred scores to per-cycle error probabilities
   and compares a common-template model with an independent-template model.
   It also reports an exact Poisson-binomial mismatch-compatibility p-value.
3. Builds the spatial null directly from every observed cluster coordinate in
   each lane. For a candidate at distance `d`, the p-value is the probability
   that a random unordered pair from that lane is on the same tile and has
   distance at most `d`.
4. Applies Benjamini-Hochberg correction across sequence-compatible candidate
   edges. Sequence-weighted BH is available as a sensitivity analysis.
5. Joins significant edges into components, retains the read pair with the
   largest total R1+R2 quality score, and writes synchronized filtered FASTQs.

The default is unweighted BH at 1% candidate-edge FDR. This is deliberately
conservative. A q-value is not the posterior probability that one read pair is
optical, and candidate-edge FDR is not exactly the same as read-level FDR after
component clustering.

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
  --candidate-audit sample.fastqoptifilter.candidates.tsv.gz \
  --removed-audit sample.fastqoptifilter.removed.tsv.gz \
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

## Outputs

- Filtered R1 and R2 FASTQs remain synchronized.
- The JSON report contains configuration, model estimates, FDR sensitivity,
  and filtering counts.
- The Markdown report is a concise human-readable summary.
- The candidate audit contains sequence likelihoods, quality-aware mismatch
  p-values, spatial p-values, BH q-values, and optional weighted-BH q-values.
- The removed audit identifies the retained representative and supporting edge
  for every removed read pair.
- The diagnostic plot compares the observed spatial distribution with the
  complete geometry-derived null.

## Important scope limits

FastqOptiFilter estimates proximity duplication from raw reads. It cannot
perfectly distinguish a true library/PCR copy from an independently generated
biological molecule with the same insert, especially for non-randomly
fragmented cfDNA. UMI-aware analysis remains preferable when UMIs are present.
For final library-complexity estimates, remap the filtered FASTQs and rerun the
same duplicate-marking pipeline used for the unfiltered data, while retaining
the FastqOptiFilter sensitivity table as uncertainty around the correction.


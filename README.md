# Linear-Time Reference-Free Variant Detection (`snpDetectLinear`)

A reference-free, linear-time variant detection and spectrum analysis suite for discovering single-nucleotide polymorphisms (SNPs) and multi-nucleotide substitutions (MNPs) directly from raw Illumina sequencing reads.

Based on a linear flank-hashing join algorithm.

---

## Overview

Unlike traditional bubble-assembly callers that construct local De Bruijn graphs or assemble contigs, `snpDetectLinear` performs a direct, linear-time hash join over differential solid $k$-mers using shared flanking anchors of decreasing length:

$$P(a) \cdot U \cdot S(a) \longleftrightarrow P(a) \cdot V \cdot S(a)$$

### Key Advantages:
1. **Linear Time $O(|D_A| + |D_B|)$**: Joins differential $k$-mers in descending flank order with a strict pairwise comparison budget.
2. **No Assembly Bottlenecks**: Avoids De Bruijn graph construction, contig extension, and bubble tip trimming.
3. **General Substitution Discovery**: Discovers both 1-bp SNPs and complex consecutive codon substitutions (e.g. *Salmonella enterica* tetracycline resistance *tetR* $\text{TTT} \to \text{GGG}$) without pre-assuming mutation length.
4. **Strand-Neutral & Sample-Preserving**: Enforces canonical $k$-mer representation without losing the identity of Strain A (Wild-Type) vs. Strain B (Variant).

---

## 1. Running Variant Detection (`snpDetectLinear.py`)

Run variant discovery between two read sets:

```bash
python snpDetectLinear.py \
  data/salmonella-enterica.reads.fna \
  data/salmonella-enterica-variant.reads.fna \
  -k 31 \
  -o output/report.txt
```

### Fast Subsampling Mode (Quick Screen)
To test or quickly inspect large datasets, limit read sampling using `-m`:

```bash
python snpDetectLinear.py readsA.fasta readsB.fasta -k 31 -m 250000
```

### Key CLI Options:
| Flag | Description | Default |
| :--- | :--- | :--- |
| `readsA`, `readsB` | Paths to input FASTA reads for Strain A (WT) and Strain B (Variant) | *Required* |
| `-k, --kmer` | $k$-mer length | `31` |
| `-m, --max-reads` | Maximum reads to sample from each file (useful for fast testing) | `None` (all) |
| `-c, --min-coverage` | Explicit coverage cutoff threshold (solid $k$-mer filter) | Auto-detected |
| `-e, --max-error` | Maximum count tolerated in the opposite strain for sequencing noise | `1` |
| `-o, --output` | Path to save the human-readable text report | Console output |
| `--max-flank` | Maximum flank size to begin adaptive search | `(k-1)//2` |
| `--min-flank` | Minimum flank anchor size to test | Auto |
| `--max-comparisons` | Strict budget of pairwise comparisons per flank level | `250,000` |
| `--min-neighbor-support` | Minimum supporting neighbor $k$-mer sides required (`0`, `1`, or `2`) | `1` |
| `--raw-candidates` | Output unclustered raw candidate records across all flank levels | `False` |

---

## 2. Visualizations & Spectrum Analysis (`plotSnpDetectLinear.py`)

`plotSnpDetectLinear.py` is the unified, self-contained visualization companion tool. It produces three distinct publication-quality plot modes without dependencies on legacy non-linear files.

### Mode A: Linear SNP Calling Multi-Panel Plot
Generates a 3-panel figure:
1. Multiplicity spectra of both strains with auto-detected noise cutoffs ($t_A, t_B$) and solid differential count metrics.
2. Flank-hashing join progression (candidate pairs compared vs. new candidate mutations discovered per flank length).
3. Sequencing depth bar profile verifying allele coverage in both strains at each discovered locus.

```bash
python plotSnpDetectLinear.py \
  data/salmonella-enterica.reads.fna \
  data/salmonella-enterica-variant.reads.fna \
  -k 31 \
  -o img/snp_linear_detection.png
```

To also export the k-mer spectrum as a reusable DataFrame CSV:
```bash
python plotSnpDetectLinear.py readsA.fasta readsB.fasta -k 31 --output-df spectrum.csv
```

---

### Mode B: Recreate `kmerAnalysis.py` Spectrum Plot (`--kmer-analysis`)
Recreates the multiplicity spectrum plot across candidate $k$ values with centered moving average smoothing, matching `img/kmer_spectrum_custom_ma5.png`:

```bash
python plotSnpDetectLinear.py --kmer-analysis
```
- **Custom range & smoothing**:
  ```bash
  python plotSnpDetectLinear.py --kmer-analysis --df plot_df/kmer_spectrum_df.csv --min-x 2 --max-x 300 --moving-average 5 -o img/kmer_spectrum_custom_ma5.png
  ```
- Defaults to reading `plot_df/kmer_spectrum_df.csv` and saving to `img/kmer_spectrum_custom_ma5.png`.

---

### Mode C: Recreate `plot_theo_dist.py` Theoretical Mixture & Overlap Band (`--theo-dist`)
Recreates the 2-panel theoretical mixture distribution and Bayesian decision overlap band figure, matching `img/kmer_spectrum_overlap_band.png`:

```bash
python plotSnpDetectLinear.py --theo-dist
```
- **Custom options**:
  ```bash
  python plotSnpDetectLinear.py --theo-dist -k 31 --max-c 260 -o img/kmer_spectrum_overlap_band.png --all-k
  ```
- Evaluates the mixture model:
  $$\text{Total}(c) = \text{Error}(c) + \text{True}(c) + \text{Repeat}_2(c) + \text{Repeat}_3(c)$$
- Computes the overlap uncertainty band $[c_{\text{low}}, c_{\text{high}}]$ where $0.01 < \frac{p(\text{real})}{p(\text{error})} < 100$, and the Bayes decision cutoff $t^*$ where ratio $= 1.0$.

---

## 3. Running the Test Suite (`testSnpDetectLinear.py`)

`testSnpDetectLinear.py` provides unit tests and simulation benchmarks validating:
1. Reverse-complement symmetry and canonical $k$-mer neutrality.
2. Clean synthetic 1-base SNP detection.
3. 3-base codon substitution detection (mirroring the *Salmonella* TetR `TTT` $\to$ `GGG` mutation).
4. Robustness against $1\%$ random Illumina substitution sequencing errors.
5. Diagnostics and level reporting.
6. Verification of `--kmer-analysis` and `--theo-dist` plotting execution.

### Run All Tests:
```bash
python -m unittest testSnpDetectLinear.py
```
*(or run directly: `python testSnpDetectLinear.py`)*

---

## Modular File Summary

```
salmonella-project/
├── snpDetectLinear.py       # Core linear-time variant detector (clean CLI & engine)
├── plotSnpDetectLinear.py   # Plotting companion (--kmer-analysis & --theo-dist flags)
├── testSnpDetectLinear.py   # Complete unit tests, error simulation & smoke tests
├── README.md                # Guide and documentation for the linear pipeline
├── img/                     # Default image output folder
├── plot_df/                 # Precomputed multiplicity dfs for fast plotting for k in {15, 23, 31, 39, 47, 55}
├── output/                  # Default putput folder
└── data/                    # Data folder
```

#!/usr/bin/env python3
"""
testSnpDetectLinear.py - Test Suite for snpDetectLinear.py.

Validates:
1. Reverse complement, canonical k-mer symmetry, and strand neutrality.
2. Synthetic single-nucleotide mutation (SNP) detection via flank hash join.
3. Multi-nucleotide / codon substitution (MNP) detection (mirroring Salmonella TetR TTT -> GGG).
4. Error robustness against 1% random Illumina substitution errors.
5. Notebook smoke test (test_snp_and_mnp_search).
6. Diagnostics and level reporting.
7. Plotting and spectrum export integration.
"""

import os
import random
import tempfile
import unittest
import snpDetectLinear
import plotSnpDetectLinear


def generate_random_genome(length: int = 1500, seed: int = 42) -> str:
    """Generate a random DNA sequence of specified length."""
    random.seed(seed)
    return "".join(random.choices("ACGT", k=length))


def simulate_illumina_reads(genome: str, read_length: int = 150, coverage: int = 30,
                            error_rate: float = 0.0, seed: int = 42) -> list:
    """
    Simulate Illumina reads from a genome:
    - Random uniform starting positions.
    - Double-stranded sequencing (50% forward, 50% reverse complement).
    - Random substitution sequencing errors at rate error_rate.
    """
    random.seed(seed)
    total_bases = len(genome) * coverage
    num_reads = total_bases // read_length
    reads = []
    L = len(genome)

    for _ in range(num_reads):
        pos = random.randint(0, L - read_length)
        frag = genome[pos : pos + read_length]

        if random.random() < 0.5:
            frag = snpDetectLinear.reverse_complement(frag)

        if error_rate > 0:
            frag_list = list(frag)
            for j in range(len(frag_list)):
                if random.random() < error_rate:
                    curr = frag_list[j]
                    choices = [b for b in "ACGT" if b != curr]
                    frag_list[j] = random.choice(choices)
            frag = "".join(frag_list)

        reads.append(frag)

    return reads


def write_reads_to_fasta(reads: list, filepath: str):
    """Write list of reads to a FASTA file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for idx, r in enumerate(reads):
            f.write(f">read_{idx}\n{r}\n")


def snp_synthetic_specifics(original_seq: str, variant_seq: str, k: int = 31, count: int = 30):
    """Build canonical sample-specific k-mers from two clean synthetic sequences."""
    def kmers(seq):
        c = {}
        for i in range(len(seq) - k + 1):
            km = snpDetectLinear.canonical_kmer(seq[i : i + k])
            c[km] = c.get(km, 0) + count
        return c

    ca, cb = kmers(original_seq), kmers(variant_seq)
    return (
        {km: c for km, c in ca.items() if km not in cb},
        {km: c for km, c in cb.items() if km not in ca}
    )


def test_snp_and_mnp_search():
    """Executable smoke tests from salmonella_notebook.ipynb."""
    left = "ACGTCAGTCCGAGGTCAGCT"
    right = "TGCACTAGGCTAGCTACGTA"
    for label, core in (("SNP", "TAT"), ("MNP", "AAA")):
        a, b = snp_synthetic_specifics(left + "TTT" + right,
                                       left + core + right, k=31)
        recs, diag = snpDetectLinear.find_substitutions_adaptive_corrected(
            a, b, k=31, max_flank=14, min_flank=12,
            max_comparisons=50_000, min_neighbor_support=1,
        )
        expected_a = "T" if label == "SNP" else "TTT"
        expected_b = "A" if label == "SNP" else "AAA"

        def found(r):
            fwd = (r["A"], r["B"])
            rev = (snpDetectLinear.reverse_complement(r["A"]), snpDetectLinear.reverse_complement(r["B"]))
            return (expected_a, expected_b) in (fwd, rev) and r["type"] == label

        assert any(found(r) for r in recs), f"Could not recover {label} {expected_a}->{expected_b}"
    return True


class TestSNPDetectLinear(unittest.TestCase):

    def test_notebook_smoke_test(self):
        """Verify the exact synthetic test from salmonella_notebook.ipynb passes."""
        self.assertTrue(test_snp_and_mnp_search())

    def test_reverse_complement_and_canonical(self):
        """Verify reverse complement and canonical k-mer logic."""
        seq = "ATGCGATC"
        self.assertEqual(snpDetectLinear.reverse_complement(seq), "GATCGCAT")
        self.assertEqual(snpDetectLinear.reverse_complement(snpDetectLinear.reverse_complement(seq)), seq)

        kmer1 = "AAACCC"
        kmer2 = "GGGTTT"
        self.assertEqual(snpDetectLinear.canonical_kmer(kmer1), kmer1)
        self.assertEqual(snpDetectLinear.canonical_kmer(kmer2), kmer1)
        self.assertEqual(snpDetectLinear.canonical_kmer(kmer1), snpDetectLinear.canonical_kmer(kmer2))

    def test_synthetic_clean_snp(self):
        """Verify clean 1-base SNP detection without errors."""
        genome_wt = generate_random_genome(1200, seed=101)
        mut_pos = 600
        orig_base = genome_wt[mut_pos]
        new_base = "G" if orig_base != "G" else "A"
        genome_var = genome_wt[:mut_pos] + new_base + genome_wt[mut_pos + 1 :]

        with tempfile.TemporaryDirectory() as tmpdir:
            file_wt = os.path.join(tmpdir, "wt.fasta")
            file_var = os.path.join(tmpdir, "var.fasta")

            reads_wt = simulate_illumina_reads(genome_wt, read_length=150, coverage=30, seed=1)
            reads_var = simulate_illumina_reads(genome_var, read_length=150, coverage=30, seed=2)

            write_reads_to_fasta(reads_wt, file_wt)
            write_reads_to_fasta(reads_var, file_var)

            variants = snpDetectLinear.detect_snps(file_wt, file_var, k=25, min_coverage=5, verbose=False)

            self.assertEqual(len(variants), 1, "Expected exactly 1 SNP to be found")
            var = variants[0]

            forward_match = (var["allele_a"] == orig_base and var["allele_b"] == new_base)
            rc_match = (var["allele_a"] == snpDetectLinear.reverse_complement(orig_base) and
                        var["allele_b"] == snpDetectLinear.reverse_complement(new_base))

            self.assertTrue(forward_match or rc_match,
                            f"Mismatch in called alleles: WT={var['allele_a']}, Var={var['allele_b']}")

    def test_synthetic_codon_substitution(self):
        """Verify 3-base codon substitution (mirroring Salmonella TetR TTT -> GGG)."""
        genome_wt = generate_random_genome(1200, seed=202)
        mut_pos = 500
        wt_triplet = "TTT"
        var_triplet = "GGG"
        genome_wt = genome_wt[:mut_pos] + wt_triplet + genome_wt[mut_pos + 3 :]
        genome_var = genome_wt[:mut_pos] + var_triplet + genome_wt[mut_pos + 3 :]

        with tempfile.TemporaryDirectory() as tmpdir:
            file_wt = os.path.join(tmpdir, "wt.fasta")
            file_var = os.path.join(tmpdir, "var.fasta")

            reads_wt = simulate_illumina_reads(genome_wt, read_length=150, coverage=35, seed=10)
            reads_var = simulate_illumina_reads(genome_var, read_length=150, coverage=35, seed=20)

            write_reads_to_fasta(reads_wt, file_wt)
            write_reads_to_fasta(reads_var, file_var)

            variants = snpDetectLinear.detect_snps(file_wt, file_var, k=25, min_coverage=5, verbose=False)

            self.assertEqual(len(variants), 1, "Expected exactly 1 codon mutation to be detected")
            var = variants[0]

            forward_match = (var["allele_a"] == wt_triplet and var["allele_b"] == var_triplet)
            rc_match = (var["allele_a"] == snpDetectLinear.reverse_complement(wt_triplet) and
                        var["allele_b"] == snpDetectLinear.reverse_complement(var_triplet))

            self.assertTrue(forward_match or rc_match,
                            f"Expected {wt_triplet} -> {var_triplet}, got {var['allele_a']} -> {var['allele_b']}")

    def test_error_robustness(self):
        """Verify robustness against 1% random Illumina substitution errors."""
        genome_wt = generate_random_genome(1500, seed=303)
        mut_pos = 750
        orig_base = genome_wt[mut_pos]
        new_base = "C" if orig_base != "C" else "T"
        genome_var = genome_wt[:mut_pos] + new_base + genome_wt[mut_pos + 1 :]

        with tempfile.TemporaryDirectory() as tmpdir:
            file_wt = os.path.join(tmpdir, "wt_err.fasta")
            file_var = os.path.join(tmpdir, "var_err.fasta")

            reads_wt = simulate_illumina_reads(genome_wt, read_length=150, coverage=40,
                                               error_rate=0.01, seed=30)
            reads_var = simulate_illumina_reads(genome_var, read_length=150, coverage=40,
                                                error_rate=0.01, seed=40)

            write_reads_to_fasta(reads_wt, file_wt)
            write_reads_to_fasta(reads_var, file_var)

            variants = snpDetectLinear.detect_snps(file_wt, file_var, k=27, verbose=False)

            self.assertEqual(len(variants), 1,
                             f"Expected exactly 1 true SNP and 0 false positives, got {len(variants)}")
            var = variants[0]
            forward_match = (var["allele_a"] == orig_base and var["allele_b"] == new_base)
            rc_match = (var["allele_a"] == snpDetectLinear.reverse_complement(orig_base) and
                        var["allele_b"] == snpDetectLinear.reverse_complement(new_base))
            self.assertTrue(forward_match or rc_match)

    def test_plotting_and_export(self):
        """Verify that plot_detection_results and export_spectrum_dataframe work seamlessly."""
        genome_wt = generate_random_genome(1200, seed=505)
        mut_pos = 600
        orig_base = genome_wt[mut_pos]
        new_base = "G" if orig_base != "G" else "A"
        genome_var = genome_wt[:mut_pos] + new_base + genome_wt[mut_pos + 1 :]

        with tempfile.TemporaryDirectory() as tmpdir:
            file_wt = os.path.join(tmpdir, "wt.fasta")
            file_var = os.path.join(tmpdir, "var.fasta")
            plot_png = os.path.join(tmpdir, "detection_plot.png")
            df_csv = os.path.join(tmpdir, "spectrum.csv")

            reads_wt = simulate_illumina_reads(genome_wt, read_length=150, coverage=30, seed=1)
            reads_var = simulate_illumina_reads(genome_var, read_length=150, coverage=30, seed=2)

            write_reads_to_fasta(reads_wt, file_wt)
            write_reads_to_fasta(reads_var, file_var)

            variants = snpDetectLinear.detect_snps(file_wt, file_var, k=25, min_coverage=5, verbose=False)
            self.assertEqual(len(variants), 1)

            # Test plot generation
            res_plot = plotSnpDetectLinear.plot_detection_results(
                variants=variants,
                output_path=plot_png,
                min_x=1,
                max_x=50
            )
            self.assertTrue(os.path.exists(plot_png), "Plot file should be created")
            self.assertGreater(os.path.getsize(plot_png), 1000, "Plot file should not be empty")

            # Test DataFrame export
            records = plotSnpDetectLinear.export_spectrum_dataframe(
                counts_a=variants.counts_a,
                counts_b=variants.counts_b,
                k=25,
                thresh_a=variants.thresh_a,
                thresh_b=variants.thresh_b,
                output_path=df_csv
            )
            self.assertTrue(os.path.exists(df_csv), "DataFrame CSV should be created")
            self.assertGreater(len(records), 0, "Spectrum records should be populated")

    def test_kmer_analysis_and_theo_dist_plots(self):
        """Verify that --kmer-analysis and --theo-dist plotting functions run and create valid figures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ma5_png = os.path.join(tmpdir, "ma5_test.png")
            overlap_png = os.path.join(tmpdir, "overlap_test.png")
            default_df = plotSnpDetectLinear.find_default_df()

            if os.path.exists(default_df):
                # 1. Test kmerAnalysis spectrum recreation
                res_ma5 = plotSnpDetectLinear.plot_kmer_analysis(
                    df_path_or_df=default_df,
                    output_path=ma5_png,
                    min_x=2,
                    max_x=300,
                    ma_width=5
                )
                self.assertTrue(os.path.exists(ma5_png), "kmerAnalysis plot should be generated")
                self.assertGreater(os.path.getsize(ma5_png), 50_000, "kmerAnalysis plot should not be empty")

                # 2. Test theoretical overlap band recreation
                res_theo = plotSnpDetectLinear.plot_theoretical_overlap_band(
                    df_path=default_df,
                    k=31,
                    max_c=260,
                    output_path=overlap_png
                )
                self.assertTrue(os.path.exists(overlap_png), "Theoretical overlap plot should be generated")
                self.assertGreater(os.path.getsize(overlap_png), 50_000, "Theoretical overlap plot should not be empty")


if __name__ == "__main__":
    unittest.main()

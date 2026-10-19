#!/usr/bin/env python3
"""
snpDetectLinear.py - Reference-Free Linear-Time SNP and Substitution Detection.

Based on the linear-time flank-hashing algorithm described in:
- salmonella_notebook.ipynb
- salmonella_jellyfish_SNP_CORRETTO (2).ipynb

Key Algorithmic Differences from snpDetect.py:
1. No De Bruijn graph construction, contig assembly, or sequence extension.
2. Direct hash join on shared left and right flanking anchors of decreasing length:
   P(a) ... U ... S(a) <-> P(a) ... V ... S(a)
3. Strictly linear-time O(|diff_A| + |diff_B|) join per flank level with a comparison budget.
4. Detects SNPs, consecutive MNPs (e.g. TTT -> GGG / AAA), and clustered substitutions.
5. Strand-neutral, preserving sample identities without swapping A and B.
6. Groups k-mer-level events into distinct biological mutation loci while supporting
   raw candidate inspection and theoretical collision diagnostics.
"""

import os
import sys
import argparse
import math
import time
from collections import Counter, defaultdict

try:
    from tqdm import tqdm
except ImportError:
    # Lightweight fallback when tqdm is not installed in the environment
    class DummyTqdm:
        def __init__(self, iterable=None, total=None, desc="", unit="", **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0

        def __iter__(self):
            if self.iterable is not None:
                for item in self.iterable:
                    yield item

        def update(self, n=1):
            self.n += n

        def refresh(self):
            pass

        def close(self):
            pass

    def tqdm(iterable=None, **kwargs):
        if iterable is not None:
            return DummyTqdm(iterable=iterable, **kwargs)
        return DummyTqdm(**kwargs)


DNA = "ACGT"
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(COMP)[::-1]


def canonical_kmer(kmer: str) -> str:
    """Return the lexicographically smaller of a k-mer and its reverse complement."""
    rc = reverse_complement(kmer)
    return kmer if kmer < rc else rc


# Aliases for notebook compatibility
snp_revcomp = reverse_complement
snp_canonical = canonical_kmer


def snp_orientations(kmer: str):
    """Return the set containing the k-mer and its reverse complement."""
    return {kmer, reverse_complement(kmer)}


def snp_neighbor_support(seq_a: str, seq_b: str, canon_a: set, canon_b: set):
    """
    Check pairs of neighboring k-mers in sample-specific sets.
    Returns (left_ok, right_ok) as integers (0 or 1).
    """
    left = any(
        canonical_kmer(base + seq_a[:-1]) in canon_a
        and canonical_kmer(base + seq_b[:-1]) in canon_b
        for base in "ACGT"
    )
    right = any(
        canonical_kmer(seq_a[1:] + base) in canon_a
        and canonical_kmer(seq_b[1:] + base) in canon_b
        for base in "ACGT"
    )
    return int(left), int(right)


def snp_random_hamming_cdf(n: int, d: int) -> float:
    """For iid uniform DNA, P(D <= d) for two random n-base strings."""
    if d < 0:
        return 0.0
    return sum(
        math.comb(n, j) * (0.75 ** j) * (0.25 ** (n - j))
        for j in range(min(d, n) + 1)
    )


def parse_fasta(filepath: str, max_reads: int = None):
    """
    Generator yielding sequences from a FASTA file in linear time O(N).
    Handles single-line and multi-line FASTA records.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        seq_buffer = []
        n_reads = 0
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped[0] == ">":
                if seq_buffer:
                    yield "".join(seq_buffer).upper()
                    n_reads += 1
                    if max_reads and n_reads >= max_reads:
                        return
                    seq_buffer = []
            else:
                seq_buffer.append(stripped)
        if seq_buffer:
            yield "".join(seq_buffer).upper()


def count_canonical_kmers(filepath: str, k: int = 31, max_reads: int = None,
                          desc: str = "Counting k-mers", show_progress: bool = True):
    """
    Count canonical k-mers from a FASTA file in linear time O(N).
    """
    counts = Counter()
    total_reads = 0
    total_bases = 0

    if max_reads:
        pbar = tqdm(total=max_reads, desc=desc, unit="reads",
                    disable=not show_progress, dynamic_ncols=True)
    else:
        file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        pbar = tqdm(total=file_size, desc=desc, unit="B", unit_scale=True,
                    unit_divisor=1024, disable=not show_progress, dynamic_ncols=True)

    bytes_read = 0
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        seq_buffer = []
        for line in f:
            line_bytes = len(line.encode("utf-8", errors="ignore"))
            bytes_read += line_bytes
            if not max_reads and show_progress:
                pbar.update(line_bytes)

            stripped = line.strip()
            if not stripped:
                continue
            if stripped[0] == ">":
                if seq_buffer:
                    seq = "".join(seq_buffer).upper()
                    total_reads += 1
                    total_bases += len(seq)
                    L = len(seq)
                    if L >= k:
                        for i in range(L - k + 1):
                            kmer = seq[i : i + k]
                            if "N" not in kmer:
                                counts[canonical_kmer(kmer)] += 1
                    if max_reads:
                        if show_progress and total_reads % 500 == 0:
                            pbar.update(500)
                        if total_reads >= max_reads:
                            seq_buffer = []
                            break
                    seq_buffer = []
            else:
                seq_buffer.append(stripped)

        if seq_buffer and (not max_reads or total_reads < max_reads):
            seq = "".join(seq_buffer).upper()
            total_reads += 1
            total_bases += len(seq)
            L = len(seq)
            if L >= k:
                for i in range(L - k + 1):
                    kmer = seq[i : i + k]
                    if "N" not in kmer:
                        counts[canonical_kmer(kmer)] += 1

    pbar.n = total_reads if max_reads else bytes_read
    pbar.refresh()
    pbar.close()
    if show_progress:
        print(f"    Loaded {total_reads:,} reads ({total_bases:,} bp, {len(counts):,} unique canonical k-mers)")
    return counts, total_reads, total_bases


def estimate_coverage_threshold(counts: Counter, user_min_count: int = None) -> int:
    """
    Determine the cutoff threshold t to distinguish genomic k-mers from error k-mers.
    If user_min_count is not provided, estimate it from the k-mer frequency spectrum trough.
    """
    if user_min_count is not None and user_min_count > 0:
        return user_min_count

    mult_freq = Counter(counts.values())
    trough_count = 5
    min_freq = float("inf")
    for c in range(2, min(25, max(mult_freq.keys(), default=2))):
        freq = mult_freq.get(c, 0)
        if freq < min_freq:
            min_freq = freq
            trough_count = c
        elif freq > min_freq * 1.5:
            break

    return max(4, trough_count)


def find_differential_kmers(counts_a: Counter, counts_b: Counter,
                            min_cov_a: int, min_cov_b: int,
                            max_error: int = 1):
    """
    Find solid k-mers specific to A (absent or near-zero in B)
    and solid k-mers specific to B (absent or near-zero in A).
    """
    diff_a = {
        km: c for km, c in counts_a.items()
        if c >= min_cov_a and counts_b.get(km, 0) <= max_error
    }
    diff_b = {
        km: c for km, c in counts_b.items()
        if c >= min_cov_b and counts_a.get(km, 0) <= max_error
    }
    return diff_a, diff_b


def find_substitutions_adaptive_corrected(
    diff_a: dict,
    diff_b: dict,
    k: int = 31,
    max_flank: int = None,
    min_flank: int = 1,
    max_comparisons: int = 250_000,
    min_neighbor_support: int = 0,
    context: int = 25,
    max_expected_random_pairs: float = None,
    verbose: bool = True
):
    """
    Assembly-free, reference-free linear-time substitution discovery.
    Direct implementation of the algorithm from salmonella_notebook.ipynb /
    salmonella_jellyfish_SNP_CORRETTO (2).ipynb.

    Compares sample-specific solid k-mers by hashing prefix and suffix of length 'flank':
        P(flank) ... U ... S(flank)  <-->  P(flank) ... V ... S(flank)

    Stops when the strict comparison budget or random-collision heuristic is reached.
    Returns (candidates, diagnostics).
    """
    if k <= 2 or k % 2 != 1:
        raise ValueError(f"k must be an odd integer >= 3, got k={k}")

    if max_flank is None:
        max_flank = (k - 1) // 2

    if not (1 <= min_flank <= max_flank and 2 * max_flank < k):
        raise ValueError(f"Flank sizes must satisfy 1 <= min_flank ({min_flank}) <= max_flank ({max_flank}) < k/2 ({k/2})")
    if max_comparisons < 1:
        raise ValueError("max_comparisons must be >= 1")
    if min_neighbor_support not in (0, 1, 2):
        raise ValueError("min_neighbor_support must be 0, 1 or 2")
    if context < 0:
        raise ValueError("context must be >= 0")

    # Validate dataset k-mers
    for dataset_name, dataset in (("A", diff_a), ("B", diff_b)):
        for seq, count in dataset.items():
            if len(seq) != k or any(b not in DNA for b in seq):
                raise ValueError(f"Dataset {dataset_name}: invalid {k}-mer: {seq}")
            if canonical_kmer(seq) != seq:
                raise ValueError(f"Dataset {dataset_name}: expected canonical k-mer: {seq}")
            if count < 1:
                raise ValueError(f"Dataset {dataset_name}: nonpositive count: {seq}")

    if not diff_a or not diff_b:
        return [], {
            "levels": [],
            "comparisons_executed": 0,
            "unique_candidates": 0,
            "stop_reason": "empty sample-specific set",
            "elapsed_seconds": 0.0
        }

    oriented_a = [(s, c) for km, c in diff_a.items() for s in snp_orientations(km)]
    oriented_b = [(s, c) for km, c in diff_b.items() for s in snp_orientations(km)]
    set_a, set_b = set(diff_a), set(diff_b)

    records_by_event = {}
    levels = []
    comparisons_executed = 0
    start_time = time.perf_counter()
    stop_reason = "minimum flank reached"

    for flank in range(max_flank, min_flank - 1, -1):
        t0 = time.perf_counter()
        variable_size = k - 2 * flank
        index = defaultdict(list)
        for s, count in oriented_a:
            index[(s[:flank], s[-flank:])].append((s, count))

        comparisons_at_level = sum(
            len(index.get((s[:flank], s[-flank:]), ()))
            for s, _ in oriented_b
        )

        iid_expected = len(oriented_a) * len(oriented_b) / (4 ** (2 * flank))
        level = {
            "flank": flank,
            "variable_bases": variable_size,
            "pairs_to_compare": comparisons_at_level,
            "iid_expected_flank_matches": iid_expected,
            "new_candidates": 0,
            "seconds": 0.0,
            "status": "pending",
        }

        if comparisons_executed + comparisons_at_level > max_comparisons:
            level["status"] = "STOP: comparison budget"
            levels.append(level)
            stop_reason = "comparison budget reached; shorter flanks NOT examined"
            break
        if (max_expected_random_pairs is not None and
                iid_expected > max_expected_random_pairs):
            level["status"] = "STOP: iid heuristic"
            levels.append(level)
            stop_reason = "iid random-collision heuristic; shorter flanks NOT examined"
            break

        before = len(records_by_event)
        for seq_b, count_b in oriented_b:
            key = (seq_b[:flank], seq_b[-flank:])
            for seq_a, count_a in index.get(key, ()):
                mismatch_positions = [i for i in range(flank, k - flank)
                                      if seq_a[i] != seq_b[i]]
                if not mismatch_positions:
                    continue
                f, last = mismatch_positions[0], mismatch_positions[-1]
                core_a, core_b = seq_a[f : last + 1], seq_b[f : last + 1]
                left_context = seq_a[max(0, f - context) : f]
                right_context = seq_a[last + 1 : last + 1 + context]

                event_fwd = (left_context, core_a, core_b, right_context)
                event_rc = (
                    reverse_complement(right_context),
                    reverse_complement(core_a),
                    reverse_complement(core_b),
                    reverse_complement(left_context),
                )
                event_key = min(event_fwd, event_rc)

                left_ok, right_ok = snp_neighbor_support(seq_a, seq_b, set_a, set_b)
                neighbor_sides = left_ok + right_ok
                if neighbor_sides < min_neighbor_support:
                    continue

                changes = len(mismatch_positions)
                span = last - f + 1
                mutation_type = (
                    "SNP" if changes == 1 and span == 1 else
                    "MNP" if changes == span else
                    "clustered substitutions"
                )

                iid_pairs_le_d = iid_expected * snp_random_hamming_cdf(variable_size, changes)
                candidate = {
                    "type": mutation_type,
                    "allele_a": event_key[1],
                    "allele_b": event_key[2],
                    "A": event_key[1],
                    "B": event_key[2],
                    "mutations": changes,
                    "span": span,
                    "flank_found": flank,
                    "count_A": count_a,
                    "count_B": count_b,
                    "depth_a": count_a,
                    "depth_b": count_b,
                    "left_flank": event_key[0],
                    "right_flank": event_key[3],
                    "left_context": event_key[0],
                    "right_context": event_key[3],
                    "left_neighbor": left_ok,
                    "right_neighbor": right_ok,
                    "supporting_sides": neighbor_sides,
                    "context_A": event_key[0] + "[" + event_key[1] + "]" + event_key[3],
                    "context_B": event_key[0] + "[" + event_key[2] + "]" + event_key[3],
                    "full_seq_a": seq_a,
                    "full_seq_b": seq_b,
                    "is_rc": (event_key == event_rc and event_fwd != event_rc),
                    "iid_expected_pairs_le_d": iid_pairs_le_d,
                    "_norm_key": event_key,
                }

                previous = records_by_event.get(event_key)
                if previous is None or (
                    neighbor_sides, flank, min(count_a, count_b)
                ) > (
                    previous["supporting_sides"], previous["flank_found"],
                    min(previous["count_A"], previous["count_B"])
                ):
                    records_by_event[event_key] = candidate

        comparisons_executed += comparisons_at_level
        level["new_candidates"] = len(records_by_event) - before
        level["seconds"] = round(time.perf_counter() - t0, 3)
        level["status"] = "searched"
        levels.append(level)
        if verbose:
            print(f"flank={flank:2d} | variable={variable_size:2d} | "
                  f"pairs={comparisons_at_level:,} | candidates={len(records_by_event):,} "
                  f"| {level['seconds']:.2f}s")

    candidates = list(records_by_event.values())
    candidates.sort(key=lambda r: (-r["supporting_sides"],
                                   -r["flank_found"],
                                   -min(r["count_A"], r["count_B"]),
                                   r["iid_expected_pairs_le_d"]))

    diagnostics = {
        "levels": levels,
        "comparisons_executed": comparisons_executed,
        "unique_candidates": len(candidates),
        "elapsed_seconds": round(time.perf_counter() - start_time, 3),
        "stop_reason": stop_reason,
        "warning": "SNP and MNP records are candidates: k-mer counts are not independent reads",
    }
    return candidates, diagnostics


def cluster_variant_candidates(candidates: list, min_overlap: int = 6) -> list:
    """
    Cluster redundant k-mer-level events resulting from shifted k-mer windows
    covering the same underlying genomic mutation locus into single representative variant calls.
    Keeps the highest-confidence, maximum-flank candidate for each distinct locus.
    """
    unique_variants = []

    for cand in candidates:
        seq_a = cand["left_context"] + cand["A"] + cand["right_context"]
        seq_a_rc = reverse_complement(seq_a)
        matched = False

        for var in unique_variants:
            v_seq = var["left_context"] + var["A"] + var["right_context"]
            v_seq_rc = reverse_complement(v_seq)

            # Check overlap around the mutation core
            anchor_len = min(min_overlap, len(cand["left_context"]), len(cand["right_context"]))
            if anchor_len > 0:
                anchor = cand["left_context"][-anchor_len:] + cand["A"] + cand["right_context"][:anchor_len]
                anchor_rc = reverse_complement(anchor)
                if anchor in v_seq or anchor_rc in v_seq or anchor in v_seq_rc:
                    matched = True
                    break

            # Check substring relationship
            if seq_a in v_seq or seq_a in v_seq_rc or v_seq in seq_a or v_seq_rc in seq_a:
                matched = True
                break

        if not matched:
            unique_variants.append(cand)

    return unique_variants


class VariantList(list):
    """Subclass of list carrying execution diagnostics for rich inspection."""
    def __init__(self, items=(), diagnostics=None):
        super().__init__(items)
        self.diagnostics = diagnostics or {}


def detect_snps(path_a: str, path_b: str, k: int = 31,
                max_reads: int = None, min_coverage: int = None,
                max_error_count: int = 1, verbose: bool = True,
                max_flank: int = None, min_flank: int = None,
                max_comparisons: int = 250_000,
                min_neighbor_support: int = 1,
                context: int = 25,
                max_expected_random_pairs: float = None,
                cluster_variants: bool = True,
                return_diagnostics: bool = False):
    """
    Main detection pipeline running in linear time O(N) using flank hash join.
    Compatible drop-in replacement for snpDetect.detect_snps.
    """
    t0 = time.time()
    t_count0 = time.time()
    counts_a, reads_a, bases_a = count_canonical_kmers(
        path_a, k=k, max_reads=max_reads, desc="Strain A (WT)", show_progress=verbose
    )
    counts_b, reads_b, bases_b = count_canonical_kmers(
        path_b, k=k, max_reads=max_reads, desc="Strain B (Var)", show_progress=verbose
    )
    kmer_counting_time = time.time() - t_count0

    thresh_a = estimate_coverage_threshold(counts_a, min_coverage)
    thresh_b = estimate_coverage_threshold(counts_b, min_coverage)

    if verbose:
        print(f"[*] Coverage cutoffs: Strain A >= {thresh_a}, Strain B >= {thresh_b}")

    diff_a, diff_b = find_differential_kmers(counts_a, counts_b, thresh_a, thresh_b, max_error_count)

    if verbose:
        print(f"[*] Candidate differential k-mers: {len(diff_a):,} in Strain A, {len(diff_b):,} in Strain B")

    if max_flank is None:
        max_flank = (k - 1) // 2

    if min_flank is None:
        # Default lower flank search bound: search down to flank 4 (variable window up to k-8)
        # to capture short indels and MNPs without excessive random collisions.
        min_flank = max(1, min(4, (k - 1) // 2 - 5))

    candidates, diagnostics = find_substitutions_adaptive_corrected(
        diff_a=diff_a,
        diff_b=diff_b,
        k=k,
        max_flank=max_flank,
        min_flank=min_flank,
        max_comparisons=max_comparisons,
        min_neighbor_support=min_neighbor_support,
        context=context,
        max_expected_random_pairs=max_expected_random_pairs,
        verbose=verbose
    )

    if cluster_variants:
        variants_out = cluster_variant_candidates(candidates)
    else:
        variants_out = candidates

    total_time = round(time.time() - t0, 2)
    diagnostics["total_time_seconds"] = total_time
    diagnostics["kmer_counting_seconds"] = round(kmer_counting_time, 2)

    result = VariantList(variants_out, diagnostics=diagnostics)
    result.counts_a = counts_a
    result.counts_b = counts_b
    result.thresh_a = thresh_a
    result.thresh_b = thresh_b
    result.diff_a = diff_a
    result.diff_b = diff_b
    result.k = k

    if verbose:
        print(f"[*] Found {len(result)} variant locus/loci ({len(candidates)} raw candidate records) "
              f"in {total_time:.2f}s total execution time.\n")

    if return_diagnostics:
        return result, diagnostics
    return result


def format_report(variants: list, name_a: str = "Strain A", name_b: str = "Strain B",
                  diagnostics: dict = None, top: int = None) -> str:
    """Format called variants into a human-readable report."""
    diag = diagnostics or getattr(variants, "diagnostics", None)
    if not variants and not diag:
        return "No confident variants detected between the two datasets."

    lines = []
    lines.append("=" * 80)
    lines.append(f"LINEAR-TIME VARIATION CALLING REPORT: {name_a} vs {name_b}")
    lines.append("=" * 80)

    total_found = len(variants)
    display_variants = variants[:top] if top is not None and top > 0 else variants

    if diag:
        lines.append(f"Pipeline Diagnostics:")
        if "total_time_seconds" in diag:
            timing_line = f"  Total execution time: {diag['total_time_seconds']:.2f}s"
            if "kmer_counting_seconds" in diag and "elapsed_seconds" in diag:
                timing_line += f" (k-mer counting: {diag['kmer_counting_seconds']:.2f}s, flank search: {diag['elapsed_seconds']:.2f}s)"
            lines.append(timing_line)
        elif "elapsed_seconds" in diag:
            lines.append(f"  Flank search time: {diag['elapsed_seconds']:.2f}s")
        lines.append(f"  Comparisons executed: {diag.get('comparisons_executed', 0):,}")
        lines.append(f"  Candidate records evaluated: {diag.get('unique_candidates', total_found):,}")
        lines.append(f"  Total variant loci detected: {total_found:,}")
        if top is not None and top < total_found:
            lines.append(f"  Showing top: {len(display_variants):,} variants (ranked by confidence)")
        lines.append(f"  Flank search stop reason: {diag.get('stop_reason', 'completed')}")
        lines.append("-" * 80)
    elif top is not None and top < total_found:
        lines.append(f"Showing top {len(display_variants):,} of {total_found:,} detected variants:")
        lines.append("-" * 80)

    if not display_variants:
        lines.append("\nNo confident variants detected between the two datasets.")
        return "\n".join(lines)

    for idx, var in enumerate(display_variants, start=1):
        lines.append(f"\n[Variant {idx}] Type: {var['type']}")
        depth_a = var.get("depth_a", var.get("count_A", 0))
        depth_b = var.get("depth_b", var.get("count_B", 0))
        allele_a = var.get("allele_a", var.get("A", ""))
        allele_b = var.get("allele_b", var.get("B", ""))
        lines.append(f"  {name_a} Allele: {allele_a} (observed k-mer depth: {depth_a}x)")
        lines.append(f"  {name_b} Allele: {allele_b} (observed k-mer depth: {depth_b}x)")

        if "flank_found" in var:
            lines.append(f"  Flank length: {var['flank_found']} bp | Supporting neighbor sides: {var.get('supporting_sides', 'N/A')}/2")

        l_ctx = var.get("left_flank", var.get("left_context", ""))
        r_ctx = var.get("right_flank", var.get("right_context", ""))
        l_disp = l_ctx[-25:] if len(l_ctx) > 25 else l_ctx
        r_disp = r_ctx[:25] if len(r_ctx) > 25 else r_ctx

        lines.append(f"\n  Genomic Context (Forward orientation):")
        lines.append(f"    {name_a}: ... {l_disp} [{allele_a}] {r_disp} ...")
        lines.append(f"    {name_b}: ... {l_disp} [{allele_b}] {r_disp} ...")

        # Complementary strand
        rc_allele_a = reverse_complement(allele_a)
        rc_allele_b = reverse_complement(allele_b)
        rc_l = reverse_complement(r_disp)
        rc_r = reverse_complement(l_disp)

        lines.append(f"  Complementary Strand:")
        lines.append(f"    {name_a}: ... {rc_l} [{rc_allele_a}] {rc_r} ...")
        lines.append(f"    {name_b}: ... {rc_l} [{rc_allele_b}] {rc_r} ...")
        lines.append("-" * 80)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Reference-free de novo SNP and variant detector from Illumina reads (Linear-Time Flank Hashing)."
    )
    parser.add_argument("readsA", help="Path to FASTA reads of Strain A (e.g. Wild-type)")
    parser.add_argument("readsB", help="Path to FASTA reads of Strain B (e.g. Variant)")
    parser.add_argument("-k", "--kmer", type=int, default=31, help="k-mer length (default: 31)")
    parser.add_argument("-m", "--max-reads", type=int, default=None,
                        help="Maximum reads to sample from each file (e.g. 250000 for ultra-fast mode)")
    parser.add_argument("-n", "--top", type=int, default=10,
                        help="Limit output report to the top N highest-confidence variants (e.g. -n 10)")
    parser.add_argument("-c", "--min-coverage", type=int, default=None,
                        help="Coverage cutoff threshold (default: auto-detected)")
    parser.add_argument("-e", "--max-error", type=int, default=1,
                        help="Maximum count allowed in opposite strain to tolerate sequencing noise (default: 1)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Optional file path to write results")
    parser.add_argument("--max-flank", type=int, default=None,
                        help="Maximum flank size to start adaptive hash join (default: (k-1)//2)")
    parser.add_argument("--min-flank", type=int, default=8,
                        help="Minimum flank size to search (default: 8) (set to None for auto)")
    parser.add_argument("--max-comparisons", type=int, default=250_000,
                        help="Strict total pairwise comparison budget (default: 250000)")
    parser.add_argument("--min-neighbor-support", type=int, default=1,
                        help="Minimum supporting neighbor k-mer sides (0, 1, or 2; default: 1)")
    parser.add_argument("--raw-candidates", action="store_true", default=False,
                        help="Output all raw unclustered candidate records across searched flank levels")

    args = parser.parse_args()

    variants = detect_snps(
        args.readsA,
        args.readsB,
        k=args.kmer,
        max_reads=args.max_reads,
        min_coverage=args.min_coverage,
        max_error_count=args.max_error,
        max_flank=args.max_flank,
        min_flank=args.min_flank,
        max_comparisons=args.max_comparisons,
        min_neighbor_support=args.min_neighbor_support,
        cluster_variants=not args.raw_candidates,
        verbose=True
    )

    report = format_report(variants, name_a="Wild-Type", name_b="Variant", top=args.top)
    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()

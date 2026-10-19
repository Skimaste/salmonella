#!/usr/bin/env python3
"""
plotSnpDetectLinear.py - Visualizations & Plotting for Linear-Time SNP & Variant Detection.

Companion plotting tool for snpDetectLinear.py.
Provides publication-quality plots:
1. Multiplicity spectra of Strain A and Strain B with error thresholds and peaks.
2. Adaptive flank-join progression (join size and candidates across flank levels).
3. Allele depth profiles for detected variants in both strains.
4. Exporting spectrum DataFrames compatible with downstream statistical scripts.
5. (--kmer-analysis flag): Recreates the kmerAnalysis.py multiplicity spectrum plot
   with centered moving average smoothing (matching img/kmer_spectrum_custom_ma5.png).
6. (--theo-dist flag): Recreates the plot_theo_dist.py theoretical mixture distribution
   and uncertainty overlap band plot (matching img/kmer_spectrum_overlap_band.png).

Only dependent on "-linear" files (snpDetectLinear) and products from them.
"""

import os
import sys
import argparse
from collections import Counter
import numpy as np
import pandas as pd
import snpDetectLinear


def find_default_df():
    """Locate kmer_spectrum_df.csv robustly regardless of current working directory."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_df", "kmer_spectrum_df.csv"),
        "./salmonella-project/plot_df/kmer_spectrum_df.csv",
        "./plot_df/kmer_spectrum_df.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "kmer_spectrum_df.csv"),
        "kmer_spectrum_df.csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return candidates[0]


def find_default_output(filename="snp_linear_detection.png"):
    """Locate output image path in img/ directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(script_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    return os.path.join(img_dir, filename)


def export_spectrum_dataframe(counts_a: Counter, counts_b: Counter = None, k: int = 31,
                              thresh_a: int = None, thresh_b: int = None,
                              output_path: str = None) -> list:
    """
    Export k-mer frequency spectra of both strains as a DataFrame CSV compatible with
    downstream distribution and theoretical modeling tools.
    """
    records = []

    def find_peak(cnts, thr):
        if not cnts:
            return 5
        mult_freq = Counter(cnts.values())
        best_p, best_f = (thr + 1) if thr else 5, 0
        for m, f in mult_freq.items():
            if m > (thr or 4) and f > best_f:
                best_f = f
                best_p = m
        return best_p

    peak_a = find_peak(counts_a, thresh_a)
    peak_b = find_peak(counts_b, thresh_b) if counts_b else None

    mult_freq_a = Counter(counts_a.values()) if counts_a else Counter()
    mult_freq_b = Counter(counts_b.values()) if counts_b else Counter()
    all_mults = sorted(set(mult_freq_a.keys()) | set(mult_freq_b.keys()))

    for m in all_mults:
        ca = mult_freq_a.get(m, 0)
        cb = mult_freq_b.get(m, 0)
        if ca > 0:
            records.append({
                "k": k,
                "multiplicity": m,
                "count": ca,
                "threshold": thresh_a,
                "genomic_peak": peak_a,
                "strain": "Strain A (WT)",
                "is_solid": thresh_a is None or m >= thresh_a
            })
        if cb > 0:
            records.append({
                "k": k,
                "multiplicity": m,
                "count": cb,
                "threshold": thresh_b,
                "genomic_peak": peak_b,
                "strain": "Strain B (Variant)",
                "is_solid": thresh_b is None or m >= thresh_b
            })

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        try:
            df = pd.DataFrame(records)
            df.to_csv(output_path, index=False)
            print(f"[*] Multiplicity spectrum DataFrame saved to: {output_path}")
        except Exception:
            import csv
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)
            print(f"[*] Multiplicity spectrum DataFrame saved to: {output_path} (via csv writer)")

    return records


def compute_moving_average(x: list, y: list, window: int):
    """
    Compute a centered moving average over multiplicity spectrum counts.
    - Centers the window so peak and valley positions remain preserved on the x-axis.
    - Evaluates across a dense multiplicity grid to handle missing multiplicities correctly.
    - Retains only positive smoothed values for log-scale plotting compatibility.
    """
    if not x or not y or window is None or window <= 1:
        return list(x), list(y)

    val_dict = dict(zip(x, y))
    min_x, max_x = min(x), max(x)
    dense_x = list(range(min_x, max_x + 1))
    dense_y = [float(val_dict.get(xi, 0)) for xi in dense_x]

    n = len(dense_y)
    half = window // 2
    smooth_x = []
    smooth_y = []

    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        sub = dense_y[left:right]
        avg = sum(sub) / len(sub) if sub else 0.0
        if avg > 0:
            smooth_x.append(dense_x[i])
            smooth_y.append(avg)

    return smooth_x, smooth_y


def plot_kmer_analysis(df_path_or_df=None, output_path: str = None, min_x: int = 2, max_x: int = 300,
                       ma_width: int = 5, show: bool = False):
    """
    Recreates the kmerAnalysis.py multiplicity spectrum plot (img/kmer_spectrum_custom_ma5.png).
    Supports centered moving average smoothing with customizable window width.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] Matplotlib not available; skipping plot generation.")
        return None

    if df_path_or_df is None:
        df_path_or_df = find_default_df()

    if isinstance(df_path_or_df, str):
        if not os.path.exists(df_path_or_df):
            print(f"[!] DataFrame file not found: {df_path_or_df}")
            return None
        sep = "\t" if df_path_or_df.endswith(".tsv") else ","
        df = pd.read_csv(df_path_or_df, sep=sep)
    else:
        df = df_path_or_df

    if output_path is None:
        output_path = find_default_output("kmer_spectrum_custom_ma5.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for k_val, group in df.groupby("k"):
        sub = group[(group["multiplicity"] >= min_x) & (group["multiplicity"] <= max_x)]
        x_vals = list(sub["multiplicity"])
        y_vals = list(sub["count"])
        if ma_width and ma_width > 1:
            x_vals, y_vals = compute_moving_average(x_vals, y_vals, ma_width)

        peak_val = sub["genomic_peak"].iloc[0] if "genomic_peak" in sub.columns else "N/A"
        lbl = f"k = {k_val} (peak @ {peak_val}x)"
        if ma_width and ma_width > 1:
            lbl += f" [MA {ma_width}]"
        ax.plot(x_vals, y_vals, marker=".", markersize=4, label=lbl)

    ax.set_yscale("log")
    ax.set_xlim(left=min_x, right=max_x)
    ax.set_xlabel("k-mer Multiplicity (Coverage Depth)")
    ax.set_ylabel("Number of Distinct k-mers (log scale)")
    title = f"k-mer Multiplicity Spectrum (x: {min_x} to {max_x})"
    if ma_width and ma_width > 1:
        title += f" [Moving Avg width={ma_width}]"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"[*] k-mer analysis spectrum plot saved to: {output_path}")

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def cont_poisson(x, mu):
    """
    Continuous evaluation of Poisson probability mass function using log-gamma:
    ln P(x; mu) = x * ln(mu) - mu - ln Gamma(x + 1)
    Avoids float overflow for x > 170.
    """
    try:
        from scipy.special import gammaln
        x = np.asarray(x, dtype=float)
        x_safe = np.maximum(x, 1e-12)
        log_p = x_safe * np.log(mu) - mu - gammaln(x_safe + 1.0)
        return np.exp(log_p)
    except ImportError:
        import math
        x = np.asarray(x, dtype=float)
        res = []
        for xi in x:
            xi_safe = max(float(xi), 1e-12)
            log_p = xi_safe * math.log(mu) - mu - math.lgamma(xi_safe + 1.0)
            res.append(math.exp(log_p))
        return np.array(res)


def compute_mixture_and_band(c_eval, lam_true, N_err_1, alpha_decay, G_true, G_rep,
                             lam_rep=None, G_rep3=None, lam_rep3=None):
    """
    Compute theoretical mixture counts and posterior odds ratio p(real) / p(error).
    Returns count_err, count_true, count_rep, count_rep3, count_total, ratio, and band values.
    """
    if lam_rep is None:
        lam_rep = 2.0 * lam_true
    if lam_rep3 is None:
        lam_rep3 = 3.0 * lam_true

    decay_rate = -np.log(max(alpha_decay, 1e-6))

    # 1. Error component count (exponential decay starting at c=1)
    count_err = N_err_1 * np.exp(-decay_rate * (c_eval - 1.0))
    count_err = np.maximum(count_err, 0.0)

    # 2. True single-copy genomic component count
    count_true = G_true * cont_poisson(c_eval, lam_true)

    # 3. 2-copy repeat genomic component count
    count_rep = G_rep * cont_poisson(c_eval, lam_rep)

    # 4. 3-copy repeat component count (if present)
    if G_rep3 is not None and G_rep3 > 0:
        count_rep3 = G_rep3 * cont_poisson(c_eval, lam_rep3)
    else:
        count_rep3 = np.zeros_like(c_eval)

    count_real = count_true + count_rep + count_rep3
    count_total = count_err + count_real

    # Posterior odds ratio p(real) / p(error)
    ratio = np.divide(count_real, count_err, out=np.full_like(count_real, 1e12), where=(count_err > 1e-50))

    # Search for band crossings in the valley between c=1 and peak
    search_mask = (c_eval >= 1.0) & (c_eval <= lam_true)
    c_search = c_eval[search_mask]
    ratio_search = ratio[search_mask]

    idx_001 = np.where(ratio_search >= 0.01)[0]
    idx_100 = np.where(ratio_search >= 1.0)[0]
    idx_10000 = np.where(ratio_search >= 100.0)[0]

    c_low = float(c_search[idx_001[0]]) if len(idx_001) > 0 else float(c_eval[0])
    c_mid = float(c_search[idx_100[0]]) if len(idx_100) > 0 else float(c_eval[len(c_eval)//2])
    c_high = float(c_search[idx_10000[0]]) if len(idx_10000) > 0 else float(c_eval[-1])

    return {
        'count_err': count_err,
        'count_true': count_true,
        'count_rep': count_rep,
        'count_rep3': count_rep3,
        'count_total': count_total,
        'count_real': count_real,
        'ratio': ratio,
        'c_low': c_low,
        'c_mid': c_mid,
        'c_high': c_high,
        'lam_true': lam_true,
        'lam_rep': lam_rep,
        'lam_rep3': lam_rep3,
        'G_true': G_true,
        'G_rep': G_rep,
        'G_rep3': G_rep3,
        'decay_rate': decay_rate,
        'alpha': alpha_decay
    }


def print_comparison_across_k(df_path=None):
    """
    Evaluate and print a structured comparison table of the overlap band [c_low, c_high]
    across all candidate k values (15, 23, 31, 39, 47, 55).
    """
    if df_path is None:
        df_path = find_default_df()
    if not os.path.exists(df_path):
        return

    df = pd.read_csv(df_path)
    k_vals = sorted(df['k'].unique())
    x_eval = np.linspace(1.0, 50.0, 5000)

    print("\n" + "=" * 96)
    print("OVERLAP BAND & DECISION BOUNDARY COMPARISON ACROSS ALL CANDIDATE k")
    print("=" * 96)
    print(f"{'k':<5} {'Peak (λ)':<10} {'alpha (c2/c1)':<15} {'c_low (0.01)':<15} {'Cutoff t* (1.0)':<18} {'c_high (100)':<15} {'Band Width':<12}")
    print("-" * 96)

    for k in k_vals:
        sub = df[(df['k'] == k) & (df['multiplicity'] <= 260)].sort_values('multiplicity')
        c = sub['multiplicity'].values
        counts = sub['count'].values

        if 'genomic_peak' in sub.columns and not pd.isna(sub['genomic_peak'].iloc[0]):
            peak_cov = float(sub['genomic_peak'].iloc[0])
        else:
            p_idx = np.argmax(counts[c >= 20])
            peak_cov = float(c[c >= 20][p_idx])

        alpha = float(counts[1] / max(counts[0], 1))
        decay_rate = -np.log(alpha)

        N_err_1 = float(counts[0])
        G_true = float(counts[(c >= peak_cov * 0.4) & (c <= peak_cov * 1.5)].sum())

        p_err = N_err_1 * np.exp(-decay_rate * (x_eval - 1.0))
        p_real = G_true * cont_poisson(x_eval, peak_cov)

        ratio = np.divide(p_real, p_err, out=np.full_like(p_real, 1e12), where=(p_err > 1e-50))

        search_mask = (x_eval >= 1.0) & (x_eval <= peak_cov)
        c_s = x_eval[search_mask]
        r_s = ratio[search_mask]

        idx_001 = np.where(r_s >= 0.01)[0]
        idx_100 = np.where(r_s >= 1.0)[0]
        idx_10000 = np.where(r_s >= 100.0)[0]

        c_low = float(c_s[idx_001[0]]) if len(idx_001) > 0 else 0.0
        c_mid = float(c_s[idx_100[0]]) if len(idx_100) > 0 else 0.0
        c_high = float(c_s[idx_10000[0]]) if len(idx_10000) > 0 else 0.0
        width = c_high - c_low

        print(f"{k:<5} {peak_cov:<10.1f} {alpha:<15.4f} {c_low:<15.2f} {c_mid:<18.2f} {c_high:<15.2f} {width:<12.2f}")

    print("=" * 96 + "\n")


def plot_all_k_comparison(df_path=None, output_path=None):
    """
    Generate an overlay comparison figure for all candidate k values (15, 23, 31, 39, 47, 55),
    matching the user's inspiration spectrum plot with log-scale y-axis and moving averages.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if df_path is None:
        df_path = find_default_df()
    if not os.path.exists(df_path):
        return None

    df = pd.read_csv(df_path)
    k_vals = sorted(df['k'].unique())

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, k in enumerate(k_vals):
        sub = df[(df['k'] == k) & (df['multiplicity'] <= 260) & (df['multiplicity'] >= 2)].sort_values('multiplicity')
        c = sub['multiplicity'].values
        counts = sub['count'].values

        if 'genomic_peak' in sub.columns and not pd.isna(sub['genomic_peak'].iloc[0]):
            peak = int(sub['genomic_peak'].iloc[0])
        else:
            p_idx = np.argmax(counts[c >= 20])
            peak = int(c[c >= 20][p_idx])

        color = colors[i % len(colors)]
        ax.plot(c, counts, marker='.', markersize=3.5, linewidth=1.2, color=color,
                label=f'k = {k} (peak @ {peak}x)')

    ax.set_yscale('log')
    ax.set_xlim(2, 260)
    ax.set_ylim(0.8, 1e6)
    ax.set_xlabel('k-mer Multiplicity (Coverage Depth)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Distinct k-mers (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Empirical k-mer Multiplicity Spectra across Candidate k (log scale)', fontsize=13, fontweight='bold')
    ax.grid(True, which='both', alpha=0.25, linestyle='--')
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.92)

    plt.tight_layout()
    if output_path is None:
        output_path = find_default_output("kmer_spectrum_all_k_log.png")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"[*] All-k spectrum comparison saved to: {output_path}")
    plt.close(fig)
    return output_path


def plot_theoretical_overlap_band(df_path=None, k: int = 31, max_c: int = 260,
                                  output_path: str = None, show: bool = False,
                                  demo_mode: bool = False, all_k: bool = False):
    """
    Generate 2-panel figure with log-scale y-axis and empirical data overlay:
    - Top panel: Number of Distinct k-mers on log scale (1 to 10^8), showing:
        * Actual data curve & points from kmer_spectrum_df.csv
        * Theoretical Mixture Model (Total)
        * Theoretical Error Component (exponential decay)
        * Theoretical True Genomic Component (single-copy Poisson)
        * Theoretical Repeat Components (2-copy and 3-copy Poissons)
        * Overlap Band [c_low, c_high] shaded in gold (0.01 < p(real)/p(error) < 100)
        * Decision Cutoff t* line where ratio = 1.0
    - Bottom panel: Posterior Odds Ratio p(real)/p(error) on log scale (10^-4 to 10^4).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] Matplotlib not available; skipping plot generation.")
        return None

    if df_path is None:
        df_path = find_default_df()

    c_eval = np.linspace(1.0, max_c, 5000)

    if demo_mode or not os.path.exists(df_path):
        # Synthetic reference demonstration matching inspiration figure
        title_prefix = f"Synthetic Reference Model (Peak ~ 41x)"
        lam_t = 41.0
        lam_r = 82.0
        lam_r3 = 123.0
        N_err_1 = 1.5e7
        alpha_decay = 0.05
        G_true = 3.5e6
        G_rep = 2.5e4
        G_rep3 = 2e3

        c_data = np.arange(1, max_c + 1)
        mix = compute_mixture_and_band(c_eval, lam_t, N_err_1, alpha_decay, G_true, G_rep, lam_r, G_rep3, lam_r3)
        count_data = compute_mixture_and_band(c_data, lam_t, N_err_1, alpha_decay, G_true, G_rep, lam_r, G_rep3, lam_r3)['count_total']
        data_lbl = "Synthetic Inspiration Dataset"
    else:
        # Real Salmonella enterica dataset
        df = pd.read_csv(df_path)
        sub = df[(df['k'] == k) & (df['multiplicity'] <= max_c)].sort_values('multiplicity')
        c_data = sub['multiplicity'].values
        count_data = sub['count'].values

        title_prefix = f"Salmonella enterica Outbreak Dataset (k = {k})"
        data_lbl = f"Actual Data (Salmonella enterica, k={k})"

        # Identify genomic peak (lambda_true)
        if 'genomic_peak' in sub.columns and not pd.isna(sub['genomic_peak'].iloc[0]):
            lam_t = float(sub['genomic_peak'].iloc[0])
        else:
            p_idx = np.argmax(count_data[c_data >= 20])
            lam_t = float(c_data[c_data >= 20][p_idx])

        lam_r = 2.0 * lam_t
        lam_r3 = 3.0 * lam_t

        # Single-copy genome size (~4.87M distinct k-mers)
        G_true = float(count_data[(c_data >= lam_t * 0.4) & (c_data <= lam_t * 1.5)].sum())
        # 2-copy repeats (~29k distinct k-mers)
        rep_mask = (c_data >= int(lam_r * 0.8)) & (c_data <= int(lam_r * 1.2))
        G_rep = float(count_data[rep_mask].sum()) if np.any(rep_mask) else 0.01 * G_true
        # 3-copy repeats (~2k distinct k-mers)
        rep3_mask = (c_data >= int(lam_r3 * 0.85)) & (c_data <= int(lam_r3 * 1.15))
        G_rep3 = float(count_data[rep3_mask].sum()) if np.any(rep3_mask) else 0.001 * G_true

        N_err_1 = float(count_data[0])
        alpha_decay = float(count_data[1] / max(count_data[0], 1))

        mix = compute_mixture_and_band(c_eval, lam_t, N_err_1, alpha_decay, G_true, G_rep, lam_r, G_rep3, lam_r3)

    c_low = mix['c_low']
    c_mid = mix['c_mid']
    c_high = mix['c_high']

    print("\n" + "=" * 78)
    print(f"K-MER MULTIPLICITY SPECTRUM & OVERLAP BAND: {title_prefix}")
    print("=" * 78)
    print(f"Single-Copy Genomic Peak (lambda_true) : {lam_t:.1f}x coverage")
    print(f"2-Copy Repeat Peak (2 * lambda_true)   : {lam_r:.1f}x coverage")
    print(f"Error Decay Factor (alpha = c2 / c1)  : {alpha_decay:.4f} (decay rate = {mix['decay_rate']:.3f})")
    print(f"Single-Copy Distinct k-mers (G_true)   : {G_true:,.0f} bp")
    print(f"2-Copy Repeat Distinct k-mers (G_rep)  : {G_rep:,.0f} bp")
    print("-" * 78)
    print(f"Overlap Band [0.01 < p(real)/p(error) < 100]: [{c_low:.2f}x, {c_high:.2f}x]")
    print(f"  - Left Bound  (ratio = 0.01, 99% Error): {c_low:.2f}x")
    print(f"  - Bayes Cutoff (ratio = 1.0,  50% Real) : {c_mid:.2f}x (Optimal Threshold t*)")
    print(f"  - Right Bound (ratio = 100,  99% Real) : {c_high:.2f}x")
    print(f"  - Uncertainty Band Width                : {c_high - c_low:.2f}x coverage units")
    print("=" * 78 + "\n")

    # -------------------------------------------------------------------------
    # Plotting: 2-Panel Figure with Y-Axis Log Scale and Data Overlay
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[2.6, 1], sharex=True)

    # 1. Top Panel: Number of Distinct k-mers on Log Scale
    ax1.plot(c_data, count_data, color='#0284c7', marker='o', markersize=3.5, linewidth=1.2,
             alpha=0.75, label=data_lbl)

    # Theoretical Curves
    ax1.plot(c_eval, mix['count_total'], color='#0f172a', linewidth=2.5,
             label='Theoretical Mixture Model (Total)')
    ax1.plot(c_eval, mix['count_err'], color='#dc2626', linestyle='--', linewidth=1.8,
             label='Theoretical Error Component')
    ax1.plot(c_eval, mix['count_true'], color='#16a34a', linestyle='--', linewidth=1.8,
             label=f'Theoretical True Genomic Component (λ={lam_t:.0f}x)')
    ax1.plot(c_eval, mix['count_rep'], color='#9333ea', linestyle='--', linewidth=1.8,
             label=f'Theoretical 2-Copy Repeat Component (2λ={lam_r:.0f}x)')
    if max_c >= 230 and mix['G_rep3'] > 0:
        ax1.plot(c_eval, mix['count_rep3'], color='#c026d3', linestyle=':', linewidth=1.5,
                 label=f'Theoretical 3-Copy Repeat Component (3λ={lam_r3:.0f}x)')

    # Overlap Band Shading
    band_label = f'Overlap Band [{c_low:.1f}x, {c_high:.1f}x] (0.01 < p(real)/p(error) < 100)'
    ax1.axvspan(c_low, c_high, facecolor='#fef08a', alpha=0.55, edgecolor='#eab308',
                linestyle='--', linewidth=1.2, label=band_label)
    ax1.axvline(c_mid, color='#ea580c', linestyle=':', linewidth=2.4,
                label=f'Decision Cutoff (t* = {c_mid:.1f}x, ratio=1.0)')

    # Annotations & Arrows
    err_y = count_data[1] if len(count_data) > 1 else N_err_1 * alpha_decay
    ax1.annotate('Error k-mers\n(sequencing errors)',
                 xy=(2, err_y), xytext=(15, 1.5e7),
                 arrowprops=dict(facecolor='black', width=1.8, headwidth=6.0, headlength=6.0),
                 fontsize=10.5, fontweight='bold')

    peak_count = count_data[c_data == int(lam_t)][0] if np.any(c_data == int(lam_t)) else max(count_data[c_data >= 20])
    ax1.annotate(f'True k-mers\n(single-copy genome, peak={lam_t:.0f}x)',
                 xy=(lam_t, peak_count), xytext=(lam_t - 22, 1.2e6),
                 arrowprops=dict(facecolor='black', width=1.8, headwidth=6.0, headlength=6.0),
                 fontsize=10.5, fontweight='bold')

    rep_y = count_data[c_data == int(lam_r)][0] if np.any(c_data == int(lam_r)) else 900
    ax1.annotate(f'Repeat k-mers\n(2-copy repeats, peak={lam_r:.0f}x)',
                 xy=(lam_r, rep_y), xytext=(lam_r + 12, 1e4),
                 arrowprops=dict(facecolor='black', width=1.8, headwidth=6.0, headlength=6.0),
                 fontsize=10.5, fontweight='bold')

    ax1.annotate(f'Overlap Band\n[{c_low:.1f}x - {c_high:.1f}x]\nUncertainty Zone',
                 xy=(c_mid, 10), xytext=(c_mid + 14, 200),
                 arrowprops=dict(arrowstyle='->', lw=1.6, color='#b45309'),
                 fontsize=9.5, fontweight='bold', color='#92400e',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef08a', edgecolor='#eab308', alpha=0.9))

    ax1.set_yscale('log')
    ax1.set_ylim(0.8, 1e8)
    ax1.set_xlim(1, max_c)
    ax1.set_ylabel('Number of Distinct k-mers (log scale)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Theoretical k-mer Multiplicity Spectrum Overlaid with Empirical Data (k = {k})', fontsize=13, fontweight='bold')
    ax1.grid(True, which='both', alpha=0.25, linestyle='--')
    ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.94)

    # 2. Bottom Panel: Posterior Odds Ratio p(real) / p(error) on Log Scale
    ratio_clipped = np.clip(mix['ratio'], 1e-4, 1e4)
    ax2.plot(c_eval, ratio_clipped, color='#7e22ce', linewidth=2.2,
             label='Posterior Odds: p(real) / p(error)')

    ax2.axhline(0.01, color='#dc2626', linestyle='--', linewidth=1.5, alpha=0.85,
                label='Ratio = 0.01 (99% Error / 1% Real)')
    ax2.axhline(1.0, color='#ea580c', linestyle=':', linewidth=2.2, alpha=0.95,
                label='Ratio = 1.0 (50% Real / 50% Error, Bayes t*)')
    ax2.axhline(100.0, color='#16a34a', linestyle='--', linewidth=1.5, alpha=0.85,
                label='Ratio = 100 (99% Real / 1% Error)')

    ax2.axvspan(c_low, c_high, facecolor='#fef08a', alpha=0.55, edgecolor='#eab308', linestyle='--')
    ax2.axvline(c_mid, color='#ea580c', linestyle=':', linewidth=2.2)

    ax2.text(2, 0.0003, 'Error-Dominated Zone\n(>99% Error)', fontsize=8.5, color='#dc2626', fontweight='bold')
    ax2.text(c_high + 4, 300, 'Real Genomic Zone\n(>99% Real)', fontsize=8.5, color='#16a34a', fontweight='bold')

    ax2.set_yscale('log')
    ax2.set_ylim(1e-4, 1e4)
    ax2.set_xlim(1, max_c)
    ax2.set_xlabel('k-mer Multiplicity (Coverage Depth)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('p(real) / p(error)\n(log scale)', fontsize=10, fontweight='bold')
    ax2.grid(True, which='both', alpha=0.25, linestyle='--')
    ax2.legend(loc='lower right', fontsize=8.5, framealpha=0.94)

    plt.tight_layout()

    if output_path is None:
        output_path = find_default_output("kmer_spectrum_overlap_band.png")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"[*] Overlap band plot saved to: {output_path}")

    if not demo_mode and os.path.exists(df_path):
        print_comparison_across_k(df_path)

    if all_k or not demo_mode:
        all_k_out = find_default_output("kmer_spectrum_all_k_log.png")
        plot_all_k_comparison(df_path=df_path, output_path=all_k_out)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_detection_results(
    variants: list,
    counts_a: Counter = None,
    counts_b: Counter = None,
    thresh_a: int = None,
    thresh_b: int = None,
    diagnostics: dict = None,
    output_path: str = None,
    name_a: str = "Strain A (WT)",
    name_b: str = "Strain B (Variant)",
    min_x: int = 1,
    max_x: int = 180,
    show: bool = False
):
    """
    Publication-quality visualization of linear SNP detection results:
    1. Multiplicity spectra of Strain A and Strain B with error thresholds and peaks.
    2. Adaptive flank search progression (join size and candidates across flank levels).
    3. Allele depth profiles for detected variants in both strains.
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] Matplotlib not available; skipping plot generation.")
        return None

    counts_a = counts_a if counts_a is not None else getattr(variants, "counts_a", None)
    counts_b = counts_b if counts_b is not None else getattr(variants, "counts_b", None)
    thresh_a = thresh_a if thresh_a is not None else getattr(variants, "thresh_a", None)
    thresh_b = thresh_b if thresh_b is not None else getattr(variants, "thresh_b", None)
    diagnostics = diagnostics if diagnostics is not None else getattr(variants, "diagnostics", {})

    if output_path is None:
        output_path = find_default_output("snp_linear_detection.png")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])

    # -------------------------------------------------------------
    # Panel 1 (Top Left): Multiplicity Spectrum Overlay
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    if counts_a and counts_b:
        mult_a = Counter(counts_a.values())
        mult_b = Counter(counts_b.values())

        x_a = [c for c in range(min_x, max_x + 1) if c in mult_a]
        y_a = [mult_a[c] for c in x_a]
        x_b = [c for c in range(min_x, max_x + 1) if c in mult_b]
        y_b = [mult_b[c] for c in x_b]

        ax1.plot(x_a, y_a, color="#0284c7", linewidth=2.0, label=f"{name_a}")
        ax1.plot(x_b, y_b, color="#f97316", linewidth=2.0, linestyle="--", label=f"{name_b}")

        if thresh_a:
            ax1.axvline(thresh_a, color="#0284c7", linestyle=":", linewidth=1.5,
                        label=f"{name_a} Cutoff $t_A$ = {thresh_a}")
        if thresh_b and thresh_b != thresh_a:
            ax1.axvline(thresh_b, color="#f97316", linestyle=":", linewidth=1.5,
                        label=f"{name_b} Cutoff $t_B$ = {thresh_b}")
        elif thresh_b and thresh_b == thresh_a:
            ax1.axvline(thresh_a, color="#64748b", linestyle=":", linewidth=1.5,
                        label=f"Solid Cutoff $t$ = {thresh_a}")

        diff_a_len = len(getattr(variants, "diff_a", {}))
        diff_b_len = len(getattr(variants, "diff_b", {}))
        if diff_a_len or diff_b_len:
            ax1.text(0.95, 0.65, f"Differential Solid k-mers:\n  {name_a}: {diff_a_len:,}\n  {name_b}: {diff_b_len:,}",
                     transform=ax1.transAxes, fontsize=9, verticalalignment="top", horizontalalignment="right",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9))

        ax1.set_yscale("log")
        ax1.set_xlim(min_x, max_x)
        ax1.set_xlabel("k-mer Multiplicity (Coverage Depth)")
        ax1.set_ylabel("Distinct k-mers (log scale)")
        ax1.set_title("A. k-mer Multiplicity Spectrum & Cutoff", fontweight="bold", loc="left")
        ax1.grid(True, which="both", alpha=0.25, linestyle="--")
        ax1.legend(loc="upper right", framealpha=0.9)
    else:
        ax1.text(0.5, 0.5, "Spectrum counts not provided", ha="center", va="center")
        ax1.set_title("A. k-mer Multiplicity Spectrum", fontweight="bold", loc="left")

    # -------------------------------------------------------------
    # Panel 2 (Top Right): Flank-Hashing Join Progression
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    levels = diagnostics.get("levels", [])
    if levels:
        flanks = [lvl["flank"] for lvl in levels]
        pairs = [lvl["pairs_to_compare"] for lvl in levels]
        new_cands = [lvl["new_candidates"] for lvl in levels]

        color_pairs = "#2563eb"
        color_cands = "#16a34a"

        ax2.plot(flanks, pairs, marker="o", color=color_pairs, linewidth=2, label="Candidate Pairs Compared")
        ax2.set_xlabel("Flank Length $a$ (bp, descending search)")
        ax2.set_ylabel("Number of Candidate Pairs", color=color_pairs)
        ax2.tick_params(axis="y", labelcolor=color_pairs)
        ax2.invert_xaxis()

        ax2_right = ax2.twinx()
        ax2_right.bar(flanks, new_cands, width=0.4, color=color_cands, alpha=0.6, label="New Candidates Discovered")
        ax2_right.set_ylabel("Candidates Discovered", color=color_cands)
        ax2_right.tick_params(axis="y", labelcolor=color_cands)

        max_comp = diagnostics.get("comparisons_executed", sum(pairs))
        stop_reason = diagnostics.get("stop_reason", "completed")
        ax2.text(0.05, 0.85, f"Comparisons Executed: {max_comp:,}\nStop: {stop_reason}",
                 transform=ax2.transAxes, fontsize=9, verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9))

        ax2.set_title("B. Linear Flank-Hashing Search Progression", fontweight="bold", loc="left")
        ax2.grid(True, alpha=0.25, linestyle="--")
    else:
        ax2.text(0.5, 0.5, "Diagnostics levels not available", ha="center", va="center")
        ax2.set_title("B. Linear Flank-Hashing Progression", fontweight="bold", loc="left")

    # -------------------------------------------------------------
    # Panel 3 (Bottom): Detected Variants & Read Depth Profile
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, :])
    if variants:
        n_show = min(len(variants), 6)
        top_vars = variants[:n_show]
        x_indices = np.arange(len(top_vars))
        width = 0.35

        depths_a = [v.get("depth_a", v.get("count_A", 0)) for v in top_vars]
        depths_b = [v.get("depth_b", v.get("count_B", 0)) for v in top_vars]
        labels = [
            f"[Variant {i+1}: {v.get('type', 'VAR')}]\n"
            f"{name_a}: {v.get('allele_a', v.get('A', ''))} ({v.get('depth_a', v.get('count_A', 0))}x)\n"
            f"{name_b}: {v.get('allele_b', v.get('B', ''))} ({v.get('depth_b', v.get('count_B', 0))}x)\n"
            f"Flank: {v.get('flank_found', 'N/A')} bp | Support: {v.get('supporting_sides', 'N/A')}/2"
            for i, v in enumerate(top_vars)
        ]

        rects1 = ax3.bar(x_indices - width/2, depths_a, width, label=f"{name_a} Depth", color="#0284c7")
        rects2 = ax3.bar(x_indices + width/2, depths_b, width, label=f"{name_b} Depth", color="#f97316")

        for rect in rects1:
            h = rect.get_height()
            if h > 0:
                ax3.annotate(f"{h:.0f}x", xy=(rect.get_x() + rect.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        for rect in rects2:
            h = rect.get_height()
            if h > 0:
                ax3.annotate(f"{h:.0f}x", xy=(rect.get_x() + rect.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        ax3.set_xticks(x_indices)
        ax3.set_xticklabels(labels, fontsize=9)
        ax3.set_ylabel("Observed k-mer Depth (x)")
        ax3.set_title("C. Discovered Variant Alleles & Sequencing Depth Verification", fontweight="bold", loc="left")
        ax3.legend(loc="upper right", framealpha=0.9)
        ax3.grid(True, axis="y", alpha=0.25, linestyle="--")
    else:
        ax3.text(0.5, 0.5, "No confident variants detected", ha="center", va="center")
        ax3.set_title("C. Discovered Variant Alleles", fontweight="bold", loc="left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"[*] Linear SNP detection plot saved to: {output_path}")

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualizations and plotting tool for linear-time reference-free SNP detection."
    )
    parser.add_argument("readsA", nargs="?", default=None,
                        help="Path to FASTA reads of Strain A (e.g. Wild-type)")
    parser.add_argument("readsB", nargs="?", default=None,
                        help="Path to FASTA reads of Strain B (e.g. Variant)")
    parser.add_argument("-k", "--kmer", type=int, default=31, help="k-mer length (default: 31)")
    parser.add_argument("-m", "--max-reads", type=int, default=None,
                        help="Maximum reads to sample (default: all)")
    parser.add_argument("-c", "--min-coverage", type=int, default=None,
                        help="Coverage cutoff threshold (default: auto-detected)")
    parser.add_argument("-e", "--max-error", type=int, default=1,
                        help="Max occurrences in opposite strain (default: 1)")
    parser.add_argument("-o", "--output-plot", "--output", type=str, default=None,
                        help="Path to save the generated plot")
    parser.add_argument("--output-df", type=str, default=None,
                        help="Optional path to save spectrum DataFrame CSV")
    parser.add_argument("--df", "--input-df", "--plot-from-df", type=str, default=None,
                        help="Path to existing spectrum DataFrame CSV (e.g. plot_df/kmer_spectrum_df.csv)")

    # Specialized plot mode flags
    parser.add_argument("--kmer-analysis", action="store_true", default=False,
                        help="Recreate kmerAnalysis.py spectrum plot (matching img/kmer_spectrum_custom_ma5.png)")
    parser.add_argument("--theo-dist", action="store_true", default=False,
                        help="Recreate plot_theo_dist.py theoretical mixture & overlap band plot (matching img/kmer_spectrum_overlap_band.png)")

    # Plotting parameter controls
    parser.add_argument("--moving-average", "--smooth", "--ma-width", type=int, default=None,
                        help="Moving average smoothing window (default: 5 for --kmer-analysis)")
    parser.add_argument("--min-x", type=int, default=None,
                        help="Min x-axis multiplicity (default: 2 for --kmer-analysis, 1 for SNP detection)")
    parser.add_argument("--max-x", type=int, default=None,
                        help="Max x-axis multiplicity (default: 300 for --kmer-analysis, 180 for SNP detection)")
    parser.add_argument("--max-c", type=int, default=260,
                        help="Max coverage on x-axis for theoretical distribution (default: 260)")
    parser.add_argument("--all-k", action="store_true", default=False,
                        help="Also generate multi-k comparison overlay plot across all k values (with --theo-dist)")
    parser.add_argument("--demo", action="store_true", default=False,
                        help="Generate the theoretical reference model matching the inspiration picture (peak ~ 41x)")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Show interactive window")

    args = parser.parse_args()

    # Mode 1: Recreate kmerAnalysis.py spectrum plot
    if args.kmer_analysis:
        df_target = args.df if args.df else find_default_df()
        out_target = args.output_plot if args.output_plot else find_default_output("kmer_spectrum_custom_ma5.png")
        min_x = args.min_x if args.min_x is not None else 2
        max_x = args.max_x if args.max_x is not None else 300
        ma_w = args.moving_average if args.moving_average is not None else 5

        if (not os.path.exists(df_target)) and args.readsA:
            print(f"[*] Building spectrum DataFrame directly from {args.readsA} using snpDetectLinear...")
            counts, _, _ = snpDetectLinear.count_canonical_kmers(args.readsA, k=args.kmer, max_reads=args.max_reads, verbose=True)
            thresh = snpDetectLinear.estimate_coverage_cutoff(counts, verbose=False)
            df_target = export_spectrum_dataframe(counts, k=args.kmer, thresh_a=thresh, output_path=args.output_df)

        plot_kmer_analysis(
            df_path_or_df=df_target,
            output_path=out_target,
            min_x=min_x,
            max_x=max_x,
            ma_width=ma_w,
            show=args.show
        )
        return

    # Mode 2: Recreate plot_theo_dist.py theoretical mixture & overlap band plot
    if args.theo_dist:
        df_target = args.df if args.df else find_default_df()
        out_target = args.output_plot if args.output_plot else find_default_output("kmer_spectrum_overlap_band.png")

        if (not os.path.exists(df_target)) and args.readsA:
            print(f"[*] Building spectrum DataFrame directly from {args.readsA} using snpDetectLinear...")
            counts, _, _ = snpDetectLinear.count_canonical_kmers(args.readsA, k=args.kmer, max_reads=args.max_reads, verbose=True)
            thresh = snpDetectLinear.estimate_coverage_cutoff(counts, verbose=False)
            df_target = export_spectrum_dataframe(counts, k=args.kmer, thresh_a=thresh, output_path=args.output_df)

        plot_theoretical_overlap_band(
            df_path=df_target,
            k=args.kmer,
            max_c=args.max_c,
            output_path=out_target,
            show=args.show,
            demo_mode=args.demo,
            all_k=args.all_k
        )
        return

    # Mode 3: Default Linear SNP detection 3-panel plot
    if not args.readsA or not args.readsB:
        parser.error("the following arguments are required: readsA, readsB (or specify --kmer-analysis / --theo-dist)")

    out_target = args.output_plot if args.output_plot else find_default_output("snp_linear_detection.png")
    min_x = args.min_x if args.min_x is not None else 1
    max_x = args.max_x if args.max_x is not None else 180

    variants = snpDetectLinear.detect_snps(
        args.readsA,
        args.readsB,
        k=args.kmer,
        max_reads=args.max_reads,
        min_coverage=args.min_coverage,
        max_error_count=args.max_error,
        verbose=True
    )

    if args.output_df:
        export_spectrum_dataframe(
            counts_a=getattr(variants, "counts_a", None),
            counts_b=getattr(variants, "counts_b", None),
            k=args.kmer,
            thresh_a=getattr(variants, "thresh_a", None),
            thresh_b=getattr(variants, "thresh_b", None),
            output_path=args.output_df
        )

    plot_detection_results(
        variants=variants,
        counts_a=getattr(variants, "counts_a", None),
        counts_b=getattr(variants, "counts_b", None),
        thresh_a=getattr(variants, "thresh_a", None),
        thresh_b=getattr(variants, "thresh_b", None),
        diagnostics=getattr(variants, "diagnostics", {}),
        output_path=out_target,
        min_x=min_x,
        max_x=max_x,
        show=args.show
    )


if __name__ == "__main__":
    main()

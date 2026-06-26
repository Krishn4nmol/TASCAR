# run_statistical_tests.py
# Statistical significance tests
# for TASCAR paper
# Paired t-test + Wilcoxon signed-rank
# across all 7 seeds
# Run after all seeds complete!

import numpy as np
from scipy import stats
import json
import os

# ─────────────────────────────────────────
# SEED RESULTS
# Fill in after all seeds complete!
# Format: (CASR_CSR, TASCAR_CSR)
# ─────────────────────────────────────────

# Seeds 42, 123, 456 from paper
# Seeds 789, 1000, 2024, 2025 from
# run_multiseed_part2.py output

WORKLOADS = ['Common', 'Significant', 'Random']

# Per-seed CSR results
# Update 2024 and 2025 after they finish!
SEED_RESULTS = {
    42: {
        'CASR':   {'Common': 90.737,
                   'Significant': 91.892,
                   'Random': 85.675},
        'TASCAR': {'Common': 71.883,
                   'Significant': 74.558,
                   'Random': 70.852},
    },
    123: {
        'CASR':   {'Common': 89.503,
                   'Significant': 93.692,
                   'Random': 90.081},
        'TASCAR': {'Common': 71.848,
                   'Significant': 74.558,
                   'Random': 70.838},
    },
    456: {
        'CASR':   {'Common': 90.525,
                   'Significant': 93.625,
                   'Random': 77.012},
        'TASCAR': {'Common': 71.851,
                   'Significant': 74.548,
                   'Random': 70.841},
    },
    789: {
        'CASR':   {'Common': 90.005,
                   'Significant': 91.826,
                   'Random': 75.399},
        'TASCAR': {'Common': 71.381,
                   'Significant': 75.048,
                   'Random': 71.019},
    },
    1000: {
        'CASR':   {'Common': 91.146,
                   'Significant': 94.302,
                   'Random': 78.905},
        'TASCAR': {'Common': 71.384,
                   'Significant': 75.038,
                   'Random': 71.022},
    },
    2024: {
        'CASR':   {'Common': 87.578,
                   'Significant': 92.795,
                   'Random': 87.213},
        'TASCAR': {'Common': 71.381,
                   'Significant': 75.048,
                   'Random': 71.019},
    },
    2025: {
        'CASR':   {'Common': 87.543,
                   'Significant': 95.735,
                   'Random': 86.113},
        'TASCAR': {'Common': 71.381,
                   'Significant': 75.048,
                   'Random': 71.019},
    },
}


def load_from_json(seed, results_path):
    """Load results from JSON if available"""
    path = f"{results_path}/casr_vs_tascar.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        result = {'CASR': {}, 'TASCAR': {}}
        for wl in WORKLOADS:
            if wl in data:
                result['CASR'][wl] = data[wl].get(
                    'CASR', {}).get(
                    'cold_start_rate', 0)
                result['TASCAR'][wl] = data[wl].get(
                    'TASCAR', {}).get(
                    'cold_start_rate', 0)
        return result
    except Exception as e:
        print(f"  Error loading seed {seed}: {e}")
        return None


def run_tests():
    print("=" * 65)
    print("Statistical Significance Tests")
    print("TASCAR vs CASR — 7 Seeds")
    print("Paired t-test + Wilcoxon")
    print("=" * 65)

    # Try to load from JSON files first
    seed_paths = {
        42:   'results_tascar',
        123:  'results_tascar_seed123',
        456:  'results_tascar_seed456',
        789:  'results_tascar_seed789',
        1000: 'results_tascar_seed1000',
        2024: 'results_tascar_seed2024',
        2025: 'results_tascar_seed2025',
    }

    print("\nLoading results...")
    for seed, path in seed_paths.items():
        loaded = load_from_json(seed, path)
        if loaded:
            # Only update if valid (non-zero)
            for wl in WORKLOADS:
                if loaded['CASR'].get(wl, 0) > 0:
                    SEED_RESULTS[seed]['CASR'][wl] = \
                        loaded['CASR'][wl]
                if loaded['TASCAR'].get(wl, 0) > 0:
                    SEED_RESULTS[seed]['TASCAR'][wl] = \
                        loaded['TASCAR'][wl]
            print(f"  Seed {seed}: loaded from {path}")
        else:
            print(f"  Seed {seed}: using hardcoded values")

    # Filter valid seeds (non-zero results)
    valid_seeds = [
        s for s in SEED_RESULTS
        if SEED_RESULTS[s]['TASCAR']['Common'] > 0]
    print(f"\nValid seeds: {valid_seeds} (n={len(valid_seeds)})")

    results = {}

    print("\n" + "=" * 65)
    print("RESULTS PER WORKLOAD")
    print("=" * 65)

    for wl in WORKLOADS:
        print(f"\n{'─'*50}")
        print(f"Workload: {wl}")
        print(f"{'─'*50}")

        casr_vals   = np.array([
            SEED_RESULTS[s]['CASR'][wl]
            for s in valid_seeds])
        tascar_vals = np.array([
            SEED_RESULTS[s]['TASCAR'][wl]
            for s in valid_seeds])
        differences = casr_vals - tascar_vals

        print(f"\nPer-seed CSR (%):")
        print(f"{'Seed':<8} {'CASR':>10} "
              f"{'TASCAR':>10} {'Diff':>10}")
        print("-" * 42)
        for i, s in enumerate(valid_seeds):
            print(f"{s:<8} {casr_vals[i]:>10.3f} "
                  f"{tascar_vals[i]:>10.3f} "
                  f"{differences[i]:>+10.3f}")

        print(f"\nDescriptive Statistics:")
        print(f"  CASR:   mean={np.mean(casr_vals):.3f}% "
              f"std={np.std(casr_vals):.3f}%")
        print(f"  TASCAR: mean={np.mean(tascar_vals):.3f}% "
              f"std={np.std(tascar_vals):.3f}%")
        print(f"  Diff:   mean={np.mean(differences):.3f}pp "
              f"std={np.std(differences):.3f}pp")

        # 95% Confidence Interval for difference
        n    = len(differences)
        se   = stats.sem(differences)
        ci   = stats.t.interval(
            0.95, df=n-1,
            loc=np.mean(differences),
            scale=se)
        print(f"\n95% CI for mean difference: "
              f"[{ci[0]:.3f}, {ci[1]:.3f}] pp")

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(
            casr_vals, tascar_vals)
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value:     {t_pval:.6f}")
        if t_pval < 0.001:
            sig = "*** (p<0.001)"
        elif t_pval < 0.01:
            sig = "** (p<0.01)"
        elif t_pval < 0.05:
            sig = "* (p<0.05)"
        else:
            sig = "not significant"
        print(f"  Significance: {sig}")

        # Wilcoxon signed-rank test
        try:
            w_stat, w_pval = stats.wilcoxon(
                casr_vals, tascar_vals,
                alternative='greater')
            print(f"\nWilcoxon signed-rank test:")
            print(f"  W-statistic: {w_stat:.4f}")
            print(f"  p-value:     {w_pval:.6f}")
            if w_pval < 0.001:
                w_sig = "*** (p<0.001)"
            elif w_pval < 0.01:
                w_sig = "** (p<0.01)"
            elif w_pval < 0.05:
                w_sig = "* (p<0.05)"
            else:
                w_sig = "not significant"
            print(f"  Significance: {w_sig}")
        except Exception as e:
            print(f"\nWilcoxon: {e}")
            w_stat = w_pval = w_sig = None

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.std(casr_vals)**2 +
             np.std(tascar_vals)**2) / 2)
        cohens_d = (np.mean(casr_vals) -
                    np.mean(tascar_vals)) / \
                   pooled_std if pooled_std > 0 else 0
        print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
        if abs(cohens_d) >= 0.8:
            effect = "large"
        elif abs(cohens_d) >= 0.5:
            effect = "medium"
        else:
            effect = "small"
        print(f"  Interpretation: {effect} effect")

        results[wl] = {
            'n': n,
            'casr_mean': float(np.mean(casr_vals)),
            'casr_std': float(np.std(casr_vals)),
            'tascar_mean': float(np.mean(tascar_vals)),
            'tascar_std': float(np.std(tascar_vals)),
            'mean_diff': float(np.mean(differences)),
            'std_diff': float(np.std(differences)),
            'ci_95_lower': float(ci[0]),
            'ci_95_upper': float(ci[1]),
            't_stat': float(t_stat),
            't_pval': float(t_pval),
            't_sig': sig,
            'w_stat': float(w_stat) if w_stat else None,
            'w_pval': float(w_pval) if w_pval else None,
            'w_sig': w_sig,
            'cohens_d': float(cohens_d),
            'effect_size': effect,
        }

    # Summary table for paper
    print("\n" + "=" * 65)
    print("SUMMARY TABLE FOR PAPER")
    print("=" * 65)
    print(f"\n{'Workload':<14} "
          f"{'CASR':>16} "
          f"{'TASCAR':>16} "
          f"{'p-value':>10} "
          f"{'Sig':>6}")
    print("-" * 65)
    for wl in WORKLOADS:
        r = results[wl]
        casr_str   = (f"{r['casr_mean']:.2f}"
                      f"±{r['casr_std']:.2f}")
        tascar_str = (f"{r['tascar_mean']:.2f}"
                      f"±{r['tascar_std']:.2f}")
        pval = r['t_pval']
        sig  = ("***" if pval < 0.001 else
                "**" if pval < 0.01 else
                "*" if pval < 0.05 else "ns")
        print(f"{wl:<14} "
              f"{casr_str:>16} "
              f"{tascar_str:>16} "
              f"{pval:>10.4f} "
              f"{sig:>6}")

    print(f"\nNote: n={len(valid_seeds)} seeds. "
          f"Paired t-test (one-tailed). "
          f"*** p<0.001, ** p<0.01, * p<0.05")

    # Save results
    os.makedirs('results_statistical', exist_ok=True)
    with open('results_statistical/significance_tests.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results_statistical/significance_tests.json")
    print("=" * 65)

    return results


if __name__ == "__main__":
    run_tests()
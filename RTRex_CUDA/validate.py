#!/usr/bin/env python3
"""
RTRExtractor CUDA — Accuracy Validation Script

Compares GPU output against sequential reference to measure
accuracy preservation of the CUDA implementation.

Usage:
    python3 validate.py <dataset_name> <epsilon>
    
    Expects:
      - RTRex_Sequential/clustering/RTRex-CPU-<name>-decomposition.txt  (sequential output)
      - RTRex_CUDA/RTRex-CUDA-<name>-decomposition.txt                  (GPU output)
"""

import sys
import os
import math
from collections import defaultdict

def load_clusters(filepath):
    """Load cluster decomposition file into list of sets of vertex ids."""
    clusters = []
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            verts = set(int(v) for v in line.split())
            clusters.append(verts)
    return clusters

def compute_stats(clusters):
    """Compute cluster statistics."""
    sizes = [len(c) for c in clusters]
    nontrivial = [s for s in sizes if s > 1]
    
    clustered_vertices = sum(nontrivial)
    total_vertices = sum(sizes)
    
    return {
        'total_clusters': len(clusters),
        'nontrivial_clusters': len(nontrivial),
        'max_size': max(sizes) if sizes else 0,
        'clustered_vertices': clustered_vertices,
        'total_vertices': total_vertices,
        'size_distribution': dict(sorted(
            [(s, sizes.count(s)) for s in set(sizes)], key=lambda x: x[0]
        )),
    }

def coverage_at_density(clusters, density_threshold):
    """
    Compute fraction of vertices in clusters of size >= 5
    with internal density >= threshold.
    (Matches the paper's Fig. 1 metric)
    """
    covered = 0
    total = 0
    for c in clusters:
        total += len(c)
        n = len(c)
        if n < 5:
            continue
        # Density = 2*E / (n*(n-1))
        # We don't have E from the decomposition file alone.
        # Use size as a proxy (larger clusters have higher density in RTRex)
        # For proper validation, use the stats file.
        covered += n
    return covered / total if total > 0 else 0

def compute_jaccard_similarity(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0

def match_clusters(cpu_clusters, gpu_clusters, threshold=0.5):
    """
    Greedy matching of GPU clusters to CPU clusters by Jaccard similarity.
    Returns list of (cpu_idx, gpu_idx, similarity) matches.
    """
    matches = []
    gpu_available = set(range(len(gpu_clusters)))
    
    for cpu_idx, cpu_cluster in enumerate(cpu_clusters):
        best_sim = 0
        best_gpu_idx = -1
        for gpu_idx in list(gpu_available):
            sim = compute_jaccard_similarity(cpu_cluster, gpu_clusters[gpu_idx])
            if sim > best_sim:
                best_sim = sim
                best_gpu_idx = gpu_idx
        
        if best_sim >= threshold:
            matches.append((cpu_idx, best_gpu_idx, best_sim))
            gpu_available.discard(best_gpu_idx)
    
    return matches

def js_divergence(hist1, hist2):
    """Jensen-Shannon divergence between two histograms (dict: key->count)."""
    all_keys = set(hist1.keys()) | set(hist2.keys())
    total1 = sum(hist1.values())
    total2 = sum(hist2.values())
    
    jsd = 0.0
    for k in all_keys:
        p = hist1.get(k, 0) / total1 if total1 > 0 else 0
        q = hist2.get(k, 0) / total2 if total2 > 0 else 0
        m = (p + q) / 2
        
        if p > 0:
            jsd += p * math.log2(p / m) if m > 0 else 0
        if q > 0:
            jsd += q * math.log2(q / m) if m > 0 else 0
    
    return jsd / 2

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 validate.py <dataset_name> <epsilon>")
        print("Example: python3 validate.py small-test 0.1")
        sys.exit(1)
    
    name = sys.argv[1]
    eps = sys.argv[2]
    
    # Expected file paths
    cpu_file = f"../RTRex_Sequential/clustering/RTRex-{name}-decomposition.txt"
    gpu_file = f"RTRex-CUDA-{name}-decomposition.txt"
    
    print("=" * 60)
    print(f"RTRExtractor CUDA Validation: {name} (eps={eps})")
    print("=" * 60)
    
    # Load
    print(f"\nLoading CPU reference: {cpu_file}")
    cpu_clusters = load_clusters(cpu_file)
    if cpu_clusters is None:
        sys.exit(1)
    
    print(f"Loading GPU output: {gpu_file}")
    gpu_clusters = load_clusters(gpu_file)
    if gpu_clusters is None:
        sys.exit(1)
    
    # Stats
    cpu_stats = compute_stats(cpu_clusters)
    gpu_stats = compute_stats(gpu_clusters)
    
    print(f"\n{'Metric':<30} {'CPU':>12} {'GPU':>12} {'Delta':>12}")
    print("-" * 68)
    
    metrics = [
        ("Total clusters", 'total_clusters'),
        ("Non-trivial clusters", 'nontrivial_clusters'),
        ("Max cluster size", 'max_size'),
        ("Clustered vertices", 'clustered_vertices'),
        ("Total vertices", 'total_vertices'),
    ]
    
    for label, key in metrics:
        cpu_val = cpu_stats[key]
        gpu_val = gpu_stats[key]
        delta = gpu_val - cpu_val
        delta_pct = (delta / cpu_val * 100) if cpu_val != 0 else 0
        print(f"{label:<30} {cpu_val:>12} {gpu_val:>12} {delta_pct:>+10.2f}%")
    
    # Size distribution comparison
    jsd = js_divergence(cpu_stats['size_distribution'],
                        gpu_stats['size_distribution'])
    print(f"\nSize distribution JS-divergence: {jsd:.6f}")
    print(f"  (0 = identical, 1 = completely different)")
    
    # Cluster matching
    print(f"\n{'Cluster Matching (Jaccard >= 0.5):':<50}")
    print("-" * 50)
    matches = match_clusters(cpu_clusters, gpu_clusters, threshold=0.5)
    
    n_cpu_matched = len(set(m[0] for m in matches))
    n_gpu_matched = len(set(m[1] for m in matches))
    
    print(f"  CPU clusters matched: {n_cpu_matched}/{cpu_stats['nontrivial_clusters']}")
    print(f"  GPU clusters matched: {n_gpu_matched}/{gpu_stats['nontrivial_clusters']}")
    
    if matches:
        sims = [m[2] for m in matches]
        avg_sim = sum(sims) / len(sims)
        print(f"  Average Jaccard similarity: {avg_sim:.4f}")
    
    # Accuracy assessment
    print(f"\n{'=' * 60}")
    print("ACCURACY ASSESSMENT")
    print("=" * 60)
    
    # Key metric: clustered vertices ratio
    if cpu_stats['clustered_vertices'] > 0:
        cv_ratio = gpu_stats['clustered_vertices'] / cpu_stats['clustered_vertices']
        print(f"Clustered vertex ratio (GPU/CPU): {cv_ratio:.4f}")
        if cv_ratio > 0.98:
            print("  ✓ Within 2% tolerance")
        elif cv_ratio > 0.95:
            print("  ~ Within 5% tolerance")
        else:
            print("  ✗ Outside 5% tolerance — investigate")
    
    if jsd < 0.05:
        print(f"Size distribution: ✓ JS-divergence {jsd:.4f} < 0.05 (excellent)")
    elif jsd < 0.10:
        print(f"Size distribution: ~ JS-divergence {jsd:.4f} < 0.10 (acceptable)")
    else:
        print(f"Size distribution: ✗ JS-divergence {jsd:.4f} >= 0.10 (diverged)")
    
    print()

if __name__ == "__main__":
    main()

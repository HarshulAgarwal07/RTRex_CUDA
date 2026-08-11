#!/usr/bin/env python3
"""
Convert SNAP-format edge lists to RTRExtractor (Escape) format.

SNAP format (input):
    # comment lines
    # Nodes: <n> Edges: <m>
    u v
    ...

Escape format (output):
    <nVertices> <nEdges>
    u v
    ...

Usage:
    python3 snap_to_rtr.py <input.snap.txt> <output.rtr.txt>

    # Convert from gzipped SNAP file:
    gunzip -c com-orkut.ungraph.txt.gz | python3 snap_to_rtr.py - output.rtr.txt
    python3 snap_to_rtr.py com-youtube.ungraph.txt youtube.rtr.txt

The script:
   - Strips comment lines (starting with # or %)
   - Handles both 0-indexed and 1-indexed SNAP graphs
   - Remaps vertex IDs to consecutive 0..N-1 if there are gaps
   - Deduplicates edges (undirected: (u,v) and (v,u) treated as one)
   - Reports stats (vertices, edges, duplicates, remapping info)
"""

import sys
import re
from collections import defaultdict


def parse_header_comments(line):
    """Try to extract node/edge counts from SNAP header comments."""
    m = re.search(r'Nodes:\s*(\d+).*?Edges:\s*(\d+)', line, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r'(\d+)\s+vertices.*?(\d+)\s+edges', line, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def convert_snap_to_rtr(input_path, output_path):
    """Main conversion routine."""

    # --- Read edges ---
    edges = []
    comments = []
    expected_n, expected_m = None, None

    if input_path == '-':
        f = sys.stdin
    else:
        f = open(input_path, 'r')

    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#') or line.startswith('%'):
                comments.append(line)
                if expected_n is None:
                    expected_n = parse_header_comments(line)
                    if expected_n:
                        expected_n, expected_m = expected_n
                continue

            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                continue

            if u == v:
                continue  # skip self-loops

            edges.append((u, v))

    # --- Detect indexing and gaps ---
    all_vertex_ids = set()
    for u, v in edges:
        all_vertex_ids.add(u)
        all_vertex_ids.add(v)

    min_id = min(all_vertex_ids)
    max_id = max(all_vertex_ids)
    n_vertices = len(all_vertex_ids)

    is_zero_indexed = (min_id == 0)
    is_consecutive = (max_id - min_id + 1 == n_vertices)

    needs_remap = not is_consecutive or not is_zero_indexed

    print(f"Input stats:")
    print(f"  Raw edges read:     {len(edges)}")
    print(f"  Unique vertex IDs:  {n_vertices}")
    print(f"  ID range:           [{min_id}, {max_id}]")
    print(f"  Zero-indexed:       {is_zero_indexed}")
    print(f"  Consecutive:        {is_consecutive}")
    print(f"  Needs remapping:    {needs_remap}")
    if expected_n:
        print(f"  SNAP header claims: {expected_n} vertices, {expected_m} edges")

    # --- Remap vertices to 0..N-1 ---
    if needs_remap:
        sorted_ids = sorted(all_vertex_ids)
        mapping = {old: new for new, old in enumerate(sorted_ids)}
        edges = [(mapping[u], mapping[v]) for u, v in edges]
        print(f"  Remapped to 0..{n_vertices - 1}")

    # --- Deduplicate undirected edges ---
    # Normalize: store as (min, max)
    edge_set = set()
    duplicate_count = 0
    for u, v in edges:
        edge = (u, v) if u < v else (v, u)
        if edge in edge_set:
            duplicate_count += 1
        else:
            edge_set.add(edge)

    n_edges = len(edge_set)
    print(f"  Unique undirected:  {n_edges}")
    print(f"  Duplicates removed: {duplicate_count}")

    # --- Write output ---
    if output_path == '-':
        out = sys.stdout
    else:
        out = open(output_path, 'w')

    with out:
        out.write(f"{n_vertices} {n_edges}\n")
        for u, v in sorted(edge_set):
            out.write(f"{u} {v}\n")

    if output_path != '-':
        print(f"\nWrote: {output_path}")
        print(f"  Format: {n_vertices} vertices, {n_edges} edges")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("Examples:")
        print("  python3 snap_to_rtr.py com-youtube.ungraph.txt youtube.rtr.txt")
        print("  gunzip -c com-orkut.ungraph.txt.gz | python3 snap_to_rtr.py - orkut.rtr.txt")
        sys.exit(1)

    convert_snap_to_rtr(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()

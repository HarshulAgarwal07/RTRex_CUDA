# RTRExtractor CUDA — Quickstart Guide

GPU-accelerated dense subgraph discovery via Triangle-Rich Sets.

## Prerequisites

### 1. NVIDIA GPU + CUDA Toolkit

**Check if you have a CUDA-capable GPU:**
```bash
# macOS (this is an Apple Silicon / AMD Mac — CUDA NOT available)
# You need a Linux or Windows machine with an NVIDIA GPU
system_profiler SPDisplaysDataType | grep Chipset

# Linux
lspci | grep -i nvidia
nvidia-smi
```

**Install CUDA Toolkit (Linux):**

| GPU Series | Architecture | CUDA Flag |
|-----------|-------------|-----------|
| RTX 20xx, T4 | Turing | `sm_75` |
| A100 | Ampere | `sm_80` |
| RTX 30xx | Ampere | `sm_86` |
| RTX 40xx | Ada Lovelace | `sm_89` |
| H100 | Hopper | `sm_90` |

```bash
# Ubuntu/Debian (CUDA 12.x)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-6

# Verify installation
nvcc --version
nvidia-smi
```

**Install CUDA Toolkit (macOS — NOT SUPPORTED):**
CUDA is **not available** on Apple Silicon Macs or modern macOS versions.
You must run this code on a Linux or Windows machine with an NVIDIA GPU.
If you're on a Mac, use a cloud GPU instance (Lambda Labs, AWS, GCP) or a remote Linux box.

### 2. C++ Build Tools

```bash
# Ubuntu/Debian
sudo apt install build-essential

# Verify
g++ --version
```

---

## Google Colab — Zero-Setup Run

Google Colab provides **free NVIDIA GPUs** with CUDA pre-installed — no local setup needed. Just open a notebook in your browser.

### What Colab provides

| GPU | Architecture | CUDA Flag | VRAM | Availability |
|-----|-------------|-----------|------|-------------|
| Tesla T4 | Turing | `sm_75` | 15 GB | Most common on free tier |
| Tesla V100 | Volta | `sm_70` | 16 GB | Occasionally on free tier |
| Tesla A100 | Ampere | `sm_80` | 40 GB | Colab Pro / Pro+ |

Colab already has `nvcc`, `g++`, `python3`, and NVIDIA drivers — no installation needed.

### Complete Colab notebook

Create a new notebook at [colab.research.google.com](https://colab.research.google.com), select **Runtime → Change runtime type → T4 GPU**, then paste each cell below:

#### Cell 1: Clone repo + auto-detect GPU architecture

```python
import subprocess, sys

# Clone the repository
!git clone https://github.com/YOUR_USERNAME/RTRExtractor_CUDA.git 2>/dev/null || echo "Repo already cloned"

# Detect GPU and set architecture flag
gpu_name = subprocess.getoutput("nvidia-smi --query-gpu=name --format=csv,noheader").strip()
print(f"Detected GPU: {gpu_name}")

if "T4" in gpu_name:
    sm_flag = "sm_75"
elif "V100" in gpu_name:
    sm_flag = "sm_70"
elif "A100" in gpu_name:
    sm_flag = "sm_80"
elif "L4" in gpu_name:
    sm_flag = "sm_89"
else:
    sm_flag = "sm_75"  # safe fallback
print(f"CUDA architecture flag: {sm_flag}")
```

#### Cell 2: Build Escape library + CUDA binary

```bash
%%bash
cd RTRExtractor_CUDA/RTR2/RTRex_Sequential
make libescape.a -j$(nproc)
cd ../RTRex_CUDA

# SM_FLAG is set from the Python cell above via %%bash env
make CUDA_ARCH=${SM_FLAG:-sm_75} -j$(nproc)
echo "=== Build complete ==="
ls -lh RTRex_CUDA
```

#### Cell 3: Download a SNAP dataset + convert to Escape format

```bash
%%bash
cd RTRExtractor_CUDA/RTR2/RTRex_CUDA

# --- Choose ONE dataset by uncommenting: ---

# YouTube (1.1M vertices, 3M edges) — ~10 sec on T4
# wget -q https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz
# gunzip -k com-youtube.ungraph.txt.gz
# python3 snap_to_rtr.py com-youtube.ungraph.txt youtube.rtr.txt

# Amazon (335K vertices, 926K edges) — ~3 sec on T4
# wget -q https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz
# gunzip -k com-amazon.ungraph.txt.gz
# python3 snap_to_rtr.py com-amazon.ungraph.txt amazon.rtr.txt

# DBLP (317K vertices, 1M edges) — ~3 sec on T4
wget -q https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz
gunzip -k com-dblp.ungraph.txt.gz
python3 snap_to_rtr.py com-dblp.ungraph.txt dblp.rtr.txt

# Orkut (3M vertices, 117M edges) — ~5 min on T4, needs A100 for speed
# Use Colab Pro for this one
# wget -q https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz
# python3 snap_to_rtr.py com-orkut.ungraph.txt orkut.rtr.txt

echo "=== Dataset ready ==="
ls -lh *.rtr.txt 2>/dev/null
```

#### Cell 4: Run RTRExtractor CUDA

```bash
%%bash
cd RTRExtractor_CUDA/RTR2/RTRex_CUDA
./RTRex_CUDA dblp.rtr.txt dblp 0.3
```

#### Cell 5: View results

```python
import os

for fname in sorted(os.listdir()):
    if fname.startswith("RTRex-CUDA-dblp-"):
        print(f"\n{'='*60}")
        print(f"FILE: {fname}")
        print('='*60)
        with open(fname) as f:
            content = f.read()
            # Print first 1500 chars (enough to see stats + first few clusters)
            print(content[:2000])
```

### Colab-specific notes

- **Session timeout**: Free Colab disconnects after ~90 min of inactivity or ~12 hours total. Download your output files before the session ends.
- **Disk space**: Colab provides ~70 GB of disk. Orkut (~1.5 GB compressed, ~2 GB uncompressed) fits easily.
- **VRAM limits**: Free-tier T4 has 15 GB. Graphs up to ~150M edges fit. Orkut (117M edges, ~8 GB on GPU) works on T4 but may approach the limit.
- **Speed on T4 vs A100**: T4 is ~3–5× slower than A100. A Colab Pro subscription gets you A100 access and longer runtimes — worth it for datasets above 50M edges.
- **Persistent storage**: Mount Google Drive to save results across sessions:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then write output with `--output /content/drive/MyDrive/rtr-results/`.

---

## Project Layout

```
RTR2/
├── RTRex_Sequential/          # Original CPU implementation (reference)
│   ├── Escape/                # Graph library (CSR, GraphIO, etc.)
│   │   ├── Graph.h, Graph.cpp
│   │   ├── GraphIO.h, GraphIO.cpp
│   │   ├── ClusterStructures.h
│   │   └── Decomposition.h    # Core RTRExtractor algorithm
│   ├── clustering/            # Executables + test data
│   │   ├── RTRex.cpp          # Sequential main
│   │   ├── nucleus.cpp        # Nucleus decomposition
│   │   ├── clean_*.txt        # Test datasets
│   │   └── Makefile
│   └── Makefile
│
├── RTRex_CUDA/                # GPU-accelerated implementation (NEW)
│   ├── gpu_common.cuh         # Shared types, device functions
│   ├── gpu_graph.cuh/.cu      # GPU CSR structures + transfer
│   ├── gpu_triangle_count.cuh/.cu  # Weighted triangle counting
│   ├── gpu_clean.cuh/.cu      # Bulk-synchronous edge cleaning
│   ├── gpu_extract.cuh/.cu    # Extraction inner loop kernels
│   ├── main.cu                # Full pipeline driver
│   ├── Makefile
│   └── validate.py            # Accuracy comparison script
│
└── RTRExtractor_Research_Paper.pdf
```

---

## Build

### Step 1: Build the Escape library (sequential dependency)

```bash
cd RTRex_Sequential

# Build libescape.a — required by both sequential and CUDA versions
make libescape.a

cd ..
```

Expected output: `RTRex_Sequential/libescape.a`

### Step 2: Build the CUDA version

```bash
cd RTRex_CUDA

# For RTX 30xx (default):
make CUDA_ARCH=sm_86

# For RTX 40xx:
make CUDA_ARCH=sm_89

# For A100:
make CUDA_ARCH=sm_80

# For RTX 20xx / T4:
make CUDA_ARCH=sm_75
```

Expected output: `RTRex_CUDA/RTRex_CUDA`

### Step 3 (optional): Build the sequential version for comparison

```bash
cd RTRex_Sequential
make clustering
cd ..
```

Expected output: `RTRex_Sequential/clustering/RTRex`

---

## Run

### GPU Version

```bash
cd RTRex_CUDA

# Basic usage:
# ./RTRex_CUDA <graph_file> <output_name> <epsilon> [gpu_device_id]
#
# Parameters:
#   graph_file:   Path to graph in Escape format
#   output_name:  Prefix for output files (decomposition, stats, freq, cut)
#   epsilon:      Cleaning threshold, typical values: 0.1, 0.3, 0.5
#   gpu_device:   CUDA device ID (default: 0)

# Small test
./RTRex_CUDA ../RTRex_Sequential/Example/small-test.txt small-test 0.3

# Paper datasets (various sizes):
./RTRex_CUDA ../RTRex_Sequential/clustering/clean_email.txt email 0.3
./RTRex_CUDA ../RTRex_Sequential/clustering/clean_astro.txt astro 0.3
./RTRex_CUDA ../RTRex_Sequential/clustering/clean_amazon.txt amazon 0.3
./RTRex_CUDA ../RTRex_Sequential/clustering/clean_youtube.txt youtube 0.3

# With specific GPU device:
./RTRex_CUDA ../RTRex_Sequential/clustering/clean_youtube.txt youtube 0.3 1
```

### Output Files

For output name `youtube`, the following files are generated:
- `RTRex-CUDA-youtube-decomposition.txt` — Vertex IDs per cluster (one cluster per line)
- `RTRex-CUDA-youtube-stats.txt` — Size and density per non-singleton cluster
- `RTRex-CUDA-youtube-freq.txt` — Cluster size histogram
- `RTRex-CUDA-youtube-cut.txt` — Inter-cluster cut edges

### Sequential Version (for accuracy comparison)

```bash
cd RTRex_Sequential/clustering

# ./RTRex <graph_file> <output_name> <epsilon> <mode>
# Mode: s = simple extraction
./RTRex ../clustering/clean_youtube.txt youtube 0.3 s
```

---

## Validate Accuracy

Compare GPU output against the sequential reference:

```bash
cd RTRex_CUDA

# 1. Run sequential version first
cd ../RTRex_Sequential/clustering
./RTRex clean_youtube.txt youtube 0.3 s
cd ../../RTRex_CUDA

# 2. Run CUDA version
./RTRex_CUDA ../RTRex_Sequential/clustering/clean_youtube.txt youtube 0.3

# 3. Compare
python3 validate.py youtube 0.3
```

The validation script reports:
- **Cluster count & size**: Total clusters, non-trivial clusters, max size
- **Coverage**: Ratio of GPU/CPU clustered vertices
- **JS-divergence**: Size distribution similarity (0 = identical)
- **Jaccard matching**: How many clusters match between CPU and GPU outputs
- **Pass/fail**: Green/yellow/red assessment at 2% and 5% tolerance thresholds

---

## Graph File Format (Escape format)

The input graph must be in Escape format:

```
<num_vertices> <num_edges>
<u1> <v1>
<u2> <v2>
...
```

Example (`small-test.txt`):
```
10 19
1 7
3 6
1 2
0 1
...
```

The first line gives `nVertices nEdges` (undirected edges). Each subsequent line is one undirected edge `u v` (0-indexed vertex IDs). The graph is assumed to be **undirected and simple** (no self-loops, no duplicate edges).

---

## Troubleshooting

### "nvcc: command not found"
CUDA toolkit is not installed or not on PATH.
```bash
# Check if CUDA is installed
ls /usr/local/cuda/bin/nvcc

# If installed but not on PATH:
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### "cudaErrorNoDevice: no CUDA-capable device is detected"
No NVIDIA GPU available. Run `nvidia-smi` to check. On cloud instances, ensure the NVIDIA driver is installed.

### "libescape.a: No such file or directory"
The Escape library wasn't built. Run `make libescape.a` in `RTRex_Sequential/` first.

### "error: identifier 'omp_get_max_threads' is undefined"
The Escape library uses OpenMP. Ensure `-fopenmp` is in the compiler flags (it's in `RTRex_Sequential/common.mk`). This only affects the sequential build; the CUDA build doesn't use Decomposition.h.

### "fatal error: Escape/Graph.h: No such file or directory"
Include path issue. The Makefile uses `-I../RTRex_Sequential`. Run `make` from within the `RTRex_CUDA/` directory.

### Memory errors on large graphs
The graph must fit in GPU VRAM. Approximate memory required:
- CSR + state arrays ≈ `3 × nEdges × 8 + nVertices × 40` bytes
- For 100M undirected edges (200M directed): ~6 GB
- For 300M edges: ~18 GB — requires A100/H100

---

## Tuning Parameters

Key parameters that affect accuracy vs. speed:

| Parameter | Location | Effect |
|-----------|----------|--------|
| `eps` | CLI argument | Higher → more aggressive cleaning, fewer/smaller clusters |
| `CUDA_ARCH` | Makefile | Must match your GPU architecture for optimal performance |
| `THREADS_PER_BLOCK` | `gpu_common.cuh:18` | 256 is optimal for Ampere; tune 128–512 |
| `maxRounds` | `gpu_clean.cuh:33, gpu_clean.cu:143` | 100 for init, 200 for extract. Higher = more accurate if not converged |

---

## References

- **Paper**: S. Basu, D. Paul-Pena, K. Qian, C. Seshadhri, E. Huang, K. Subbian. "Covering a Graph with Dense Subgraph Families, via Triangle-Rich Sets." arXiv:2407.16850, 2024.
- **Sequential code**: `RTRex_Sequential/clustering/RTRex.cpp`
- **Core algorithm**: `RTRex_Sequential/Escape/Decomposition.h` (function `disjointExtract`)

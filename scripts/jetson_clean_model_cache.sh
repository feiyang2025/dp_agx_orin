#!/bin/bash
# Clean tinygrad/model caches so models recompile for the local GPU arch.
# Run once after flashing AGX Orin (JP6) or switching JetPack versions:
#   Xavier = sm_72 (CUDA 11) -> Orin = sm_87 (CUDA 12)
# First modeld start will re-JIT automatically (slower first run only).
set -e

echo "Cleaning tinygrad JIT cache..."
rm -rf ~/.cache/tinygrad

echo "Cleaning precompiled model pickles..."
rm -f selfdrive/modeld/models/*_tinygrad.pkl
rm -f selfdrive/modeld/models/*.pkl.tmp 2>/dev/null || true

# TensorRT engine caches are also arch-specific
find . -name "*.engine" -path "*modeld*" -delete 2>/dev/null || true
rm -rf ~/.cache/tensorrt_engine_cache* 2>/dev/null || true

echo "Done. Models will recompile on next modeld launch (expect a slow first boot)."

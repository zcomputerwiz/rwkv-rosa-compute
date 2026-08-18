# Development Rules & Guidelines

## 1. Upstream Isolation
Do not modify files in `external/RWKV-LM` or `external/rosa_soft` submodules unless explicitly directed by a later research phase task.

## 2. Compatibility First
Any optimization must retain a reference implementation and semantic comparison test.

## 3. No Silent Semantic Changes
Do not alter suffix horizon (512), bit width (4), tie-breaking recency rules (latest occurrence wins), unmatched symbol zeroing, value routing, or signed representation conventions (`{-1.0, 0.0, +1.0}`) without adding or updating compatibility tests.

## 4. CPU Oracle
The simple CPU reference implementation (`blinkdl_rosa_4bit_reference`) is the source of truth for correctness.

## 5. CUDA as Implementation Detail
CUDA kernels must reproduce the reference behavior; CUDA is an optimization, not the semantic specification.

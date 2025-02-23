# KU_Unsloth_Challenge_Solutions
This repository contains the solution for the challenge given by Unsloth.ai.

## 1. Challenge A :- Convert nf4 to Triton. [Difficulty: Hard] [Max points: 14]

## Challenge Objective

**Goal:**  
Implement a single-kernel dequantization function for nf4–quantized tensors (converting them to fp16 or bf16) that is at least **1.15× faster** than the reference unsloth fast_dequantize.

**Constraints:**
- Use a single fused kernel (no `torch.compile`).
- Achieve a significant speedup without resorting to large intermediate buffers.
- Leverage techniques like custom inline PTX if needed.

## Our Approach

### High-Level Triton Optimizations
- **Single Triton Kernel:**  
  Designed a kernel that fuses both the dequantization of the per–block statistics (absmax and nested_absmax) and the actual weight dequantization.
- **Tiling & Loop Unrolling:**  
  Experimented with having each thread process multiple output elements (tiling) to amortize global memory access overhead.
- **Vectorized Global Memory Access:**  
  Loaded multiple packed nf4 values at once by interpreting groups of bytes as 32-bit words to reduce the number of memory transactions.
- **Shared Memory Caching:**  
  Cached scaling factors computed from absmax and nested_absmax in shared memory to avoid redundant global loads.
- **Inline PTX:**  
  Incorporated inline PTX (using the UBFE instruction) for the critical bit extraction step to reduce the instruction count for extracting the 4-bit quantized value.

### Ultra-Low-Level Triton & Custom Approaches
- Explored asynchronous copies (if supported) to prefetch scaling factors and weight data, aiming to hide global memory latency.
- Considered a complete custom CUDA solution as an alternative, but then refocused on optimizing the Triton kernel.

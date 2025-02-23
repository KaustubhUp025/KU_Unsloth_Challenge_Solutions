# KU_Unsloth_Challenge_Solutions
This repository contains the solution for the challenge given by Unsloth.ai.

## 1. Challenge A :- Convert nf4 to Triton. 

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

## 2. Challenge B :-  Make QLoRA work with FSDP2 

## Our Approach


## Environment and Setup
- **Clean Runtime:**  
  Set environment variables (e.g. for CUDA and bitsandbytes) and cleared the module cache to ensure a clean runtime.
- **GPU Configuration:**  
  Configured the script to run on 2 GPUs using `CUDA_VISIBLE_DEVICES` and a per-process device map based on the `LOCAL_RANK` environment variable.

## Model Loading and Quantization
- **Model Loading:**  
  Loaded the Llama 3.1 8B model using a 4‑bit quantized configuration from Unsloth’s HF page.
- **Freezing Base Model:**  
  Frozen the base model to ensure that the quantized (and integer‑dtype) parameters are not updated during fine‑tuning.

## PEFT/LoRA Integration
- **LoRA Adapters:**  
  Applied LoRA adapters (using the PEFT library) to the model, adding new, small trainable floating‑point parameters (the LoRA weights) while keeping the base frozen.
- **Parameter Freezing:**  
  Ensured that any non‑floating point parameters remain frozen.

## Handling Frozen Quantized Weights
- **Buffer Conversion:**  
  Converted the frozen, non‑floating point (quantized) parameters to buffers. This prevents FSDP from attempting to flatten these integer‑dtype parameters, which could cause shape mismatches.
- **GPU Transfer:**  
  After conversion, moved the entire model (including its buffers) to the appropriate GPU.

## FSDP Wrapping
- **Ignoring Self-Attention Submodules:**  
  Marked self‑attention submodules (identified by “self_attn” in their names) to be ignored by FSDP, ensuring that FSDP does not modify the large, frozen base weights in the attention projections.
- **Custom FSDP Policies:**  
  Defined FSDP policies (with mixed-precision and offloading) and manually applied FSDP wrapping only to eligible submodules (e.g. LlamaDecoderLayer modules with trainable parameters that are not marked to be ignored).

## torch.compile Integration
- **Optimizing Trainable Parts:**  
  Added functionality to recursively search for submodules containing LoRA adapter parameters (e.g. attributes like `lora_A` or `lora_B`) and compile them using `torch.compile()` to further optimize the trainable components.

## 3. Challenge C :- Make torch.compile work without graph breaks for QLoRA 

## Our Approach

## Restored Original Numerical Behavior

We reverted our custom dequantization and matmul functions to exactly mimic the original BitsAndBytes behavior (i.e. using `F.dequantize_4bit(weight, quant_state).to(x.dtype).t()`), ensuring that training loss remains correct.

## Isolated Low-Level Operations from Compilation

- **Low-Level Calls Isolation:**  
  Wrapped the low-level BitsAndBytes calls (such as dequantization and transpose) in functions decorated with `@torch._dynamo.disable` so that TorchDynamo doesn’t try to trace them.

- **Eager Transpose Execution:**  
  Patched the transpose method (`.t()`) of BitsAndBytes’ `Params4bit` to run in eager mode, preventing graph breaks from user-defined methods.

## Addressed Dynamic Behavior in the Model

- **Disabling Past Key Caches:**  
  In the attention module, we disabled the use of past key caches during training (by setting `past_key_value = None` when training), which reduces dynamic behavior that leads to guard failures and frequent recompilations.

- **Skipping PEFT’s LoRA Code:**  
  We added a skip directive for PEFT’s LoRA code to prevent TorchDynamo from compiling that part of the code. This helps avoid many guard failures due to dynamic parameter sizes:

  ```python
  torch._dynamo.config.skipfiles.add("peft/tuners/lora")

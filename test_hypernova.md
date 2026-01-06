# HyperNova-60B Real-World Validation

## Configuration
- **Model:** MultiverseComputingCAI/HyperNova-60B
- **GPU:** 2x RTX 5090
- **Tensor Parallel:** 2
- **Max Model Len:** 131,072
- **Max Num Seqs:** 8
- **Max Batched Tokens:** 65,536
- **GPU Utilization:** 0.90

## Model Architecture (from HuggingFace config)
- **Layers:** 32
- **Attention Heads:** 64
- **KV Heads:** 8
- **Head Dimension:** 64
- **Hidden Size:** 2,880 (actual) vs 4,096 (calculated as 64×64)
- **Quantization:** MXFP4 (4-bit)
- **Type:** MoE with 80 experts, 4 active per token

## Real-World Results (from logs)

### Memory Usage (binary units from nvidia-smi)
- **Total VRAM:** 32607 MiB = 31.85 GiB per GPU
- **Used:** 28416 MiB = 27.75 GiB per GPU
- **Free:** 4191 MiB = 4.09 GiB per GPU

### Memory Usage (decimal units for calculator)
- **Total VRAM:** 34.2 GB per GPU (32607 MiB × 1024² ÷ 1e9)
- **Available (90%):** 30.78 GB per GPU
- **Model Loading:** 16.9066 GiB = 18.15 GB per GPU
- **KV Available:** 7.69 GiB = 8.26 GB per GPU
- **KV Cache Tokens:** 251,728 tokens

## Calculator Expected Results

### KV Cache Calculation
```
KV heads per GPU: 8 / 2 = 4
Bytes per token per layer: 2 (K+V) × 4 × 64 × 2 (dtype) = 1,024 bytes
Bytes per token: 1,024 × 32 layers = 32,768 bytes = 32 KB
```

### Verification
```
Real KV available: 8.26 GB = 8,260,000,000 bytes
Real tokens: 251,728
Bytes per token: 8,260,000,000 / 251,728 = 32,818 bytes

Expected tokens: 8,260,000,000 / 32,768 = 252,075 tokens
Error: 252,075 - 251,728 = 347 tokens (0.14% error) ✓
```

**Result:** KV cache calculation is **CORRECT** ✓

## Overhead Analysis

### Calculator's Overhead Estimate
```
Attn heads per GPU: 64 / 2 = 32
Hidden size per GPU: 32 × 64 = 2,048 (calculator assumes this)
Actual hidden size per GPU: 2,880 / 2 = 1,440 (from config)

Prefill tokens: 65,536
Decode tokens: 8
Total: 65,544 tokens

Activation bytes (calculator): 65,544 × 2,048 × 2 = 268,468,224 bytes = 0.268 GB
Overhead (2x buffers): 0.268 × 2 = 0.537 GB
CUDA graphs: 0.268 GB
Total overhead (calculator): 0.805 GB
```

### Real-World Overhead
```
Available: 30.78 GB
Weights: 18.15 GB
KV available: 8.26 GB
Everything else: 30.78 - 18.15 - 8.26 = 4.37 GB

Missing overhead: 4.37 - 0.805 = 3.565 GB
```

**Result:** Calculator underestimates overhead by **3.565 GB**

### Overhead Sources (not in calculator)
1. **CUDA Graphs:** Much larger than estimated (~2-3 GB vs 0.268 GB)
   - Captures multiple batch sizes and sequence lengths
   - Logs show 83 mixed prefill-decode graphs + 4 decode graphs
2. **MoE Overhead:** Expert routing, gating networks (80 experts)
3. **Framework Overhead:** PyTorch, CUDA runtime, torch.compile cache
4. **PagedAttention:** Block managers, page tables
5. **Hidden Size Discrepancy:** Using 2,048 instead of 1,440

## Workaround

Users should add **3.5 GB** to the "Extra Overhead (GB)" field for this configuration to match real-world results.

## Potential Fixes

1. **Add Hidden Size field** to the calculator for accurate activation overhead
2. **Improve CUDA graphs estimation** - current formula significantly underestimates
3. **Add MoE overhead calculation** - detect MoE models and add expert overhead
4. **Add framework overhead constant** (~1-2 GB for PyTorch/CUDA/vLLM)
5. **Make overhead padding more prominent** with suggested values

## Summary

| Metric | Calculator | Real-World | Match |
|--------|-----------|------------|-------|
| KV Cache Tokens | 252,075 | 251,728 | ✓ 0.14% error |
| Bytes per Token | 32,768 | 32,818 | ✓ 0.15% error |
| KV Available | 11.83 GB* | 8.26 GB | ✗ Without overhead adjustment |
| KV Available | 8.28 GB** | 8.26 GB | ✓ With 3.5 GB extra overhead |

*Without extra overhead padding
**With 3.5 GB extra overhead padding

## Conclusion

The calculator's **core KV cache math is accurate** but **overhead estimation needs improvement**, especially for:
- MoE models
- CUDA graphs (significantly underestimated)
- Large context windows with chunked prefill
- Models where hidden_size ≠ num_attention_heads × head_dim

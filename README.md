# vLLM VRAM Calculator

A sophisticated web-based calculator for planning GPU memory allocation when deploying Large Language Models (LLMs) with vLLM. This tool helps you optimize your inference setup by calculating memory requirements, analyzing capacity constraints, and generating optimized vLLM launch commands.

🔗 **[Live Demo](https://taylorelley.github.io/vllm-vram-calc/)** (GitHub Pages)

## Features

### GPU Configuration
- **GPU Presets Dropdown**: Easy selection from 10+ common GPUs:
  - **Consumer**: RTX 3090, 4070 Ti, 4080, 4090, 5090
  - **Professional**: A6000, A100 (40/80GB), L40S, H100
  - Uses actual usable VRAM from nvidia-smi (decimal GB)
- **Tensor Parallelism**: Support for multi-GPU deployments with automatic per-GPU calculations
- **Memory Utilization Control**: Configurable GPU memory utilization (typically 0.85-0.95)
- **Consistent Units**: All calculations use decimal GB (1 GB = 1,000³ bytes) to match GPU specs

### Model Configuration
- **HuggingFace Integration**: Automatically fetch model configurations from HuggingFace Hub
  - Extracts architecture, layer count, KV heads, head dimensions
  - Estimates model size from safetensors metadata
  - Detects quantization from model tags
  - Simply enter a model ID and click "Fetch"
- **Manual Configuration**: Full control over:
  - Model weights (GB)
  - Number of layers
  - KV heads (for Grouped Query Attention)
  - Head dimension

### Quantization Support
Comprehensive quantization calculation with support for:
- **FP16/BF16** (16-bit float)
- **FP8** (8-bit float)
- **MXFP4** (4-bit MX format)
- **AWQ** (4-bit Activation-aware Weight Quantization)
- **GPTQ** (4-bit GPT Quantization)
- **BitsAndBytes** (NF4, INT8)
- **EXL2** (variable bit)
- **GGUF** (variable bit)

The calculator estimates:
- Weight size based on bits per parameter
- Scale/zero-point overhead for group-wise quantization
- Total model size including quantization metadata

### vLLM Configuration
Fine-tune your deployment with:
- **Max Model Length**: Maximum context window
- **Max Num Seqs**: Concurrent sequences
- **Max Batched Tokens**: Tokens per forward pass
- **KV Cache Dtype**: BF16/FP16 or FP8 compression
- **CUDA Graphs**: Enable/disable (affects ~2.5GB overhead)
- **Overhead Estimate**: Account for activations and buffers

### Memory Analysis
The calculator provides detailed breakdowns:
- **Per-GPU Memory Usage**:
  - Model weights (distributed across GPUs)
  - KV cache (estimated based on active tokens)
  - CUDA graphs overhead
  - Framework overhead
  - Free headroom
- **Visual Memory Bar**: Color-coded visualization of memory allocation
- **Capacity Analysis**:
  - Maximum tokens that fit in KV cache
  - Max context length for single sequence
  - Average context per sequence
  - Recommended max-num-seqs for optimal throughput
- **vLLM Comparison**: Shows values in both GB (decimal) and GiB (binary) for easy comparison with vLLM's output

### Command Generation
Automatically generates optimized vLLM launch commands with appropriate flags:
- `--tensor-parallel-size`
- `--max-model-len`
- `--max-num-seqs`
- `--max-num-batched-tokens`
- `--gpu-memory-utilization`
- `--enable-chunked-prefill`
- `--enforce-eager` (when CUDA graphs disabled)
- `--disable-custom-all-reduce` (for multi-GPU)

## Usage

### Quick Start
1. Open the [live demo](https://taylorelley.github.io/vllm-vram-calc/) or `index.html` locally
2. Select a GPU from the dropdown (e.g., RTX 5090)
3. Enter a HuggingFace model ID and click "Fetch" to auto-populate configuration
   - Or manually configure model parameters
4. Adjust vLLM configuration as needed (max-model-len, max-num-seqs, etc.)
5. Review the memory breakdown and capacity analysis
6. Check the vLLM comparison values (in GiB) to match expected output
7. Copy the generated vLLM command

### Example Workflow

#### Using HuggingFace Integration (Recommended)
1. Select GPU from dropdown: **RTX 5090 (32GB)**
2. Set number of GPUs (TP size): **2**
3. Enter model ID: `MultiverseComputingCAI/HyperNova-60B`
4. Click **"Fetch"**
5. Review auto-populated configuration (layers, KV heads, weights, etc.)
6. Adjust max-num-seqs based on recommended value
7. Check memory status (should show ✓ Configuration looks good)
8. Compare with vLLM's expected values using the GiB display
9. Copy generated command

#### Manual Configuration
1. Select GPU from dropdown: **RTX 5090 (32GB)** (or enter custom VRAM)
2. Set number of GPUs (TP size): **2**
3. Enter model weights: **36.3 GB**
4. Configure layers: **32**, KV heads: **8**, head dimension: **64**
5. Set quantization method (e.g., **MXFP4**)
6. Configure base params: **60** billion
7. Click "Apply Estimate" to use calculated model size
8. Adjust max-model-len: **131072** and max-num-seqs: **8**
9. Review capacity analysis and vLLM comparison values

### Understanding the Output

**Status Indicators**:
- ✓ **Configuration looks good** (green): Safe configuration with headroom
- ⚡ **Tight** (yellow): May OOM under load, consider reducing max-num-seqs
- ⚠ **OOM Risk** (red): Will likely fail, reduce context/seqs or enable FP8 KV cache

**Key Metrics**:
- **Max Tokens in Cache**: Total KV cache capacity across all sequences
- **Max Context (1 seq)**: Maximum context length for a single request
- **Avg Context (all seqs)**: Average context per sequence when all slots filled
- **Recommended max-num-seqs**: Optimal concurrency for ~32K average context

## Technical Details

### Units: GB vs GiB
The calculator uses **decimal GB** (1 GB = 1,000,000,000 bytes) throughout all calculations to match:
- GPU manufacturer specifications (nvidia-smi reports in MiB but marketed as GB)
- Standard storage conventions
- HuggingFace model sizes

**Important**: vLLM's logs report memory in **binary GiB** (1 GiB = 1,073,741,824 bytes).

**Conversion**: 1 GB (decimal) ≈ 0.931 GiB (binary)

The calculator displays both values in the status section for easy comparison with vLLM output:
```
💡 vLLM comparison: Available 28.66 GiB (binary) • KV cache 7.69 GiB
```

### KV Cache Calculation
The calculator computes KV cache requirements using:
```
bytes_per_token_per_layer = 2 (K+V) × kv_heads_per_gpu × head_dim × dtype_bytes
total_bytes_per_token = bytes_per_token_per_layer × num_layers
```

With tensor parallelism, KV heads are distributed across GPUs, reducing per-GPU cache size.

### Memory Layout
Per-GPU memory is divided into:
1. **Fixed Overhead**:
   - Model weights (total / num_gpus)
   - CUDA graphs (~2.5GB when enabled)
   - Framework overhead (configurable)
2. **Dynamic KV Cache**: Fills remaining space up to `gpu_memory_utilization` limit

### Quantization Overhead
Group-wise quantization methods (AWQ, GPTQ, MXFP4) store additional metadata:
- **Scale factors**: 2 bytes per group
- **Zero points**: 2 bytes per group (for asymmetric quantization)
- **Groups**: `num_parameters / group_size`

## Browser Compatibility

Works on all modern browsers:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

No build step or dependencies required - just open `index.html`.

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! See TODO.md for planned features and improvements.

## Acknowledgments

Built for the vLLM community to simplify production LLM deployments. Special thanks to the vLLM team for their excellent inference engine.

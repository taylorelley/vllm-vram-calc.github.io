# vLLM VRAM Calculator

A sophisticated web-based calculator for planning GPU memory allocation when deploying Large Language Models (LLMs) with vLLM. This tool helps you optimize your inference setup by calculating memory requirements, analyzing capacity constraints, and generating optimized vLLM launch commands.

## Features

### GPU Configuration
- **GPU Presets**: Quick selection for common GPUs (RTX 4090, RTX 5090, A6000, H100)
- **Tensor Parallelism**: Support for multi-GPU deployments with automatic per-GPU calculations
- **Memory Utilization Control**: Configurable GPU memory utilization (typically 0.85-0.95)

### Model Configuration
- **HuggingFace Integration**: Automatically fetch model configurations from HuggingFace Hub
  - Extracts architecture, layer count, KV heads, head dimensions
  - Estimates model size from safetensors metadata
  - Detects quantization from model tags
- **Model Presets**: Built-in presets for popular models (HyperNova-60B, gpt-oss-120b, Qwen3-80B-AWQ, Llama-70B-FP16)
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
1. Open `index.html` in a modern web browser
2. Select a GPU preset or enter custom VRAM
3. Either:
   - Enter a HuggingFace model ID and click "Fetch" to auto-populate
   - Select a model preset
   - Manually configure model parameters
4. Adjust vLLM configuration as needed
5. Review the memory breakdown and capacity analysis
6. Copy the generated vLLM command

### Example Workflow

#### Using HuggingFace Integration
1. Enter model ID: `MultiverseComputingCAI/HyperNova-60B`
2. Click "Fetch"
3. Review auto-populated configuration
4. Adjust max-num-seqs based on recommended value
5. Check memory status (should show ✓ Configuration looks good)
6. Copy generated command

#### Manual Configuration
1. Select GPU: RTX 5090 (32GB)
2. Set number of GPUs (TP size): 2
3. Enter model weights: 34.2 GB
4. Configure layers, KV heads, head dimension
5. Set quantization method (e.g., MXFP4)
6. Adjust max-model-len and max-num-seqs
7. Review capacity analysis

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

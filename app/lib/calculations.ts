import type { GPUConfig, ModelConfig, QuantizationConfig, VLLMConfig, CalculationResult } from './types';

const DECIMAL_GB = 1_000_000_000;

export function calculateVRAM(
  gpu: GPUConfig,
  model: ModelConfig,
  quant: QuantizationConfig,
  vllm: VLLMConfig
): CalculationResult {
  // Parse inputs
  const gpuVramGB = gpu.vram;
  const numGpus = gpu.numGpus;
  const gpuUtilization = gpu.utilization;

  const modelWeightsGB = model.weightsGB;
  const numLayers = model.numLayers;
  const kvHeads = model.kvHeads;
  const headDim = model.headDim;

  const maxModelLen = vllm.maxModelLen;
  const maxNumSeqs = vllm.maxNumSeqs;
  const maxBatchedTokens = vllm.maxBatchedTokens;

  const kvCacheDtypeBytes = vllm.kvCacheDtype === 'fp8' ? 1 : 2;
  const cudaGraphsEnabled = vllm.cudaGraphs;
  const overheadPaddingGB = vllm.overheadPadding;

  // Calculate available VRAM per GPU
  const totalVramBytes = gpuVramGB * DECIMAL_GB;
  const availableVramBytes = totalVramBytes * gpuUtilization;
  const availableVramGB = availableVramBytes / DECIMAL_GB;

  // Calculate model weights per GPU (distributed via TP)
  const weightsPerGpuGB = modelWeightsGB / numGpus;
  const weightsPerGpuBytes = weightsPerGpuGB * DECIMAL_GB;

  // CUDA graphs memory (per GPU)
  const cudaGraphsGB = cudaGraphsEnabled ? 2.5 : 0;
  const cudaGraphsBytes = cudaGraphsGB * DECIMAL_GB;

  // Overhead padding
  const overheadBytes = overheadPaddingGB * DECIMAL_GB;

  // Calculate KV cache memory per token
  // With tensor parallelism, KV heads are distributed across GPUs
  const kvHeadsPerGpu = Math.ceil(kvHeads / numGpus);

  // KV cache formula: 2 (K+V) × kv_heads_per_gpu × head_dim × dtype_bytes
  const bytesPerTokenPerLayer = 2 * kvHeadsPerGpu * headDim * kvCacheDtypeBytes;
  const bytesPerToken = bytesPerTokenPerLayer * numLayers;

  // KV cache per sequence
  const bytesPerSeq = bytesPerToken * maxModelLen;

  // Available memory for KV cache
  const kvAvailableBytes = availableVramBytes - weightsPerGpuBytes - cudaGraphsBytes - overheadBytes;

  // Check if we're over capacity
  if (kvAvailableBytes <= 0) {
    return {
      availableVramPerGpu: availableVramGB,
      weightsPerGpu: weightsPerGpuGB,
      cudaGraphsMemory: cudaGraphsGB,
      overheadMemory: overheadPaddingGB,
      kvBytesPerToken: bytesPerToken,
      kvBytesPerSeq: bytesPerSeq,
      totalKVCacheMemory: 0,
      maxTokensForKV: 0,
      maxConcurrentSequences: 0,
      totalBatchedTokens: 0,
      freeMemory: kvAvailableBytes / DECIMAL_GB,
      memoryUsagePercent: 100,
      isOverCapacity: true,
      warnings: ['Model weights exceed available VRAM. Reduce model size or increase GPU count.'],
      command: '',
    };
  }

  // Calculate maximum tokens that can fit in KV cache
  const maxTokensForKV = Math.floor(kvAvailableBytes / bytesPerToken);

  // Calculate capacity based on user's max_num_seqs and max_model_len
  const tokensPerSeq = maxModelLen;
  const maxConcurrentSeqs = Math.floor(maxTokensForKV / tokensPerSeq);

  // Actual concurrent sequences (limited by user's max_num_seqs)
  const actualMaxNumSeqs = Math.min(maxConcurrentSeqs, maxNumSeqs);

  // Total KV cache memory used
  const totalKVCacheBytes = actualMaxNumSeqs * bytesPerSeq;
  const totalKVCacheGB = totalKVCacheBytes / DECIMAL_GB;

  // Free memory
  const usedMemoryBytes = weightsPerGpuBytes + cudaGraphsBytes + overheadBytes + totalKVCacheBytes;
  const freeMemoryBytes = availableVramBytes - usedMemoryBytes;
  const freeMemoryGB = freeMemoryBytes / DECIMAL_GB;

  // Memory usage percentage
  const memoryUsagePercent = (usedMemoryBytes / availableVramBytes) * 100;

  // Calculate total batched tokens (min of max_batched_tokens and capacity)
  const capacityBatchedTokens = actualMaxNumSeqs * tokensPerSeq;
  const totalBatchedTokens = Math.min(maxBatchedTokens, capacityBatchedTokens);

  // Warnings
  const warnings: string[] = [];
  if (actualMaxNumSeqs < maxNumSeqs) {
    warnings.push(
      `KV cache can only fit ${actualMaxNumSeqs} sequences (you requested ${maxNumSeqs}). ` +
      `Consider reducing max_model_len or increasing GPU memory.`
    );
  }
  if (memoryUsagePercent > 95) {
    warnings.push('Memory usage is very high (>95%). Consider reducing batch size or context length.');
  }
  if (kvHeadsPerGpu * numGpus > kvHeads) {
    warnings.push(
      `With ${numGpus} GPUs, some GPUs will have ${kvHeadsPerGpu} KV heads while the model has ${kvHeads}. ` +
      `This may cause slight imbalance.`
    );
  }

  // Generate vLLM command
  const command = generateVLLMCommand({
    gpu,
    model,
    vllm,
    actualMaxNumSeqs,
    totalBatchedTokens,
  });

  return {
    availableVramPerGpu: availableVramGB,
    weightsPerGpu: weightsPerGpuGB,
    cudaGraphsMemory: cudaGraphsGB,
    overheadMemory: overheadPaddingGB,
    kvBytesPerToken: bytesPerToken,
    kvBytesPerSeq: bytesPerSeq,
    totalKVCacheMemory: totalKVCacheGB,
    maxTokensForKV,
    maxConcurrentSequences: actualMaxNumSeqs,
    totalBatchedTokens,
    freeMemory: freeMemoryGB,
    memoryUsagePercent,
    isOverCapacity: false,
    warnings,
    command,
  };
}

function generateVLLMCommand(params: {
  gpu: GPUConfig;
  model: ModelConfig;
  vllm: VLLMConfig;
  actualMaxNumSeqs: number;
  totalBatchedTokens: number;
}): string {
  const { gpu, model, vllm, actualMaxNumSeqs, totalBatchedTokens } = params;

  let cmd = 'vllm serve';

  if (model.name) {
    cmd += ` <span class="command-model">${model.name}</span>`;
  } else {
    cmd += ' <span class="command-placeholder">&lt;model-name&gt;</span>';
  }

  cmd += ` \\<br>&nbsp;&nbsp;--max-model-len <span class="command-value">${vllm.maxModelLen}</span>`;
  cmd += ` \\<br>&nbsp;&nbsp;--max-num-seqs <span class="command-value">${actualMaxNumSeqs}</span>`;

  if (totalBatchedTokens !== vllm.maxBatchedTokens) {
    cmd += ` \\<br>&nbsp;&nbsp;--max-num-batched-tokens <span class="command-value">${totalBatchedTokens}</span>`;
  }

  if (vllm.kvCacheDtype !== 'auto') {
    cmd += ` \\<br>&nbsp;&nbsp;--kv-cache-dtype <span class="command-value">${vllm.kvCacheDtype}</span>`;
  }

  if (gpu.numGpus > 1) {
    cmd += ` \\<br>&nbsp;&nbsp;--tensor-parallel-size <span class="command-value">${gpu.numGpus}</span>`;
  }

  if (!vllm.cudaGraphs) {
    cmd += ` \\<br>&nbsp;&nbsp;--enforce-eager`;
  }

  cmd += ` \\<br>&nbsp;&nbsp;--gpu-memory-utilization <span class="command-value">${gpu.utilization.toFixed(2)}</span>`;

  return cmd;
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;

  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      func(...args);
    };

    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(later, wait);
  };
}

export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return '0 B';

  const k = 1000; // Decimal (not binary)
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

export function formatNumber(num: number, decimals: number = 2): string {
  return num.toFixed(decimals);
}

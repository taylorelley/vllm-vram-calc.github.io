export interface GPUConfig {
  vram: number;
  numGpus: number;
  utilization: number;
}

export interface ModelConfig {
  weightsGB: number;
  numLayers: number;
  kvHeads: number;
  headDim: number;
  attnHeads: number;
  name?: string;
  maxContextLength?: number;
}

export interface QuantizationConfig {
  method: string;
  bits: number;
  baseParams: number;
  groupSize: number;
}

export interface VLLMConfig {
  maxModelLen: number;
  maxNumSeqs: number;
  maxBatchedTokens: number;
  kvCacheDtype: string;
  activationDtype: string;
  cudaGraphs: boolean;
  overheadPadding: number;
}

export interface CalculationResult {
  // Per GPU
  availableVramPerGpu: number;
  weightsPerGpu: number;
  cudaGraphsMemory: number;
  overheadMemory: number;

  // KV Cache
  kvBytesPerToken: number;
  kvBytesPerSeq: number;
  totalKVCacheMemory: number;
  maxTokensForKV: number;

  // Capacity
  maxConcurrentSequences: number;
  totalBatchedTokens: number;

  // Status
  freeMemory: number;
  memoryUsagePercent: number;
  isOverCapacity: boolean;
  warnings: string[];

  // Command
  command: string;
}

export interface HuggingFaceModelInfo {
  modelId: string;
  name?: string;
  weightsGB?: number;
  numLayers?: number;
  kvHeads?: number;
  headDim?: number;
  attnHeads?: number;
  maxContextLength?: number;
  quantMethod?: string;
  quantBits?: number;
  baseParams?: number;
}

export interface CachedModelData {
  data: {
    modelInfo: any;
    config: any;
  };
  timestamp: number;
}

export interface SavedConfiguration {
  gpu: GPUConfig;
  model: ModelConfig;
  quant: QuantizationConfig;
  vllm: VLLMConfig;
  timestamp: number;
}

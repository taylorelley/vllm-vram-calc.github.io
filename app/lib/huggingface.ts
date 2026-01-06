import type { HuggingFaceModelInfo, CachedModelData } from './types';

const HF_CACHE_PREFIX = 'vllm_calc_hf_';
const HF_CACHE_TTL = 7 * 24 * 60 * 60 * 1000; // 7 days

export function getCachedModel(modelId: string): CachedModelData['data'] | null {
  if (typeof window === 'undefined') return null;

  try {
    const cacheKey = HF_CACHE_PREFIX + modelId;
    const cached = localStorage.getItem(cacheKey);
    if (!cached) return null;

    const { data, timestamp }: CachedModelData = JSON.parse(cached);
    const age = Date.now() - timestamp;

    if (age > HF_CACHE_TTL) {
      localStorage.removeItem(cacheKey);
      return null;
    }

    return data;
  } catch (error) {
    console.error('Cache read error:', error);
    return null;
  }
}

export function cacheModel(modelId: string, data: any): void {
  if (typeof window === 'undefined') return;

  try {
    const cacheKey = HF_CACHE_PREFIX + modelId;
    const cacheData: CachedModelData = {
      data,
      timestamp: Date.now(),
    };
    localStorage.setItem(cacheKey, JSON.stringify(cacheData));

    // Clean old cache entries
    cleanOldCache();
  } catch (error) {
    console.error('Cache write error:', error);
  }
}

function cleanOldCache(): void {
  if (typeof window === 'undefined') return;

  try {
    const cacheEntries: Array<{ key: string; timestamp: number }> = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith(HF_CACHE_PREFIX)) {
        const item = JSON.parse(localStorage.getItem(key)!);
        cacheEntries.push({ key, timestamp: item.timestamp });
      }
    }

    // Sort by timestamp descending
    cacheEntries.sort((a, b) => b.timestamp - a.timestamp);

    // Remove old entries beyond 50
    for (let i = 50; i < cacheEntries.length; i++) {
      localStorage.removeItem(cacheEntries[i].key);
    }
  } catch (error) {
    console.error('Cache cleanup error:', error);
  }
}

export async function fetchHuggingFaceModel(modelId: string): Promise<{
  modelInfo: any;
  config: any;
}> {
  modelId = modelId.trim();
  if (!modelId) {
    throw new Error('Please enter a model ID');
  }

  // Create abort controller for timeout (10 seconds)
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10000);

  try {
    const modelInfoUrl = `https://huggingface.co/api/models/${modelId}`;
    const configUrl = `https://huggingface.co/${modelId}/resolve/main/config.json`;

    // Fetch both in parallel with timeout
    const [modelInfoRes, configRes] = await Promise.all([
      fetch(modelInfoUrl, { signal: controller.signal }),
      fetch(configUrl, { signal: controller.signal }),
    ]);

    clearTimeout(timeoutId);

    if (!modelInfoRes.ok) {
      if (modelInfoRes.status === 404) {
        throw new Error(`Model "${modelId}" not found. Please check the spelling and try again.`);
      } else if (modelInfoRes.status === 403) {
        throw new Error(`Model "${modelId}" is private or gated. You may need to authenticate.`);
      } else {
        throw new Error(`Failed to fetch model (HTTP ${modelInfoRes.status})`);
      }
    }

    const modelInfo = await modelInfoRes.json();

    let config = null;
    if (configRes.ok) {
      config = await configRes.json();
    }

    return { modelInfo, config };
  } catch (error: any) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error('Request timed out. Please check your connection and try again.');
    }
    throw error;
  }
}

export function extractModelConfig(modelInfo: any, config: any): HuggingFaceModelInfo {
  const result: HuggingFaceModelInfo = {
    modelId: modelInfo.id || modelInfo.modelId,
    name: modelInfo.id || modelInfo.modelId,
  };

  if (!config) {
    return result;
  }

  // Extract size
  if (modelInfo.safetensors?.total) {
    result.weightsGB = modelInfo.safetensors.total / 1_000_000_000;
  }

  // Extract architecture details
  result.numLayers =
    config.num_hidden_layers || config.n_layer || config.num_layers || config.n_layers;

  result.kvHeads =
    config.num_key_value_heads || config.num_kv_heads || config.kv_heads || config.num_attention_heads;

  result.attnHeads = config.num_attention_heads || config.n_head || config.num_heads;

  result.headDim = config.head_dim || (config.hidden_size && result.attnHeads
    ? config.hidden_size / result.attnHeads
    : undefined);

  result.maxContextLength =
    config.max_position_embeddings || config.max_seq_len || config.n_positions || config.model_max_length;

  // Extract quantization info
  if (config.quantization_config) {
    const qconfig = config.quantization_config;
    result.quantMethod = qconfig.quant_method || qconfig.method;
    result.quantBits = qconfig.bits || qconfig.w_bit;
    result.baseParams = config.num_parameters;
  }

  return result;
}

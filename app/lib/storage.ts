import type { SavedConfiguration, GPUConfig, ModelConfig, QuantizationConfig, VLLMConfig } from './types';

const CONFIG_STORAGE_KEY = 'vllm_calc_config_v1';
const CONFIG_TTL = 30 * 24 * 60 * 60 * 1000; // 30 days

export function saveConfiguration(
  gpu: GPUConfig,
  model: ModelConfig,
  quant: QuantizationConfig,
  vllm: VLLMConfig
): void {
  if (typeof window === 'undefined') return;

  try {
    const config: SavedConfiguration = {
      gpu,
      model,
      quant,
      vllm,
      timestamp: Date.now(),
    };

    localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(config));
  } catch (error) {
    console.error('Failed to save configuration:', error);
  }
}

export function loadConfiguration(): SavedConfiguration | null {
  if (typeof window === 'undefined') return null;

  try {
    const saved = localStorage.getItem(CONFIG_STORAGE_KEY);
    if (!saved) return null;

    const config: SavedConfiguration = JSON.parse(saved);

    // Check if saved config is recent (within 30 days)
    const age = Date.now() - (config.timestamp || 0);
    if (age > CONFIG_TTL) {
      localStorage.removeItem(CONFIG_STORAGE_KEY);
      return null;
    }

    return config;
  } catch (error) {
    console.error('Failed to load configuration:', error);
    return null;
  }
}

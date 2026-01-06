'use client';

import { GPUConfig } from '../lib/types';
import { GPU_PRESETS } from '../lib/constants';
import styles from './GPUConfig.module.css';

interface GPUConfigProps {
  config: GPUConfig;
  onChange: (config: GPUConfig) => void;
}

export default function GPUConfigComponent({ config, onChange }: GPUConfigProps) {
  const handlePreset = (vram: number) => {
    onChange({ ...config, vram });
  };

  return (
    <div className={styles.panel} role="region" aria-labelledby="gpu-config-title">
      <div className={styles.panelHeader}>
        <div className={styles.panelIcon} aria-hidden="true">
          🖥
        </div>
        <h2 className={styles.panelTitle} id="gpu-config-title">
          GPU Configuration
        </h2>
      </div>

      <div className={styles.formGroup}>
        <label htmlFor="gpu-preset">GPU Preset</label>
        <div className={styles.presetGroup}>
          <div className={styles.presetGroupLabel}>Consumer GPUs</div>
          <div className={styles.presetButtons}>
            {GPU_PRESETS.consumer.map((gpu) => (
              <button
                key={gpu.name}
                type="button"
                className={styles.presetBtn}
                onClick={() => handlePreset(gpu.vram)}
                aria-label={`Set GPU to ${gpu.name}`}
              >
                {gpu.name}
              </button>
            ))}
          </div>
        </div>
        <div className={styles.presetGroup}>
          <div className={styles.presetGroupLabel}>Professional GPUs</div>
          <div className={styles.presetButtons}>
            {GPU_PRESETS.professional.map((gpu) => (
              <button
                key={gpu.name}
                type="button"
                className={styles.presetBtn}
                onClick={() => handlePreset(gpu.vram)}
                aria-label={`Set GPU to ${gpu.name}`}
              >
                {gpu.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className={styles.inputRow}>
        <div className={styles.formGroup}>
          <label htmlFor="gpu-vram">VRAM per GPU (GB)</label>
          <input
            type="number"
            id="gpu-vram"
            value={config.vram}
            onChange={(e) => onChange({ ...config, vram: parseFloat(e.target.value) || 0 })}
            min="1"
            max="192"
            step="0.1"
            aria-label="VRAM per GPU in decimal gigabytes"
            aria-describedby="gpu-vram-hint"
          />
          <p className="hint" id="gpu-vram-hint">
            1 GB = 1,000³ bytes (decimal)
          </p>
        </div>
        <div className={styles.formGroup}>
          <label htmlFor="num-gpus">Number of GPUs (TP size)</label>
          <input
            type="number"
            id="num-gpus"
            value={config.numGpus}
            onChange={(e) => onChange({ ...config, numGpus: parseInt(e.target.value) || 1 })}
            min="1"
            max="8"
            step="1"
            aria-label="Number of GPUs for tensor parallelism"
          />
        </div>
      </div>

      <div className={styles.formGroup}>
        <label htmlFor="gpu-utilization">GPU Memory Utilization</label>
        <input
          type="number"
          id="gpu-utilization"
          value={config.utilization}
          onChange={(e) => onChange({ ...config, utilization: parseFloat(e.target.value) || 0 })}
          min="0.5"
          max="0.99"
          step="0.01"
          aria-label="GPU memory utilization ratio"
          aria-describedby="gpu-util-hint"
        />
        <p className="hint" id="gpu-util-hint">
          Typically 0.85-0.95. Lower = more headroom for spikes
        </p>
      </div>
    </div>
  );
}

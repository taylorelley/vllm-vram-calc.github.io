// Verification script for HyperNova-60B calculator accuracy
// Run with: node verify_calculation.js

const config = {
  // GPU Config
  gpuVram: 34.2,          // GB (RTX 5090: 32607 MiB = 34.2 GB decimal)
  numGpus: 2,
  gpuUtilization: 0.90,

  // Model Config
  modelWeights: 36.3,     // GB total (16.91 GiB per GPU × 2)
  numLayers: 32,
  kvHeads: 8,
  headDim: 64,
  attnHeads: 64,

  // vLLM Config
  maxModelLen: 131072,
  maxNumSeqs: 8,
  maxBatchedTokens: 65536,
  kvCacheDtypeBytes: 2,   // FP16/BF16
  activationDtypeBytes: 2,
  cudaGraphsEnabled: true,
  overheadPadding: 1.0    // GB (new default)
};

console.log('='.repeat(80));
console.log('HyperNova-60B vLLM VRAM Calculator Verification');
console.log('='.repeat(80));

// Calculate per-GPU values
const availableVram = config.gpuVram * config.gpuUtilization;
const weightsPerGpu = config.modelWeights / config.numGpus;

console.log('\n📊 Per-GPU Memory Breakdown:');
console.log(`  Total VRAM:              ${config.gpuVram.toFixed(2)} GB`);
console.log(`  Available (${(config.gpuUtilization * 100).toFixed(0)}%):        ${availableVram.toFixed(2)} GB`);
console.log(`  Model Weights:           ${weightsPerGpu.toFixed(2)} GB`);

// Activation overhead
const attnHeadsPerGpu = Math.ceil(config.attnHeads / config.numGpus);
const hiddenSizePerGpu = attnHeadsPerGpu * config.headDim;
const prefillTokens = config.maxBatchedTokens;
const decodeTokens = config.maxNumSeqs;
const activationTokens = prefillTokens + decodeTokens;
const activationBytes = activationTokens * hiddenSizePerGpu * config.activationDtypeBytes;
const overheadEstimate = (activationBytes * 2) / 1e9;

console.log(`  Activation Overhead:     ${overheadEstimate.toFixed(3)} GB`);
console.log(`    (${activationTokens.toLocaleString()} tokens × ${hiddenSizePerGpu} hidden × ${config.activationDtypeBytes} bytes × 2 buffers)`);

// CUDA graphs (NEW CALCULATION)
const cudaGraphMultiplier = 10;
const cudaGraphsSize = config.cudaGraphsEnabled ? (activationBytes / 1e9) * cudaGraphMultiplier : 0;

console.log(`  CUDA Graphs:             ${cudaGraphsSize.toFixed(3)} GB`);
console.log(`    (${(activationBytes / 1e9).toFixed(3)} GB × ${cudaGraphMultiplier} for multi-graph capture)`);

// Total overhead
const totalOverhead = overheadEstimate + config.overheadPadding;

console.log(`  Extra Overhead:          ${config.overheadPadding.toFixed(3)} GB`);
console.log(`  Total Fixed Overhead:    ${(weightsPerGpu + cudaGraphsSize + totalOverhead).toFixed(3)} GB`);

// KV cache calculation
const kvHeadsPerGpu = Math.ceil(config.kvHeads / config.numGpus);
const bytesPerTokenPerLayer = 2 * kvHeadsPerGpu * config.headDim * config.kvCacheDtypeBytes;
const bytesPerToken = bytesPerTokenPerLayer * config.numLayers;

console.log('\n🧮 KV Cache Calculation:');
console.log(`  KV Heads per GPU:        ${kvHeadsPerGpu}`);
console.log(`  Bytes per token/layer:   ${bytesPerTokenPerLayer.toLocaleString()} bytes`);
console.log(`  Bytes per token:         ${bytesPerToken.toLocaleString()} bytes (${(bytesPerToken / 1024).toFixed(1)} KB)`);

// Available for KV cache
const fixedOverhead = weightsPerGpu + cudaGraphsSize + totalOverhead;
const kvAvailable = Math.max(0, availableVram - fixedOverhead);
const kvAvailableBytes = kvAvailable * 1e9;

console.log(`  Available for KV:        ${kvAvailable.toFixed(2)} GB`);

// Max tokens
const maxTokensInCache = Math.floor(kvAvailableBytes / bytesPerToken);

console.log(`  Max Tokens in Cache:     ${maxTokensInCache.toLocaleString()} tokens`);

// Comparison with real-world
const realKvAvailableGiB = 7.69;
const realKvAvailableGB = realKvAvailableGiB * 1.073741824;
const realMaxTokens = 251728;

console.log('\n✅ Real-World Comparison:');
console.log(`  Real KV Available:       ${realKvAvailableGB.toFixed(2)} GB (${realKvAvailableGiB} GiB)`);
console.log(`  Calculator KV Available: ${kvAvailable.toFixed(2)} GB`);
console.log(`  Difference:              ${Math.abs(kvAvailable - realKvAvailableGB).toFixed(2)} GB (${((Math.abs(kvAvailable - realKvAvailableGB) / realKvAvailableGB) * 100).toFixed(1)}%)`);

console.log(`\n  Real Max Tokens:         ${realMaxTokens.toLocaleString()}`);
console.log(`  Calculator Max Tokens:   ${maxTokensInCache.toLocaleString()}`);
console.log(`  Difference:              ${Math.abs(maxTokensInCache - realMaxTokens).toLocaleString()} tokens (${((Math.abs(maxTokensInCache - realMaxTokens) / realMaxTokens) * 100).toFixed(2)}%)`);

// Final verdict
const kvError = Math.abs(kvAvailable - realKvAvailableGB);
const tokenError = Math.abs(maxTokensInCache - realMaxTokens);
const tokenErrorPct = (tokenError / realMaxTokens) * 100;

console.log('\n' + '='.repeat(80));
if (tokenErrorPct < 2.0 && kvError < 0.5) {
  console.log('✅ CALCULATOR IS ACCURATE - Within acceptable tolerances!');
} else if (tokenErrorPct < 5.0 && kvError < 1.0) {
  console.log('⚠️  CALCULATOR IS ACCEPTABLE - Minor adjustments may be needed');
} else {
  console.log('❌ CALCULATOR NEEDS IMPROVEMENT - Significant discrepancies');
}
console.log('='.repeat(80));

// Memory usage estimate
const estimatedKvUsage = (Math.min(activationTokens, maxTokensInCache) * bytesPerToken) / 1e9;
const totalUsage = weightsPerGpu + estimatedKvUsage + cudaGraphsSize + totalOverhead;

console.log('\n📈 Estimated Total Usage:');
console.log(`  Weights:                 ${weightsPerGpu.toFixed(2)} GB`);
console.log(`  KV Cache (active):       ${estimatedKvUsage.toFixed(2)} GB`);
console.log(`  CUDA Graphs:             ${cudaGraphsSize.toFixed(2)} GB`);
console.log(`  Overhead:                ${totalOverhead.toFixed(2)} GB`);
console.log(`  Total:                   ${totalUsage.toFixed(2)} GB (${((totalUsage / config.gpuVram) * 100).toFixed(1)}% of ${config.gpuVram} GB)`);

const realUsageGiB = 28.416;
const realUsageGB = realUsageGiB * 1.073741824;
console.log(`\n  Real Usage:              ${realUsageGB.toFixed(2)} GB (${realUsageGiB} GiB from nvidia-smi)`);
console.log(`  Difference:              ${Math.abs(totalUsage - realUsageGB).toFixed(2)} GB`);

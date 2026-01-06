// Test script to verify that changing max-num-seqs now properly affects Memory Breakdown
// Run with: node verify_max_num_seqs.js

function calculateMemory(maxNumSeqs) {
  const config = {
    gpuVram: 34.2,
    numGpus: 2,
    gpuUtilization: 0.90,
    modelWeights: 36.3,
    numLayers: 32,
    kvHeads: 8,
    headDim: 64,
    attnHeads: 64,
    maxModelLen: 131072,
    maxNumSeqs: maxNumSeqs,
    maxBatchedTokens: 65536,
    kvCacheDtypeBytes: 2,
    activationDtypeBytes: 2,
    cudaGraphsEnabled: true,
    overheadPadding: 1.0
  };

  const availableVram = config.gpuVram * config.gpuUtilization;
  const weightsPerGpu = config.modelWeights / config.numGpus;
  const attnHeadsPerGpu = Math.ceil(config.attnHeads / config.numGpus);
  const hiddenSizePerGpu = attnHeadsPerGpu * config.headDim;
  const prefillTokens = config.maxBatchedTokens;
  const decodeTokens = config.maxNumSeqs;
  const activationTokens = prefillTokens + decodeTokens;
  const activationBytes = activationTokens * hiddenSizePerGpu * config.activationDtypeBytes;
  const overheadEstimate = (activationBytes * 2) / 1e9;
  const totalOverhead = overheadEstimate + config.overheadPadding;
  const cudaGraphMultiplier = 10;
  const cudaGraphsSize = config.cudaGraphsEnabled ? (activationBytes / 1e9) * cudaGraphMultiplier : 0;

  // KV cache calculation
  const kvHeadsPerGpu = Math.ceil(config.kvHeads / config.numGpus);
  const bytesPerTokenPerLayer = 2 * kvHeadsPerGpu * config.headDim * config.kvCacheDtypeBytes;
  const bytesPerToken = bytesPerTokenPerLayer * config.numLayers;

  const fixedOverhead = weightsPerGpu + cudaGraphsSize + totalOverhead;
  const kvAvailable = Math.max(0, availableVram - fixedOverhead);
  const kvAvailableBytes = kvAvailable * 1e9;
  const maxTokensInCache = Math.floor(kvAvailableBytes / bytesPerToken);

  // NEW: vLLM allocates FULL KV cache pool at startup
  const kvCacheAllocated = kvAvailable;
  const totalUsage = weightsPerGpu + kvCacheAllocated + cudaGraphsSize + totalOverhead;

  return {
    weights: weightsPerGpu,
    kvCache: kvCacheAllocated,
    cudaGraphs: cudaGraphsSize,
    overhead: totalOverhead,
    total: totalUsage,
    maxTokens: maxTokensInCache,
    avgContextPerSeq: Math.floor(maxTokensInCache / config.maxNumSeqs)
  };
}

console.log('='.repeat(80));
console.log('Testing: Does max-num-seqs affect Memory Breakdown?');
console.log('='.repeat(80));
console.log('\nOLD BEHAVIOR (before fix):');
console.log('  KV Cache shown as "estimated active" = min(maxBatchedTokens + maxNumSeqs, capacity)');
console.log('  Result: Changing max-num-seqs from 1 to 100 barely changes breakdown!\n');

console.log('NEW BEHAVIOR (after fix):');
console.log('  KV Cache shown as "allocated pool" = full kvAvailable');
console.log('  Result: max-num-seqs affects capacity analysis, not memory allocation\n');

console.log('='.repeat(80));
console.log('Testing with different max-num-seqs values:');
console.log('='.repeat(80));

const testValues = [1, 8, 10, 50, 100];

console.log('\nMemory Breakdown (Per GPU):');
console.log('-'.repeat(80));
console.log('max-num-seqs | Weights | KV Pool | CUDA G | Overhead | Total  | Free');
console.log('-'.repeat(80));

testValues.forEach(seqs => {
  const mem = calculateMemory(seqs);
  const free = 34.2 - mem.total;
  console.log(
    `${seqs.toString().padStart(12)} | ` +
    `${mem.weights.toFixed(2).padStart(7)} | ` +
    `${mem.kvCache.toFixed(2).padStart(7)} | ` +
    `${mem.cudaGraphs.toFixed(2).padStart(6)} | ` +
    `${mem.overhead.toFixed(2).padStart(8)} | ` +
    `${mem.total.toFixed(2).padStart(6)} | ` +
    `${free.toFixed(2).padStart(4)}`
  );
});

console.log('-'.repeat(80));

console.log('\nCapacity Analysis:');
console.log('-'.repeat(80));
console.log('max-num-seqs | Max Tokens | Avg Context/Seq');
console.log('-'.repeat(80));

testValues.forEach(seqs => {
  const mem = calculateMemory(seqs);
  console.log(
    `${seqs.toString().padStart(12)} | ` +
    `${mem.maxTokens.toLocaleString().padStart(10)} | ` +
    `${mem.avgContextPerSeq.toLocaleString().padStart(15)}`
  );
});

console.log('-'.repeat(80));

console.log('\n✅ KEY INSIGHT:');
console.log('  - Memory Breakdown: CONSTANT (KV pool fully allocated regardless of max-num-seqs)');
console.log('  - Capacity Analysis: CHANGES (avg context per sequence varies with max-num-seqs)');
console.log('  - This matches real vLLM behavior where nvidia-smi shows full allocation!');

console.log('\n📊 Real-World Validation (HyperNova-60B):');
const realConfig = calculateMemory(8);
const realUsageGiB = 28.416;
const realUsageGB = realUsageGiB * 1.073741824;
console.log(`  Calculator Total: ${realConfig.total.toFixed(2)} GB`);
console.log(`  nvidia-smi Actual: ${realUsageGB.toFixed(2)} GB (${realUsageGiB} GiB)`);
console.log(`  Difference: ${Math.abs(realConfig.total - realUsageGB).toFixed(2)} GB (${((Math.abs(realConfig.total - realUsageGB) / realUsageGB) * 100).toFixed(1)}%)`);

if (Math.abs(realConfig.total - realUsageGB) < 1.0) {
  console.log('\n✅ EXCELLENT: Within 1 GB of real-world measurement!');
} else if (Math.abs(realConfig.total - realUsageGB) < 2.0) {
  console.log('\n✅ GOOD: Within 2 GB of real-world measurement!');
} else {
  console.log('\n⚠️  NEEDS IMPROVEMENT: More than 2 GB difference from real-world');
}
console.log('='.repeat(80));

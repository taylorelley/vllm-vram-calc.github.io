# TODO

Development roadmap and planned improvements for vLLM VRAM Calculator.

## High Priority

### Features
- [ ] **Save/Load Configurations**
  - Export configuration as JSON
  - Import saved configurations
  - URL parameter support for sharing configurations
  - LocalStorage for auto-saving last used configuration

- [ ] **Multi-Node Support**
  - Pipeline parallelism calculations
  - Cross-node tensor parallelism
  - Network bandwidth considerations

- [ ] **Advanced KV Cache Modes**
  - PagedAttention block calculations
  - Prefix caching overhead estimation
  - Sliding window attention support

### Bug Fixes
- [ ] **Validation**
  - Add input validation for all numeric fields
  - Warn when max-batched-tokens > max-model-len
  - Validate TP size doesn't exceed KV heads
  - Check for unrealistic configurations (e.g., 1000B model on 24GB GPU)

- [ ] **HuggingFace Integration**
  - Better error handling for private/gated models
  - Support for models with non-standard config.json
  - Fallback to model card parsing when config unavailable
  - Cache fetched configs (with TTL)

## Medium Priority

### UI/UX Improvements
- [ ] **Dark/Light Mode Toggle**
  - Add theme switcher
  - Respect system preferences
  - Persist user choice

- [ ] **Responsive Design**
  - Improve mobile layout
  - Better tablet experience
  - Touch-friendly controls

- [ ] **Accessibility**
  - Add ARIA labels
  - Keyboard navigation
  - Screen reader support
  - Color-blind friendly palette

- [ ] **Interactive Help**
  - Tooltips for technical terms
  - Inline documentation
  - Example scenarios
  - Video walkthrough

### Calculations
- [ ] **Mixture of Experts (MoE) Support**
  - Add MoE-specific parameters (num_experts, experts_per_token)
  - Calculate active parameter vs total parameter overhead
  - Expert parallelism strategies

- [ ] **Speculative Decoding**
  - Draft model memory requirements
  - Combined memory calculation
  - Acceptance rate impact

- [ ] **LoRA Adapters**
  - Adapter memory overhead
  - Multi-adapter capacity
  - Dynamic adapter loading

- [ ] **Batch Size Optimization**
  - Suggest optimal batch sizes for throughput
  - Latency vs throughput tradeoffs
  - Dynamic batching simulation

### Command Generation
- [ ] **Copy to Clipboard Button**
  - One-click command copying
  - Copy individual flags
  - Success feedback

- [ ] **Docker Command Support**
  - Generate Docker run commands
  - Environment variable format
  - Kubernetes YAML export

- [ ] **Configuration File Export**
  - Export as vLLM config YAML/JSON
  - Ray Serve deployment configs
  - Skypilot task YAML

### Data & Presets
- [ ] **Expanded Model Presets**
  - Add more popular models (Mistral, Gemma, Phi, etc.)
  - Organize by model family
  - Filter by size/type
  - Community-contributed presets

- [ ] **GPU Database**
  - Comprehensive GPU specs
  - Power consumption estimates
  - Cost per hour (cloud providers)
  - Availability indicators

## Low Priority

### Analytics & Monitoring
- [ ] **Cost Estimation**
  - Calculate cloud GPU costs
  - Compare different GPU configurations
  - TCO analysis (on-prem vs cloud)

- [ ] **Performance Predictions**
  - Estimated tokens/second
  - Latency estimates
  - Throughput projections
  - Bottleneck identification

- [ ] **Comparison Mode**
  - Side-by-side configuration comparison
  - Diff view for changes
  - Optimization suggestions

### Advanced Features
- [ ] **Batch Processing**
  - Upload multiple models for comparison
  - CSV export of results
  - Bulk optimization

- [ ] **API Mode**
  - REST API for programmatic access
  - Webhook support
  - Batch job processing

- [ ] **Visualization Enhancements**
  - Interactive memory timeline
  - 3D memory visualization
  - Animated transitions
  - Export charts as images

- [ ] **Integration with vLLM**
  - Direct integration with vLLM profiler
  - Real-time memory monitoring
  - Live configuration adjustment

### Code Quality
- [ ] **Refactoring**
  - Split monolithic HTML into separate files
  - Extract JavaScript into modules
  - CSS organization and theming
  - TypeScript migration for type safety

- [ ] **Testing**
  - Unit tests for calculation functions
  - Integration tests for HF API
  - Visual regression tests
  - Cross-browser testing

- [ ] **Documentation**
  - JSDoc comments
  - Architecture documentation
  - Calculation methodology docs
  - API documentation (if API added)

- [ ] **Build System**
  - Optional build step for optimization
  - Minification and bundling
  - Asset optimization
  - CI/CD pipeline

## Future Considerations

### Community Features
- [ ] Share configurations with community
- [ ] User accounts and saved presets
- [ ] Community voting on presets
- [ ] Discussion forum integration

### Educational Content
- [ ] Interactive tutorials
- [ ] Best practices guide
- [ ] Common pitfall warnings
- [ ] Case studies

### Platform Support
- [ ] Desktop app (Electron)
- [ ] Mobile app (React Native)
- [ ] CLI version
- [ ] Browser extension

## Known Issues

- [ ] HuggingFace API rate limiting not handled
- [ ] Very large context windows (>1M) may show incorrect estimates
- [ ] KV cache estimate assumes uniform token distribution
- [ ] CUDA graphs size is approximate (varies by batch size)

## Ideas for Exploration

- Integration with Weights & Biases for experiment tracking
- Support for other inference engines (TensorRT-LLM, Text Generation Inference)
- Multi-model serving calculations (multiple models on same GPU)
- Flash Attention version detection and memory impact
- Automatic optimization solver (find cheapest config meeting requirements)
- Integration with model compression tools
- Real-time collaboration features
- Plugin system for custom calculations

## Completed

- [x] Initial single-page calculator
- [x] GPU and model presets
- [x] Tensor parallelism support
- [x] KV cache calculation
- [x] Memory visualization
- [x] Command generation
- [x] HuggingFace integration
- [x] Quantization estimation
- [x] Capacity analysis
- [x] Status indicators

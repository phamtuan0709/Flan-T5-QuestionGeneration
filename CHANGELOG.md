# Changelog

All notable changes to the Bloom-QG project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-10-24

### Added
- **LearningQ Dataset Support** (~230K educational samples)
  - `prepare_learningq.py` with ML-based answer extraction using SentenceTransformers
  - Extended Bloom annotation with 84 keywords across 6 cognitive levels
  - Multi-source data loading (Khan Academy, TED-Ed, experiments)
  - DPK-style processing pipeline with progress tracking
- **Smart Context Handling** for long video transcripts
  - Auto-detection of dataset type (SQuAD vs LearningQ)
  - Context truncation for sequences >384 tokens
  - Context length statistics logging
- **Enhanced Documentation**
  - Comprehensive README with LearningQ instructions
  - Updated Makefile with LearningQ commands
  - CHANGELOG.md for version tracking
- **Professional Project Structure**
  - Moved config files to root (README, requirements.txt, setup.py, etc.)
  - Created data/ and checkpoints/ directories at root
  - Created examples/ directory for sample data
  - Added .gitkeep files for empty directories
- **Testing Infrastructure**
  - tests/ directory structure
  - Unit tests for models, data processing, and integration
  - pytest configuration
- **Updated Dependencies**
  - `data-prep-kit>=0.2.0` for scalable data processing
  - `sentence-transformers>=2.2.0` for answer extraction
- **Enhanced Setup Verification**
  - Updated `verify_setup.py` to check new dependencies
  - Better error messages and warnings

### Changed
- **Default Dataset**: Changed from SQuAD (87K) to LearningQ (230K)
- **Dataset Class**: `BloomQGDataset` now supports both SQuAD and LearningQ formats
- **Training Script**: Default `--data_path` changed to `data/learningq_train.json`
- **Makefile**:
  - `make prepare` now prepares LearningQ instead of SQuAD
  - Added `make test` for running unit tests
  - Updated all training commands with new parameters
  - Added emojis for better readability

### Removed
- **SQuAD v1.1 Support** (deprecated in favor of LearningQ)
  - Removed `prepare_squad.py` script
  - Removed SQuAD references from documentation
  - Removed SQuAD-specific examples
- **Redundant Documentation Files**
  - Deleted `FILE_TREE.md` (364 lines)
  - Deleted `PROJECT_SUMMARY.md` (370 lines)
  - Deleted `INDEX.md` (280 lines)
  - Deleted `ARCHITECTURE.md` (~19KB)
  - Deleted `QUICKSTART.md` (~9KB)
  - Total reduction: ~15,000 lines of duplicate/outdated documentation

### Fixed
- Context length handling for educational video transcripts
- Dataset auto-detection logic
- Import paths after directory restructure

## [0.1.0] - 2025-10-23

### Added
- Initial release with hybrid BloomBERT + FLAN-T5 architecture
- **Neural Models**:
  - `BloomEncoder`: Frozen BloomBERT wrapper (768-dim output)
  - `FusionLayer`: 6 learnable prototypes + MLP (2.4M params)
  - `FlanGenerator`: FLAN-T5 with LoRA adapters (4.1M params)
  - `HybridModel`: Complete pipeline integration (6.5M trainable params)
- **Data Processing**:
  - SQuAD v1.1 annotation with Bloom heuristics
  - PyTorch Dataset with custom collate_fn
  - Keyword-based Bloom classification (~70-80% accuracy)
- **Training Infrastructure**:
  - GPU training with FP16 mixed precision
  - Gradient clipping and accumulation
  - Checkpoint saving (best + per-epoch)
  - Validation split and early stopping
- **Inference**:
  - MPS/CPU support for Apple Silicon
  - Interactive mode for single questions
  - Batch mode for multiple samples
  - Test mode with example files
- **Evaluation**:
  - BLEU (1-4 gram)
  - ROUGE (1, 2, L)
  - METEOR metrics
  - Per-Bloom-level statistics
- **Baseline**:
  - Pure FLAN-T5 implementation for comparison
- **Documentation**:
  - README with architecture diagrams
  - Installation instructions
  - Usage examples
  - Performance benchmarks
- **Configuration**:
  - requirements.txt with all dependencies
  - setup.py for package installation
  - Makefile with common commands
  - LICENSE (MIT)

### Notes
- Trained on SQuAD v1.1 (~87K samples)
- Expected training time: 8-10 hours on RTX 3090
- Requires CUDA 11.8+ or Apple Silicon with MPS

---

## Version History

- **0.2.0** (2025-10-24): LearningQ dataset migration, professional restructure
- **0.1.0** (2025-10-23): Initial release with SQuAD support

## Future Plans

### [0.3.0] - Planned
- [ ] Multi-GPU training support (DistributedDataParallel)
- [ ] ONNX export for inference optimization
- [ ] Fine-tuned Bloom classifier (replace keyword heuristics)
- [ ] Evaluation on external test sets
- [ ] Model quantization for edge deployment

### [0.4.0] - Planned
- [ ] REST API for inference
- [ ] Web UI demo
- [ ] Docker containerization
- [ ] Cloud deployment scripts (AWS/Azure/GCP)
- [ ] Continuous integration (GitHub Actions)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Links

- **Repository**: [GitHub](https://github.com/yourusername/bloom-qg)
- **Issues**: [Issue Tracker](https://github.com/yourusername/bloom-qg/issues)
- **Documentation**: [Docs](https://github.com/yourusername/bloom-qg/blob/main/README.md)

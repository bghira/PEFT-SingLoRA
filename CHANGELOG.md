# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.1] - 2025-07-21

### Changed
- New release

## [0.1.0] - 2024-XX-XX

### Added
- Initial implementation of SingLoRA layer
- PEFT framework compatibility
- Support for nn.Linear layers
- Ramp-up function for gradual adaptation
- Multi-adapter support
- Merge/unmerge functionality
- Comprehensive test suite
- Examples for basic usage and fine-tuning
- Full documentation

### Features
- 50% parameter reduction compared to standard LoRA
- Drop-in replacement for PEFT LoRA
- Automatic registration with `setup_singlora()`
- Compatible with existing PEFT workflows
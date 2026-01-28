# Documentation Index

Complete guide to all documentation in this project.

## Getting Started

1. **[QUICKSTART.md](QUICKSTART.md)** - Get running in 5 minutes
   - Step-by-step setup
   - First experiment
   - Common tasks

2. **[../README.md](../README.md)** - Complete reference guide
   - Full setup instructions
   - All commands and options
   - Configuration details
   - Troubleshooting

## Understanding Results

3. **[RESULTS_GUIDE.md](RESULTS_GUIDE.md)** - How to interpret results
   - Understanding metrics
   - Reading degradation curves
   - Robustness statistics
   - Example analyses

4. **[USAGE_SUMMARY.md](USAGE_SUMMARY.md)** - Quick command reference
   - Common commands
   - Output locations
   - Key files

## Project Documentation

5. **[WEEK3_DELIVERABLES.md](WEEK3_DELIVERABLES.md)** - Week 3 implementation details
   - Corruption modules
   - Pipeline architecture
   - Configuration schemas

6. **[CROSS_DOMAIN_NOTE.md](CROSS_DOMAIN_NOTE.md)** - Domain shift notes
   - Implementation approach
   - Differences from corruption

7. **[VERIFICATION_RESULTS.md](VERIFICATION_RESULTS.md)** - Test results
   - Verification status
   - Test coverage

## Code Structure

### CLI Tools (`src/cli/`)
- `run_baseline.py` - Baseline experiments
- `run_corruption.py` - Single corruption experiments
- `run_severity_grid.py` - Severity grid experiments
- `analyze_severity_grid.py` - Result analysis and visualization
- `summarize.py` - Result summarization

### Core Modules (`src/`)
- `corruptions/` - Corruption implementations
- `datasets/` - Dataset loaders
- `models/` - Model definitions
- `pipelines/` - Experiment pipelines
- `common/` - Shared utilities

## Quick Links

- **New to the project?** → Start with [QUICKSTART.md](QUICKSTART.md)
- **Running experiments?** → See [../README.md](../README.md) Running Experiments section
- **Interpreting results?** → Read [RESULTS_GUIDE.md](RESULTS_GUIDE.md)
- **Need a command?** → Check [USAGE_SUMMARY.md](USAGE_SUMMARY.md)
- **Understanding code?** → See [WEEK3_DELIVERABLES.md](WEEK3_DELIVERABLES.md)

## File Locations

### Configuration Files
- `configs/*.yaml` - Experiment configurations

### Output Files
- `outputs/runs/` - Single experiment results
- `outputs/severity_grids/` - Severity grid results
- `outputs/severity_grids/severity_grid_summary.yaml` - Grid summary

### Test Files
- `../tests/test_corruptions.py` - Corruption module tests
- `../tests/test_corruption_pipeline.py` - Pipeline integration tests

## Getting Help

1. Check the relevant documentation file above
2. Look at example configs in `../configs/`
3. Run tests to verify setup: `python tests/test_corruptions.py`
4. Check console output for error messages

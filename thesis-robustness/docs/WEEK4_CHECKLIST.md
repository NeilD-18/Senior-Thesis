# Week 4 Checklist: Pilot Experiments

## Pre-Flight Checks

- [ ] Virtual environment is activated
- [ ] All dependencies installed (`pip install -e .`)
- [ ] Datasets are available (Adult auto-downloads, IMDB/Airbnb local)
- [ ] Config files exist in `configs/` directory
- [ ] Output directories exist (`outputs/severity_grids/`)

## Adult Dataset Pilots

- [ ] **Additive Noise Pilot**
  - [ ] Run severity grid (0.0-1.0, 11 points)
  - [ ] Verify degradation curve looks reasonable
  - [ ] Check runtime (< 60 min)
  - [ ] Verify reproducibility (re-run with same seed)

- [ ] **Missingness Pilot**
  - [ ] Run severity grid
  - [ ] Verify missing value handling
  - [ ] Check degradation pattern

- [ ] **Class Imbalance Pilot**
  - [ ] Run severity grid
  - [ ] Verify imbalance ratio matches severity
  - [ ] Check that test set remains balanced

## IMDB Dataset Pilots

- [ ] **Token Dropout Pilot**
  - [ ] Run severity grid
  - [ ] Verify sparse matrix handling
  - [ ] Check degradation pattern
  - [ ] Verify TF-IDF format preserved

## Airbnb Dataset Pilots

- [ ] **Noise Pilot**
  - [ ] Run severity grid
  - [ ] Verify RMSE/MAE metrics
  - [ ] Check log-transformed target handling

- [ ] **Missingness Pilot**
  - [ ] Run severity grid
  - [ ] Verify missing value imputation
  - [ ] Check regression metrics

## Analysis & Validation

- [ ] Generate degradation curves for all pilots
- [ ] Document runtime for each experiment type
- [ ] Verify factor levels are appropriate
- [ ] Confirm experimental design with advisor
- [ ] Create pilot results summary document

## Deliverables

- [ ] Pilot results summary (`docs/WEEK4_PILOT_SUMMARY.md`)
- [ ] Degradation curves for all pilots
- [ ] Runtime estimates for full experiments
- [ ] Confirmed experimental design
- [ ] Validation decisions documented

## Notes

- Use smaller `n-points` (5-7) for initial pilots if runtime is a concern
- Focus on validating that corruption works correctly, not on perfect results
- Document any issues or unexpected behaviors
- Confirm severity ranges and spacing with advisor

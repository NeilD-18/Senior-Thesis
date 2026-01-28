# Cross-Domain Shift Implementation Note

## Methods Section 3.4.2

The Methods section describes cross-domain shift as:
- Train on IMDB
- Evaluate on Amazon without target-domain fine-tuning

## Current Implementation

The cross-domain shift is **not** implemented as a corruption module because it's fundamentally different:
- **Corruption**: Modifies training data with controlled noise
- **Domain shift**: Uses different datasets for train vs test

## Implementation Approach

To properly implement cross-domain shift per Methods 3.3:
1. Fit TF-IDF vectorizer on IMDB training data
2. Apply same vectorizer to:
   - IMDB test data
   - Amazon evaluation data

**Current status**: The Amazon loader (`src/datasets/amazon.py`) currently fits its own vectorizer. For the domain shift experiment, you'll need to:
- Save the IMDB vectorizer after fitting
- Load and apply it to Amazon data

This is a preprocessing/pipeline concern rather than a corruption module, and should be handled in Week 4 pilot experiments.

## Week 3 Scope

Week 3 deliverables focus on **synthetic corruptions**:
- ✅ Additive noise
- ✅ Missingness  
- ✅ Class imbalance
- ✅ Token dropout

Cross-domain shift will be handled as a separate experimental configuration in the full pipeline.

# Persistent Semantic Issues - Action Plan

## Current Configuration

I've enabled **debugging mode** in your API:

### Changes Made:
1. ✅ **Greedy Search ENABLED** (`USE_GREEDY_SEARCH = True`)
2. ✅ **Raw Features** (`APPLY_NORMALIZATION = False`)
3. ✅ **Vocab Verification** at startup
4. ✅ **I3D Case-Insensitive Loading**
5. ✅ **Linear Interpolation** for frames

---

## Immediate Next Steps

### 1. Start the Server

```bash
python main.py
```

### 2. Check Startup Logs Carefully

Look for these sections:

#### A. Vocabulary Verification
```
======================================================================
VOCAB VERIFICATION: First 100 Words
======================================================================
  [   0] -> '<PAD>' ***CRITICAL***
  [   1] -> '<SOS>' ***CRITICAL***
  [   2] -> '<EOS>' ***CRITICAL***
  [ 284] -> 'buy' ***CRITICAL***
  [1729] -> 'pour' ***CRITICAL***

Critical Index Verification:
  Index  284: Expected 'buy', Got '???' [OK or MISMATCH]
  Index 1729: Expected 'pour', Got '???' [OK or MISMATCH]
```

**If you see [MISMATCH]**: This is the problem! Vocab indices are wrong.

#### B. I3D Loading
```
Loading I3D weights from rgb_imagenet.pt...
Loaded 148 I3D weight tensors
```

**If you see warnings about missing keys**: I3D not loaded properly.

### 3. Upload Test Video

Watch for:

#### A. Feature Stats
```
RAW Features - Mean: 0.123456, Std: 0.234567
ZERO-NORMALIZATION: Using RAW features (no StandardScaler)
```

#### B. Greedy Search Output
```
======================================================================
GREEDY SEARCH (Temperature 0.1)
======================================================================
Step  0: Token    4 = 'i              ' (prob: 0.8523)
Step  1: Token   15 = 'want           ' (prob: 0.7234)
Step  2: Token   89 = 'water          ' (prob: 0.6891)
Step  3: Token    2 = '<EOS>          ' (prob: 0.9123)

Greedy Output: i want water
Average Confidence: 0.7918
```

---

## Interpreting Greedy Search Results

### ✅ GOOD Signs (Model is Working):
- Token probabilities > 0.5
- Words make semantic sense
- Coherent sentence
- Natural EOS token

### ❌ BAD Signs (Model Has Issues):
- Token probabilities < 0.3
- Random unrelated words
- "pour push was silk..." pattern continues
- No clear EOS

---

## If Issues Persist After Checking Logs

### Root Cause Analysis:

#### Scenario 1: Vocab Mismatch
**Symptom**: `[MISMATCH]` in vocab verification

**Cause**: Model was trained with different vocab.pkl

**Fix**: 
- Get the EXACT vocab.pkl used during training
- Or retrain model with current vocab.pkl

#### Scenario 2: Feature Distribution Mismatch
**Symptom**: Greedy search shows low probabilities (<0.3) for all tokens

**Cause**: Training used different normalization

**Try**:
```python
# In main.py line 772
APPLY_NORMALIZATION = True  # Try with StandardScaler
```

Then restart server and test again.

#### Scenario 3: Model Never Learned
**Symptom**: Random predictions regardless of input

**Cause**: Model didn't converge during training

**Evidence**:
- Training loss never went below 3.0
- Validation accuracy very low
- Model outputs same words for different signs

**Fix**: Model needs retraining

---

## Diagnostic Script

Run this to check all components:

```bash
python diagnose_semantic_issues.py
```

This checks:
- Vocabulary mappings
- Model-vocab size alignment
- I3D weights
- Feature extraction

---

## Configuration Matrix to Try

If greedy search still shows bad output, try these combinations:

### Config 1 (Current - RAW + Greedy):
```python
APPLY_NORMALIZATION = False
USE_GREEDY_SEARCH = True
```

### Config 2 (StandardScaler + Greedy):
```python
APPLY_NORMALIZATION = True
USE_GREEDY_SEARCH = True
```

### Config 3 (RAW + Beam Search):
```python
APPLY_NORMALIZATION = False
USE_GREEDY_SEARCH = False
```

### Config 4 (StandardScaler + Beam Search):
```python
APPLY_NORMALIZATION = True
USE_GREEDY_SEARCH = False
```

**Test each config** and note which produces best output.

---

## Critical Questions to Answer

Based on greedy search logs, answer these:

1. **What is the first token predicted?**
   - If wrong from Step 0, model has fundamental issues
   - If Step 0 correct but later wrong, decoding issue

2. **What are the token probabilities?**
   - >0.5 = confident (good)
   - 0.3-0.5 = uncertain (concerning)
   - <0.3 = random (bad)

3. **Do vocab indices match?**
   - Index 284 = "buy"? [OK/MISMATCH]
   - Index 1729 = "pour"? [OK/MISMATCH]

4. **Are I3D features non-zero?**
   - "Loaded 148 I3D weight tensors" = good
   - Warnings about missing keys = bad

---

## What to Share for Further Help

If issues persist, share:

1. **Startup logs** (vocab verification section)
2. **Greedy search output** (full step-by-step)
3. **Feature statistics** (RAW features mean/std)
4. **Training information**:
   - What vocab size was used during training?
   - What feature normalization was used?
   - What was the final training loss?

---

## Files Created for Debugging

1. **diagnose_semantic_issues.py** - Comprehensive diagnostic script
2. **TROUBLESHOOTING_GUIDE.md** - Detailed troubleshooting steps
3. **greedy_search.py** - Greedy decoder for debugging
4. **SEMANTIC_ALIGNMENT_FIX.md** - All alignment fixes documentation

---

## Summary

**Current Status**: Debugging mode enabled with greedy search

**Next Action**: 
1. Start server: `python main.py`
2. Check startup logs for vocab verification
3. Upload test video
4. Examine greedy search step-by-step output
5. Share results for further diagnosis

The greedy search output will tell us exactly what the model is predicting and why! 🔍

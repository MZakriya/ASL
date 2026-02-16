# Persistent Semantic Issues - Troubleshooting Guide

## Current Status
Issues persist even after implementing all alignment fixes. This suggests a deeper problem with the model or training data.

---

## Immediate Debugging Steps

### Step 1: Run Diagnostic Script

```bash
python diagnose_semantic_issues.py
```

This will check:
- ✓ Vocabulary mappings (index 284 = "buy", 1729 = "pour")
- ✓ Model-vocab size alignment
- ✓ I3D weights loaded correctly
- ✓ Feature extraction statistics
- ✓ Model architecture

**Look for**:
- `✗ MISMATCH` in vocab indices
- `✗ MISMATCH` in model-vocab sizes
- `⚠ WARNING` in feature statistics

---

### Step 2: Test with Greedy Search (Already Enabled)

```bash
python main.py
# Upload a test video
```

**Check startup logs for**:
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
  Index  284: Expected 'buy', Got 'buy' [OK]
  Index 1729: Expected 'pour', Got 'pour' [OK]
```

**Check prediction logs for**:
```
======================================================================
GREEDY SEARCH (Temperature 0.1)
======================================================================
Step  0: Token    4 = 'i              ' (prob: 0.8523)
Step  1: Token   15 = 'want           ' (prob: 0.7234)
Step  2: Token   89 = 'water          ' (prob: 0.6891)
```

---

## Common Root Causes

### 1. Model-Vocab Mismatch

**Symptom**: Model predicts index 1729 but vocab says it's "block" not "pour"

**Cause**: Model was trained with different vocab.pkl than current one

**Fix**:
- Get the EXACT vocab.pkl used during training
- Or retrain model with current vocab.pkl

---

### 2. Feature Distribution Mismatch

**Symptom**: Model outputs random words regardless of input

**Cause**: Training used different feature normalization than inference

**Solutions to try**:

**A. Try StandardScaler** (in `main.py` line 772):
```python
APPLY_NORMALIZATION = True  # Enable StandardScaler
```

**B. Check feature stats** in logs:
```
RAW Features - Mean: X, Std: Y, Min: Z, Max: W
```

Compare with training logs. If very different, normalization mismatch.

---

### 3. I3D Features Are Zeros

**Symptom**: Model ignores motion, only uses pose

**Cause**: I3D weights not loaded correctly

**Check logs for**:
```
Loading I3D weights from rgb_imagenet.pt...
Loaded 148 I3D weight tensors
[No "Missing keys" warnings]
```

**If warnings appear**:
```bash
python download_weights.py
# Or download manually
```

---

### 4. Model Was Trained Incorrectly

**Symptom**: Even greedy search with temp 0.1 produces nonsense

**Cause**: Model never learned proper sign-to-text mapping

**Evidence**:
- All token probabilities are low (<0.3)
- Predictions are random regardless of input
- Loss during training never went below 3.0

**Fix**: Model needs retraining with:
- Correct data preprocessing
- Proper vocabulary
- Sufficient training epochs
- Validated on held-out test set

---

### 5. Vocabulary Shift Error

**Symptom**: Predictions are semantically related but wrong words

**Example**: Model predicts "drink" when sign is "water"

**Cause**: Vocab indices shifted by N positions

**Check**:
```python
# In startup logs
Index  284: Expected 'buy', Got 'sell' [MISMATCH]
Index 1729: Expected 'pour', Got 'fill' [MISMATCH]
```

**Fix**: Regenerate vocab.pkl from training data with correct ordering

---

## Diagnostic Checklist

Run through this checklist:

### ✓ Vocabulary
- [ ] Index 0 = `<PAD>`
- [ ] Index 1 = `<SOS>`
- [ ] Index 2 = `<EOS>`
- [ ] Index 3 = `<UNK>`
- [ ] Index 284 = `buy`
- [ ] Index 1729 = `pour`

### ✓ Model Loading
- [ ] Model file exists (`best_model.pth`)
- [ ] Model vocab size matches vocab.pkl size
- [ ] No missing keys warnings
- [ ] Input dimension = 2653

### ✓ I3D Loading
- [ ] `rgb_imagenet.pt` exists
- [ ] Loaded 148 weight tensors
- [ ] No missing keys warnings

### ✓ Feature Extraction
- [ ] RAW features mean in reasonable range (-1 to 1)
- [ ] RAW features std in reasonable range (0.1 to 2)
- [ ] No NaN or Inf values

### ✓ Greedy Search Output
- [ ] Token probabilities > 0.5 (confident)
- [ ] Words make semantic sense
- [ ] EOS reached naturally (not forced)

---

## Configuration Matrix to Try

If issues persist, try these combinations:

| Config | APPLY_NORMALIZATION | USE_GREEDY_SEARCH | Expected Result |
|--------|---------------------|-------------------|-----------------|
| **1** | False | True | RAW features + greedy (current) |
| **2** | True | True | StandardScaler + greedy |
| **3** | False | False | RAW features + beam search |
| **4** | True | False | StandardScaler + beam search |

**Try each config** and note which produces best output.

---

## Advanced Debugging

### Check Model Predictions Directly

```python
# In Python console
import torch
import pickle

# Load vocab
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Load model
checkpoint = torch.load('best_model.pth', map_location='cpu')

# Check output layer weights
if 'fc_out.weight' in checkpoint:
    output_weights = checkpoint['fc_out.weight']
elif 'generator.weight' in checkpoint:
    output_weights = checkpoint['generator.weight']

print(f"Output layer shape: {output_weights.shape}")
print(f"Vocab size: {len(vocab.itos)}")

# Check if they match
if output_weights.shape[0] == len(vocab.itos):
    print("✓ Sizes match")
else:
    print(f"✗ MISMATCH: {output_weights.shape[0]} vs {len(vocab.itos)}")
```

---

## If Nothing Works

### Likely Causes:
1. **Model was trained with different vocab** - Need original vocab.pkl
2. **Model was trained with different features** - Need exact preprocessing
3. **Model didn't converge during training** - Need retraining
4. **Data quality issues** - Signs don't match text labels

### Next Steps:
1. **Get training logs** - Check what vocab size, feature dim, normalization was used
2. **Get training code** - Verify preprocessing matches inference
3. **Get training vocab.pkl** - Use exact same vocabulary
4. **Validate training data** - Ensure signs match text labels
5. **Consider retraining** - With verified data and preprocessing

---

## Quick Fixes to Try Now

### Fix 1: Disable All Penalties
In `advanced_predict_translation.py`, temporarily disable:
- Bias token penalties
- Frequency penalties
- N-gram blocking

This tests if penalties are causing issues.

### Fix 2: Use Temperature 1.0
In `greedy_search.py`, change:
```python
temperature=1.0  # Instead of 0.1
```

This shows raw model probabilities without sharpening.

### Fix 3: Check First Token Only
Look at greedy search Step 0 output:
```
Step  0: Token    4 = 'i' (prob: 0.8523)
```

If first token is already wrong, model has fundamental issues.
If first token is correct but later tokens wrong, it's a decoding issue.

---

## Expected Greedy Search Output

### Good (Model Working):
```
Step  0: Token    4 = 'i              ' (prob: 0.8523)
Step  1: Token   15 = 'want           ' (prob: 0.7234)
Step  2: Token   89 = 'water          ' (prob: 0.6891)
Step  3: Token    2 = '<EOS>          ' (prob: 0.9123)
```
- High probabilities (>0.5)
- Coherent sentence
- Natural EOS

### Bad (Model Broken):
```
Step  0: Token 1729 = 'pour           ' (prob: 0.2341)
Step  1: Token  187 = 'was            ' (prob: 0.1823)
Step  2: Token 2341 = 'silk           ' (prob: 0.1456)
Step  3: Token  892 = 'two            ' (prob: 0.1234)
```
- Low probabilities (<0.3)
- Random words
- No coherent meaning

If you see the "Bad" pattern, **model needs retraining**.

---

## Contact Points

If you've tried everything above and issues persist:

1. **Share greedy search logs** - Full step-by-step output
2. **Share diagnostic script output** - All checks
3. **Share training logs** - Vocab size, feature dim, loss curve
4. **Share training code** - Preprocessing and normalization

This will help identify the exact mismatch between training and inference.

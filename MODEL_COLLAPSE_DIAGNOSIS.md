# MODEL COLLAPSE DIAGNOSIS

## 🚨 Critical Issue Identified

Your model is experiencing **MODEL COLLAPSE** - it predicts the same token ("pour", index 1729) repeatedly with 100% confidence.

---

## Evidence from Logs

```
Step  0: Token 1729 = 'pour' (prob: 0.9847)
Step  1: Token 1729 = 'pour' (prob: 1.0000)
Step  2: Token 1729 = 'pour' (prob: 1.0000)
...
Step 19: Token 1729 = 'pour' (prob: 1.0000)

Output: "pour pour pour pour pour pour pour pour pour pour..."
```

---

## What Model Collapse Means

**Model collapse** occurs when:
1. The decoder gets stuck in a feedback loop
2. It predicts the same token repeatedly
3. Confidence is artificially high (100%)
4. The model ignores the input video entirely

**This is NOT**:
- ❌ A vocab mismatch (vocab is correct: index 1729 = "pour" ✓)
- ❌ A normalization issue (features are reasonable)
- ❌ An I3D loading problem (I3D loaded successfully)

**This IS**:
- ✅ A fundamental model training issue
- ✅ The decoder learned to always predict token 1729
- ✅ Autoregressive feedback amplifies this bias

---

## Why This Happens

### Root Cause: Training Data Imbalance

**Most Likely**: Token 1729 ("pour") appeared **extremely frequently** in your training data.

**Evidence**:
- Model predicts "pour" with 98.47% confidence from Step 0
- This confidence increases to 100% in subsequent steps
- The model learned: "When in doubt, predict 'pour'"

### Secondary Causes:

1. **Insufficient Training Diversity**:
   - Training data had too many "pour" examples
   - Model never learned other signs properly

2. **Loss Function Issue**:
   - Cross-entropy loss didn't penalize repetition
   - Model found local minimum: "always predict 'pour'"

3. **No Regularization**:
   - No dropout or label smoothing during training
   - Model overfit to dominant token

---

## Immediate Fixes Implemented

### Fix 1: Repetition Blocking in Greedy Search

I've updated `greedy_search.py` to block repetition:

```python
# Block immediate repetition
if len(sequence) > 1:
    last_token = sequence[-1]
    probs[last_token] *= 0.05  # 95% penalty
    
    # Penalize last 3 tokens
    for prev_token in sequence[-3:]:
        probs[prev_token] *= 0.3
```

**Test this now**:
```bash
# Restart server
python main.py

# Upload test video again
```

**Expected**: Should now predict different tokens (though may still be semantically wrong)

---

## Long-Term Solutions

### Solution 1: Retrain with Balanced Data ⭐ RECOMMENDED

**Steps**:
1. **Analyze training data**:
   ```python
   # Count token frequencies
   from collections import Counter
   token_counts = Counter(all_training_labels)
   print(token_counts.most_common(20))
   ```

2. **Check if "pour" dominates**:
   - If "pour" appears >10% of the time, data is imbalanced

3. **Balance the dataset**:
   - **Undersample**: Remove excess "pour" examples
   - **Oversample**: Add more examples of rare signs
   - **Augment**: Create variations of rare signs

4. **Retrain with**:
   - Label smoothing (0.1)
   - Dropout (0.3)
   - Class weights (inverse frequency)

### Solution 2: Add Repetition Penalty to Training

Modify training loss to penalize repetition:

```python
# In training loop
def repetition_aware_loss(logits, targets, prev_tokens):
    loss = F.cross_entropy(logits, targets)
    
    # Penalize if predicting same token as previous
    for i in range(1, len(targets)):
        if targets[i] == targets[i-1]:
            loss += 0.5  # Repetition penalty
    
    return loss
```

### Solution 3: Use Beam Search with Diversity

Enable beam search with diversity penalty:

```python
# In main.py
USE_GREEDY_SEARCH = False
```

Then in `advanced_predict_translation.py`, ensure diversity penalty is active.

---

## Testing the Repetition Block Fix

### Step 1: Restart Server
```bash
python main.py
```

### Step 2: Upload Same Video

### Step 3: Check Greedy Search Output

**Before (Model Collapse)**:
```
Step  0: Token 1729 = 'pour' (prob: 0.9847)
Step  1: Token 1729 = 'pour' (prob: 1.0000)
Step  2: Token 1729 = 'pour' (prob: 1.0000)
```

**After (Repetition Blocked)**:
```
Step  0: Token 1729 = 'pour' (prob: 0.9847)
Step  1: Token  187 = 'was' (prob: 0.6234)  ← Different token!
Step  2: Token 2341 = 'silk' (prob: 0.5123)  ← Different token!
```

**What to look for**:
- ✅ Different tokens at each step
- ✅ Probabilities vary (not all 1.0000)
- ⚠️ Tokens may still be semantically wrong (but at least diverse)

---

## If Repetition Block Doesn't Help

### Scenario A: Still Repeats "pour"

**Cause**: Model is SO biased toward "pour" that even with 95% penalty, it's still the top choice

**Fix**: Increase penalty in `greedy_search.py`:
```python
probs[last_token] *= 0.01  # 99% penalty instead of 95%
```

### Scenario B: Now Repeats Different Tokens

**Example**: "pour was was was was..."

**Cause**: Model has multiple dominant tokens

**Fix**: Increase n-gram blocking:
```python
# Block last 5 tokens instead of 3
for prev_token in sequence[-5:]:
    probs[prev_token] *= 0.1
```

### Scenario C: Random Tokens, Low Confidence

**Example**: 
```
Step  0: Token 1729 = 'pour' (prob: 0.3234)
Step  1: Token  892 = 'two' (prob: 0.2123)
Step  2: Token 3421 = 'hands' (prob: 0.1856)
```

**Cause**: Model never learned proper sign-to-text mapping

**Fix**: **Model needs complete retraining** with:
- Balanced dataset
- Proper validation
- Verified data quality

---

## Checking Training Data Quality

### Run This Analysis:

```python
import pickle
import json
from collections import Counter

# Load your training data
with open('train_data.json', 'r') as f:
    train_data = json.load(f)

# Count token frequencies
all_tokens = []
for item in train_data:
    tokens = item['text'].split()  # Or however you store labels
    all_tokens.extend(tokens)

token_counts = Counter(all_tokens)

print("Top 20 most frequent tokens:")
for token, count in token_counts.most_common(20):
    percentage = (count / len(all_tokens)) * 100
    print(f"  {token:15s}: {count:5d} ({percentage:5.2f}%)")

# Check if any token dominates
max_token, max_count = token_counts.most_common(1)[0]
if (max_count / len(all_tokens)) > 0.10:
    print(f"\n⚠️ WARNING: '{max_token}' appears {max_count/len(all_tokens)*100:.1f}% of the time!")
    print("This is likely causing model collapse.")
```

---

## Expected Results After Fix

### Good (Repetition Blocked):
```
Step  0: Token 1729 = 'pour' (prob: 0.9847)
Step  1: Token  187 = 'was' (prob: 0.6234)
Step  2: Token 2341 = 'silk' (prob: 0.5123)
Step  3: Token  892 = 'two' (prob: 0.4567)
```
- Different tokens
- Varying probabilities
- May still be semantically wrong, but at least diverse

### Bad (Still Collapsed):
```
Step  0: Token 1729 = 'pour' (prob: 0.9847)
Step  1: Token 1729 = 'pour' (prob: 0.8234)
Step  2: Token 1729 = 'pour' (prob: 0.7123)
```
- Still repeating despite penalty
- Model is extremely biased
- **Needs retraining**

---

## Summary

### What We Know:
1. ✅ Vocab is correct (index 1729 = "pour")
2. ✅ I3D loaded successfully
3. ✅ Features are reasonable
4. ❌ **Model has collapsed** - always predicts "pour"

### What We Fixed:
1. ✅ Added repetition blocking to greedy search
2. ✅ Penalize last token by 95%
3. ✅ Penalize last 3 tokens by 70%

### What You Need to Do:
1. **Test repetition block**: Restart server, upload video
2. **Check if tokens diversify**: Look for different tokens in greedy output
3. **If still repeats**: Increase penalty or **retrain model**
4. **Analyze training data**: Check if "pour" dominates (>10%)
5. **Retrain with balanced data**: Use undersampling/oversampling

### Ultimate Fix:
**Retrain the model** with:
- Balanced training data (no token >5%)
- Label smoothing (0.1)
- Dropout (0.3)
- Repetition penalty in loss
- Proper validation set

---

## Next Steps

1. **Restart server** with repetition block fix
2. **Upload test video**
3. **Share greedy search output** - let's see if tokens diversify
4. **If still collapsed**: Model needs retraining (no quick fix possible)

The repetition block is a **band-aid** - it will force diversity, but won't fix the underlying issue that the model never learned proper sign-to-text mapping. 🔧

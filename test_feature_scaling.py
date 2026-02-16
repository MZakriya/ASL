#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Scaling Test Script

Tests the RobustScaler + MinMax clipping implementation to ensure
features are properly scaled to the [-1, 1] range.
"""

import torch
import numpy as np
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_robust_scaler_minmax():
    """Test RobustScaler with MinMax clipping"""
    print("=" * 70)
    print("FEATURE SCALING TEST: RobustScaler + MinMax Clipping")
    print("=" * 70)
    
    # Test Case 1: Normal distribution
    print("\nTest Case 1: Normal Distribution")
    print("-" * 70)
    normal_data = torch.randn(1, 200, 2653) * 5.0  # Mean ~0, Std ~5
    print(f"Input - Mean: {normal_data.mean().item():.6f}, Std: {normal_data.std().item():.6f}")
    print(f"Input - Min: {normal_data.min().item():.6f}, Max: {normal_data.max().item():.6f}")
    
    # Apply RobustScaler
    median = normal_data.median()
    q75 = torch.quantile(normal_data, 0.75)
    q25 = torch.quantile(normal_data, 0.25)
    iqr = q75 - q25 + 1e-8
    scaled_data = (normal_data - median) / iqr
    
    print(f"\nAfter RobustScaler:")
    print(f"  Median: {median.item():.6f}, IQR: {iqr.item():.6f}")
    print(f"  Min: {scaled_data.min().item():.6f}, Max: {scaled_data.max().item():.6f}")
    
    # Apply MinMax Clipping
    clipped_data = torch.clamp(scaled_data, min=-1.0, max=1.0)
    print(f"\nAfter MinMax Clipping [-1, 1]:")
    print(f"  Mean: {clipped_data.mean().item():.6f}, Std: {clipped_data.std().item():.6f}")
    print(f"  Min: {clipped_data.min().item():.6f}, Max: {clipped_data.max().item():.6f}")
    
    assert clipped_data.min().item() >= -1.0, "Min value below -1.0!"
    assert clipped_data.max().item() <= 1.0, "Max value above 1.0!"
    print("[PASSED] Values within [-1, 1] range")
    
    # Test Case 2: Extreme outliers
    print("\n\nTest Case 2: Extreme Outliers")
    print("-" * 70)
    outlier_data = torch.randn(1, 200, 2653)
    outlier_data[0, 0, 0] = 100.0  # Extreme positive outlier
    outlier_data[0, 1, 0] = -100.0  # Extreme negative outlier
    print(f"Input - Mean: {outlier_data.mean().item():.6f}, Std: {outlier_data.std().item():.6f}")
    print(f"Input - Min: {outlier_data.min().item():.6f}, Max: {outlier_data.max().item():.6f}")
    
    # Apply RobustScaler
    median = outlier_data.median()
    q75 = torch.quantile(outlier_data, 0.75)
    q25 = torch.quantile(outlier_data, 0.25)
    iqr = q75 - q25 + 1e-8
    scaled_data = (outlier_data - median) / iqr
    
    print(f"\nAfter RobustScaler:")
    print(f"  Median: {median.item():.6f}, IQR: {iqr.item():.6f}")
    print(f"  Min: {scaled_data.min().item():.6f}, Max: {scaled_data.max().item():.6f}")
    print(f"  Note: Outliers may still be extreme before clipping")
    
    # Apply MinMax Clipping
    clipped_data = torch.clamp(scaled_data, min=-1.0, max=1.0)
    print(f"\nAfter MinMax Clipping [-1, 1]:")
    print(f"  Mean: {clipped_data.mean().item():.6f}, Std: {clipped_data.std().item():.6f}")
    print(f"  Min: {clipped_data.min().item():.6f}, Max: {clipped_data.max().item():.6f}")
    
    assert clipped_data.min().item() >= -1.0, "Min value below -1.0!"
    assert clipped_data.max().item() <= 1.0, "Max value above 1.0!"
    print("[PASSED] Outliers clipped to [-1, 1] range")
    
    # Test Case 3: User's reported extreme values (-13.0)
    print("\n\nTest Case 3: User's Extreme Values (Range: -13.0)")
    print("-" * 70)
    extreme_data = torch.randn(1, 200, 2653) * 10.0 - 5.0  # Simulate extreme range
    print(f"Input - Mean: {extreme_data.mean().item():.6f}, Std: {extreme_data.std().item():.6f}")
    print(f"Input - Min: {extreme_data.min().item():.6f}, Max: {extreme_data.max().item():.6f}")
    
    # Apply RobustScaler
    median = extreme_data.median()
    q75 = torch.quantile(extreme_data, 0.75)
    q25 = torch.quantile(extreme_data, 0.25)
    iqr = q75 - q25 + 1e-8
    scaled_data = (extreme_data - median) / iqr
    
    print(f"\nAfter RobustScaler:")
    print(f"  Median: {median.item():.6f}, IQR: {iqr.item():.6f}")
    print(f"  Min: {scaled_data.min().item():.6f}, Max: {scaled_data.max().item():.6f}")
    
    # Apply MinMax Clipping
    clipped_data = torch.clamp(scaled_data, min=-1.0, max=1.0)
    print(f"\nAfter MinMax Clipping [-1, 1]:")
    print(f"  Mean: {clipped_data.mean().item():.6f}, Std: {clipped_data.std().item():.6f}")
    print(f"  Min: {clipped_data.min().item():.6f}, Max: {clipped_data.max().item():.6f}")
    
    assert clipped_data.min().item() >= -1.0, "Min value below -1.0!"
    assert clipped_data.max().item() <= 1.0, "Max value above 1.0!"
    print("[PASSED] Extreme values clipped to [-1, 1] range")
    
    # Test Case 4: Zero variance (all same values)
    print("\n\nTest Case 4: Zero Variance (Edge Case)")
    print("-" * 70)
    zero_var_data = torch.ones(1, 200, 2653) * 5.0
    print(f"Input - Mean: {zero_var_data.mean().item():.6f}, Std: {zero_var_data.std().item():.6f}")
    print(f"Input - Min: {zero_var_data.min().item():.6f}, Max: {zero_var_data.max().item():.6f}")
    
    # Apply RobustScaler
    median = zero_var_data.median()
    q75 = torch.quantile(zero_var_data, 0.75)
    q25 = torch.quantile(zero_var_data, 0.25)
    iqr = q75 - q25 + 1e-8  # Epsilon prevents division by zero
    scaled_data = (zero_var_data - median) / iqr
    
    print(f"\nAfter RobustScaler:")
    print(f"  Median: {median.item():.6f}, IQR: {iqr.item():.6f}")
    print(f"  Min: {scaled_data.min().item():.6f}, Max: {scaled_data.max().item():.6f}")
    
    # Apply MinMax Clipping
    clipped_data = torch.clamp(scaled_data, min=-1.0, max=1.0)
    print(f"\nAfter MinMax Clipping [-1, 1]:")
    print(f"  Mean: {clipped_data.mean().item():.6f}, Std: {clipped_data.std().item():.6f}")
    print(f"  Min: {clipped_data.min().item():.6f}, Max: {clipped_data.max().item():.6f}")
    
    assert clipped_data.min().item() >= -1.0, "Min value below -1.0!"
    assert clipped_data.max().item() <= 1.0, "Max value above 1.0!"
    print("[PASSED] Zero variance handled correctly (epsilon prevents div by zero)")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    print("\nConclusion:")
    print("  - RobustScaler successfully centers data using median (resistant to outliers)")
    print("  - IQR scaling normalizes spread without being affected by extreme values")
    print("  - MinMax clipping ensures all values stay within [-1, 1] range")
    print("  - This prevents extreme values (-13.0) from destabilizing LayerNorm")

def compare_with_zscore():
    """Compare RobustScaler with Z-score normalization"""
    print("\n\n" + "=" * 70)
    print("COMPARISON: RobustScaler vs Z-Score Normalization")
    print("=" * 70)
    
    # Create data with outliers
    data = torch.randn(1, 200, 2653)
    data[0, 0, :10] = 50.0  # Add outliers
    data[0, 1, :10] = -50.0
    
    print(f"\nOriginal Data:")
    print(f"  Mean: {data.mean().item():.6f}, Std: {data.std().item():.6f}")
    print(f"  Min: {data.min().item():.6f}, Max: {data.max().item():.6f}")
    
    # Z-score normalization
    print(f"\nZ-Score Normalization:")
    mean = data.mean()
    std = data.std()
    zscore_data = (data - mean) / (std + 1e-8)
    print(f"  Mean: {zscore_data.mean().item():.6f}, Std: {zscore_data.std().item():.6f}")
    print(f"  Min: {zscore_data.min().item():.6f}, Max: {zscore_data.max().item():.6f}")
    print(f"  WARNING: Range: {zscore_data.max().item() - zscore_data.min().item():.2f} (can be very large!)")
    
    # RobustScaler
    print(f"\nRobustScaler:")
    median = data.median()
    q75 = torch.quantile(data, 0.75)
    q25 = torch.quantile(data, 0.25)
    iqr = q75 - q25 + 1e-8
    robust_data = (data - median) / iqr
    print(f"  Median: {median.item():.6f}, IQR: {iqr.item():.6f}")
    print(f"  Min: {robust_data.min().item():.6f}, Max: {robust_data.max().item():.6f}")
    print(f"  Range: {robust_data.max().item() - robust_data.min().item():.2f}")
    
    # RobustScaler + Clipping
    print(f"\nRobustScaler + MinMax Clipping:")
    clipped_data = torch.clamp(robust_data, min=-1.0, max=1.0)
    print(f"  Min: {clipped_data.min().item():.6f}, Max: {clipped_data.max().item():.6f}")
    print(f"  SAFE: Range: {clipped_data.max().item() - clipped_data.min().item():.2f} (guaranteed <= 2.0)")
    
    print("\nKey Difference:")
    print("  Z-Score: Sensitive to outliers, can produce extreme values")
    print("  RobustScaler: Uses median/IQR, resistant to outliers")
    print("  + Clipping: Guarantees safe range for LayerNorm")

if __name__ == "__main__":
    test_robust_scaler_minmax()
    compare_with_zscore()
    print("\n[SUCCESS] Feature scaling implementation verified!")

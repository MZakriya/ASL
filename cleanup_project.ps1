# ASLR Project Cleanup Script
Write-Host "Starting ASLR Project Cleanup..." -ForegroundColor Cyan

# Documentation files to delete
$docFiles = @(
    "ACTION_PLAN.md",
    "CALIBRATION_SUMMARY.md",
    "EPOCH_10_REFINEMENT.md",
    "EPOCH_10_ROLLBACK.md",
    "EPOCH_10_SEMANTIC_FIX.md",
    "IMPLEMENTATION_SUMMARY.md",
    "MODEL_COLLAPSE_DIAGNOSIS.md",
    "MODE_COLLAPSE_FIX.md",
    "SEMANTIC_ALIGNMENT_FIX.md",
    "SEMANTIC_DRIFT_FIX_SUMMARY.md",
    "SEMANTIC_RERANKING_FIX.md",
    "TRAINING_ALIGNMENT_SUMMARY.md",
    "TROUBLESHOOTING_GUIDE.md"
)

# Test/Debug scripts to delete
$testFiles = @(
    "audit_model_vocab.py",
    "diagnose_semantic_issues.py",
    "download_simple.py",
    "download_weights.py",
    "inspect_model.py",
    "inspect_vocab.py",
    "local_finetune_overfit.py",
    "test_api.py",
    "test_feature_scaling.py",
    "test_model_load.py",
    "test_phase7_arch.py",
    "verify_decoder.py",
    "verify_vocab_mapping.py"
)

$deletedCount = 0

# Delete documentation files
Write-Host "`nDeleting documentation files..." -ForegroundColor Yellow
foreach ($file in $docFiles) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "  ✓ Deleted: $file" -ForegroundColor Green
        $deletedCount++
    }
}

# Delete test/debug files
Write-Host "`nDeleting test/debug scripts..." -ForegroundColor Yellow
foreach ($file in $testFiles) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "  ✓ Deleted: $file" -ForegroundColor Green
        $deletedCount++
    }
}

# Delete __pycache__ directory
Write-Host "`nDeleting cache directory..." -ForegroundColor Yellow
if (Test-Path "__pycache__") {
    Remove-Item "__pycache__" -Recurse -Force
    Write-Host "  ✓ Deleted: __pycache__/" -ForegroundColor Green
    $deletedCount++
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Cleanup Complete!" -ForegroundColor Green
Write-Host "Total items deleted: $deletedCount" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

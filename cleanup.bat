@echo off
echo Cleaning up ASLR project...

REM Delete documentation files
del /f /q ACTION_PLAN.md 2>nul
del /f /q CALIBRATION_SUMMARY.md 2>nul
del /f /q EPOCH_10_REFINEMENT.md 2>nul
del /f /q EPOCH_10_ROLLBACK.md 2>nul
del /f /q EPOCH_10_SEMANTIC_FIX.md 2>nul
del /f /q IMPLEMENTATION_SUMMARY.md 2>nul
del /f /q MODEL_COLLAPSE_DIAGNOSIS.md 2>nul
del /f /q MODE_COLLAPSE_FIX.md 2>nul
del /f /q SEMANTIC_ALIGNMENT_FIX.md 2>nul
del /f /q SEMANTIC_DRIFT_FIX_SUMMARY.md 2>nul
del /f /q SEMANTIC_RERANKING_FIX.md 2>nul
del /f /q TRAINING_ALIGNMENT_SUMMARY.md 2>nul
del /f /q TROUBLESHOOTING_GUIDE.md 2>nul
echo Documentation files deleted.

REM Delete test/debug scripts
del /f /q audit_model_vocab.py 2>nul
del /f /q diagnose_semantic_issues.py 2>nul
del /f /q download_simple.py 2>nul
del /f /q download_weights.py 2>nul
del /f /q inspect_model.py 2>nul
del /f /q inspect_vocab.py 2>nul
del /f /q local_finetune_overfit.py 2>nul
del /f /q test_api.py 2>nul
del /f /q test_feature_scaling.py 2>nul
del /f /q test_model_load.py 2>nul
del /f /q test_phase7_arch.py 2>nul
del /f /q verify_decoder.py 2>nul
del /f /q verify_vocab_mapping.py 2>nul
echo Test/debug scripts deleted.

REM Delete cache directory
rmdir /s /q __pycache__ 2>nul
echo Cache directory deleted.

echo.
echo Cleanup complete!
echo.
pause

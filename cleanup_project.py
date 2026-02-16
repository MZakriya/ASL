import os
import shutil

print("Starting ASLR Project Cleanup...")
print("=" * 50)

# Documentation files to delete
doc_files = [
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
]

# Test/Debug scripts to delete
test_files = [
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
]

deleted_count = 0

# Delete documentation files
print("\n📄 Deleting documentation files...")
for file in doc_files:
    if os.path.exists(file):
        try:
            os.remove(file)
            print(f"  ✓ Deleted: {file}")
            deleted_count += 1
        except Exception as e:
            print(f"  ✗ Failed to delete {file}: {e}")

# Delete test/debug files
print("\n🧪 Deleting test/debug scripts...")
for file in test_files:
    if os.path.exists(file):
        try:
            os.remove(file)
            print(f"  ✓ Deleted: {file}")
            deleted_count += 1
        except Exception as e:
            print(f"  ✗ Failed to delete {file}: {e}")

# Delete __pycache__ directory
print("\n🗑️  Deleting cache directory...")
if os.path.exists("__pycache__"):
    try:
        shutil.rmtree("__pycache__")
        print(f"  ✓ Deleted: __pycache__/")
        deleted_count += 1
    except Exception as e:
        print(f"  ✗ Failed to delete __pycache__: {e}")

# Delete cleanup scripts
print("\n🧹 Cleaning up cleanup scripts...")
cleanup_scripts = ["cleanup.bat", "cleanup_project.ps1", "cleanup_project.py"]
for file in cleanup_scripts:
    if os.path.exists(file):
        try:
            os.remove(file)
            print(f"  ✓ Deleted: {file}")
            deleted_count += 1
        except Exception as e:
            print(f"  ✗ Failed to delete {file}: {e}")

print("\n" + "=" * 50)
print(f"✅ Cleanup Complete!")
print(f"📊 Total items deleted: {deleted_count}")
print("=" * 50)

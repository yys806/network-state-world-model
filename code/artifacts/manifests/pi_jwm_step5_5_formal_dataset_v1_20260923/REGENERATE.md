# STEP 5.5 Formal Dataset v1 regeneration

Run from the repository root with the project Python environment. Replace `<OUT>` with a new, empty, local artifact path (the collector refuses to overwrite an existing directory):

```powershell
$env:PYTHONPATH='<REPO_ROOT>\code\src;<REPO_ROOT>\code\scripts'
python code/scripts/collect_step5_5_formal_raw_v1.py --output-dir <OUT>
python code/scripts/build_step5_5_formal_dataset_v1.py --output-dir <OUT>
python code/scripts/step5_5_formal_dataset_cpu_acceptance_v1.py --manifest <OUT>\formal_dataset_manifest.json --output <OUT>\formal_interface_cpu_smoke_receipt.json --device cpu --dry-run
```

This is the exact sequence used for v1 generation; it is a documented regeneration command and has not been rerun during context/record synchronization. Large Raw and package files remain local-only; portable identity, size/hash summary, split, coverage, acceptance, deterministic rebuild, normalization and CPU smoke receipts are stored beside this file. A deterministic rebuild reuses the retained Raw sources and frozen split rather than rerunning simulator collection.

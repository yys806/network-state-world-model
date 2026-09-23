# PI-JWM Formal Dataset v1 Share Bundle

Dataset: PI-JWM Formal Dataset v1

## Frozen protocol

- H = 2
- L = 4
- 60 trajectories: 48 train, 12 validation
- 96 transitions per trajectory
- 4416 train windows, 1104 validation windows, 5520 total windows
- Topology: `radius_knn`, radius = 1000 m, k = 2
- Actions: Route, Comm, Comp, Mobility

The split is trajectory-level. Normalization statistics come from the train split only; validation trajectories do not contribute to normalization. There is no `locked_test`. Future Return birth follows the frozen fixed-support boundary. This is a Formal Dataset artifact, not a performance result.

Compatible code version: `0408fcdcc4125748daddc73e88e256ac5f991e4d`

AirFogSim is the data-generation source. It is not the PI-JWM framework. No license is declared by this bundle.

## Portable loading

After extracting this bundle anywhere, point to the extracted `formal_dataset_manifest.json`; no machine-specific absolute path is required:

```python
from pathlib import Path
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset

manifest = Path("/path/to/extracted/training/formal_dataset_manifest.json")
interface = FormalTrainingInterface.from_manifest(manifest)
dataset = FullFormalShardDataset(interface)
print(len(interface.train_indices), len(interface.validation_indices))  # 4416 1104
```

The training bundle contains the full sharded packages and runtime package files. The separate raw add-on contains the 60 raw trajectories and the provenance needed to inspect or rebuild the dataset.

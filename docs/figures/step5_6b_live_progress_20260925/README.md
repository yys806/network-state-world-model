# STEP 5.6B training progress — interim snapshot

This is a **read-only snapshot**, not a final result. The active run was not
modified or paused. Snapshot time: 2026-09-25 13:35 UTC (21:35 Beijing).

- Run ID: `pi_jwm_formal_train_v1_seed5601_20260924T112424Z`
- Training source Git SHA: `6e15ec2da0e3a6e0561dc821d0aaef90696a2387`
- Copied heartbeat: `RUNNING`, 2646/5520 completed steps
- Full prior-only validations: step 1104 and step 2208, 1104 validation windows each
- `L_Val`: 0.1766124568 → 0.0801012691
- Dataset manifest SHA-256: `6392a08b31340812463be9e3f5f78891d39933c54d50c71cca1984b3bf9448bc`
- Formal config SHA-256: `a806c320f1d238a3997a94ca74af5641169a512a9551af7f3ff04c0ee2447660`

The three PNGs show per-step training losses, full validation losses by
horizon, and Motion/CSI raw MAE/RMSE by horizon. The training objective changes
from posterior-assisted H1 to prior-dominant H1 at step 552, to H2 at 1104,
and to H4 at 2208. Therefore, training-loss heights across stage boundaries
are not directly comparable. Motion raw errors aggregate variables with
different units; the plot does not label them as meters. CSI raw errors are dB.

The exact copied-log hashes and plot hashes are in `snapshot_receipt.json`.
Original live logs and checkpoints stay on the server. A local read-only log
copy is under `code/artifacts/audit/pi_jwm_step5_6b_progress_snapshot_20260925/`
and is not committed. To update the plots later, copy a fresh set of the same
four log/manifest files into a **new** local snapshot directory and run:

```powershell
python code/scripts/plot_step5_6b_training_progress_v1.py <new_snapshot_dir>
```

This snapshot establishes a process record only. It is not a locked-test,
baseline, Planner, or final performance claim.

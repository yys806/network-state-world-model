# Third-Party References

`reference/` contains local third-party simulators and reference implementations. They are inputs to PI-JWM development, not PI-JWM framework modules.

## AirFogSim

Keep the AirFogSim checkout at `代码/reference/AirFogSim/` and manage its dependencies in an independent environment:

```powershell
conda activate airfogsim
cd D:\shen\PKU\PIJWM\代码\reference\AirFogSim\examples
```

The checkout is intentionally ignored by the PI-JWM repository because it has its own Git history, dependencies, generated outputs, and local modifications. PI-JWM scripts may read simulator data from it, but reusable model code must remain under `代码/src/pi_jwm/`.

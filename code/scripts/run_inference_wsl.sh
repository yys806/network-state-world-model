#!/usr/bin/env bash
# PIJWM / V2U4Real 推理启动脚本（WSL 专用，2026-11-16 建）
#
# 为什么需要它：
#   1) open3d 需要 libusb-1.0.so.0，WSL 里 apt 装不上（sudo 要密码），
#      库放在仓库内 .syslib/（已加 .gitignore），靠 LD_LIBRARY_PATH 指过去。
#   2) 直接跑 opencood/tools/inference.py 时项目根不在 sys.path，必须给 PYTHONPATH。
#   3) box_overlaps 是 Cython 扩展，需先编译（本脚本首次运行会自动编译）。
#
# 用法：
#   bash code/scripts/run_inference_wsl.sh --model_dir <ckpt目录> [--fusion_method intermediate]
set -euo pipefail

REPO=/mnt/d/shen/PKU/PIJWM
V2U4=$REPO/code/reference/V2U4Real

export LD_LIBRARY_PATH=$REPO/.syslib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$V2U4:${PYTHONPATH:-}

# 首次运行：编译 Cython 的 box_overlaps（缺 .so 时）
if ! ls "$V2U4"/opencood/utils/box_overlaps*.so >/dev/null 2>&1; then
  echo "[setup] 编译 box_overlaps ..."
  ( cd "$V2U4" && "$REPO/.venv/bin/python" opencood/utils/setup.py build_ext --inplace )
fi

cd "$V2U4"
exec "$REPO/.venv/bin/python" opencood/tools/inference.py "$@"

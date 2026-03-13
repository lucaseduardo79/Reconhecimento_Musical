#!/bin/bash
set -e

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║     🎵  Partitura IA — lucaseduardo79                ║"
echo "╚══════════════════════════════════════════════════════╝"

GPU_INFO=$(python3 -c "
import torch
if torch.cuda.is_available():
    print('GPU: ' + torch.cuda.get_device_name(0))
else:
    print('CPU (CUDA não disponível)')
" 2>/dev/null)
echo "  ⚡  $GPU_INFO"

# ── Download dos checkpoints do oemer (só na primeira vez) ───────────────────
CHECKPOINT_DIR="${OEMER_CHECKPOINT_DIR:-/opt/conda/lib/python3.11/site-packages/oemer/checkpoints}"
if [ ! -d "$CHECKPOINT_DIR/unet_big" ]; then
    echo ""
    echo "  📥  Baixando modelos oemer (~400 MB) — só ocorre uma vez..."
    python3 -c "from oemer.download import download; download()"
    echo "  ✅  Modelos baixados."
else
    echo "  ✅  Modelos oemer já presentes."
fi

echo ""
exec python3 /workspace/partitura_ia.py

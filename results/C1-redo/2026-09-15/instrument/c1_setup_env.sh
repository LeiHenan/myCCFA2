#!/usr/bin/env bash
# C1 redo -- environment bootstrap on the new box (A800-SXM4-80GB, sm80, 18 vCPU, 1TB RAM)
# Goal: vLLM 0.29.0 (same version as the archived C-1 runs) WITHOUT disturbing the
#       pre-installed torch 2.12.1+cu130, which is already sm80-capable.
set -u

PIP=/root/miniconda3/bin/pip
PY=/root/miniconda3/bin/python
export PIP_INDEX_URL=http://mirrors.aliyun.com/pypi/simple
export PIP_TRUSTED_HOST=mirrors.aliyun.com

echo "########## 0. disk / identity ##########"
df -h / /root/autodl-tmp 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'cap',torch.cuda.get_device_capability())"

echo "########## 1. vLLM wheel (no deps) ##########"
$PIP install --no-deps "vllm==0.29.0" 2>&1 | tail -5

echo "########## 2. declared requirements ##########"
$PY - <<'PYEOF'
import importlib.metadata as md
try:
    reqs = md.requires('vllm') or []
except Exception as e:
    print("METADATA_MISSING", e); raise SystemExit(1)
skip_prefix = ('torch', 'torchvision', 'torchaudio', 'flashinfer', 'xformers', 'vllm')
keep, skipped = [], []
for r in reqs:
    if r.split(';')[0].strip().startswith(skip_prefix) or 'extra ==' in r:
        skipped.append(r)
    else:
        keep.append(r)
print("KEEP:")
for k in keep: print("   ", k)
print("SKIPPED(torch-family/optional):")
for s in skipped: print("   ", s)
open('/root/autodl-tmp/vllm_keep_reqs.txt', 'w').write("\n".join(k for k in keep if 'extra ==' not in k))
PYEOF

echo "########## 3. install the rest ##########"
$PIP install -r /root/autodl-tmp/vllm_keep_reqs.txt 2>&1 | tail -25

echo "########## 4. torch must be untouched ##########"
$PY -c "import torch;print('torch now',torch.__version__,'cuda',torch.version.cuda)"

echo "########## 5. import vllm, repair missing modules iteratively ##########"
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  out=$($PY -c "import vllm; print('VLLM_OK', vllm.__version__)" 2>&1)
  echo "--- attempt $i ---"
  echo "$out" | tail -3
  if echo "$out" | grep -q VLLM_OK; then
    echo "IMPORT_OK"
    break
  fi
  mod=$(echo "$out" | grep -oE "No module named '[^']+'" | head -1 | sed "s/No module named '//;s/'//")
  if [ -z "$mod" ]; then
    echo "UNRESOLVED_NON_MODULE_ERROR"; break
  fi
  pkg=$(echo "$mod" | cut -d. -f1)
  echo ">>> installing missing: $pkg"
  $PIP install "$pkg" 2>&1 | tail -3
done

echo "########## 6. final verify ##########"
$PY - <<'PYEOF'
import vllm, torch
print("vllm", vllm.__version__, "| torch", torch.__version__, "| cap", torch.cuda.get_device_capability())
from vllm import LLM, SamplingParams
print("LLM import OK")
import vllm.model_executor.models.registry as reg
names = sorted(reg.ModelRegistry.get_supported_archs())
print("n_archs", len(names))
for want in ("Qwen3ForCausalLM","Qwen3_5ForConditionalGeneration","FalconH1ForCausalLM"):
    print(want, want in names)
PYEOF
echo "########## DONE ##########"

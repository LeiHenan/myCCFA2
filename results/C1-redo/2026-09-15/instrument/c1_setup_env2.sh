#!/usr/bin/env bash
# C1 redo -- vLLM 0.29.0 dependency install, take 2.
#
# Why take 2: `pip install -r <pinned reqs>` hung in do_poll with a frozen cache
# (523 MB -> 523 MB over 25 s, 0 established connections). The pinned set asks for
# versions the aliyun mirror may not carry, which sends the resolver into
# backtracking plus stalled connections.
#
# Take 2 strategy:
#   * strip every version specifier -> "latest available on the mirror", so the
#     resolver has nothing to backtrack over;
#   * install ONE package per pip invocation under a hard `timeout`, so a single
#     unreachable package can never freeze the whole environment again;
#   * fail fast on network (--timeout 15 --retries 2);
#   * then repair via the import loop, and finally prove torch was not disturbed.
set -u

PIP=/root/miniconda3/bin/pip
PY=/root/miniconda3/bin/python
export PIP_INDEX_URL=http://mirrors.aliyun.com/pypi/simple
export PIP_TRUSTED_HOST=mirrors.aliyun.com
PIPFLAGS="--timeout 15 --retries 2 --no-input"

echo "########## 0. state ##########"
$PY -c "import torch;print('torch before',torch.__version__,torch.version.cuda)"
$PY -c "import vllm;print('vllm',vllm.__version__)" 2>&1 | tail -1

echo "########## 1. build unpinned package list from vLLM metadata ##########"
$PY - <<'PYEOF' > /root/autodl-tmp/pkgs_unpinned.txt
import importlib.metadata as md, re
reqs = md.requires('vllm') or []
skip = ('torch', 'torchvision', 'torchaudio', 'flashinfer', 'xformers', 'vllm')
seen, out = set(), []
for r in reqs:
    if 'extra ==' in r:
        continue
    body = r.split(';')[0].strip()
    name = re.split(r'[<>=!\[ ]', body)[0].strip()
    extra = ''
    m = re.match(r'^[A-Za-z0-9_.\-]+(\[[^\]]+\])', body)
    if m:
        extra = m.group(1)
    if not name or name.lower().startswith(skip):
        continue
    key = (name + extra).lower()
    if key in seen:
        continue
    seen.add(key)
    out.append(name + extra)
print("\n".join(out))
PYEOF
echo "packages: $(wc -l < /root/autodl-tmp/pkgs_unpinned.txt)"

echo "########## 2. install one at a time, fail fast ##########"
ok=0; failed=""
while read -r p; do
  [ -z "$p" ] && continue
  if timeout 240 $PIP install -q $PIPFLAGS "$p" >/tmp/piplog 2>&1; then
    ok=$((ok+1)); echo "  OK   $p"
  else
    failed="$failed $p"; echo "  FAIL $p  :: $(tail -1 /tmp/piplog | cut -c1-140)"
  fi
done < /root/autodl-tmp/pkgs_unpinned.txt
echo "installed=$ok failed=[$failed]"

echo "########## 3. import loop ##########"
for i in $(seq 1 14); do
  out=$($PY -c "import vllm; print('VLLM_OK', vllm.__version__)" 2>&1)
  echo "--- attempt $i: $(echo "$out" | tail -1 | cut -c1-160)"
  if echo "$out" | grep -q VLLM_OK; then echo "IMPORT_OK"; break; fi
  mod=$(echo "$out" | grep -oE "No module named '[^']+'" | head -1 | sed "s/No module named '//;s/'//")
  [ -z "$mod" ] && { echo "NON_MODULE_ERROR"; break; }
  pkg=$(echo "$mod" | cut -d. -f1)
  echo ">>> installing missing: $pkg"
  timeout 240 $PIP install -q $PIPFLAGS "$pkg" 2>&1 | tail -2
done

echo "########## 4. torch integrity ##########"
$PY -c "import torch;print('torch after',torch.__version__,torch.version.cuda,'cap',torch.cuda.get_device_capability())"

echo "########## 5. capability probe ##########"
$PY - <<'PYEOF'
import vllm, torch
print("vllm", vllm.__version__, "| torch", torch.__version__)
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
import inspect
# which spec methods does this build accept?
try:
    from vllm.config.speculative import SpeculativeConfig
    import typing
    f = SpeculativeConfig.model_fields.get('method')
    print("spec method field:", f)
except Exception as e:
    print("spec cfg probe failed:", type(e).__name__, e)
print("EngineArgs has disable_log_stats:", 'disable_log_stats' in inspect.signature(EngineArgs).parameters)
PYEOF
echo "########## DONE2 ##########"

"""Step-count instrumentation, injected via PYTHONPATH so that EVERY python
process in the tree -- including the spawned EngineCore child -- applies it.

v2, after a segfault: the first version did `from vllm.v1.engine.core import
EngineCore` at interpreter start-up.  That eager import chain reaches the stdlib
`readline` extension, which segfaults in this container because the locale is
broken (`LC_ALL=en_US.UTF-8` cannot be set), killing EngineCore before it could
serve anything:

    !!!!!!! Segfault encountered !!!!!!!
      File "<unknown>", line 0, in _rl_init_locale
      File ".../Modules/readline.c", line 1306, in setup_readline

So the patch is now installed as a *lazy* meta-path hook: nothing is imported at
start-up, and EngineCore is only wrapped once vLLM itself imports it.

The counter counts ENGINE ITERATIONS, which include prefill and scheduling-only
iterations.  It is NOT a decode-step counter -- the earlier C-1 work mislabelled
exactly this quantity as "real steps" (see 审核_原方案错误清单 4.2).
"""
import os
import sys

_LOG = os.environ.get("C1R_STEP_FILE")
_STATUS = os.environ.get("C1R_STEP_STATUS")
_TARGET = "vllm.v1.engine.core"


def _note(msg):
    if not _STATUS:
        return
    try:
        with open(_STATUS, "a") as f:
            f.write(f"{msg} pid={os.getpid()}\n")
    except Exception:
        pass


def _patch(module):
    def make(orig, name):
        def wrapper(self, *args, **kwargs):
            try:
                with open(_LOG, "a") as f:
                    f.write("1\n")
            except Exception:
                pass
            return orig(self, *args, **kwargs)
        wrapper.__name__ = name
        return wrapper

    hit = []
    core = getattr(module, "EngineCore", None)
    if core is None:
        _note("NO_ENGINECORE_ATTR")
        return
    for name in ("step", "step_with_batch_queue"):
        orig = getattr(core, name, None)
        if orig is None or getattr(orig, "_c1r_patched", False):
            continue
        w = make(orig, name)
        w._c1r_patched = True
        setattr(core, name, w)
        hit.append(name)
    _note(f"PATCHED {hit}" if hit else "NO_STEP_METHOD_FOUND")


if _LOG:
    try:
        import importlib.abc

        class _Loader(importlib.abc.Loader):
            def __init__(self, inner):
                self._inner = inner

            def create_module(self, spec):
                return self._inner.create_module(spec)

            def exec_module(self, module):
                self._inner.exec_module(module)
                try:
                    _patch(module)
                except Exception as e:  # noqa: BLE001
                    _note(f"PATCH_FAILED {type(e).__name__}: {e}")

        class _Finder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname != _TARGET:
                    return None
                for finder in sys.meta_path:
                    if finder is self:
                        continue
                    try:
                        spec = finder.find_spec(fullname, path, target)
                    except Exception:
                        continue
                    if spec is not None and spec.loader is not None:
                        spec.loader = _Loader(spec.loader)
                        return spec
                return None

        sys.meta_path.insert(0, _Finder())
        _note("HOOK_INSTALLED")
    except Exception as e:  # noqa: BLE001
        _note(f"HOOK_INSTALL_FAILED {type(e).__name__}: {e}")

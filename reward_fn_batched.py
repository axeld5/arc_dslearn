from concurrent.futures import ThreadPoolExecutor
from reward_fn import (
    extract_python_code, SOLVE_RE, IMPORT_RE,
    equivalent, safe_exec
)
from json_utils import from_jsonable
import ast, json, inspect, re

DSL_NAMES = None
def init_dsl_names():
    global DSL_NAMES
    if DSL_NAMES is None:
        import arc_dsl.dsl as dsl
        DSL_NAMES = set(dsl.__dict__.keys())
init_dsl_names()

def _score_one(code_raw, shot_list):
    """Exactly the same three checks as before, just isolated so we can
    thread-pool it."""
    code  = extract_python_code(code_raw)
    shots = from_jsonable(shot_list)
    r     = 0.0

    # 1.  solve() present
    r += 0.1 if SOLVE_RE.search(code) else -1.0

    # 2.  import/identifier hygiene
    try:
        tree        = ast.parse(code)
        bad_imports = bool(IMPORT_RE.search(code))
        names       = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        unknown     = names - {"I", "O"} - {f"x{i}" for i in range(1, 100)} - DSL_NAMES
        if not bad_imports and not unknown:
            r += 0.1
    except SyntaxError:
        pass

    # 3.  functional correctness
    try:
        for shot in shots:
            inp  = json.loads(shot["inputs"])  if isinstance(shot["inputs"],  str) else shot["inputs"]
            exp  = json.loads(shot["output"])  if isinstance(shot["output"],  str) else shot["output"]
            exp  = from_jsonable(exp)
            mod  = safe_exec(code, inp)
            res  = mod.solve(inp) if callable(getattr(mod, "solve", None)) else object()
            if not equivalent(res, exp, inp):
                break
        else:                                   # -- no break → all good
            r += 0.8
    except Exception:
        pass

    return r

def reward_fn(completions, shots, **_):
    """
    Batched version: takes the same `completions` (list[str])
    and `shots` (list[list[shot]]) as TRL hands to us, but
    evaluates in a ThreadPool so Python’s GIL isn’t blocking
    the I/O-heavy `ast.parse` / `exec`.
    """
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(_score_one, completions, shots))
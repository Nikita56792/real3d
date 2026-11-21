#!/usr/bin/env python
"""Per-op estimated cost (MLComputePlan) grouped into encoder / neck / head buckets.
Usage: cost_plan.py model.mlpackage [CPU_AND_GPU|CPU_AND_NE|ALL] [--top N]"""
import re
import sys
from collections import Counter, defaultdict

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

from compute_plan import dev_name


def bucket(name: str, idx: int, first_neck: int) -> str:
    n = name.lower()
    if "fusion" in n:
        return "neck.fusion"
    if "reassemble" in n or "neck_convs" in n or "neck.convs" in n:
        return "neck.reassemble+convs"
    if "head" in n:
        return "head"
    return "encoder" if idx < first_neck else "neck/head(other)"


def main():
    path = sys.argv[1]
    cu = getattr(ct.ComputeUnit, sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else ct.ComputeUnit.CPU_AND_GPU
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
    compiled = ct.models.utils.compile_model(path)
    plan = MLComputePlan.load_from_path(compiled, compute_units=cu)
    ops = [op for op in plan.model_structure.program.functions["main"].block.operations if op.operator_name != "const"]
    names = [op.outputs[0].name if op.outputs else "" for op in ops]
    first_neck = next((i for i, n in enumerate(names) if re.search("reassemble|neck", n.lower())), len(ops))
    rows = []
    for i, op in enumerate(ops):
        c = plan.get_estimated_cost_for_mlprogram_operation(op)
        w = c.weight if c is not None else 0.0
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        dev = dev_name(u.preferred_compute_device) if u else "?"
        rows.append((w, op.operator_name, names[i], dev, bucket(names[i], i, first_neck)))
    tot = sum(r[0] for r in rows) or 1.0
    by_b, by_t, by_d = defaultdict(float), defaultdict(float), Counter()
    for w, t, n, d, b in rows:
        by_b[b] += w
        by_t[t] += w
        by_d[d] += 1
    print(f"== {path} {cu}  ops={len(ops)} devices={dict(by_d)} total_weight={tot:.3f}")
    print("by bucket:")
    for b, w in sorted(by_b.items(), key=lambda kv: -kv[1]):
        print(f"  {b:28s} {100 * w / tot:6.1f} %")
    print("by op type:")
    for t, w in sorted(by_t.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {t:28s} {100 * w / tot:6.1f} %")
    print(f"top {top} ops:")
    for w, t, n, d, b in sorted(rows, key=lambda r: -r[0])[:top]:
        print(f"  {100 * w / tot:5.2f} %  {t:24s} {d:4s} {b:22s} {n}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Print where each MIL op of a compiled Core ML model is scheduled (ANE / GPU / CPU) for a given
compute-unit setting, plus a summary by op type. Usage: compute_plan.py model.mlpackage [ALL|CPU_AND_NE|CPU_AND_GPU]"""
import sys
from collections import Counter, defaultdict

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan
from coremltools.models.compute_device import MLNeuralEngineComputeDevice, MLGPUComputeDevice, MLCPUComputeDevice


def dev_name(d):
    if isinstance(d, MLNeuralEngineComputeDevice):
        return "ANE"
    if isinstance(d, MLGPUComputeDevice):
        return "GPU"
    if isinstance(d, MLCPUComputeDevice):
        return "CPU"
    return type(d).__name__


def main():
    path = sys.argv[1]
    cu = getattr(ct.ComputeUnit, sys.argv[2]) if len(sys.argv) > 2 else ct.ComputeUnit.ALL
    verbose = "-v" in sys.argv
    compiled = ct.models.utils.compile_model(path)
    plan = MLComputePlan.load_from_path(compiled, compute_units=cu)
    prog = plan.model_structure.program
    fn = prog.functions["main"]
    ops = fn.block.operations
    by_dev = Counter()
    by_type_dev = defaultdict(Counter)
    unsupported = []
    for op in ops:
        if op.operator_name == "const":
            continue
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if u is None:
            dev = "?"
        else:
            dev = dev_name(u.preferred_compute_device)
        by_dev[dev] += 1
        by_type_dev[op.operator_name][dev] += 1
        if dev != "ANE" and u is not None:
            supported = [dev_name(d) for d in u.supported_compute_devices]
            unsupported.append((op.operator_name, [o.name for o in op.outputs][:1], dev, supported))
        if verbose:
            print(f"{op.operator_name:32s} {[o.name for o in op.outputs][0]:40s} -> {dev}")
    print(f"\n== {path}  compute_units={cu}")
    print("ops by device:", dict(by_dev))
    print("ops by type/device:")
    for t, c in sorted(by_type_dev.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {t:36s} {dict(c)}")
    if unsupported:
        print(f"\nops NOT on ANE ({len(unsupported)}):")
        for name, outs, dev, supported in unsupported[:60]:
            print(f"  {name:32s} {outs[0]:40s} -> {dev}  (supported: {supported})")


if __name__ == "__main__":
    main()

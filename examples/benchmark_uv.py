"""Benchmark experimental UV wrapper and bare XSPEC on matching trial vectors."""

import argparse
import json
import time
import tracemalloc
from unittest.mock import patch

import numpy as np
from sherpa.astro import xspec

from sherpa_plasma._uv_model import _TripletPumpingModel


def _time_calls(model, vectors, lo, hi):
    start = time.perf_counter()
    for vector in vectors:
        model.calc(vector, lo, hi)
    return (time.perf_counter() - start) / len(vectors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--bins", type=int, default=10000)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.bins < 2 or args.iterations < 1:
        parser.error("require at least two bins and one iteration")
    xspec.set_xsxset("APECROOT", args.root)
    child = xspec.XSvvapec("bare")
    child.kT = .43
    wrapped_child = xspec.XSvvapec("wrapped")
    wrapped_child.kT = .43
    start = time.perf_counter()
    wrapper = _TripletPumpingModel(wrapped_child, "Na")
    initialization = time.perf_counter() - start
    lohi = np.linspace(.8, 1.4, args.bins + 1)
    lo, hi = lohi[:-1], lohi[1:]
    bare = np.array([par.val for par in child.pars])
    combined = np.array([par.val for par in wrapper.pars])
    child.startup(cache=True)
    wrapper.startup(cache=True)
    try:
        child.calc(bare, lo, hi)
        wrapper.calc(combined, lo, hi)
        unchanged_bare = [bare.copy() for _ in range(args.iterations)]
        unchanged_wrapped = [combined.copy() for _ in range(args.iterations)]
        pumping = [combined.copy() for _ in range(args.iterations)]
        varying_bare = [bare.copy() for _ in range(args.iterations)]
        varying_wrapped = [combined.copy() for _ in range(args.iterations)]
        for i in range(args.iterations):
            pumping[i][-1] = (i % 5) * .5
            varying_bare[i][0] = .42 + (i % 7) * .002
            varying_wrapped[i][0] = varying_bare[i][0]
            varying_wrapped[i][-1] = 1.
        result = {
            "bins": args.bins,
            "iterations": args.iterations,
            "wrapper_initialization_seconds": initialization,
            "bare_unchanged_seconds": _time_calls(child, unchanged_bare, lo, hi),
            "wrapper_zero_seconds": _time_calls(wrapper, unchanged_wrapped, lo, hi),
            "wrapper_pumping_only_seconds": _time_calls(wrapper, pumping, lo, hi),
            "bare_varying_temperature_seconds": _time_calls(child, varying_bare, lo, hi),
            "wrapper_varying_temperature_seconds": _time_calls(wrapper, varying_wrapped, lo, hi),
        }
        wrapper.cache_clear()
        tracemalloc.start()
        with patch.object(type(wrapped_child), "calc", autospec=True,
                          side_effect=type(wrapped_child).calc) as calculate, \
             patch("sherpa_plasma._uv_model._read_native_lines") as read_lines:
            for vector in pumping:
                wrapper.calc(vector, lo, hi)
            result["pumping_only_child_calc_calls"] = calculate.call_count
            result["pumping_only_native_line_reads"] = read_lines.call_count
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["pumping_only_python_traced_peak_bytes"] = peak
        print(json.dumps(result, indent=2))
    finally:
        wrapper.teardown()
        child.teardown()


if __name__ == "__main__":
    main()

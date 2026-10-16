"""Compare isolated pre/post-refactor builds on identical local problems.

Each package directory must contain a separately built cpo3 Python package.
Workers run serially (alternating build order), with one numerical thread.
Loading/verification are excluded from solver timing. Results are append-only
and resumable; each record includes exact vectors/metrics and work counters.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def worker(args):
    sys.path.insert(0, args.package)
    sys.path.insert(1, args.harness)
    import cpo3
    import numpy as np
    from scipy import sparse
    from solver_benchmarks.datasets.registry import get_dataset
    from solver_benchmarks.transforms.cones import qp_to_nonnegative_cone

    def load(spec):
        if spec["loader"] == "mps":
            with (
                gzip.open(spec["path"], "rb") as source,
                tempfile.NamedTemporaryFile(suffix=".mps") as expanded,
            ):
                expanded.write(source.read())
                expanded.flush()
                return cpo3.read_mps(expanded.name)
        dataset = get_dataset(spec["dataset"])(args.harness)
        qp = dataset.load_problem(spec["name"]).qp
        a, b, z = qp_to_nonnegative_cone(qp)
        cones = []
        if z:
            cones.append(cpo3.ZeroCone(int(z)))
        if a.shape[0] > z:
            cones.append(cpo3.NonnegativeCone(int(a.shape[0] - z)))
        return cpo3.Problem(P=qp["P"], c=qp["q"], A=a, b=b, cones=cones)

    def run(spec, config, batch):
        problem = load(spec)
        settings = cpo3.Settings(**config)
        samples = []
        for _ in range(batch):
            start = time.perf_counter()
            try:
                sol = cpo3.solve(problem, settings=settings)
            except RuntimeError as exc:
                return {"status": "error", "error": str(exc)}
            samples.append(time.perf_counter() - start)
        elapsed = sum(samples) / len(samples)
        info = sol.info
        vectors = hashlib.sha256()
        for array in [sol.x, sol.y, sol.s]:
            vectors.update(np.asarray(array, dtype=np.float64).tobytes())
        result = {
            "status": sol.status.value,
            "samples": samples,
            "elapsed": elapsed,
            "solve_time": info.solve_time,
            "vectors_sha256": vectors.hexdigest(),
            "n": problem.variable_count,
            "m": problem.constraint_count,
            "cvxopt_init_used": info.cvxopt_init_used,
        }
        for field in [
            "iters",
            "factorizations",
            "primary_solves",
            "refinement_solves",
            "refinement_failures",
            "refinement_settled",
            "inertia_failures",
        ]:
            result[field] = getattr(info, field)
        for field in ["pobj", "dobj", "pres", "dres", "comp"]:
            result[field] = float(getattr(info, field)).hex()
        return result

    # Initialize factorization/library dispatch before any measured solve.
    cpo3.solve(
        cpo3.Problem(
            P=sparse.eye(1, format="csc"),
            c=[-1.0],
            A=sparse.csc_matrix([[-1.0]]),
            b=[0.0],
            cones=[cpo3.NonnegativeCone(1)],
        ),
        settings=cpo3.Settings(presolve=False),
    )
    for line in sys.stdin:
        request = json.loads(line)
        try:
            result = run(request["problem"], request["settings"], request.get("batch", 1))
        except Exception as exc:  # noqa: BLE001 — record loader failures for audit
            result = {"status": "harness_error", "error": repr(exc)}
        print(json.dumps(result), flush=True)


def manifest(args):
    sys.path.insert(0, args.harness)
    from solver_benchmarks.datasets.registry import get_dataset

    specs = []
    for path in sorted(Path(args.data).glob("*/*.mps.gz")):
        specs.append(
            {
                "id": path.parent.name + "/" + path.name.removesuffix(".mps.gz"),
                "loader": "mps",
                "dataset": path.parent.name,
                "name": path.stem,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    for name in ["mpc_qpbenchmark", "dc_opf"]:
        dataset = get_dataset(name)(args.harness)
        for spec in dataset.list_problems():
            specs.append(
                {
                    "id": name + "/" + spec.name,
                    "loader": "dataset",
                    "dataset": name,
                    "name": spec.name,
                    "path": str(spec.path),
                    "sha256": hashlib.sha256(spec.path.read_bytes()).hexdigest(),
                }
            )
    return specs


def compare(args):
    import selectors

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    specs = manifest(args)
    settings = {"presolve": False, "max_iters": 100, "time_limit": 60.0}
    description = {
        "problems": specs,
        "settings": settings,
        "repeats": args.repeats,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "threads": 1,
        "verification_tolerance": 1e-6,
        "baseline_binary_sha256": hashlib.sha256(
            (Path(args.baseline) / "cpo3/_core.abi3.so").read_bytes()
        ).hexdigest(),
        "candidate_binary_sha256": hashlib.sha256(
            (Path(args.candidate) / "cpo3/_core.abi3.so").read_bytes()
        ).hexdigest(),
    }
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == description, (
            "Run configuration changed"
        )
    else:
        manifest_path.write_text(json.dumps(description, indent=2) + "\n")
    path = output / "results.jsonl"
    done = (
        {
            (r["problem"], r["variant"], r["repeat"])
            for r in (json.loads(l) for l in path.read_text().splitlines())
            if r["result"]["status"] != "harness_error"
        }
        if path.exists()
        else set()
    )
    processes = {}

    def start(variant):
        env = dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            RAYON_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1",
        )
        log = (output / (variant + "-stderr.log")).open("a")
        proc = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "worker",
                "--package",
                getattr(args, variant),
                "--harness",
                args.harness,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            env=env,
        )
        processes[variant] = (proc, log)

    try:
        for i, spec in enumerate(specs):
            for repeat in range(args.repeats):
                order = (
                    ["baseline", "candidate"]
                    if (i + repeat) % 2 == 0
                    else ["candidate", "baseline"]
                )
                for variant in order:
                    if (spec["id"], variant, repeat) in done:
                        continue
                    if variant not in processes:
                        start(variant)
                    proc, log = processes[variant]
                    proc.stdin.write(
                        json.dumps({"problem": spec, "settings": settings}) + "\n"
                    )
                    proc.stdin.flush()
                    with selectors.DefaultSelector() as selector:
                        selector.register(proc.stdout, selectors.EVENT_READ)
                        ready = selector.select(120)
                    if ready:
                        line = proc.stdout.readline()
                        result = (
                            json.loads(line) if line else {"status": "worker_failed"}
                        )
                    else:
                        result = {"status": "external_timeout"}
                    if result["status"] in ["worker_failed", "external_timeout"]:
                        proc.kill()
                        proc.wait()
                        log.close()
                        del processes[variant]
                    record = {
                        "problem": spec["id"],
                        "dataset": spec["dataset"],
                        "variant": variant,
                        "repeat": repeat,
                        "result": result,
                    }
                    with path.open("a") as f:
                        f.write(json.dumps(record) + "\n")
            print(f"{i + 1}/{len(specs)} {spec['id']}", flush=True)
    finally:
        for proc, log in processes.values():
            proc.stdin.close()
            proc.wait()
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--package", required=True)
    w.add_argument("--harness", required=True)
    c = sub.add_parser("compare")
    for option in ["baseline", "candidate", "harness", "data", "output"]:
        c.add_argument("--" + option, required=True)
    c.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    (worker if args.command == "worker" else compare)(args)

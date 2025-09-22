#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import hashlib
import zlib
import math
from datetime import datetime
from pathlib import Path
from typing import Counter, Dict, Iterable, List, Sequence, Tuple, Optional

_COMBINE_OPTS: Dict[str, float | int] = {
    "bayes_iters": 2,
    "hmm_stay_prob": 0.95,
    "super_threshold": 0.75,
    "trim_k": 1,
}

def run_minipro(part: str, outfile: Path, minipro_path: str = "minipro", timeout: int | None = None, extra_args: Sequence[str] | None = None, force: bool = False) -> subprocess.CompletedProcess:
    args = [minipro_path, "-p", part, "-r", str(outfile)]
    if extra_args:
        args.extend(extra_args)
    if force:
        args.append("-y")
    return subprocess.run(args, check=False, capture_output=True, text=True, timeout=timeout)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def load_dumps(paths: Sequence[Path]) -> Tuple[List[bytearray], List[int]]:
    bufs: List[bytearray] = []
    lengths: List[int] = []
    for p in paths:
        data = p.read_bytes()
        bufs.append(bytearray(data))
        lengths.append(len(data))
    return bufs, lengths

def pick_target_length(lengths: Sequence[int], override: Optional[int]) -> int:
    length_counts = collections.Counter(lengths)
    if override is not None:
        target = min(override, min(lengths))
        if any(L < target for L in lengths):
            print(f"WARNING: Some inputs shorter than --length {override}; using min available length {target}")
        return target
    if len(length_counts) > 1:
        common_len, _ = length_counts.most_common(1)[0]
        target = common_len
        print(f"WARNING: Input lengths differ: {dict(length_counts)}; using length {target}")
        if any(L < target for L in lengths):
            target = min(lengths)
            print(f"NOTE: Reduced to min length {target} because some inputs shorter than common length.")
        return target
    return lengths[0]

def truncate_buffers(buffers: Sequence[bytearray], target_len: int) -> List[bytearray]:
    if any(len(b) != target_len for b in buffers):
        return [b[:target_len] for b in buffers]
    return list(buffers)

def combine(buffers: Sequence[bytearray], target_len: int, mode: str) -> Tuple[bytearray, int, int, float]:
    t0 = time.time()
    if mode == "byte":
        combined, disagree, ties = combine_majority_byte(buffers, target_len)
    elif mode == "bit":
        combined, disagree, ties = combine_majority_bit(buffers, target_len)
    elif mode == "bayes":
        iters = int(_COMBINE_OPTS.get("bayes_iters", 2))
        combined, disagree, ties = combine_bayes(buffers, target_len, iters=iters)
    elif mode == "hmm":
        stay_prob = float(_COMBINE_OPTS.get("hmm_stay_prob", 0.95))
        combined, disagree, ties = combine_hmm(buffers, target_len, stay_prob=stay_prob)
    elif mode == "super":
        thr = float(_COMBINE_OPTS.get("super_threshold", 0.75))
        combined, disagree, ties = combine_supermajority_bit(buffers, target_len, threshold=thr)
    elif mode == "medoid":
        combined, disagree, ties = combine_medoid(buffers, target_len)
    elif mode == "trim":
        k = int(_COMBINE_OPTS.get("trim_k", 1))
        combined, disagree, ties = combine_trimmed_majority(buffers, target_len, trim_k=k)
    else:
        combined, disagree, ties = combine_majority_bit(buffers, target_len)
    dt = time.time() - t0
    return combined, disagree, ties, dt

def majority_byte(values: Iterable[int]) -> Tuple[int, int, bool]:
    counter: Counter[int] = collections.Counter(values)
    if not counter:
        return 0, 0, False
    most_common = counter.most_common()
    top_count = most_common[0][1]
    winners = [b for b, c in most_common if c == top_count]
    tie = len(winners) > 1
    winner = min(winners)
    return winner, top_count, tie

def combine_majority_byte(buffers: Sequence[bytearray], length: int) -> Tuple[bytearray, int, int]:
    nbuf = len(buffers)
    out = bytearray(length)
    disagree = 0
    ties = 0
    for i in range(length):
        vals = [buf[i] for buf in buffers]
        winner, count, tie = majority_byte(vals)
        out[i] = winner
        if count != nbuf:
            disagree += 1
        if tie:
            ties += 1
    return out, disagree, ties

def combine_majority_bit(buffers: Sequence[bytearray], length: int) -> Tuple[bytearray, int, int]:
    nbuf = len(buffers)
    out = bytearray(length)
    disagree = 0
    tie_positions = 0
    half = nbuf / 2.0
    for i in range(length):
        bit_sum = 0
        val0 = buffers[0][i]
        result = 0
        tied_here = False
        for bit in range(8):
            ones = sum((buf[i] >> bit) & 1 for buf in buffers)
            if ones > half:
                result |= (1 << bit)
            elif ones == half:
                tied_here = True
                if (val0 >> bit) & 1:
                    result |= (1 << bit)

        out[i] = result
        if any(buf[i] != result for buf in buffers):
            disagree += 1
        if tied_here:
            tie_positions += 1
    return out, disagree, tie_positions

def combine_supermajority_bit(buffers: Sequence[bytearray], length: int, threshold: float = 0.75) -> Tuple[bytearray, int, int]:
    nbuf = len(buffers)
    if nbuf == 0:
        return bytearray(length), 0, 0
    out = bytearray(length)
    disagree = 0
    tie_positions = 0
    half = nbuf / 2.0
    thr = min(0.999999, max(0.5, float(threshold)))
    need = math.ceil(thr * nbuf)
    for i in range(length):
        val0 = buffers[0][i]
        result = 0
        tied_here = False
        for bit in range(8):
            ones = sum((buf[i] >> bit) & 1 for buf in buffers)
            zeros = nbuf - ones
            if ones >= need:
                result |= (1 << bit)
            elif zeros >= need:
                pass
            else:
                if ones > half:
                    result |= (1 << bit)
                elif ones == half:
                    tied_here = True
                    if (val0 >> bit) & 1:
                        result |= (1 << bit)
        out[i] = result
        if any(buf[i] != result for buf in buffers):
            disagree += 1
        if tied_here:
            tie_positions += 1
    return out, disagree, tie_positions

def combine_medoid(buffers: Sequence[bytearray], length: int) -> Tuple[bytearray, int, int]:
    nbuf = len(buffers)
    if nbuf == 0:
        return bytearray(length), 0, 0
    majority_out, _, tie_positions = combine_majority_bit(buffers, length)
    dists = [hamming_distance(buf[:length], majority_out) for buf in buffers]
    best_idx = min(range(nbuf), key=lambda k: dists[k])
    out = bytearray(buffers[best_idx][:length])
    disagree = sum(1 for i in range(length) if any(buf[i] != out[i] for buf in buffers))
    return out, disagree, tie_positions

def combine_trimmed_majority(buffers: Sequence[bytearray], length: int, trim_k: int = 1) -> Tuple[bytearray, int, int]:
    nbuf = len(buffers)
    if nbuf == 0:
        return bytearray(length), 0, 0
    if nbuf == 1 or trim_k <= 0:
        return combine_majority_bit(buffers, length) if nbuf > 1 else (bytearray(buffers[0][:length]), 0, 0)
    init, _, _ = combine_majority_bit(buffers, length)
    errs = _estimate_dump_bit_error_rates(buffers, length, initial=init)
    idxs = list(range(nbuf))
    idxs.sort(key=lambda i: errs[i], reverse=True)
    k = min(int(trim_k), max(0, nbuf - 2))
    keep = sorted(set(range(nbuf)) - set(idxs[:k]))
    kept_buffers = [buffers[i] for i in keep]
    if len(kept_buffers) >= 2:
        out, disagree, ties = combine_majority_bit(kept_buffers, length)
    else:
        best_idx = min(range(nbuf), key=lambda i: errs[i])
        out = bytearray(buffers[best_idx][:length])
        disagree = sum(1 for i in range(length) if any(buf[i] != out[i] for buf in buffers))
        ties = 0
    return out, disagree, ties

def _hamming_distance_bytes(a: bytes, b: bytes) -> int:
    return hamming_distance(a, b)

def _estimate_dump_bit_error_rates(buffers: Sequence[bytearray], length: int, initial: Optional[bytes] = None) -> List[float]:
    total_bits = max(1, length * 8)
    if initial is None:
        initial, _, _ = combine_majority_bit(buffers, length)
    errs: List[float] = []
    for buf in buffers:
        dist = _hamming_distance_bytes(buf[:length], initial)
        e = dist / total_bits
        e = min(0.499999, max(1.0 / total_bits, e))
        errs.append(e)
    return errs

def _weights_from_error_rates(errs: Sequence[float]) -> List[float]:
    ws: List[float] = []
    for e in errs:
        ws.append(math.log((1.0 - e) / e))
    return ws

def combine_bayes(buffers: Sequence[bytearray], length: int, iters: int = 2) -> Tuple[bytearray, int, int]:
    nbuf = len(buffers)
    if nbuf == 0:
        return bytearray(length), 0, 0
    out, _, _ = combine_majority_bit(buffers, length)
    errs = _estimate_dump_bit_error_rates(buffers, length, initial=out)
    weights = _weights_from_error_rates(errs)
    tol = 1e-15
    for _ in range(max(1, iters)):
        tie_positions = 0
        new_out = bytearray(length)
        for i in range(length):
            val0 = buffers[0][i]
            res = 0
            tie_here = False
            for bit in range(8):
                s = 0.0
                for k, buf in enumerate(buffers):
                    b = (buf[i] >> bit) & 1
                    s += weights[k] * (1 if b == 1 else -1)
                if abs(s) <= tol:
                    tie_here = True
                    if (val0 >> bit) & 1:
                        res |= (1 << bit)
                elif s > 0:
                    res |= (1 << bit)
            new_out[i] = res
            if tie_here:
                tie_positions += 1
        out = new_out
        errs = _estimate_dump_bit_error_rates(buffers, length, initial=out)
        weights = _weights_from_error_rates(errs)
    disagree = 0
    for i in range(length):
        if any(buf[i] != out[i] for buf in buffers):
            disagree += 1
    ties = 0
    tol = 1e-15
    for i in range(length):
        tie_here = False
        val0 = buffers[0][i]
        for bit in range(8):
            s = 0.0
            for k, buf in enumerate(buffers):
                b = (buf[i] >> bit) & 1
                s += weights[k] * (1 if b == 1 else -1)
            if abs(s) <= tol:
                tie_here = True
                break
        if tie_here:
            ties += 1
    return out, disagree, ties

def combine_hmm(buffers: Sequence[bytearray], length: int, stay_prob: float = 0.95) -> Tuple[bytearray, int, int]:
    nbuf = len(buffers)
    if nbuf == 0:
        return bytearray(length), 0, 0
    stay_prob = max(0.5, min(0.999999, stay_prob))
    change_prob = 1.0 - stay_prob
    log_stay = math.log(stay_prob)
    log_change = math.log(change_prob)
    log_pi = math.log(0.5)

    init, _, _ = combine_majority_bit(buffers, length)
    errs = _estimate_dump_bit_error_rates(buffers, length, initial=init)
    log_e = [math.log(e) for e in errs]
    log_1e = [math.log(1.0 - e) for e in errs]

    bitplanes: List[List[int]] = [[0] * length for _ in range(8)]
    tie_positions = 0
    tol = 1e-18

    for bit in range(8):
        emit0 = [0.0] * length
        emit1 = [0.0] * length
        for i in range(length):
            s0 = 0.0
            s1 = 0.0
            for k, buf in enumerate(buffers):
                b = (buf[i] >> bit) & 1
                if b == 0:
                    s0 += log_1e[k]
                    s1 += log_e[k]
                else:
                    s0 += log_e[k]
                    s1 += log_1e[k]
            emit0[i] = s0
            emit1[i] = s1

        dp0 = [float('-inf')] * length
        dp1 = [float('-inf')] * length
        bp0 = [0] * length
        bp1 = [0] * length

        dp0[0] = log_pi + emit0[0]
        dp1[0] = log_pi + emit1[0]
        tie_flags = [False] * length
        if abs(emit0[0] - emit1[0]) <= tol:
            tie_flags[0] = True

        for i in range(1, length):
            cand0_from0 = dp0[i-1] + log_stay
            cand0_from1 = dp1[i-1] + log_change
            if abs(cand0_from0 - cand0_from1) <= tol:
                bp0[i] = 0 if cand0_from0 >= cand0_from1 else 1
                tie_flags[i] = True
                prev0 = max(cand0_from0, cand0_from1)
            elif cand0_from0 >= cand0_from1:
                bp0[i] = 0
                prev0 = cand0_from0
            else:
                bp0[i] = 1
                prev0 = cand0_from1
            dp0[i] = prev0 + emit0[i]

            cand1_from0 = dp0[i-1] + log_change
            cand1_from1 = dp1[i-1] + log_stay
            if abs(cand1_from0 - cand1_from1) <= tol:
                bp1[i] = 0 if cand1_from0 > cand1_from1 else 1
                tie_flags[i] = True
                prev1 = max(cand1_from0, cand1_from1)
            elif cand1_from0 >= cand1_from1:
                bp1[i] = 0
                prev1 = cand1_from0
            else:
                bp1[i] = 1
                prev1 = cand1_from1
            dp1[i] = prev1 + emit1[i]

            if abs(emit0[i] - emit1[i]) <= tol:
                tie_flags[i] = True

        if dp1[-1] > dp0[-1]:
            state = 1
        elif dp0[-1] > dp1[-1]:
            state = 0
        else:
            ones = sum((buf[length-1] >> bit) & 1 for buf in buffers)
            state = 1 if ones >= (nbuf - ones) else 0
            tie_flags[-1] = True

        seq = [0] * length
        seq[length - 1] = state
        for i in range(length - 1, 0, -1):
            if state == 0:
                state = bp0[i]
            else:
                state = bp1[i]
            seq[i - 1] = state

        for i, s in enumerate(seq):
            bitplanes[bit][i] = s
        tie_positions += sum(1 for f in tie_flags if f)

    out = bytearray(length)
    for i in range(length):
        val = 0
        for bit in range(8):
            if bitplanes[bit][i]:
                val |= (1 << bit)
        out[i] = val

    disagree = sum(1 for i in range(length) if any(buf[i] != out[i] for buf in buffers))

    ties = 0
    for i in range(length):
        tie_here = False
        for bit in range(8):
            s0 = 0.0
            s1 = 0.0
            for k, buf in enumerate(buffers):
                b = (buf[i] >> bit) & 1
                if b == 0:
                    s0 += log_1e[k]
                    s1 += log_e[k]
                else:
                    s0 += log_e[k]
                    s1 += log_1e[k]
            if abs(s0 - s1) <= tol:
                tie_here = True
                break
        if tie_here:
            ties += 1

    return out, disagree, ties

def hamming_distance(a: bytes, b: bytes) -> int:
    return sum((x ^ y).bit_count() for x, y in zip(a, b))

def compute_mismatch_stats(buffers: Sequence[bytearray], combined: bytes) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for idx, buf in enumerate(buffers):
        stats[f"dump_{idx}_byte_mismatches"] = sum(1 for x, y in zip(buf, combined) if x != y)
        stats[f"dump_{idx}_bit_mismatches"] = hamming_distance(buf, combined)
    return stats

def compute_disagree_offsets(buffers: Sequence[bytearray], combined: bytes) -> List[int]:
    length = len(combined)
    offs: List[int] = []
    for i in range(length):
        if any(buf[i] != combined[i] for buf in buffers):
            offs.append(i)
    return offs

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multiple minipro dumps with majority consensus")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("-p", "--part", help="Minipro part name (e.g., AT28C256) when performing hardware dumps")
    src.add_argument("--inputs", nargs="*", help="Existing dump files to combine (skip running minipro)")

    p.add_argument("-n", "--count", type=int, default=0, help="Number of dumps to read with minipro (requires --part). Default 0")
    p.add_argument("-o", "--output", type=str, default="combined.bin", help="Output combined dump path")
    p.add_argument("--out-dir", type=str, default="dumps", help="Directory to store individual dumps when reading")
    p.add_argument("--mode", choices=["byte", "bit", "bayes", "hmm", "super", "medoid", "trim"], default="byte", help="Consensus mode: 'byte' (per-byte majority), 'bit' (per-bit majority), 'bayes' (Bayesian weighted voting), 'hmm' (Viterbi smoothing), 'super' (supermajority per-bit), 'medoid' (closest input to majority), or 'trim' (drop worst K then majority)")
    p.add_argument("--bayes-iters", type=int, default=2, help="Iterations of Bayesian reweighting (mode=bayes)")
    p.add_argument("--hmm-stay-prob", type=float, default=0.95, help="HMM stay probability for bit stability (mode=hmm; 0.5-0.999999)")
    p.add_argument("--super-threshold", type=float, default=0.75, help="Supermajority threshold in [0.5,1.0] for mode=super (e.g., 0.75)")
    p.add_argument("--trim-k", type=int, default=1, help="Number of worst dumps to drop by estimated BER for mode=trim")
    p.add_argument("--until-agree", action="store_true", help="Continuously read dumps until full agreement (0 disagreements) is achieved")
    p.add_argument("--max-reads", type=int, default=0, help="Maximum total reads when using --until-agree (0 = unlimited)")
    p.add_argument("--min-reads", type=int, default=2, help="Minimum number of reads before evaluating agreement in --until-agree mode")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between reads in --until-agree mode")
    p.add_argument("--minipro", default="minipro", help="Path to minipro executable")
    p.add_argument("--timeout", type=int, default=None, help="Per-read timeout seconds")
    p.add_argument("--extra-arg", action="append", default=[], help="Extra argument to pass to minipro; repeatable")
    p.add_argument("--force-minipro", action="store_true", help="Pass -y to minipro to skip confirmation prompts")
    p.add_argument("--report", type=str, default=None, help="Write a JSON report with statistics to this path")
    p.add_argument("--disagreements-out", type=str, default=None, help="Write newline-separated disagreeing byte offsets to this file")
    p.add_argument("--force", action="store_true", help="Overwrite existing output files")
    p.add_argument("--length", type=int, default=None, help="Override truncate length when input dump sizes differ")
    p.add_argument("--no-early-stop", action="store_true", help="With --count, do not stop early even if 100% reconstructed")

    return p.parse_args(argv)

def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    _COMBINE_OPTS["bayes_iters"] = int(max(1, args.bayes_iters)) if hasattr(args, "bayes_iters") else 2
    if hasattr(args, "hmm_stay_prob"):
        try:
            _COMBINE_OPTS["hmm_stay_prob"] = float(args.hmm_stay_prob)
        except Exception:
            _COMBINE_OPTS["hmm_stay_prob"] = 0.95
    if hasattr(args, "super_threshold"):
        try:
            _COMBINE_OPTS["super_threshold"] = float(args.super_threshold)
        except Exception:
            _COMBINE_OPTS["super_threshold"] = 0.75
    if hasattr(args, "trim_k"):
        try:
            _COMBINE_OPTS["trim_k"] = int(args.trim_k)
        except Exception:
            _COMBINE_OPTS["trim_k"] = 1
    if getattr(args, "length", None) is not None and args.length <= 0:
        print(f"WARNING: Ignoring non-positive --length {args.length}; using auto-detected length.")
        args.length = None
    tempdir_ctx: Optional[tempfile.TemporaryDirectory] = None
    temp_out_dir: Optional[Path] = None
    if args.part:
        tempdir_ctx = tempfile.TemporaryDirectory(prefix="minipro_dumps_")
        temp_out_dir = Path(tempdir_ctx.name)
        print(f"NOTE: Using temporary directory for hardware reads: {temp_out_dir}")

    try:
        out_path = Path(args.output)
        if out_path.exists() and not args.force:
            print(f"ERROR: Output path exists: {out_path} (use --force to overwrite)", file=sys.stderr)
            return 2

        dump_paths: List[Path] = []

        if args.inputs:
            for pat in args.inputs:
                for p in sorted(Path().glob(pat)) if any(ch in pat for ch in "*?[]") else [Path(pat)]:
                    if p.exists():
                        dump_paths.append(p)
                    else:
                        print(f"WARNING: input file not found: {p}", file=sys.stderr)

        if args.part and args.count and args.count > 0:
            out_dir = temp_out_dir if temp_out_dir is not None else Path(args.out_dir)
            ensure_dir(out_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            early_stopped = False
            last_combined: Optional[bytes] = None
            last_len = 0
            for i in range(1, args.count + 1):
                fname = f"dump_{timestamp}_{i:03}.bin"
                path = out_dir / fname
                print(f"[minipro] Reading {args.part} -> {path}")
                proc = run_minipro(args.part, path, minipro_path=args.minipro, timeout=args.timeout, extra_args=args.extra_arg, force=args.force_minipro)
                if proc.returncode != 0:
                    print("ERROR: minipro failed:", file=sys.stderr)
                    sys.stderr.write(proc.stdout or "")
                    sys.stderr.write(proc.stderr or "")
                    return proc.returncode
                dump_paths.append(path)
                buffers, lengths = load_dumps(dump_paths)
                target_len = pick_target_length(lengths, args.length)
                buffers = truncate_buffers(buffers, target_len)
                combined, disagree, ties, dt = combine(buffers, target_len, args.mode)
                total_bytes = max(1, target_len)
                agree_bytes = total_bytes - disagree
                agree_pct = 100.0 * agree_bytes / total_bytes
                print(
                    f"[stats] reads={i}/{args.count} inputs={len(buffers)} len={target_len} "
                    f"disagreements={disagree} reconstructed={agree_bytes} ({agree_pct:.4f}%)"
                )
                last_combined = combined
                last_len = target_len
                if disagree == 0 and len(buffers) >= 2 and not args.no_early_stop and not args.until_agree:
                    print("Achieved 100% reconstruction; stopping further reads due to --count early stop.")
                    early_stopped = True
                    break

        if args.until_agree:
            if not args.part:
                print("ERROR: --until-agree requires --part to read from hardware", file=sys.stderr)
                return 2
            out_dir = temp_out_dir if temp_out_dir is not None else Path(args.out_dir)
            ensure_dir(out_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if not dump_paths and args.count <= 0:
                seed_reads = max(args.min_reads, 2)
            else:
                seed_reads = 0

            total_reads = 0
            iterations = 0
            stopped_reason = ""
            full_agreement = False
            paths: List[Path] = list(dump_paths)

            def do_read(idx: int) -> Optional[Path]:
                fname = f"dump_{timestamp}_{idx:03}.bin"
                path = out_dir / fname
                print(f"[minipro] Reading {args.part} -> {path}")
                proc = run_minipro(args.part, path, minipro_path=args.minipro, timeout=args.timeout, extra_args=args.extra_arg, force=args.force_minipro)
                if proc.returncode != 0:
                    print("ERROR: minipro failed:", file=sys.stderr)
                    sys.stderr.write(proc.stdout or "")
                    sys.stderr.write(proc.stderr or "")
                    return None
                return path

            while seed_reads > 0:
                p = do_read(total_reads + 1)
                if p is None:
                    return 1
                paths.append(p)
                total_reads += 1
                seed_reads -= 1

                buffers, lengths = load_dumps(paths)
                target_len = pick_target_length(lengths, args.length)
                buffers = truncate_buffers(buffers, target_len)
                combined, disagree, ties, dt = combine(buffers, target_len, args.mode)
                total_bytes = max(1, target_len)
                agree_bytes = total_bytes - disagree
                agree_pct = 100.0 * agree_bytes / total_bytes
                print(
                    f"[stats] reads={total_reads} inputs={len(buffers)} len={target_len} "
                    f"disagreements={disagree} reconstructed={agree_bytes} ({agree_pct:.4f}%)"
                )
                if args.sleep > 0:
                    time.sleep(args.sleep)

            loop_start = time.time()
            while True:
                if not paths:
                    print("ERROR: No input dumps available to evaluate", file=sys.stderr)
                    return 2
                buffers, lengths = load_dumps(paths)
                target_len = pick_target_length(lengths, args.length)
                buffers = truncate_buffers(buffers, target_len)
                combined, disagree, ties, dt = combine(buffers, target_len, args.mode)
                iterations += 1

                total_bytes = max(1, target_len)
                agree_bytes = total_bytes - disagree
                agree_pct = 100.0 * agree_bytes / total_bytes
                tie_pct = 100.0 * ties / total_bytes
                elapsed = time.time() - loop_start
                print(
                    f"[stats] reads={total_reads} iter={iterations} inputs={len(buffers)} len={target_len} "
                    f"disagreements={disagree} reconstructed={agree_bytes} ({agree_pct:.4f}%)"
                )
                if disagree == 0:
                    full_agreement = True
                    stopped_reason = "full_agreement"
                    break
                if args.max_reads and (total_reads >= args.max_reads):
                    stopped_reason = "max_reads_reached"
                    break
                p = do_read(total_reads + 1)
                if p is None:
                    return 1
                paths.append(p)
                total_reads += 1
                if args.sleep > 0:
                    time.sleep(args.sleep)

            dump_paths = paths
            buffers, lengths = load_dumps(dump_paths)
            target_len = pick_target_length(lengths, args.length)
            buffers = truncate_buffers(buffers, target_len)
            combined, disagree, ties, dt = combine(buffers, target_len, args.mode)
            out_path.write_bytes(combined)
            print(f"Wrote combined dump: {out_path} ({len(combined)} bytes)")

            if args.disagreements_out:
                offsets = compute_disagree_offsets(buffers, combined)
                Path(args.disagreements_out).write_text("\n".join(str(i) for i in offsets) + ("\n" if offsets else ""))
                print(f"Wrote {len(offsets)} disagreeing offsets to: {args.disagreements_out}")

            stats = {
                "mode": args.mode,
                "inputs": [str(p) for p in dump_paths],
                "output": str(out_path),
                "length": len(combined),
                "disagreements_count": disagree,
                "ties_count": ties,
                "num_inputs": len(buffers),
                "full_agreement": full_agreement,
                "iterations": iterations,
                "stopped_reason": stopped_reason,
            }
            stats.update(compute_mismatch_stats(buffers, combined))

            print(f"Bytes with disagreements: {disagree}")
            if ties:
                print(f"Bytes with ties: {ties} (resolved deterministically)")

            total_bytes = max(1, len(combined))
            total_bits = total_bytes * 8
            agree_bytes = total_bytes - disagree
            agree_pct = 100.0 * agree_bytes / total_bytes
            tie_pct = 100.0 * ties / total_bytes

            per_dump = []
            for idx in range(len(buffers)):
                b_key = f"dump_{idx}_byte_mismatches"
                bit_key = f"dump_{idx}_bit_mismatches"
                b_mis = stats.get(b_key, 0)
                bit_mis = stats.get(bit_key, 0)
                per_dump.append({
                    "index": idx,
                    "byte_mismatches": b_mis,
                    "byte_mismatch_percent": (100.0 * b_mis / total_bytes) if total_bytes else 0.0,
                    "bit_mismatches": bit_mis,
                    "bit_error_rate": (bit_mis / total_bits) if total_bits else 0.0,
                })

            best = min(per_dump, key=lambda d: d["bit_mismatches"]) if per_dump else None
            worst_all = max(per_dump, key=lambda d: d["bit_mismatches"]) if per_dump else None

            md5 = hashlib.md5(combined).hexdigest()
            crc32 = zlib.crc32(combined) & 0xFFFFFFFF

            stats.update({
                "agreement_bytes": agree_bytes,
                "agreement_percent": agree_pct,
                "tie_percent": tie_pct,
                "hashes": {"md5": md5, "crc32": f"{crc32:08x}"},
                "per_dump": per_dump,
            })

            print("Consensus summary:")
            print(f"  Inputs: {len(buffers)} | Mode: {args.mode} | Length: {total_bytes} bytes")
            print(f"  Reconstruction: disagreements={disagree} | reconstructed={agree_bytes}/{total_bytes} ({agree_pct:.4f}%)")
            if ties:
                print(f"  Ties: {ties} ({tie_pct:.6f}%)")
            print(f"  Checksums: MD5 {md5} | CRC32 {crc32:08x}")

            if args.report:
                Path(args.report).write_text(json.dumps(stats, indent=2))
                print(f"Wrote report: {args.report}")

            return 0

        if not dump_paths:
            print("ERROR: No input dumps provided or collected. Use --inputs or specify --part and --count.", file=sys.stderr)
            return 2

        print(f"Combining {len(dump_paths)} dumps...")
        buffers, lengths = load_dumps(dump_paths)
        target_len = pick_target_length(lengths, args.length)
        buffers = truncate_buffers(buffers, target_len)

        combined, disagree, ties, dt = combine(buffers, target_len, args.mode)

        out_path.write_bytes(combined)
        print(f"Wrote combined dump: {out_path} ({len(combined)} bytes) in {dt:.2f}s")

        if args.disagreements_out:
            offsets = compute_disagree_offsets(buffers, combined)
            Path(args.disagreements_out).write_text("\n".join(str(i) for i in offsets) + ("\n" if offsets else ""))
            print(f"Wrote {len(offsets)} disagreeing offsets to: {args.disagreements_out}")

        stats = {
            "mode": args.mode,
            "inputs": [str(p) for p in dump_paths],
            "output": str(out_path),
            "length": len(combined),
            "disagreements_count": disagree,
            "ties_count": ties,
            "num_inputs": len(buffers),
            "full_agreement": (disagree == 0),
            "iterations": 1,
            "stopped_reason": "full_agreement" if disagree == 0 else "one_shot",
        }
        stats.update(compute_mismatch_stats(buffers, combined))

        print(f"Bytes with disagreements: {disagree}")
        if ties:
            print(f"Bytes with ties: {ties} (resolved deterministically)")

        total_bytes = max(1, len(combined))
        total_bits = total_bytes * 8
        agree_bytes = total_bytes - disagree
        agree_pct = 100.0 * agree_bytes / total_bytes
        tie_pct = 100.0 * ties / total_bytes

        per_dump = []
        for idx in range(len(buffers)):
            b_key = f"dump_{idx}_byte_mismatches"
            bit_key = f"dump_{idx}_bit_mismatches"
            b_mis = stats.get(b_key, 0)
            bit_mis = stats.get(bit_key, 0)
            per_dump.append({
                "index": idx,
                "byte_mismatches": b_mis,
                "byte_mismatch_percent": (100.0 * b_mis / total_bytes) if total_bytes else 0.0,
                "bit_mismatches": bit_mis,
                "bit_error_rate": (bit_mis / total_bits) if total_bits else 0.0,
            })

        best = min(per_dump, key=lambda d: d["bit_mismatches"]) if per_dump else None
        worst_all = max(per_dump, key=lambda d: d["bit_mismatches"]) if per_dump else None

        md5 = hashlib.md5(combined).hexdigest()
        crc32 = zlib.crc32(combined) & 0xFFFFFFFF

        stats.update({
            "agreement_bytes": agree_bytes,
            "agreement_percent": agree_pct,
            "tie_percent": tie_pct,
            "hashes": {"md5": md5, "crc32": f"{crc32:08x}"},
            "per_dump": per_dump,
        })

        print("Consensus summary:")
        print(f"  Inputs: {len(buffers)} | Mode: {args.mode} | Length: {total_bytes} bytes")
        print(f"  Reconstruction: disagreements={disagree} | reconstructed={agree_bytes}/{total_bytes} ({agree_pct:.4f}%)")
        if ties:
            print(f"  Ties: {ties} ({tie_pct:.6f}%)")
        print(f"  Checksums: MD5 {md5} | CRC32 {crc32:08x}")

        if args.report:
            Path(args.report).write_text(json.dumps(stats, indent=2))
            print(f"Wrote report: {args.report}")

        return 0
    finally:
        if tempdir_ctx is not None:
            tempdir_ctx.cleanup()

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

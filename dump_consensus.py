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
from datetime import datetime
from pathlib import Path
from typing import Counter, Dict, Iterable, List, Sequence, Tuple, Optional

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

def hamming_distance(a: bytes, b: bytes) -> int:
    return sum((x ^ y).bit_count() for x, y in zip(a, b))

def compute_mismatch_stats(buffers: Sequence[bytearray], combined: bytes) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for idx, buf in enumerate(buffers):
        stats[f"dump_{idx}_byte_mismatches"] = sum(1 for x, y in zip(buf, combined) if x != y)
        stats[f"dump_{idx}_bit_mismatches"] = hamming_distance(buf, combined)
    return stats

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multiple minipro dumps with majority consensus")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("-p", "--part", help="Minipro part name (e.g., AT28C256) when performing hardware dumps")
    src.add_argument("--inputs", nargs="*", help="Existing dump files to combine (skip running minipro)")

    p.add_argument("-n", "--count", type=int, default=0, help="Number of dumps to read with minipro (requires --part). Default 0")
    p.add_argument("-o", "--output", type=str, default="combined.bin", help="Output combined dump path")
    p.add_argument("--out-dir", type=str, default="dumps", help="Directory to store individual dumps when reading")
    p.add_argument("--mode", choices=["byte", "bit"], default="byte", help="Consensus mode: per-byte or per-bit majority")
    p.add_argument("--until-agree", action="store_true", help="Continuously read dumps until full agreement (0 disagreements) is achieved")
    p.add_argument("--max-reads", type=int, default=0, help="Maximum total reads when using --until-agree (0 = unlimited)")
    p.add_argument("--min-reads", type=int, default=2, help="Minimum number of reads before evaluating agreement in --until-agree mode")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between reads in --until-agree mode")
    p.add_argument("--minipro", default="minipro", help="Path to minipro executable")
    p.add_argument("--timeout", type=int, default=None, help="Per-read timeout seconds")
    p.add_argument("--extra-arg", action="append", default=[], help="Extra argument to pass to minipro; repeatable")
    p.add_argument("--force_minipro", action="store_true", help="Pass -y to minipro to skip confirmation prompts")
    p.add_argument("--report", type=str, default=None, help="Write a JSON report with statistics to this path")
    p.add_argument("--force", action="store_true", help="Overwrite existing output files")
    p.add_argument("--length", type=int, default=None, help="Override truncate length when input dump sizes differ")

    return p.parse_args(argv)

def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
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
                    (
                        f"[progress] iter={iterations} inputs={len(buffers)} len={target_len} "
                        f"disagree={disagree} agree={agree_pct:.4f}% ties={ties} ({tie_pct:.6f}%) "
                        f"combine={dt:.3f}s elapsed={elapsed:.1f}s reads={total_reads}"
                    )
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

            stats = {
                "mode": args.mode,
                "inputs": [str(p) for p in dump_paths],
                "output": str(out_path),
                "length": len(combined),
                "disagree_offsets": disagree,
                "tie_offsets": ties,
                "num_inputs": len(buffers),
                "full_agreement": full_agreement,
                "iterations": iterations,
                "stopped_reason": stopped_reason,
            }
            stats.update(compute_mismatch_stats(buffers, combined))

            print(f"Offsets with disagreements: {disagree}")
            if ties:
                print(f"Offsets with ties: {ties} (resolved deterministically)")

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
            print(f"  Full-agreement bytes: {agree_bytes}/{total_bytes} ({agree_pct:.4f}%)")
            if ties:
                print(f"  Tie bytes: {ties} ({tie_pct:.6f}%)")
            if best is not None and worst_all is not None:
                print(
                    "  Best dump vs combined: #{} | byte-mis: {} ({:.4f}%) | bit-mis: {} (BER {:.6e})".format(
                        best["index"],
                        best["byte_mismatches"],
                        best["byte_mismatch_percent"],
                        best["bit_mismatches"],
                        best["bit_error_rate"],
                    )
                )
                print(
                    "  Worst dump vs combined: #{} | byte-mis: {} ({:.4f}%) | bit-mis: {} (BER {:.6e})".format(
                        worst_all["index"],
                        worst_all["byte_mismatches"],
                        worst_all["byte_mismatch_percent"],
                        worst_all["bit_mismatches"],
                        worst_all["bit_error_rate"],
                    )
                )
            print(f"  Checksums: MD5 {md5} | CRC32 {crc32:08x}")

            bit_keys = sorted([k for k in stats if k.endswith("bit_mismatches")])
            worst = sorted(((k, stats[k]) for k in bit_keys), key=lambda kv: kv[1], reverse=True)[:3]
            if worst:
                print("Top bit-mismatch counts:")
                for k, v in worst:
                    print(f"  {k}: {v}")

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

        stats = {
            "mode": args.mode,
            "inputs": [str(p) for p in dump_paths],
            "output": str(out_path),
            "length": len(combined),
            "disagree_offsets": disagree,
            "tie_offsets": ties,
            "num_inputs": len(buffers),
            "full_agreement": (disagree == 0),
            "iterations": 1,
            "stopped_reason": "full_agreement" if disagree == 0 else "one_shot",
        }
        stats.update(compute_mismatch_stats(buffers, combined))

        print(f"Offsets with disagreements: {disagree}")
        if ties:
            print(f"Offsets with ties: {ties} (resolved deterministically)")

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
        print(f"  Full-agreement bytes: {agree_bytes}/{total_bytes} ({agree_pct:.4f}%)")
        if ties:
            print(f"  Tie bytes: {ties} ({tie_pct:.6f}%)")
        if best is not None and worst_all is not None:
            print(
                "  Best dump vs combined: #{} | byte-mis: {} ({:.4f}%) | bit-mis: {} (BER {:.6e})".format(
                    best["index"],
                    best["byte_mismatches"],
                    best["byte_mismatch_percent"],
                    best["bit_mismatches"],
                    best["bit_error_rate"],
                )
            )
            print(
                "  Worst dump vs combined: #{} | byte-mis: {} ({:.4f}%) | bit-mis: {} (BER {:.6e})".format(
                    worst_all["index"],
                    worst_all["byte_mismatches"],
                    worst_all["byte_mismatch_percent"],
                    worst_all["bit_mismatches"],
                    worst_all["bit_error_rate"],
                )
            )
        print(f"  Checksums: MD5 {md5} | CRC32 {crc32:08x}")

        bit_keys = sorted([k for k in stats if k.endswith("bit_mismatches")])
        worst = sorted(((k, stats[k]) for k in bit_keys), key=lambda kv: kv[1], reverse=True)[:3]
        if worst:
            print("Top bit-mismatch counts:")
            for k, v in worst:
                print(f"  {k}: {v}")

        if args.report:
            Path(args.report).write_text(json.dumps(stats, indent=2))
            print(f"Wrote report: {args.report}")

        return 0
    finally:
        if tempdir_ctx is not None:
            tempdir_ctx.cleanup()

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

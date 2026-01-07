#!/usr/bin/env python3
# compare_gpt2_weights.py

import os
import argparse
import numpy as np

HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4  # int32

def read_header(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=HEADER_INTS)
    if header.size != HEADER_INTS:
        raise RuntimeError(f"{path}: cannot read full header (got {header.size} ints)")
    return header

def param_layout_from_header(header: np.ndarray):
    # matches your C code:
    # header[2]=maxT, [3]=V, [4]=L, [5]=NH, [6]=C, [7]=Vp
    maxT = int(header[2])
    V = int(header[3])
    L = int(header[4])
    NH = int(header[5])
    C = int(header[6])
    Vp = int(header[7])

    # sizes exactly like fill_in_parameter_sizes()
    sizes = []
    names = []

    def add(name, n):
        names.append(name)
        sizes.append(int(n))

    add("wte", Vp * C)
    add("wpe", maxT * C)
    add("ln1w", L * C)
    add("ln1b", L * C)
    add("qkvw", L * (3 * C) * C)
    add("qkvb", L * (3 * C))
    add("attprojw", L * C * C)
    add("attprojb", L * C)
    add("ln2w", L * C)
    add("ln2b", L * C)
    add("fcw", L * (4 * C) * C)
    add("fcb", L * (4 * C))
    add("fcprojw", L * C * (4 * C))
    add("fcprojb", L * C)
    add("lnfw", C)
    add("lnfb", C)

    total = sum(sizes)

    # offsets (in floats)
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)

    cfg = dict(maxT=maxT, V=V, Vp=Vp, L=L, NH=NH, C=C, total=total)
    layout = list(zip(names, offsets, sizes))
    return cfg, layout

def file_num_floats(path: str) -> int:
    sz = os.path.getsize(path)
    if sz < HEADER_BYTES:
        raise RuntimeError(f"{path}: file too small ({sz} bytes)")
    data_bytes = sz - HEADER_BYTES
    if data_bytes % 4 != 0:
        raise RuntimeError(f"{path}: data bytes not multiple of 4: {data_bytes}")
    return data_bytes // 4

def memmap_weights(path: str, n_floats: int) -> np.memmap:
    return np.memmap(path, dtype=np.float32, mode="r", offset=HEADER_BYTES, shape=(n_floats,))

def chunked_metrics(w_base, w_other, chunk_elems: int = 8_000_000):
    # Returns cosine, rel_l2, max_abs
    dot = 0.0
    n0 = 0.0
    n1 = 0.0
    diff2 = 0.0
    max_abs = 0.0

    N = w_base.shape[0]
    for i in range(0, N, chunk_elems):
        a = np.asarray(w_base[i:i+chunk_elems], dtype=np.float64)
        b = np.asarray(w_other[i:i+chunk_elems], dtype=np.float64)

        dot += float(np.dot(a, b))
        n0 += float(np.dot(a, a))
        n1 += float(np.dot(b, b))

        d = b - a
        diff2 += float(np.dot(d, d))
        max_abs = max(max_abs, float(np.max(np.abs(d))))

    cos = dot / (np.sqrt(n0) * np.sqrt(n1) + 1e-30)
    rel_l2 = np.sqrt(diff2) / (np.sqrt(n0) + 1e-30)
    return cos, rel_l2, max_abs

def per_tensor_metrics(w_base, w_other, layout, chunk_elems: int = 2_000_000):
    rows = []
    for name, off, size in layout:
        a = w_base[off:off+size]
        b = w_other[off:off+size]
        cos, rel_l2, max_abs = chunked_metrics(a, b, chunk_elems=chunk_elems)
        rows.append((name, size, cos, rel_l2, max_abs))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gpt2_trained.bin")
    ap.add_argument("--u1",   default="gpt2_update1_trained.bin")
    ap.add_argument("--u2",   default="gpt2_update2_trained.bin")
    ap.add_argument("--chunk", type=int, default=8_000_000, help="chunk size in floats")
    ap.add_argument("--per_tensor", action="store_true", help="also print per-tensor metrics")
    args = ap.parse_args()

    # read headers and check compatibility
    hb = read_header(args.base)
    h1 = read_header(args.u1)
    h2 = read_header(args.u2)

    # quick header check (magic/version + key config)
    key_idx = [0,1,2,3,4,5,6,7]
    if not np.array_equal(hb[key_idx], h1[key_idx]):
        raise RuntimeError("base vs update1 header/config mismatch (maxT/V/L/NH/C/Vp differ)")
    if not np.array_equal(hb[key_idx], h2[key_idx]):
        raise RuntimeError("base vs update2 header/config mismatch (maxT/V/L/NH/C/Vp differ)")

    cfg, layout = param_layout_from_header(hb)

    # validate file sizes
    nb = file_num_floats(args.base)
    n1 = file_num_floats(args.u1)
    n2 = file_num_floats(args.u2)
    if nb != cfg["total"] or n1 != cfg["total"] or n2 != cfg["total"]:
        raise RuntimeError(
            f"float counts mismatch: expected {cfg['total']} "
            f"got base={nb}, u1={n1}, u2={n2}"
        )

    w_base = memmap_weights(args.base, cfg["total"])
    w_u1   = memmap_weights(args.u1,   cfg["total"])
    w_u2   = memmap_weights(args.u2,   cfg["total"])

    print("=== Config ===")
    print(cfg)
    print()

    print("=== Overall similarity vs base ===")
    cos1, rel1, max1 = chunked_metrics(w_base, w_u1, chunk_elems=args.chunk)
    cos2, rel2, max2 = chunked_metrics(w_base, w_u2, chunk_elems=args.chunk)

    print(f"update1 vs base: cosine={cos1:.8f}  rel_l2={rel1:.8e}  max_abs={max1:.8e}")
    print(f"update2 vs base: cosine={cos2:.8f}  rel_l2={rel2:.8e}  max_abs={max2:.8e}")

    if args.per_tensor:
        print("\n=== Per-tensor similarity vs base ===")
        rows1 = per_tensor_metrics(w_base, w_u1, layout)
        rows2 = per_tensor_metrics(w_base, w_u2, layout)

        print("\n-- update1 vs base --")
        for name, size, cos, rel_l2, max_abs in rows1:
            print(f"{name:10s}  n={size:12d}  cos={cos:.8f}  rel_l2={rel_l2:.3e}  max_abs={max_abs:.3e}")

        print("\n-- update2 vs base --")
        for name, size, cos, rel_l2, max_abs in rows2:
            print(f"{name:10s}  n={size:12d}  cos={cos:.8f}  rel_l2={rel_l2:.3e}  max_abs={max_abs:.3e}")

if __name__ == "__main__":
    main()


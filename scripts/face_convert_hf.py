#!/usr/bin/env python3
"""
face_convert_hf.py

Output structure under --out-dir:
  onnx/
    fp32/
      detection.onnx
      recognition.onnx
  rknn/
    fp32/
      detection.rknn      (if rknn-toolkit2 available)
      recognition.rknn    (if rknn-toolkit2 available)
  tensorrt/
    fp16/
      detection.engine    (if trtexec and shape provided)
      recognition.engine  (if trtexec and shape provided)

Optional: upload the entire folder to Hugging Face Hub.

Example:
  python Lumen/scripts/face_convert_hf.py \
    --input-dir /models/face \
    --out-dir /models/face_export \
    --target-platform rk3588 \
    --trt-input-name input \
    --det-shape 1x3x640x640 \
    --rec-shape 1x3x112x112 \
    --hf-repo-id your-org/face-two-heads --private

Notes:
- Running on Linux distributions only
- TensorRT builds require `trtexec` in PATH and the correct input tensor name and static shape.
- RKNN requires `rknn-toolkit2`. If missing, RKNN conversion is skipped.
"""

import argparse
import os
import sys
import shutil
import subprocess
from typing import Optional, List

# Optional deps
try:
    from rknn.api import RKNN  # type: ignore
except Exception:
    RKNN = None

try:
    from huggingface_hub import HfApi, create_repo, upload_folder  # type: ignore
except Exception:
    HfApi = None
    create_repo = None
    upload_folder = None


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def which(cmd: str) -> Optional[str]:
    try:
        return shutil.which(cmd)
    except Exception:
        return None


def run_cmd(cmd: List[str]) -> int:
    print(f"[CMD] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    return proc.returncode


def copy_file(src: str, dst: str) -> bool:
    try:
        ensure_dir(os.path.dirname(dst))
        shutil.copy2(src, dst)
        print(f"[COPY] {src} -> {dst}")
        return True
    except Exception as ex:
        eprint(f"[COPY] Failed {src} -> {dst}: {ex}")
        return False


def build_rknn_fp32(onnx_path: str, out_path: str, target_platform: str) -> bool:
    if RKNN is None:
        eprint("[RKNN] rknn-toolkit2 not available; skipping:", onnx_path)
        return False
    ensure_dir(os.path.dirname(out_path))
    rknn = RKNN()
    try:
        rknn.config(target_platform=target_platform)
        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            eprint(f"[RKNN] load_onnx failed: {onnx_path}")
            return False
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            eprint(f"[RKNN] build failed: {onnx_path}")
            return False
        ret = rknn.export_rknn(out_path)
        if ret != 0:
            eprint(f"[RKNN] export_rknn failed: {out_path}")
            return False
        print(f"[RKNN] Saved: {out_path}")
        return True
    except Exception as ex:
        eprint(f"[RKNN] Error on {onnx_path}: {ex}")
        return False
    finally:
        try:
            rknn.release()
        except Exception:
            pass


def parse_shape(s: Optional[str]) -> Optional[str]:
    """
    Accepts strings like "1x3x640x640". Returns the same if valid, else None.
    """
    if not s:
        return None
    parts = s.lower().split("x")
    if len(parts) != 4:
        return None
    try:
        _ = [int(p) for p in parts]
        return s
    except Exception:
        return None


def build_trt_fp16(trtexec_path: str, onnx_path: str, engine_path: str, input_name: str, shape_str: str) -> bool:
    ensure_dir(os.path.dirname(engine_path))
    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--shapes={input_name}:{shape_str}",
    ]
    code = run_cmd(cmd)
    ok = code == 0 and os.path.isfile(engine_path)
    if ok:
        print(f"[TRT FP16] Saved: {engine_path}")
    else:
        eprint(f"[TRT FP16] Failed (code {code}): {engine_path}")
    return ok


def hf_upload(repo_id: Optional[str], folder_path: str, token: Optional[str], private: bool, branch: Optional[str]) -> bool:
    if not repo_id:
        print("[HF] No repo_id provided; skipping upload.")
        return False
    if HfApi is None or (create_repo is None and upload_folder is None):
        eprint("[HF] huggingface_hub not available; cannot upload.")
        return False
    try:
        # Create repo if needed
        if create_repo is not None:
            create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
        else:
            api = HfApi()
            api.create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
        # Upload folder
        if upload_folder is not None:
            upload_folder(
                repo_id=repo_id,
                folder_path=folder_path,
                token=token,
                commit_message="Upload converted FP32 ONNX, FP32 RKNN, FP16 TensorRT artifacts",
                revision=branch or "main",
            )
        else:
            api = HfApi()
            api.upload_folder(
                repo_id=repo_id,
                folder_path=folder_path,
                token=token,
                commit_message="Upload converted FP32 ONNX, FP32 RKNN, FP16 TensorRT artifacts",
                revision=branch or "main",
            )
        print(f"[HF] Uploaded {folder_path} -> {repo_id}@{branch or 'main'}")
        return True
    except Exception as ex:
        eprint(f"[HF] Upload failed: {ex}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX FP32 (copy), RKNN FP32 (no quant), TensorRT FP16 (trtexec).")
    parser.add_argument("--input-dir", required=True, help="Directory containing detection.onnx and recognition.onnx")
    parser.add_argument("--detection-onnx", default="detection.onnx", help="Filename for detection ONNX inside input-dir")
    parser.add_argument("--recognition-onnx", default="recognition.onnx", help="Filename for recognition ONNX inside input-dir")
    parser.add_argument("--out-dir", default=None, help="Output directory. Default: <input-dir>/export")
    parser.add_argument("--target-platform", default="rk3588", help="RKNN target platform (e.g., rk3566, rk3568, rk3588, rv1106)")

    # TensorRT
    parser.add_argument("--trt-input-name", default="input", help="TensorRT input tensor name for both heads (can be overridden per head)")
    parser.add_argument("--det-trt-input-name", default=None, help="Override TensorRT input name for detection")
    parser.add_argument("--rec-trt-input-name", default=None, help="Override TensorRT input name for recognition")
    parser.add_argument("--det-shape", default=None, help="Detection static shape as 'NxCxHxW' for trtexec --shapes (e.g., 1x3x640x640)")
    parser.add_argument("--rec-shape", default=None, help="Recognition static shape as 'NxCxHxW' for trtexec --shapes (e.g., 1x3x112x112)")

    # Hugging Face Hub (optional)
    parser.add_argument("--hf-repo-id", default=None, help="Hugging Face repo id (e.g., 'org/name'). If omitted, no upload.")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", None), help="HF token (or set HF_TOKEN env).")
    parser.add_argument("--branch", default=None, help="Repo branch to push to (default: main)")
    parser.add_argument("--private", action="store_true", help="Create repo as private")

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    det_onnx = os.path.join(input_dir, args.detection_onnx)
    rec_onnx = os.path.join(input_dir, args.recognition_onnx)
    if not os.path.isfile(det_onnx) or not os.path.isfile(rec_onnx):
        eprint(f"Missing input ONNX files. detection: {det_onnx}, recognition: {rec_onnx}")
        sys.exit(1)

    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(input_dir, "export")
    onnx_fp32_dir = os.path.join(out_dir, "onnx", "fp32")
    rknn_fp32_dir = os.path.join(out_dir, "rknn", "fp32")
    trt_fp16_dir = os.path.join(out_dir, "tensorrt", "fp16")
    for d in [onnx_fp32_dir, rknn_fp32_dir, trt_fp16_dir]:
        ensure_dir(d)

    # 1) Copy ONNX as FP32
    copy_file(det_onnx, os.path.join(onnx_fp32_dir, "detection.onnx"))
    copy_file(rec_onnx, os.path.join(onnx_fp32_dir, "recognition.onnx"))

    # 2) RKNN FP32 (no quant)
    det_rknn_fp32 = os.path.join(rknn_fp32_dir, "detection.rknn")
    rec_rknn_fp32 = os.path.join(rknn_fp32_dir, "recognition.rknn")
    build_rknn_fp32(det_onnx, det_rknn_fp32, args.target_platform)
    build_rknn_fp32(rec_onnx, rec_rknn_fp32, args.target_platform)

    # 3) TensorRT FP16 (trtexec)
    trtexec = which("trtexec")
    if trtexec:
        det_shape = parse_shape(args.det_shape)
        rec_shape = parse_shape(args.rec_shape)
        det_input_name = args.det_trt_input_name or args.trt_input_name
        rec_input_name = args.rec_trt_input_name or args.trt_input_name

        if det_shape and det_input_name:
            det_engine_fp16 = os.path.join(trt_fp16_dir, "detection.engine")
            build_trt_fp16(trtexec, det_onnx, det_engine_fp16, det_input_name, det_shape)
        else:
            eprint("[TRT FP16] Skipped detection: need --det-shape and --trt-input-name (or --det-trt-input-name).")

        if rec_shape and rec_input_name:
            rec_engine_fp16 = os.path.join(trt_fp16_dir, "recognition.engine")
            build_trt_fp16(trtexec, rec_onnx, rec_engine_fp16, rec_input_name, rec_shape)
        else:
            eprint("[TRT FP16] Skipped recognition: need --rec-shape and --trt-input-name (or --rec-trt-input-name).")
    else:
        eprint("[TRT] 'trtexec' not found in PATH; skipping TensorRT engine builds.")

    # 4) Optional: upload to HF
    if args.hf_repo_id:
        hf_upload(
            repo_id=args.hf_repo_id,
            folder_path=out_dir,
            token=args.hf_token,
            private=args.private,
            branch=args.branch,
        )

    print("\n[Done] Artifacts in:", out_dir)
    print("Structure:")
    for root, dirs, files in os.walk(out_dir):
        rel = os.path.relpath(root, out_dir)
        indent = "  " * (0 if rel == "." else rel.count(os.sep) + 1)
        for d in sorted(dirs):
            print(f"{indent}{d}/")
        for f in sorted(files):
            print(f"{indent}{f}")


if __name__ == "__main__":
    main()

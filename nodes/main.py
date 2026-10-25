"""promboard.triposg — TripoSG(VAST-AI) 이미지 → 3D 형상(GLB) 노드.

모델 코드는 VAST-AI-Research/TripoSG 의 triposg 패키지를 nodes/triposg 로 옮긴 것이다(NOTICE — 트랜스포머 일부는
Tencent Hunyuan Community License). 공식 예제(scripts/inference_triposg.py)와 바꾼 점:
  - 배경 지우기: RMBG-1.4(비상업) 대신 BiRefNet 공식 ONNX(MIT)
  - 메시 뽑기: diso(소스 빌드만 있는 CUDA 확장)가 필요한 플래시 디코더 대신 skimage 마칭 큐브(use_flash_decoder=False)
  - 면 줄이기: pymeshlab(GPL) 대신 trimesh 의 quadric 줄이기(fast-simplification, MIT)
"""
from __future__ import annotations

import os
import sys
import time

REPO = "VAST-AI/TripoSG"
BIREFNET = "birefnet/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx"
OOM_HELP = "그래픽카드 메모리가 모자랍니다. 다른 프로그램이 그래픽카드를 쓰고 있으면 닫아 주세요."
NO_SURFACE = "형상을 만들지 못했습니다. 다른 시드로 다시 해 보거나 물체가 더 크게 나온 사진을 넣어 주세요."


def _import_triposg():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)  # 옮겨 온 코드가 `from triposg...` 로 자기 자신을 부른다
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline

    return TripoSGPipeline


def image_to_shape(ctx, inputs, settings):
    import numpy as np
    import torch
    import trimesh

    from .cutout import prepare

    if ctx.device != "cuda":
        # 옮겨 온 메시 뽑기 코드가 CUDA 를 못 박아 쓴다(expand_edge_region_fast).
        raise ValueError("TripoSG 는 NVIDIA 그래픽카드(CUDA)가 있어야 돕니다.")
    steps = int(settings["steps"])
    guidance = float(settings["guidance"])
    faces = int(settings["faces"])
    seed = int(settings["seed"])
    started = time.monotonic()

    ctx.progress(0.02, "cutout")
    bg = str(settings["background"])
    image, removed = prepare(ctx.in_path(inputs["image"]), bg, ctx.model_path(BIREFNET) if bg != "none" else None)
    image.save(ctx.out_path("input-prepared.png"))
    ctx.log(f"input prepared {image.size} (background removed: {removed}) in {time.monotonic() - started:.1f}s")
    ctx.check_cancel()

    TripoSGPipeline = _import_triposg()
    dtype = torch.float16
    pipe = None
    try:
        ctx.progress(0.08, "load-model")
        model_dir = os.path.dirname(ctx.model_path(f"{REPO}/model_index.json"))
        pipe = TripoSGPipeline.from_pretrained(model_dir).to(ctx.device, dtype)
        ctx.log(f"TripoSG loaded in {time.monotonic() - started:.1f}s")

        def on_step(_pipe, i, _t, kw):
            ctx.check_cancel()
            ctx.progress(0.12 + 0.7 * (i + 1) / steps, "shape")
            return kw

        gen_started = time.monotonic()
        out = pipe(
            image=image,
            generator=torch.Generator(device=ctx.device).manual_seed(seed),
            num_inference_steps=steps,
            guidance_scale=guidance,
            use_flash_decoder=False,
            callback_on_step_end=on_step,
        ).samples[0]
        ctx.log(f"sampling + surface in {time.monotonic() - gen_started:.1f}s")
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower():
            raise RuntimeError(OOM_HELP) from exc
        if isinstance(exc, AttributeError) and "NoneType" in str(exc):
            # 마칭 큐브가 면을 못 찾으면 파이프라인이 (None, None) 으로 메시를 만들다 멈춘다
            raise ValueError(NO_SURFACE) from exc
        raise
    finally:
        if pipe is not None:
            del pipe
        if ctx.device == "cuda":
            torch.cuda.empty_cache()
    ctx.check_cancel()
    if out[0] is None:
        raise ValueError(NO_SURFACE)
    ctx.progress(0.88, "mesh")
    mesh = trimesh.Trimesh(out[0].astype(np.float32), np.ascontiguousarray(out[1]))
    before = int(mesh.faces.shape[0])
    if faces > 0 and before > faces:
        mesh = mesh.simplify_quadric_decimation(face_count=faces)
    dst = ctx.out_path("shape.glb")
    mesh.export(dst)
    ctx.log(f"mesh {before} -> {mesh.faces.shape[0]} faces, done in {time.monotonic() - started:.1f}s")
    ctx.progress(1.0, "done")
    return {"model": dst}


NODES = {"image-to-shape": image_to_shape}

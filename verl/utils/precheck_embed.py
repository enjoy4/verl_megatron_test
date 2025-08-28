# verl/verl/utils/precheck_embed.py
import os
import torch
import torch.distributed as dist
from megatron.core import mpu
import numpy as np

@torch.no_grad()
def stash_expected_windows_to_meta(container, divisor: int = 4):
    """
    Minimal version:
    - assumes container.non_tensor_batch['multi_modal_inputs'] is a numpy.ndarray of dicts
    - each dict may have torch.Tensor under keys 'pixel_values' and/or 'pixel_values_videos'
    - writes scalar counts into container.meta_info:
        - expected_image_windows
        - expected_video_windows
    """
    if not hasattr(container, "meta_info") or not isinstance(container.meta_info, dict):
        raise TypeError("container.meta_info must be a dict")
    nb = getattr(container, "non_tensor_batch", {}) or {}

    exp_img = 0
    exp_vid = 0

    mmi = nb.get("multi_modal_inputs", None)
    if isinstance(mmi, np.ndarray):
        for item in mmi.tolist():  # ndarray of dicts
            if not isinstance(item, dict):
                continue
            t = item.get("pixel_values", None)
            if isinstance(t, torch.Tensor):
                exp_img += int(t.shape[0]) // divisor
            t = item.get("pixel_values_videos", None)
            if isinstance(t, torch.Tensor):
                exp_vid += int(t.shape[0]) // divisor

    container.meta_info["expected_image_windows"] = int(exp_img)
    container.meta_info["expected_video_windows"] = int(exp_vid)
    return int(exp_img), int(exp_vid)

@torch.no_grad()
def mm_precheck_and_sync(data, module, image_token_id=None, video_token_id=None):
    """
    Precheck based solely on meta_info['expected_image_windows'|'expected_video_windows'].
    Do NOT read multi_modal_inputs here (it may be absent at compute_log_prob time).

    Returns: (skip: bool, reason: str, detail: dict)
    """
    # verbosity
    verbose_all = os.getenv("MM_PRECHECK_VERBOSE_ALL_RANKS", "0") == "1"
    dist_ready = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if dist_ready else -1
    is_master = (not dist_ready) or (rank == 0)
    def log(msg: str):
        if is_master or verbose_all:
            print(f"[MM-PRECHECK][rank={rank}] {msg}")

    log("start precheck multimodal data shape inconsistency")

    batch = getattr(data, "batch", {})
    input_ids = batch.get("input_ids", None)
    if not isinstance(input_ids, torch.Tensor):
        log("bypass: input_ids tensor is missing")
        return False, "", {}
    log(f"input_ids: shape={tuple(input_ids.shape)}, dtype={input_ids.dtype}, device={input_ids.device}")

    # Read expected window counts from meta_info only
    meta = getattr(data, "meta_info", {}) if hasattr(data, "meta_info") else {}
    exp_img = int(meta.get("expected_image_windows", 0) or 0) * 4
    exp_vid = int(meta.get("expected_video_windows", 0) or 0) * 4
    log(f"expected mask tokens(from meta*4): img={exp_img}, vid={exp_vid}")

    # If no expectations are provided, bypass to avoid false positives
    if exp_img == 0 and exp_vid == 0:
        log("bypass: no expected_* provided in meta_info")
        return False, "", {}

    # Resolve token ids (use module attributes if available; fallback to defaults)
    image_token_id = 151655
    video_token_id = 151656

    # Observed counts from input_ids
    got_img = int(torch.count_nonzero(input_ids == image_token_id).item())
    got_vid = int(torch.count_nonzero(input_ids == video_token_id).item())
    #log(f"observed mask_true: got_img={got_img}, got_vid={got_vid}")

    has_img = exp_img > 0
    has_vid = exp_vid > 0
    bad_img = (has_img and (got_img != exp_img))
    bad_vid = (has_vid and (got_vid != exp_vid))
    bad_local = int(bad_img or bad_vid)
    #log(f"flags(local): has_img={has_img}, has_vid={has_vid}, bad_img={bad_img}, bad_vid={bad_vid}, bad_local={bad_local}")

    # Sync decision across MP (TP×PP) and DP (MAX == logical OR)
    if dist_ready:
        dev = input_ids.device
        t = torch.tensor([bad_local], device=dev, dtype=torch.int32)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=mpu.get_data_parallel_group())
        bad_any = bool(t.item())
        #log(f"after all_reduce: bad_any={int(t.item())}")
    else:
        bad_any = bool(bad_local)
        #log(f"dist not initialized -> bad_any={bad_any}")

    if bad_any:
        parts = []
        if bad_img: parts.append(f"image mismatch: mask_true={got_img}, expected={exp_img * 4}")
        if bad_vid: parts.append(f"video mismatch: mask_true={got_vid}, expected={exp_vid * 4}")
        reason = "; ".join(parts) or "mm precheck failed"
        detail = {
            "has_img": has_img, "has_vid": has_vid,
            "got_img": got_img, "exp_img": exp_img,
            "got_vid": got_vid, "exp_vid": exp_vid,
        }
        log(f"[skip] {reason} | detail={detail}")
        return True, reason, detail

    if has_img or has_vid:
        log(f"[pass] img({got_img}/{exp_img}) vid({got_vid}/{exp_vid})")
    return False, "", {}
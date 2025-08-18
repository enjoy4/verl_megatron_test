#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage examples:

# Use defaults
#   python mix_mm.py --out_dir ./mixed_mm
#
# Specify all parquet files explicitly
#   python mix_mm.py \
#     --out_dir ./mixed_mm \
#     --text_train  /path/to/text_train.parquet \
#     --text_test   /path/to/text_test.parquet \
#     --img_train   /path/to/img_train.parquet \
#     --img_test    /path/to/img_test.parquet \
#     --video_train /path/to/video_train.parquet \
#     --video_test  /path/to/video_test.parquet
"""

import argparse
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets

# Default paths (can be overridden by CLI args)
TEXT_TRAIN  = "/media/damoxing/yoyo-ni/baige_debug/gsm8k/train.parquet"
TEXT_TEST   = "/media/damoxing/yoyo-ni/baige_debug/gsm8k/test.parquet"

IMG_TRAIN   = "/media/datasets/vlm_data/r1_data_final/robopoint_train_9728.parquet"
IMG_TEST    = "/media/datasets/vlm_data/r1_data/robopoint-test-sample-500.parquet"

VIDEO_TRAIN = "/media/datasets/vlm_data/r1_data_final/egolife_train_7168_new.parquet"
VIDEO_TEST  = "/media/datasets/vlm_data/r1_data_final/egolife_test_512_new.parquet"

EXTRA_KEYS = ["index", "answer", "question", "split"]


def norm_extra(e, i):
    """Normalize extra_info and ensure 'index' is an int if possible."""
    r = {k: None for k in EXTRA_KEYS}
    if isinstance(e, dict):
        r.update({k: e.get(k) for k in EXTRA_KEYS})
    idx = r["index"]
    if idx is None:
        r["index"] = i
    else:
        f = float(idx)
        r["index"] = int(f) if f.is_integer() else f
    return r


def map_text(row, idx):
    return {
        "prompt": row.get("prompt"),
        "ability": row.get("ability"),
        "reward_model": row.get("reward_model"),
        "extra_info": norm_extra(row.get("extra_info"), idx),
        "data_source": row.get("data_source"),
    }


def map_img(row, idx):
    return {
        "prompt": row.get("prompt"),
        "images": row.get("images"),
        "reward_model": row.get("reward_model"),
        "extra_info": norm_extra(row.get("extra_info"), idx),
        "data_source": row.get("data_source"),
    }


def map_video(row, idx):
    return {
        "prompt": row.get("prompt"),
        "videos": row.get("videos"),
        "reward_model": row.get("reward_model"),
        "extra_info": norm_extra(row.get("extra_info"), idx),
        "ability": row.get("ability"),
        "data_source": row.get("data_source"),
    }


def map_with_index(ds, fn):
    """Apply mapping function with enumerated index to keep stable index field."""
    return Dataset.from_generator(lambda: (fn(r, i) for i, r in enumerate(ds)))


def mix_multimodal_datasets(
    out_dir,
    text_train,
    text_test,
    img_train,
    img_test,
    video_train,
    video_test,
    seed: int = 42,
):
    """
    Load text/image/video parquet datasets, map to a unified schema,
    concatenate and shuffle them, then write mixed train/val parquet files.
    Returns (train_path, val_path, train_len, val_len).
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load raw datasets
    text_tr = load_dataset("parquet", data_files=str(Path(text_train).expanduser()), split="train")
    text_va = load_dataset("parquet", data_files=str(Path(text_test).expanduser()),  split="train")

    img_tr  = load_dataset("parquet", data_files=str(Path(img_train).expanduser()),  split="train")
    img_va  = load_dataset("parquet", data_files=str(Path(img_test).expanduser()),   split="train")

    vid_tr  = load_dataset("parquet", data_files=str(Path(video_train).expanduser()), split="train")
    vid_va  = load_dataset("parquet", data_files=str(Path(video_test).expanduser()),  split="train")

    # Map to unified schema
    text_tr = map_with_index(text_tr, map_text)
    text_va = map_with_index(text_va, map_text)

    img_tr  = map_with_index(img_tr,  map_img)
    img_va  = map_with_index(img_va,  map_img)

    vid_tr  = map_with_index(vid_tr,  map_video)
    vid_va  = map_with_index(vid_va,  map_video)

    # Concatenate and shuffle
    train = concatenate_datasets([text_tr, img_tr, vid_tr]).shuffle(seed=seed)
    val   = concatenate_datasets([text_va, img_va, vid_va]).shuffle(seed=seed)

    # Save parquet
    train_path = out_dir / "mixed_train.parquet"
    val_path   = out_dir / "mixed_val.parquet"
    train.to_parquet(train_path)
    val.to_parquet(val_path)

    return train_path, val_path, len(train), len(val)


def main(args):
    train_path, val_path, n_tr, n_va = mix_multimodal_datasets(
        out_dir=args.out_dir,
        text_train=args.text_train,
        text_test=args.text_test,
        img_train=args.img_train,
        img_test=args.img_test,
        video_train=args.video_train,
        video_test=args.video_test,
        seed=42,
    )
    print("saved:", train_path)
    print("saved:", val_path)
    print("train rows:", n_tr, "val rows:", n_va)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./mixed_mm")
    ap.add_argument("--text_train",  default=TEXT_TRAIN)
    ap.add_argument("--text_test",   default=TEXT_TEST)
    ap.add_argument("--img_train",   default=IMG_TRAIN)
    ap.add_argument("--img_test",    default=IMG_TEST)
    ap.add_argument("--video_train", default=VIDEO_TRAIN)
    ap.add_argument("--video_test",  default=VIDEO_TEST)
    args = ap.parse_args()
    main(args)

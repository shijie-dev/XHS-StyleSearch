# """
# extract.py - XHS-StyleSearch 特征提取脚本
# 使用 CLIP ViT-B/32 对小红书图片提取 512 维语义特征向量
# 输出: features.npy (特征矩阵) + metadata.json (元数据列表)
# """

import os
import json
import glob
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch

# ========== 配置 ==========
IMAGE_DIR = "../MediaCrawler/data/xhs/images"          # MediaCrawler 下载的图片目录
JSONL_DIR = "../MediaCrawler/data/xhs/jsonl"           # MediaCrawler 输出的 jsonl 目录
OUTPUT_DIR = "xhs_stylesearch/data"    # 特征输出目录
CLIP_MODEL = "ViT-B/32"
BATCH_SIZE = 32                         # 每批处理图片数，显存不够就调小
# ==========================


def load_note_metadata(jsonl_dir: str) -> dict:
    """从 jsonl 文件读取帖子元数据，构建 note_id -> meta 的映射"""
    note_meta = {}
    jsonl_files = glob.glob(os.path.join(jsonl_dir, "search_contents_*.jsonl"))

    if not jsonl_files:
        print(f"[警告] 在 {jsonl_dir} 下未找到 search_contents_*.jsonl 文件")
        return note_meta

    for jsonl_path in jsonl_files:
        print(f"[加载] 读取元数据: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    note_id = item.get("note_id", "")
                    if note_id:
                        note_meta[note_id] = {
                            "note_id":       note_id,
                            "title":         item.get("title", ""),
                            "desc":          item.get("desc", ""),
                            "nickname":      item.get("nickname", ""),
                            "user_id":       item.get("user_id", ""),
                            "liked_count":   item.get("liked_count", "0"),
                            "collected_count": item.get("collected_count", "0"),
                            "comment_count": item.get("comment_count", "0"),
                            "note_url":      item.get("note_url", ""),
                            "tag_list":      item.get("tag_list", ""),
                            "source_keyword": item.get("source_keyword", ""),
                        }
                except json.JSONDecodeError:
                    continue

    print(f"[加载] 共读取 {len(note_meta)} 条帖子元数据")
    return note_meta


def collect_image_paths(image_dir: str) -> list:
    """遍历图片目录，收集所有图片路径"""
    supported_ext = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    image_paths = []

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    for note_folder in sorted(os.listdir(image_dir)):
        folder_path = os.path.join(image_dir, note_folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            ext = Path(fname).suffix.lower()
            if ext in supported_ext:
                image_paths.append(os.path.join(folder_path, fname))

    print(f"[收集] 共找到 {len(image_paths)} 张图片")
    return image_paths


def extract_features(image_paths: list, note_meta: dict, output_dir: str):
    """批量提取 CLIP 特征并保存"""
    try:
        import clip
    except ImportError:
        raise ImportError(
            "请先安装 CLIP:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )

    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[模型] 加载 CLIP {CLIP_MODEL}，使用设备: {device}")
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    model.eval()

    all_features = []
    all_metadata = []
    failed = []

    total = len(image_paths)
    for batch_start in range(0, total, BATCH_SIZE):
        batch_paths = image_paths[batch_start: batch_start + BATCH_SIZE]
        batch_tensors = []
        batch_valid_paths = []

        for img_path in batch_paths:
            try:
                img = preprocess(Image.open(img_path).convert("RGB"))
                batch_tensors.append(img)
                batch_valid_paths.append(img_path)
            except Exception as e:
                print(f"[跳过] 图片读取失败: {img_path} | 原因: {e}")
                failed.append(img_path)

        if not batch_tensors:
            continue

        # 批量推理
        batch_input = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            features = model.encode_image(batch_input)
            features = features / features.norm(dim=-1, keepdim=True)  # L2 归一化
            features = features.cpu().numpy().astype(np.float32)

        for i, img_path in enumerate(batch_valid_paths):
            # 从路径里提取 note_id（倒数第二级目录名）
            note_id = Path(img_path).parent.name
            meta = note_meta.get(note_id, {})

            all_features.append(features[i])
            all_metadata.append({
                **meta,
                "note_id":    note_id,
                "image_path": img_path,
                "image_name": Path(img_path).name,
            })

        done = min(batch_start + BATCH_SIZE, total)
        print(f"[进度] {done}/{total} 张已处理")

    if not all_features:
        print("[错误] 没有成功提取任何特征，请检查图片目录和格式")
        return

    # 保存特征矩阵
    feature_matrix = np.stack(all_features, axis=0)  # shape: (N, 512)
    features_path = os.path.join(output_dir, "features.npy")
    np.save(features_path, feature_matrix)
    print(f"\n[保存] 特征矩阵: {features_path}  shape={feature_matrix.shape}")

    # 保存元数据
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    print(f"[保存] 元数据列表: {metadata_path}  共 {len(all_metadata)} 条")

    if failed:
        print(f"\n[警告] 以下 {len(failed)} 张图片处理失败:")
        for p in failed:
            print(f"  {p}")

    print("\n✅ 特征提取完成！")


def main():
    parser = argparse.ArgumentParser(description="XHS-StyleSearch 特征提取")
    parser.add_argument("--image_dir", default=IMAGE_DIR, help="图片根目录")
    parser.add_argument("--jsonl_dir", default=JSONL_DIR, help="jsonl 元数据目录")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="特征输出目录")
    args = parser.parse_args()

    note_meta = load_note_metadata(args.jsonl_dir)
    image_paths = collect_image_paths(args.image_dir)
    extract_features(image_paths, note_meta, args.output_dir)


if __name__ == "__main__":
    main()
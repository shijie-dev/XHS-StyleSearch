# """
# search.py - XHS-StyleSearch 检索 + Gradio Web Demo
# 依赖: extract.py 和 index.py 已运行完毕
# 运行: python search.py
# """

import os
import json
import numpy as np
import torch
import faiss
import gradio as gr
from PIL import Image
from pathlib import Path

# ========== 配置 ==========
DATA_DIR     = "xhs_stylesearch/data"
FEATURES_FILE = os.path.join(DATA_DIR, "features.npy")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")
INDEX_FILE    = os.path.join(DATA_DIR, "xhs.index")
CLIP_MODEL   = "ViT-B/32"
TOP_K        = 9          # 默认返回 Top-9
# ==========================


# ---------- 全局加载（只加载一次）----------
print("[启动] 加载 CLIP 模型...")
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(CLIP_MODEL, device=device)
model.eval()
print(f"[启动] CLIP 加载完成，设备: {device}")

print("[启动] 加载 FAISS 索引...")
index = faiss.read_index(INDEX_FILE)
print(f"[启动] 索引加载完成，共 {index.ntotal} 条向量")

print("[启动] 加载元数据...")
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)
print(f"[启动] 元数据加载完成，共 {len(metadata)} 条")


# ---------- 核心检索函数 ----------
def extract_query_feature(pil_image: Image.Image) -> np.ndarray:
    """对查询图片提取 CLIP 特征向量"""
    tensor = preprocess(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)


def search(query_image: Image.Image, top_k: int = TOP_K):
    """
    输入一张 PIL 图片，返回 Top-K 最相似结果
    Returns: list of (pil_image, caption)
    """
    if query_image is None:
        return []

    query_feat = extract_query_feature(query_image)
    distances, indices = index.search(query_feat, k=top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
        if idx < 0 or idx >= len(metadata):
            continue

        meta = metadata[idx]
        img_path = meta.get("image_path", "")

        # 读取本地图片
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception:
            pil_img = Image.new("RGB", (200, 200), color=(220, 220, 220))

        # 构建图片说明
        title    = meta.get("title", "无标题")[:30]
        nickname = meta.get("nickname", "")
        liked    = meta.get("liked_count", "0")
        keyword  = meta.get("source_keyword", "").strip()
        caption  = (
            f"#{rank+1}  相似度: {score:.3f}\n"
            f"📝 {title}\n"
            f"👤 {nickname}  ❤️ {liked}"
            + (f"  🔖 {keyword}" if keyword else "")
        )

        results.append((pil_img, caption))

    return results


# ---------- Gradio UI ----------
def gradio_search(query_image, top_k):
    if query_image is None:
        return []
    top_k = int(top_k)
    results = search(query_image, top_k=top_k)
    # Gradio Gallery 接受 (image, caption) 列表
    return results


with gr.Blocks(title="XHS StyleSearch", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 👗 XHS-StyleSearch · 小红书穿搭风格以图搜图
        上传一张穿搭图片，自动从图片库中找出风格最相近的帖子。
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Image(
                type="pil",
                label="上传查询图片",
                height=380,
            )
            top_k_slider = gr.Slider(
                minimum=1, maximum=20, value=9, step=1,
                label="返回数量 Top-K"
            )
            search_btn = gr.Button("🔍 开始搜索", variant="primary")

        with gr.Column(scale=2):
            gallery_output = gr.Gallery(
                label="最相似穿搭",
                columns=3,
                height=580,
                object_fit="cover",
                show_label=True,
            )

    search_btn.click(
        fn=gradio_search,
        inputs=[query_input, top_k_slider],
        outputs=gallery_output,
    )

    # 支持上传后自动触发
    query_input.change(
        fn=gradio_search,
        inputs=[query_input, top_k_slider],
        outputs=gallery_output,
    )

    gr.Markdown(
        """
        ---
        **数据来源**: 小红书图文帖子（MediaCrawler 采集）  
        **检索模型**: OpenAI CLIP ViT-B/32 + FAISS IndexFlatIP  
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,       # 自动打开浏览器
    )
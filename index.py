# """
# index.py - XHS-StyleSearch FAISS 索引构建脚本
# 读取 extract.py 输出的 features.npy，构建 FAISS IndexFlatIP 索引
# 输出: xhs.index (FAISS 索引文件)
# """

import os
import argparse
import numpy as np
import faiss

# ========== 配置 ==========
DATA_DIR = "xhs_stylesearch/data"       # extract.py 的输出目录
FEATURES_FILE = "features.npy"          # 特征矩阵文件名
INDEX_FILE = "xhs.index"               # 输出的 FAISS 索引文件名
FEATURE_DIM = 512                       # CLIP ViT-B/32 输出维度
# ==========================


def build_index(data_dir: str):
    features_path = os.path.join(data_dir, FEATURES_FILE)
    index_path = os.path.join(data_dir, INDEX_FILE)

    # 1. 读取特征矩阵
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"找不到特征文件: {features_path}\n"
            f"请先运行 extract.py 生成特征矩阵"
        )

    print(f"[加载] 读取特征矩阵: {features_path}")
    features = np.load(features_path).astype(np.float32)
    n, dim = features.shape
    print(f"[加载] 特征矩阵 shape: ({n}, {dim})")

    if dim != FEATURE_DIM:
        raise ValueError(
            f"特征维度不匹配，期望 {FEATURE_DIM}，实际 {dim}\n"
            f"请确认使用的是 CLIP ViT-B/32 模型"
        )

    # 2. 验证向量已归一化（extract.py 做了 L2 归一化，内积 == 余弦相似度）
    norms = np.linalg.norm(features, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        print("[归一化] 检测到向量未归一化，正在执行 L2 归一化...")
        faiss.normalize_L2(features)
    else:
        print("[归一化] 向量已归一化，跳过")

    # 3. 构建 IndexFlatIP（内积检索，归一化后等价于余弦相似度）
    print(f"[构建] 创建 IndexFlatIP，维度={dim}")
    index = faiss.IndexFlatIP(dim)

    # 4. 添加向量
    index.add(features)
    print(f"[构建] 已添加 {index.ntotal} 条向量到索引")

    # 5. 保存索引
    faiss.write_index(index, index_path)
    index_size_mb = os.path.getsize(index_path) / 1024 / 1024
    print(f"\n[保存] FAISS 索引: {index_path}  ({index_size_mb:.1f} MB)")

    # 6. 验证：做一次自检索，确认索引正常
    print("\n[验证] 执行自检索（用第0张图检索 Top-5）...")
    query = features[0:1]
    distances, indices = index.search(query, k=5)
    print(f"  Top-5 索引编号: {indices[0].tolist()}")
    print(f"  Top-5 相似度分数: {[round(float(d), 4) for d in distances[0]]}")
    print("  ✅ 第一个结果应为自身（编号0，分数≈1.0）")

    print("\n✅ 索引构建完成！")
    print(f"   向量总数: {index.ntotal}")
    print(f"   索引文件: {index_path}")


def main():
    parser = argparse.ArgumentParser(description="XHS-StyleSearch FAISS 索引构建")
    parser.add_argument(
        "--data_dir",
        default=DATA_DIR,
        help=f"特征数据目录（默认: {DATA_DIR}）"
    )
    args = parser.parse_args()

    build_index(args.data_dir)


if __name__ == "__main__":
    main()
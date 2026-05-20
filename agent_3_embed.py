"""
Agent 3 — 向量 Embedding & 入库
================================
用途：读取 processed/chunks.json，生成向量后批量写入 ChromaDB。

Embedding 模型选择（在下方 EMBED_MODE 配置）：
  "local"  → BAAI/bge-m3（免费，中文最优，GTX1050 约 30min/千条）
  "openai" → text-embedding-3-small（$0.02/百万 token，最快）
  "ollama" → nomic-embed-text via Ollama（本地，适中）

运行：
  python agent_3_embed.py                    # 默认 local 模式
  python agent_3_embed.py --mode openai      # OpenAI 模式（需 OPENAI_API_KEY）
  python agent_3_embed.py --mode ollama      # Ollama 模式
  python agent_3_embed.py --reset            # 清空已有向量库重新入库
  python agent_3_embed.py --batch_size 32   # 调整批次大小（内存不足时减小）
"""

import json
import argparse
import os
import sys
import time
from pathlib import Path

CHUNKS_FILE = Path(__file__).parent / "processed" / "chunks.json"
DB_DIR = Path(__file__).parent / "db" / "chroma"

# ─── 配置区 ───────────────────────────────────────────────────────────────────
EMBED_MODE = "local"       # "local" | "openai" | "ollama"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL = "http://localhost:11434"
COLLECTION_NAME = "weibo_posts"
DEFAULT_BATCH_SIZE = 32    # 本地模式建议 16-32；OpenAI 可用 100+
# ──────────────────────────────────────────────────────────────────────────────


def get_embedder(mode: str):
    """返回 embedding 函数 fn(texts: list[str]) -> list[list[float]]"""

    if mode == "local":
        print("[agent_3] 加载本地模型 BAAI/bge-m3（首次下载约 2.5GB）...")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("  pip install sentence-transformers")
            sys.exit(1)
        model = SentenceTransformer("BAAI/bge-m3")
        print(f"  模型加载完毕")

        def embed_fn(texts):
            vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return vecs.tolist()

        return embed_fn, 1024   # bge-m3 维度

    elif mode == "openai":
        if not OPENAI_API_KEY:
            print("  请设置环境变量 OPENAI_API_KEY 或在脚本中填入")
            sys.exit(1)
        try:
            from openai import OpenAI
        except ImportError:
            print("  pip install openai")
            sys.exit(1)
        client = OpenAI(api_key=OPENAI_API_KEY)

        def embed_fn(texts):
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [item.embedding for item in resp.data]

        return embed_fn, 1536

    elif mode == "ollama":
        try:
            import requests as req
        except ImportError:
            print("  pip install requests")
            sys.exit(1)

        def embed_fn(texts):
            vecs = []
            for text in texts:
                r = req.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": text},
                    timeout=30,
                )
                vecs.append(r.json()["embedding"])
            return vecs

        return embed_fn, 768

    else:
        raise ValueError(f"未知 mode: {mode}")


def load_chroma(reset: bool = False):
    """初始化 ChromaDB 客户端和 collection"""
    try:
        import chromadb
    except ImportError:
        print("  pip install chromadb")
        sys.exit(1)

    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"  已清空旧向量库 {COLLECTION_NAME}")
        except Exception:
            pass

    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"  已有向量库，当前 {collection.count()} 条")
    except Exception:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"  新建向量库 {COLLECTION_NAME}")

    return collection


def get_existing_ids(collection) -> set:
    """获取已入库的 ID 集合，支持增量入库"""
    try:
        results = collection.get(include=[])
        return set(results["ids"])
    except Exception:
        return set()


def batch_upsert(collection, chunks: list[dict], embed_fn, batch_size: int):
    """分批生成向量并写入 ChromaDB"""
    existing_ids = get_existing_ids(collection)
    new_chunks = [c for c in chunks if str(c["id"]) not in existing_ids]

    if not new_chunks:
        print("  所有 chunk 已入库，无需更新")
        return

    print(f"  需入库：{len(new_chunks)} 条（跳过已有 {len(existing_ids)} 条）")

    total = len(new_chunks)
    done = 0
    errors = 0

    for i in range(0, total, batch_size):
        batch = new_chunks[i: i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [str(c["id"]) for c in batch]
        metas = []
        for c in batch:
            # ChromaDB metadata 只支持 str/int/float/bool
            m = {k: (str(v) if not isinstance(v, (int, float, bool)) else v)
                 for k, v in c["metadata"].items()}
            metas.append(m)

        try:
            vecs = embed_fn(texts)
            collection.upsert(
                ids=ids,
                embeddings=vecs,
                documents=texts,
                metadatas=metas,
            )
            done += len(batch)
            pct = done / total * 100
            print(f"  进度：{done}/{total} ({pct:.1f}%)", end="\r", flush=True)
        except Exception as e:
            errors += 1
            print(f"\n  [warn] 批次 {i//batch_size + 1} 失败：{e}")
            time.sleep(2)

    print(f"\n  入库完成：{done} 条，失败 {errors} 批")


def main():
    parser = argparse.ArgumentParser(description="微博向量入库 Agent")
    parser.add_argument("--mode", default=EMBED_MODE, choices=["local", "openai", "ollama"])
    parser.add_argument("--reset", action="store_true", help="清空向量库重新入库")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    if not CHUNKS_FILE.exists():
        print(f"[agent_3] 找不到 chunks 文件：{CHUNKS_FILE}")
        print("  请先运行 agent_2_clean.py")
        return

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[agent_3] 读取 {len(chunks)} 个 chunk")

    embed_fn, dim = get_embedder(args.mode)
    print(f"  Embedding 模式：{args.mode}，向量维度：{dim}")

    collection = load_chroma(reset=args.reset)
    batch_upsert(collection, chunks, embed_fn, args.batch_size)

    final_count = collection.count()
    print(f"\n[agent_3] 向量库最终记录数：{final_count}")
    print(f"  数据库路径：{DB_DIR}")


if __name__ == "__main__":
    main()

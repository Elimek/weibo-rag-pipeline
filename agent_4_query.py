"""
Agent 4 — 微博知识库查询
========================
用途：从 ChromaDB 检索微博，支持：
  1. 语义搜索：自然语言提问
  2. 精确时间过滤：某年 / 某月 / 某日 / 某小时
  3. 话题过滤：#标签#
  4. 交互式问答模式（接入 LLM 总结）

查询示例：
  # 交互模式（推荐）
  python agent_4_query.py

  # 命令行直接查询
  python agent_4_query.py --query "找工作时的焦虑"
  python agent_4_query.py --query "旅行计划" --year 2024
  python agent_4_query.py --date 2024-06-15           # 查某天所有微博
  python agent_4_query.py --date 2024-06-15 --hour 22  # 查某天晚上10点
  python agent_4_query.py --topic "生活"               # 查某话题所有微博
  python agent_4_query.py --query "压力" --top_k 10   # 返回更多结果
"""

import json
import argparse
import os
import sys
import re
from pathlib import Path
from datetime import datetime

DB_DIR = Path(__file__).parent / "db" / "chroma"
COLLECTION_NAME = "weibo_posts"

# ─── LLM 配置（可选，用于问答摘要） ──────────────────────────────────────────
LLM_MODE = "none"         # "none" | "openai" | "ollama"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"
OLLAMA_MODEL = "qwen2.5:7b"    # 本地中文首选
OLLAMA_BASE_URL = "http://localhost:11434"
# ──────────────────────────────────────────────────────────────────────────────


def load_collection():
    try:
        import chromadb
    except ImportError:
        print("  pip install chromadb")
        sys.exit(1)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        col = client.get_collection(COLLECTION_NAME)
        print(f"[agent_4] 向量库：{col.count()} 条记录")
        return col
    except Exception:
        print(f"[agent_4] 向量库不存在：{DB_DIR}")
        print("  请先运行 agent_3_embed.py")
        sys.exit(1)


def get_embedder():
    """与 agent_3 保持一致的本地 embedder"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-m3")
        def fn(text):
            return model.encode([text], normalize_embeddings=True).tolist()[0]
        return fn
    except ImportError:
        # 降级：如果没有 sentence_transformers，尝试 OpenAI
        if OPENAI_API_KEY:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            def fn(text):
                resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
                return resp.data[0].embedding
            return fn
        print("  未找到 sentence-transformers，请安装：pip install sentence-transformers")
        sys.exit(1)


def build_where_filter(year=None, month=None, day=None, hour=None,
                        date_str=None, topic=None) -> dict | None:
    """构建 ChromaDB metadata 过滤条件"""
    conditions = []

    if date_str:
        # 解析 YYYY-MM-DD
        parts = date_str.split("-")
        if len(parts) >= 1 and parts[0]:
            conditions.append({"year": {"$eq": int(parts[0])}})
        if len(parts) >= 2 and parts[1]:
            conditions.append({"month": {"$eq": int(parts[1])}})
        if len(parts) >= 3 and parts[2]:
            conditions.append({"day": {"$eq": int(parts[2])}})
    else:
        if year:
            conditions.append({"year": {"$eq": int(year)}})
        if month:
            conditions.append({"month": {"$eq": int(month)}})
        if day:
            conditions.append({"day": {"$eq": int(day)}})

    if hour is not None:
        conditions.append({"hour": {"$eq": int(hour)}})

    if topic:
        # topics 字段是 "|tag1|tag2|" 格式，用 $contains 模糊匹配
        conditions.append({"topics": {"$contains": topic}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def semantic_search(collection, embed_fn, query: str, where: dict | None,
                    top_k: int = 5) -> list[dict]:
    """语义向量搜索"""
    query_vec = embed_fn(query)
    kwargs = {
        "query_embeddings": [query_vec],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)
    hits = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        hits.append({
            "text": doc,
            "metadata": meta,
            "score": round(1 - dist, 4),   # cosine similarity
        })
    return hits


def exact_search(collection, where: dict, limit: int = 50) -> list[dict]:
    """精确 metadata 过滤（无语义，按时间顺序）"""
    results = collection.get(
        where=where,
        limit=limit,
        include=["documents", "metadatas"],
    )
    hits = []
    for i, doc in enumerate(results["documents"]):
        meta = results["metadatas"][i]
        hits.append({"text": doc, "metadata": meta, "score": None})
    # 按时间排序
    hits.sort(key=lambda h: h["metadata"].get("created_at", ""))
    return hits


def format_hit(hit: dict, idx: int) -> str:
    """格式化单条结果用于显示"""
    meta = hit["metadata"]
    dt = meta.get("datetime_str") or meta.get("date_str") or meta.get("created_at", "")
    score_str = f"  相似度 {hit['score']:.3f}" if hit["score"] else ""
    topics = meta.get("topics", "")
    topic_str = f"  话题：{topics}" if topics else ""
    region = meta.get("region", "")
    region_str = f"  发布地：{region}" if region else ""
    likes = meta.get("likes", 0)

    lines = [
        f"── #{idx+1} ──────────────────────────",
        f"时间：{dt}{score_str}",
    ]
    if topic_str:
        lines.append(topic_str)
    if region_str:
        lines.append(region_str)
    lines.append(f"👍 {likes}")
    lines.append("")
    lines.append(hit["text"])
    return "\n".join(lines)


def llm_answer(question: str, hits: list[dict]) -> str:
    """用 LLM 基于检索结果回答问题"""
    context = "\n\n".join(
        f"[{h['metadata'].get('datetime_str', '')}] {h['text']}"
        for h in hits
    )
    system_prompt = (
        "你是用户本人的微博知识库助手。以下是从用户历史微博中检索到的相关内容，"
        "请基于这些内容回答问题。如果内容不足以回答，请说明。"
        "回答时引用具体时间和微博原文。"
    )
    user_msg = f"问题：{question}\n\n检索到的微博：\n{context}"

    if LLM_MODE == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content

    elif LLM_MODE == "ollama":
        import requests
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
            },
            timeout=60,
        )
        return resp.json()["message"]["content"]

    return ""


def run_query(collection, embed_fn, args) -> list[dict]:
    """执行一次查询"""
    where = build_where_filter(
        year=args.year, month=args.month, day=args.day, hour=args.hour,
        date_str=args.date, topic=args.topic,
    )

    if args.query:
        hits = semantic_search(collection, embed_fn, args.query, where, args.top_k)
    elif where:
        # 纯时间/话题查询，不需要语义搜索
        hits = exact_search(collection, where, limit=args.top_k)
    else:
        print("  请提供 --query 或时间/话题过滤条件")
        return []

    return hits


def interactive_mode(collection, embed_fn):
    """交互式查询循环"""
    print("\n" + "=" * 50)
    print("微博知识库交互查询")
    print("输入自然语言问题，或特殊命令：")
    print("  :year 2024          → 过滤年份")
    print("  :date 2024-06-15    → 查某天")
    print("  :topic 香港         → 查话题")
    print("  :clear              → 清除过滤条件")
    print("  :quit / q           → 退出")
    print("=" * 50 + "\n")

    active_filters = {}

    class FakeArgs:
        query = None
        year = None
        month = None
        day = None
        hour = None
        date = None
        topic = None
        top_k = 5

    while True:
        try:
            user_input = input("🔍 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() in (":quit", "q", "exit", "quit"):
            print("再见！")
            break

        # 解析特殊命令
        if user_input.startswith(":"):
            parts = user_input[1:].split()
            cmd = parts[0].lower() if parts else ""
            val = parts[1] if len(parts) > 1 else ""

            if cmd == "clear":
                active_filters.clear()
                print("  已清除所有过滤条件")
            elif cmd == "year" and val:
                active_filters["year"] = int(val)
                print(f"  设置年份过滤：{val}")
            elif cmd == "month" and val:
                active_filters["month"] = int(val)
                print(f"  设置月份过滤：{val}")
            elif cmd == "date" and val:
                active_filters["date"] = val
                print(f"  设置日期过滤：{val}")
            elif cmd == "hour" and val:
                active_filters["hour"] = int(val)
                print(f"  设置小时过滤：{val}:00")
            elif cmd == "topic" and val:
                active_filters["topic"] = val
                print(f"  设置话题过滤：{val}")
            elif cmd == "filters":
                print(f"  当前过滤条件：{active_filters}")
            else:
                print(f"  未知命令：{cmd}")
            continue

        # 构建查询参数
        fa = FakeArgs()
        fa.query = user_input
        fa.year = active_filters.get("year")
        fa.month = active_filters.get("month")
        fa.day = active_filters.get("day")
        fa.hour = active_filters.get("hour")
        fa.date = active_filters.get("date")
        fa.topic = active_filters.get("topic")
        fa.top_k = 5

        # 显示当前过滤状态
        if active_filters:
            filter_str = "  | ".join(f"{k}={v}" for k, v in active_filters.items())
            print(f"  [过滤：{filter_str}]")

        hits = run_query(collection, embed_fn, fa)

        if not hits:
            print("  未找到相关微博\n")
            continue

        print(f"\n  找到 {len(hits)} 条结果：\n")
        for i, h in enumerate(hits):
            print(format_hit(h, i))
            print()

        # LLM 摘要
        if LLM_MODE != "none":
            print("── AI 总结 ──────────────────────────")
            answer = llm_answer(user_input, hits)
            print(answer)
            print()


def main():
    parser = argparse.ArgumentParser(description="微博知识库查询 Agent")
    parser.add_argument("--query", "-q", default="", help="语义查询内容")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--month", type=int, default=None)
    parser.add_argument("--day", type=int, default=None)
    parser.add_argument("--hour", type=int, default=None)
    parser.add_argument("--date", default=None, help="日期 YYYY-MM-DD")
    parser.add_argument("--topic", default=None, help="话题关键词")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    args = parser.parse_args()

    collection = load_collection()
    embed_fn = get_embedder()

    # 无参数 or --interactive → 进入交互模式
    if args.interactive or (not args.query and not args.date and not args.topic
                             and args.year is None):
        interactive_mode(collection, embed_fn)
        return

    # 单次查询
    hits = run_query(collection, embed_fn, args)
    if not hits:
        print("未找到相关微博")
        return

    print(f"\n找到 {len(hits)} 条结果：\n")
    for i, h in enumerate(hits):
        print(format_hit(h, i))
        print()

    if LLM_MODE != "none" and args.query:
        print("── AI 总结 ──────────────────────────")
        print(llm_answer(args.query, hits))


if __name__ == "__main__":
    main()

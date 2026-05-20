"""
Agent 2 — 清洗 & 分块
======================
用途：读取 raw/weibo_raw.json，进行：
  1. 文本清洗（去乱码表情、去多余空白）
  2. 连续发帖合并（5分钟内的相邻原创微博合并为一块）
  3. 元数据注入（year/month/day/hour/weekday/topics）
  4. 输出 processed/chunks.json

运行：
  python agent_2_clean.py
  python agent_2_clean.py --merge_window 10   # 10分钟合并窗口
  python agent_2_clean.py --no_merge          # 不合并，每条独立成块
"""

import json
import re
import argparse
from pathlib import Path
from datetime import datetime, timezone

RAW_FILE = Path(__file__).parent / "raw" / "weibo_raw.json"
OUT_FILE = Path(__file__).parent / "processed" / "chunks.json"
STATS_FILE = Path(__file__).parent / "processed" / "clean_stats.json"

# 需要去除的无意义模式
JUNK_PATTERNS = [
    r"\[[\u4e00-\u9fa5a-zA-Z]+\]",   # [偷笑] [doge] 等表情代码
    r"http[s]?://t\.cn/\S+",           # 短链接
    r"http[s]?://\S+",                 # 其他链接
    r"[\u200b-\u200f\ufeff]",           # 零宽字符
    r"\s{3,}",                          # 三个以上连续空白
]


def clean_text(text: str) -> str:
    """清洗微博正文"""
    for pattern in JUNK_PATTERNS:
        text = re.sub(pattern, " ", text)
    # 保留 #话题# 格式，压缩空白
    text = re.sub(r" +", " ", text).strip()
    return text


def parse_dt(iso_str: str) -> datetime | None:
    """解析 ISO 时间字符串"""
    try:
        return datetime.fromisoformat(iso_str)
    except Exception:
        return None


def build_metadata(post: dict, dt: datetime | None) -> dict:
    """构建供 ChromaDB 过滤的元数据"""
    meta = {
        "id": str(post.get("id", "")),
        "source": post.get("source", ""),
        "topics": "|".join(post.get("topics", [])),    # ChromaDB 不支持 list，用 | 分隔
        "is_repost": 1 if post.get("is_repost") else 0,
        "likes": int(post.get("likes", 0)),
        "reposts": int(post.get("reposts", 0)),
        "comments": int(post.get("comments", 0)),
        "region": post.get("region_name", ""),
        "created_at": post.get("created_at", ""),
    }
    if dt:
        meta["year"] = dt.year
        meta["month"] = dt.month
        meta["day"] = dt.day
        meta["hour"] = dt.hour
        meta["minute"] = dt.minute
        meta["weekday"] = dt.weekday()   # 0=周一
        meta["yearmonth"] = int(f"{dt.year}{dt.month:02d}")    # 202501
        meta["date_str"] = dt.strftime("%Y-%m-%d")
        meta["datetime_str"] = dt.strftime("%Y-%m-%d %H:%M")
    return meta


def merge_posts(posts: list[dict], window_minutes: int) -> list[dict]:
    """
    将同一时间窗口内的连续原创微博合并成一个 chunk。
    转发微博不参与合并，独立成块。
    """
    if window_minutes <= 0:
        return posts

    merged = []
    buffer = []
    buffer_dt = None

    def flush_buffer():
        if not buffer:
            return
        if len(buffer) == 1:
            merged.append(buffer[0])
        else:
            # 合并文本，用 \n\n 分隔
            combined_text = "\n\n".join(p["_clean_text"] for p in buffer)
            combined_topics = list({t for p in buffer for t in p.get("topics", [])})
            base = buffer[0].copy()
            base["_clean_text"] = combined_text
            base["topics"] = combined_topics
            base["_merged_count"] = len(buffer)
            base["_merged_ids"] = [p.get("id", "") for p in buffer]
            merged.append(base)
        buffer.clear()

    for post in posts:
        if post.get("is_repost"):
            flush_buffer()
            merged.append(post)
            buffer_dt = None
            continue

        dt = parse_dt(post.get("created_at", ""))
        if dt is None:
            flush_buffer()
            merged.append(post)
            continue

        if buffer_dt is None:
            buffer.append(post)
            buffer_dt = dt
        else:
            diff = abs((dt - buffer_dt).total_seconds()) / 60
            if diff <= window_minutes:
                buffer.append(post)
                # 更新窗口锚点为最新时间
                buffer_dt = dt
            else:
                flush_buffer()
                buffer.append(post)
                buffer_dt = dt

    flush_buffer()
    return merged


def build_chunk_text(post: dict) -> str:
    """构建最终写入向量库的文本，包含时间前缀，方便召回时展示"""
    dt = parse_dt(post.get("created_at", ""))
    prefix = ""
    if dt:
        prefix = f"[{dt.strftime('%Y年%m月%d日 %H:%M')}]"

    text = post.get("_clean_text", "")

    # 如果是转发，附上原文
    if post.get("is_repost") and post.get("retweeted"):
        rt = post["retweeted"]
        rt_clean = clean_text(rt.get("text", ""))
        text = f"{text}\n//转发自 @{rt['user']}: {rt_clean}"

    return f"{prefix} {text}".strip()


def main():
    parser = argparse.ArgumentParser(description="微博清洗分块 Agent")
    parser.add_argument("--merge_window", type=int, default=5, help="合并时间窗口（分钟），0=不合并")
    parser.add_argument("--no_merge", action="store_true", help="不合并，等同于 --merge_window 0")
    parser.add_argument("--min_len", type=int, default=5, help="最短有效正文字数（过短的过滤掉）")
    args = parser.parse_args()

    merge_window = 0 if args.no_merge else args.merge_window

    if not RAW_FILE.exists():
        print(f"[agent_2] 找不到原始数据：{RAW_FILE}")
        print("  请先运行 agent_1_scrape.py")
        return

    print(f"[agent_2] 读取原始数据：{RAW_FILE}")
    with open(RAW_FILE, encoding="utf-8") as f:
        raw_posts = json.load(f)

    print(f"  原始微博数：{len(raw_posts)}")

    # Step 1: 清洗文本
    cleaned = []
    skipped = 0
    for post in raw_posts:
        text = clean_text(post.get("text", ""))
        if len(text.replace(" ", "")) < args.min_len:
            skipped += 1
            continue
        post = post.copy()
        post["_clean_text"] = text
        cleaned.append(post)

    print(f"  清洗后：{len(cleaned)} 条（过滤 {skipped} 条过短）")

    # Step 2: 按时间排序（升序）
    def sort_key(p):
        dt = parse_dt(p.get("created_at", ""))
        return dt if dt else datetime.min.replace(tzinfo=timezone.utc)

    cleaned.sort(key=sort_key)

    # Step 3: 合并连续微博
    if merge_window > 0:
        merged = merge_posts(cleaned, merge_window)
        merged_count = sum(1 for p in merged if p.get("_merged_count", 1) > 1)
        print(f"  合并后：{len(merged)} 块（其中 {merged_count} 块是合并块，窗口={merge_window}分钟）")
    else:
        merged = cleaned
        print(f"  不合并，保持 {len(merged)} 块")

    # Step 4: 构建最终 chunks
    chunks = []
    for post in merged:
        dt = parse_dt(post.get("created_at", ""))
        chunk_text = build_chunk_text(post)
        meta = build_metadata(post, dt)
        chunks.append({
            "id": meta["id"] or f"chunk_{len(chunks)}",
            "text": chunk_text,
            "metadata": meta,
            "merged_count": post.get("_merged_count", 1),
            "merged_ids": post.get("_merged_ids", []),
        })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # 统计
    years = {}
    for c in chunks:
        y = c["metadata"].get("year", "unknown")
        years[y] = years.get(y, 0) + 1

    stats = {
        "total_raw": len(raw_posts),
        "total_chunks": len(chunks),
        "skipped": skipped,
        "merge_window_minutes": merge_window,
        "by_year": dict(sorted(years.items())),
        "processed_at": datetime.now().isoformat(),
    }
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[agent_2] 完成！共 {len(chunks)} 个 chunk")
    print(f"  已保存：{OUT_FILE}")
    print(f"  按年分布：{stats['by_year']}")


if __name__ == "__main__":
    main()

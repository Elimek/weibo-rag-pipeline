"""
Pipeline 编排脚本 — 一键运行全流程
=====================================
用途：按顺序运行 4 个 Agent，支持跳过已完成的步骤。

使用：
  python run_pipeline.py                        # 全流程运行
  python run_pipeline.py --skip-scrape          # 跳过抓取（已有 raw/weibo_raw.json）
  python run_pipeline.py --only-query           # 直接进查询界面
  python run_pipeline.py --embed-mode openai    # 使用 OpenAI Embedding
  python run_pipeline.py --uid 1234567890       # 指定微博 UID
  python run_pipeline.py --reset-db             # 重建向量库
"""

import subprocess
import sys
import argparse
import json
import time
from pathlib import Path

BASE = Path(__file__).parent
RAW_FILE = BASE / "raw" / "weibo_raw.json"
CHUNKS_FILE = BASE / "processed" / "chunks.json"
DB_DIR = BASE / "db" / "chroma"


def run_step(name: str, cmd: list[str]) -> bool:
    """运行一个步骤，返回是否成功"""
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n  [失败] {name} 退出码：{result.returncode}")
        return False
    print(f"\n  [完成] {name}")
    return True


def check_deps():
    """检查核心依赖"""
    missing = []
    try:
        import chromadb
    except ImportError:
        missing.append("chromadb")
    try:
        import requests
    except ImportError:
        missing.append("requests")

    if missing:
        print(f"缺少依赖，请运行：pip install {' '.join(missing)}")
        print("完整依赖：pip install -r requirements.txt")
        sys.exit(1)


def print_banner():
    print("""
╔══════════════════════════════════════════════════╗
║         微博知识库 RAG Pipeline                   ║
║  Agent1:抓取 → Agent2:清洗 → Agent3:入库 → Agent4:查询  ║
╚══════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description="微博 RAG 全流程编排")
    parser.add_argument("--skip-scrape", action="store_true", help="跳过抓取（raw 文件已存在）")
    parser.add_argument("--skip-clean", action="store_true", help="跳过清洗（chunks 文件已存在）")
    parser.add_argument("--skip-embed", action="store_true", help="跳过入库（向量库已存在）")
    parser.add_argument("--only-query", action="store_true", help="直接进入查询界面")
    parser.add_argument("--uid", default="", help="微博 UID")
    parser.add_argument("--max-pages", type=int, default=0, help="抓取最大页数（0=全量）")
    parser.add_argument("--embed-mode", default="local", choices=["local", "openai", "ollama"])
    parser.add_argument("--reset-db", action="store_true", help="重建向量库")
    parser.add_argument("--merge-window", type=int, default=5, help="合并窗口分钟数")
    parser.add_argument("--query", "-q", default="", help="直接执行一次查询")
    args = parser.parse_args()

    print_banner()
    check_deps()

    py = sys.executable

    if args.only_query:
        run_step("Agent 4 — 查询界面", [py, str(BASE / "agent_4_query.py"), "-i"])
        return

    # ── Step 1: 抓取 ──────────────────────────────────────────────────────────
    if args.skip_scrape or RAW_FILE.exists():
        if RAW_FILE.exists():
            size = RAW_FILE.stat().st_size // 1024
            try:
                with open(RAW_FILE, encoding="utf-8") as f:
                    cnt = len(json.load(f))
                print(f"[跳过] 抓取（raw/weibo_raw.json 已存在：{cnt} 条，{size}KB）")
            except Exception:
                print(f"[跳过] 抓取（raw/weibo_raw.json 已存在，{size}KB）")
        else:
            print("[跳过] 抓取（--skip-scrape）")
    else:
        cmd1 = [py, str(BASE / "agent_1_scrape.py")]
        if args.uid:
            cmd1 += ["--uid", args.uid]
        if args.max_pages:
            cmd1 += ["--max_pages", str(args.max_pages)]
        if not run_step("Agent 1 — 数据抓取", cmd1):
            print("抓取失败，终止。")
            sys.exit(1)

    # ── Step 2: 清洗 ──────────────────────────────────────────────────────────
    if args.skip_clean or (CHUNKS_FILE.exists() and not args.skip_scrape):
        if CHUNKS_FILE.exists():
            try:
                with open(CHUNKS_FILE, encoding="utf-8") as f:
                    cnt = len(json.load(f))
                print(f"[跳过] 清洗（processed/chunks.json 已存在：{cnt} 块）")
            except Exception:
                print(f"[跳过] 清洗（processed/chunks.json 已存在）")
        else:
            print("[跳过] 清洗（--skip-clean）")
    else:
        cmd2 = [py, str(BASE / "agent_2_clean.py"),
                "--merge_window", str(args.merge_window)]
        if not run_step("Agent 2 — 清洗分块", cmd2):
            print("清洗失败，终止。")
            sys.exit(1)

    # ── Step 3: 入库 ──────────────────────────────────────────────────────────
    db_exists = DB_DIR.exists() and any(DB_DIR.iterdir()) if DB_DIR.exists() else False

    if args.skip_embed or (db_exists and not args.reset_db):
        print(f"[跳过] 向量入库（db 已存在，使用 --reset-db 强制重建）")
    else:
        cmd3 = [py, str(BASE / "agent_3_embed.py"), "--mode", args.embed_mode]
        if args.reset_db:
            cmd3.append("--reset")
        if not run_step("Agent 3 — 向量入库", cmd3):
            print("入库失败，终止。")
            sys.exit(1)

    # ── Step 4: 查询 ──────────────────────────────────────────────────────────
    if args.query:
        run_step("Agent 4 — 单次查询", [py, str(BASE / "agent_4_query.py"), "-q", args.query])
    else:
        run_step("Agent 4 — 交互查询界面", [py, str(BASE / "agent_4_query.py"), "-i"])


if __name__ == "__main__":
    main()

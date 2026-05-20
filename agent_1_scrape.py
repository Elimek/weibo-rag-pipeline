"""
Agent 1 — 微博数据抓取（桌面版 API）
=====================================
用途：调用 weibo.com 桌面 API 下载全部历史微博（不限 200 条上限）。

使用前：
  在下方 MY_COOKIES 填入你的 SUB 和 SUBP（从 weibo.com 复制）

运行：
  python agent_1_scrape.py                          # 全量抓取
  python agent_1_scrape.py --page 1 --max_pages 5   # 只抓前5页测试
"""

import requests, json, time, argparse, os, sys, re
from datetime import datetime
from pathlib import Path

# ─── 配置区 ───────────────────────────────────────────────────────────────────
# 桌面版 weibo.com Cookie（从 Brave → weibo.com → F12 → Cookies 复制）
MY_COOKIES = {
    # 从 Brave DevTools 复制：weibo.com → F12 → Application → Cookies
    "SUB": "YOUR_SUB_COOKIE_HERE",
    "SUBP": "YOUR_SUBP_COOKIE_HERE",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.0.0 Brave/125.0.0.0",
    "Referer": "https://weibo.com/",
    "Accept": "application/json, text/plain, */*",
    "X-Requested-With": "XMLHttpRequest",
}

OUTPUT_DIR = Path(__file__).parent / "raw"
OUTPUT_FILE = OUTPUT_DIR / "weibo_raw.json"
SLEEP = 2            # 秒，桌面版反爬更严，加长间隔
MAX_RETRIES = 3
# ──────────────────────────────────────────────────────────────────────────────


def get_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    s.cookies.update(MY_COOKIES)
    return s


def verify_login(session) -> dict:
    """验证登录并获取 UID"""
    # 首页提取 uid
    r = session.get("https://weibo.com/", timeout=10)
    m = re.search(r'"uid":(\d+)', r.text)
    if m:
        uid = m.group(1)
        print(f"  UID = {uid}")
        # 顺便验证一下微博列表 API 是否可用
        check = session.get(
            "https://weibo.com/ajax/statuses/mymblog",
            params={"uid": uid, "page": 1, "feature": 0},
            timeout=10
        )
        if check.json().get("ok"):
            return {"uid": uid}
    raise ValueError("无法获取 UID 或 Cookie 无效")


def fetch_page(session, uid: str, page: int) -> dict:
    """抓取单页微博列表（桌面版API，每页20条）"""
    url = "https://weibo.com/ajax/statuses/mymblog"
    params = {"uid": uid, "page": page, "feature": 0}
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            print(f"  [warn] page={page} status={r.status_code}, retry {attempt+1}")
        except Exception as e:
            print(f"  [error] page={page} attempt={attempt+1}: {e}")
        time.sleep(2 ** attempt)
    return {}


def normalize_weibo(item: dict) -> dict:
    """标准化单条微博"""
    text_raw = item.get("text_raw", item.get("text", ""))
    topics = re.findall(r"#([^#]+)#", text_raw)

    # 时间解析
    created = item.get("created_at", "")
    try:
        dt = datetime.strptime(created, "%a %b %d %H:%M:%S %z %Y")
        iso_time = dt.isoformat()
    except:
        iso_time = created

    # 转发
    retweeted = None
    rt = item.get("retweeted_status")
    if rt:
        rt_text = rt.get("text_raw", rt.get("text", ""))
        rt_user_obj = rt.get("user") or {}
        retweeted = {
            "user": rt_user_obj.get("screen_name", "") if isinstance(rt_user_obj, dict) else "",
            "text": rt_text,
        }

    return {
        "id": str(item.get("id", "")),
        "mid": str(item.get("mid", "")),
        "text": text_raw,
        "created_at": iso_time,
        "created_at_raw": created,
        "source": item.get("source", ""),
        "topics": topics,
        "likes": item.get("attitudes_count", 0),
        "reposts": item.get("reposts_count", 0),
        "comments": item.get("comments_count", 0),
        "is_repost": retweeted is not None,
        "retweeted": retweeted,
        "pic_ids": item.get("pic_ids", []),
        "region_name": item.get("region_name", ""),
        "visible": item.get("visible", {}),
    }


def scrape_all(session, uid: str, max_pages: int = 0, start_page: int = 1) -> list[dict]:
    """抓取全部微博"""
    all_posts = []
    seen_ids = set()
    page = start_page
    empty_pages = 0

    print(f"\n[agent_1] 开始抓取 uid={uid}")
    print(f"  API：weibo.com/ajax/statuses/mymblog（桌面版）")
    print(f"  每页 20 条，间隔 {SLEEP}s")

    while True:
        print(f"  → 第 {page} 页...", end=" ", flush=True)
        data = fetch_page(session, uid, page)

        if not data or not data.get("ok"):
            print(f"API 异常，停止。")
            break

        post_list = data.get("data", {}).get("list", [])
        total = data.get("data", {}).get("total", 0)

        if not post_list:
            empty_pages += 1
            if empty_pages >= 3:
                print("连续3页空，停止。")
                break
            print(f"空页（{empty_pages}/3），继续...")
            page += 1
            time.sleep(SLEEP)
            continue
        empty_pages = 0

        new_count = 0
        for post in post_list:
            pid = str(post.get("id", ""))
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_posts.append(normalize_weibo(post))
                new_count += 1

        pct = len(all_posts) / total * 100 if total else 0
        print(f"新增 {new_count} 条（累计 {len(all_posts)}/{total}，{pct:.1f}%）")

        if max_pages > 0 and page >= start_page + max_pages - 1:
            print(f"  已达最大页数限制")
            break

        if total > 0 and len(all_posts) >= total:
            print("  已抓取全部！")
            break

        page += 1
        time.sleep(SLEEP)

    return all_posts, total


def main():
    parser = argparse.ArgumentParser(description="微博抓取 Agent（桌面版 API）")
    parser.add_argument("--uid", default="", help="微博 UID")
    parser.add_argument("--max_pages", type=int, default=0, help="最多几页（0=全量）")
    parser.add_argument("--page", type=int, default=1, help="起始页")
    args = parser.parse_args()

    session = get_session()

    if not args.uid:
        print("[agent_1] 验证登录...")
        info = verify_login(session)
        args.uid = info["uid"]
        print(f"  UID = {args.uid}")

    posts, total = scrape_all(session, args.uid, args.max_pages, args.page)

    if not posts:
        print("[agent_1] 未获取到微博")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)

    print(f"\n[agent_1] 完成！共 {len(posts)} 条（总 {total} 条）")
    print(f"  已保存：{OUTPUT_FILE}")

    dates = [p["created_at"] for p in posts if p["created_at"]]
    if dates:
        print(f"  时间范围：{min(dates)[:10]} ~ {max(dates)[:10]}")

    summary = {
        "uid": args.uid,
        "total_remote": total,
        "total_scraped": len(posts),
        "pages_scraped": max(1, (len(posts) + 19) // 20),
        "earliest": min(dates) if dates else "",
        "latest": max(dates) if dates else "",
        "scraped_at": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "scrape_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

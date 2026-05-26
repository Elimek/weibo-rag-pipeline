"""
时间感知查询模块 — 为微博 RAG 系统提供"去年今日"和跨年日期比较能力
=====================================================================

主要功能：
  1. parse_date_query()       — 解析中文日期短语（"去年今日"、"三天前"等）
  2. query_same_day_across_years() — 查询同月同日跨所有年份的结果
  3. query_date_range()       — 按日期范围精确查询
  4. build_time_timeline()    — 将按年分组的结果格式化为可读时间线
  5. has_time_intent()        — 快速检测查询中是否包含时间意图

用法示例：
  from time_filter import parse_date_query, query_same_day_across_years

  parsed = parse_date_query("去年今日")
  if parsed and parsed["type"] == "same_day_last_year":
      results = query_same_day_across_years(collection, month, day)
      timeline = build_time_timeline(results)
      print(timeline)
"""

import re
from datetime import datetime, date


# ─── 辅助工具 ────────────────────────────────────────────────────────────────


def _today() -> tuple:
    """返回 (year, month, day)，方便测试时 mock 或替换"""
    d = date.today()
    return d.year, d.month, d.day


def _extract_number(text: str) -> int | None:
    """从文本中提取第一个数字（含中文数字），失败返回 None"""
    # 先规范化中文数字
    normalized = _normalize_chinese_numbers(text)
    try:
        nums = re.findall(r"\d+", normalized)
        return int(nums[0]) if nums else None
    except (ValueError, IndexError):
        return None


# 中文数字 → 阿拉伯数字 映射表
_CHINESE_DIGIT_MAP = {
    "零": "0", "一": "1", "二": "2", "两": "2",
    "三": "3", "四": "4", "五": "5",
    "六": "6", "七": "7", "八": "8", "九": "9",
}


def _normalize_chinese_numbers(text: str) -> str:
    """将字符串中的中文数字替换为阿拉伯数字，其余不变"""
    result = []
    for ch in text:
        result.append(_CHINESE_DIGIT_MAP.get(ch, ch))
    return "".join(result)


# ─── 1. 中文日期短语解析 ─────────────────────────────────────────────────────


def parse_date_query(query_str: str) -> dict | None:
    """
    解析中文日期短语，返回结构化的查询参数字典。

    支持短语：
      "去年今日"       → {"type": "same_day_last_year"}
      "前年今日"       → {"type": "same_day_n_years_ago", "years": 2}
      "N年前的今天"    → {"type": "same_day_n_years_ago", "years": N}
      "2024年6月15日"  → {"type": "exact_date", "year": 2024, "month": 6, "day": 15}
      "上个月今天"     → {"type": "last_month_same_day"}
      "三天前" / "三天前"  → {"type": "days_ago", "days": 3}
      "今年"           → {"type": "year", "year": 2026}
      "去年"           → {"type": "year", "year": 2025}
      "这个月"         → {"type": "month", "year": 2026, "month": 5}
      "2024年"         → {"type": "year", "year": 2024}
      "2024-06"        → {"type": "month_in_year", "year": 2024, "month": 6}

    无法解析时返回 None。
    """
    if not query_str or not isinstance(query_str, str):
        return None

    text = query_str.strip()
    # 生成中文数字归一化版本（用于后续正则匹配，如 "三天前" → "3天前"）
    normalized = _normalize_chinese_numbers(text)

    try:
        # 当前日期（用于相对计算）
        now_year, now_month, now_day = _today()

        # ── 1.1 精确短语匹配（支持嵌入在查询中的情况）─────────────────────

        if "去年今日" in text:
            return {"type": "same_day_last_year"}

        if "前年今日" in text:
            return {"type": "same_day_n_years_ago", "years": 2}

        # ── 1.2 "N年前的今天"（支持中文数字：三年前的今天）─────────────
        m = re.search(r"(\d+)年前的今天", normalized)
        if m:
            years = int(m.group(1))
            return {"type": "same_day_n_years_ago", "years": years}

        # ── 1.3 "YYYY年M月D日" 或 "YYYY年MM月DD日" ─────────────────────
        m = re.search(
            r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?", normalized
        )
        if m:
            return {
                "type": "exact_date",
                "year": int(m.group(1)),
                "month": int(m.group(2)),
                "day": int(m.group(3)),
            }

        # ── 1.3b "YYYY年M月"（无日，纯年+月）───────────────────────────
        # 匹配 "2025年5月", "2025年5月的微博", "2025年5月全部微博" 等
        m = re.search(
            r"(\d{4})\s*年\s*(\d{1,2})\s*月", normalized
        )
        if m:
            return {
                "type": "month_in_year",
                "year": int(m.group(1)),
                "month": int(m.group(2)),
            }

        # ── 1.4 "上个月今天" ───────────────────────────────────────────
        if "上个月今天" in text:
            # 上个月的 year/month/day（day 可能超过上个月天数，后续需裁剪）
            if now_month == 1:
                prev_year = now_year - 1
                prev_month = 12
            else:
                prev_year = now_year
                prev_month = now_month - 1
            # 确保 day 不超出目标月的最大天数
            max_day = _days_in_month(prev_year, prev_month)
            safe_day = min(now_day, max_day)
            return {
                "type": "exact_date",
                "year": prev_year,
                "month": prev_month,
                "day": safe_day,
            }

        # ── 1.5 "N天前"（支持中文数字：三天前）─────────────────────────
        m = re.search(r"(\d+)天前", normalized)
        if m:
            days = int(m.group(1))
            return {"type": "days_ago", "days": days}

        # ── 1.6 "今年" ─────────────────────────────────────────────────
        if text == "今年":
            return {"type": "year", "year": now_year}

        # ── 1.7 "去年" ─────────────────────────────────────────────────
        if text == "去年":
            return {"type": "year", "year": now_year - 1}

        # ── 1.8 "这个月" ───────────────────────────────────────────────
        if text == "这个月":
            return {"type": "month", "year": now_year, "month": now_month}

        # ── 1.9 "YYYY年"（仅年份，支持中文数字）────────────────────────
        m = re.match(r"^(\d{4})\s*年$", normalized)
        if m:
            return {"type": "year", "year": int(m.group(1))}

        # ── 1.10 "YYYY-MM" ────────────────────────────────────────────
        m = re.match(r"^(\d{4})-(\d{1,2})$", normalized)
        if m:
            return {
                "type": "month_in_year",
                "year": int(m.group(1)),
                "month": int(m.group(2)),
            }

        # ── 1.11 "上个月"（指整个上个月） ──────────────────────────────
        if text == "上个月":
            if now_month == 1:
                prev_year = now_year - 1
                prev_month = 12
            else:
                prev_year = now_year
                prev_month = now_month - 1
            return {"type": "month", "year": prev_year, "month": prev_month}

        # 无法解析
        return None

    except Exception:
        # 任何解析异常都返回 None
        return None


# ─── 2. 跨年同日查询 ───────────────────────────────────────────────────────


def query_same_day_across_years(
    collection, month: int, day: int, top_k_per_year: int = 10
) -> dict:
    """
    查询 ChromaDB 中所有年份的同一月/日记录，按年份分组。

    参数：
      collection      — ChromaDB collection 对象
      month, day      — 月份和日期
      top_k_per_year  — 每年最多返回条数

    返回：
      {year: [{"text": str, "metadata": dict, "score": None}, ...], ...}
    """
    if not collection:
        return {}

    try:
        # 过滤条件：匹配 month 和 day
        where_filter = {"$and": [
            {"month": {"$eq": month}},
            {"day": {"$eq": day}},
        ]}

        # 使用 get() 获取所有匹配记录（不设 limit 以覆盖全部年份）
        results = collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )

        # 按年份分组
        grouped: dict[int, list[dict]] = {}
        for i, doc in enumerate(results["documents"]):
            meta = results["metadatas"][i]
            yr = meta.get("year")
            if yr is None:
                continue
            yr = int(yr)

            entry = {"text": doc, "metadata": meta, "score": None}

            if yr not in grouped:
                grouped[yr] = []
            grouped[yr].append(entry)

        # 每年按 created_at 排序并截断
        for yr in grouped:
            grouped[yr].sort(
                key=lambda h: h["metadata"].get("created_at", ""),
            )
            if top_k_per_year > 0:
                grouped[yr] = grouped[yr][:top_k_per_year]

        # 按年份降序排列（最近的在前面）
        sorted_grouped = dict(sorted(grouped.items(), key=lambda x: x[0], reverse=True))

        return sorted_grouped

    except Exception:
        return {}


# ─── 3. 日期范围查询 ────────────────────────────────────────────────────────


def query_date_range(
    collection,
    year: int,
    month: int | None = None,
    day: int | None = None,
    top_k: int = 50,
) -> list[dict]:
    """
    按日期范围精确查询，返回按时间排序的结果列表。

    参数：
      collection  — ChromaDB collection 对象
      year        — 年份（必填）
      month       — 月份（可选，不填则查全年）
      day         — 日期（可选，需要 month 同时提供）
      top_k       — 最多返回条数

    返回：
      [{"text": str, "metadata": dict, "score": None}, ...]
    """
    if not collection:
        return []

    try:
        conditions = [{"year": {"$eq": int(year)}}]

        if month is not None:
            conditions.append({"month": {"$eq": int(month)}})
        if day is not None and month is not None:
            conditions.append({"day": {"$eq": int(day)}})

        where_filter = (
            {"$and": conditions} if len(conditions) > 1 else conditions[0]
        )

        results = collection.get(
            where=where_filter,
            limit=top_k,
            include=["documents", "metadatas"],
        )

        hits = []
        for i, doc in enumerate(results["documents"]):
            meta = results["metadatas"][i]
            hits.append({"text": doc, "metadata": meta, "score": None})

        # 按 datetime_str 或 date_str 排序
        hits.sort(
            key=lambda h: (
                h["metadata"].get("datetime_str")
                or h["metadata"].get("date_str")
                or h["metadata"].get("created_at", "")
            )
        )

        return hits

    except Exception:
        return []


# ─── 4. 跨年时间线格式化 ────────────────────────────────────────────────────


def build_time_timeline(results_by_year: dict) -> str:
    """
    将按年分组的结果格式化为可读的 Markdown 时间线。

    输入示例：
      {2025: [{"text": "今天很开心", "metadata": {"month": 5, "day": 20}}, ...],
       2024: [{"text": "去年今天在干嘛", "metadata": {"month": 5, "day": 20}}, ...]}

    输出示例：
      ## 📅 跨年同日回顾 — 5月20日

      ### 2025年
      - 2025-05-20 12:30: 今天很开心

      ### 2024年
      - 2024-05-20 08:15: 去年今天在干嘛
    """
    if not results_by_year:
        return "（无相关记录）"

    try:
        # 从第一条记录中提取 month/day 作为标题信息
        first_year = next(iter(results_by_year.values()))
        sample_month = None
        sample_day = None
        if first_year:
            meta = first_year[0].get("metadata", {})
            sample_month = meta.get("month")
            sample_day = meta.get("day")

        # 构建标题
        if sample_month and sample_day:
            title = f"跨年同日回顾 — {sample_month}月{sample_day}日"
        else:
            title = "跨年同日回顾"

        lines = [f"## 📅 {title}", ""]

        for year in sorted(results_by_year.keys(), reverse=True):
            entries = results_by_year[year]
            if not entries:
                continue

            lines.append(f"### {year}年\n")

            for entry in entries:
                meta = entry.get("metadata", {})
                dt = (
                    meta.get("datetime_str")
                    or meta.get("date_str")
                    or f"{meta.get('year', '?')}-{meta.get('month', '?'):02d}-{meta.get('day', '?'):02d}"
                )
                text = entry.get("text", "")
                # 截取过长文本并换行缩进
                if len(text) > 100:
                    text = text[:100] + "…"

                lines.append(f"- {dt}: {text}")

            lines.append("")

        return "\n".join(lines)

    except Exception:
        return "（格式化时间线时出错）"


# ─── 5. 时间意图检测 ────────────────────────────────────────────────────────


# 常见时间关键词（用于快速检测是否包含时间意图）
_TIME_KEYWORDS = [
    "去年", "前年", "今年", "明年",
    "今天", "昨天", "前天", "明天", "后天",
    "上个月", "这个月", "下个月",
    "上星期", "这星期", "下星期", "上周", "这周", "下周",
    "天前", "月前", "年前",
    "年", "月", "日", "天",
    "时候", "期间", "阶段", "时期",
    "最近", "之前", "之后", "以前", "以后",
    "年初", "月底", "年末", "月初",
    "春季", "夏季", "秋季", "冬季",
    "生日", "新年", "春节", "国庆", "元旦", "圣诞",
]

# 日期格式正则模式（如 "2024年", "2024-06", "2024年6月"）
_DATE_PATTERNS = [
    re.compile(r"\d{4}\s*年"),           # 2024年
    re.compile(r"\d{4}-\d{1,2}"),        # 2024-06
    re.compile(r"\d{4}/\d{1,2}"),        # 2024/06
    re.compile(r"\d{1,2}月\d{1,2}日"),   # 6月15日
]


def has_time_intent(query_str: str) -> bool:
    """
    快速检测查询字符串中是否包含时间相关意图。

    返回 True 表示可能涉及时间范围、日期比较或回顾性查询。
    适用于在调度层决定是否调用 parse_date_query / query_same_day_across_years。
    """
    if not query_str or not isinstance(query_str, str):
        return False

    text = query_str.strip().lower()

    # 检查关键词
    for kw in _TIME_KEYWORDS:
        if kw in text:
            return True

    # 检查日期格式模式
    for pattern in _DATE_PATTERNS:
        if pattern.search(text):
            return True

    return False


# ─── 内部辅助函数 ────────────────────────────────────────────────────────────


def _days_in_month(year: int, month: int) -> int:
    """返回指定年月的天数"""
    from calendar import monthrange
    return monthrange(year, month)[1]


# ─── 调试 / 测试 ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # 快速测试 parse_date_query
    test_cases = [
        "去年今日",
        "前年今日",
        "3年前的今天",
        "三年前的今天",   # 中文数字
        "2024年6月15日",
        "上个月今天",
        "三天前",         # 中文数字
        "3天前",          # 阿拉伯数字
        "今年",
        "去年",
        "这个月",
        "2024年",
        "2024-06",
        "上个月",
        "不可识别的输入",
        "",
    ]

    print("=" * 50)
    print("parse_date_query 测试\n")
    for tc in test_cases:
        result = parse_date_query(tc)
        print(f"  {tc:20s} → {result}")
    print("=" * 50)
    print(f"\nhas_time_intent 测试：")
    print(f"  '去年今日的动态'     → {has_time_intent('去年今日的动态')}")
    print(f"  '找工作焦虑'         → {has_time_intent('找工作焦虑')}")
    print(f"  '2024年去了哪'       → {has_time_intent('2024年去了哪')}")
    print(f"  '今天天气真好'       → {has_time_intent('今天天气真好')}")
    print()

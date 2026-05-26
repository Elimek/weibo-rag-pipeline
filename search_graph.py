"""
搜索管线 — LangGraph 状态机
============================
7 节点有向图：ParseQuery → Route → Search → Merge → Report → Critic → 自愈/返回

使用：
  from search_graph import build_search_graph, SearchState
  graph = build_search_graph(collection, model, embed_fn)
  result = graph.invoke({"query": "去年今日", "top_k": 8})
"""

import re
from datetime import date, datetime
from typing import TypedDict, Annotated, Optional, Literal
from operator import add

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

try:
    from time_filter import parse_date_query, query_same_day_across_years, has_time_intent
    HAS_TIME_FILTER = True
except ImportError:
    HAS_TIME_FILTER = False

try:
    from critic import generate_critic_report, compute_faithfulness
    HAS_CRITIC = True
except ImportError:
    HAS_CRITIC = False


# ─── 状态类型 ─────────────────────────────────────────────────────────────────
class SearchState(TypedDict):
    query: str                           # 用户原始查询
    semantic_text: str                   # 剥离日期后的语义文本
    date_filter: dict                    # {"year": ..., "month": ..., "day": ...}
    query_type: str                      # pure_date | semantic | mixed | same_day_last_year | same_day_n_years_ago
    top_k: int                           # 返回数量
    hits: list                           # 检索结果
    report_html: str                     # 生成的 HTML 报告
    critic_score: float                  # Critic 忠实性分数 (0-1)
    critic_report: str                   # Critic 详细报告文本
    retry_count: int                     # 已重试次数
    max_retries: int                     # 最大重试次数
    error: Optional[str]                 # 错误信息


# ─── 全局引擎引用（由 build_search_graph 注入）─────────────────────────────────
_collection = None
_model = None
_embed_fn = None


# ─── 节点实现 ─────────────────────────────────────────────────────────────────

def parse_query_node(state: SearchState) -> dict:
    """解析查询：提取日期信息，分离语义文本"""
    q = state.get("query", "")
    top_k = state.get("top_k", 8)
    result = {
        "semantic_text": q,
        "date_filter": {},
        "query_type": "semantic",
    }

    if not q:
        return result

    if HAS_TIME_FILTER:
        time_info = parse_date_query(q)
        if time_info:
            remaining = _strip_date(q, time_info)
            result["semantic_text"] = remaining
            result["query_type"] = time_info.get("type", "semantic")
            df = {}
            if time_info.get("year"): df["year"] = time_info["year"]
            if time_info.get("month"): df["month"] = time_info["month"]
            if time_info.get("day"): df["day"] = time_info["day"]
            result["date_filter"] = df

            # 修正 query_type
            t = time_info.get("type", "")
            if t in ("same_day_last_year", "same_day_n_years_ago"):
                result["query_type"] = t
            elif df and not remaining.strip():
                result["query_type"] = "pure_date"
            elif df and remaining.strip():
                result["query_type"] = "mixed"
            else:
                result["query_type"] = "semantic"

    return result


def _strip_date(query: str, time_info: dict) -> str:
    """从查询中剥离日期部分"""
    remaining = query
    t = time_info.get("type", "")
    if t in ("same_day_last_year", "same_day_n_years_ago"):
        remaining = re.sub(r'(去年|前年|\d+\s*年前的)\s*今日?', '', remaining).strip()
        remaining = re.sub(r'\d+\s*年前的今天', '', remaining).strip()
    elif t == "exact_date":
        remaining = re.sub(r'\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日?', '', remaining).strip()
    elif t == "month_in_year":
        remaining = re.sub(r'\d{4}\s*年\s*\d{1,2}\s*月', '', remaining).strip()
    elif t == "year":
        remaining = re.sub(r'\d{4}\s*年', '', remaining).strip()
    remaining = re.sub(r'^[的]+', '', remaining).strip()
    remaining = re.sub(r'(全部|所有)?微博?$', '', remaining).strip()
    return remaining


def route_search(state: SearchState) -> Literal["metadata_search", "semantic_search", "same_day_search"]:
    """条件路由：基于 query_type 选择搜索路径"""
    qt = state.get("query_type", "semantic")
    if qt == "pure_date":
        return "metadata_search"
    elif qt in ("same_day_last_year", "same_day_n_years_ago"):
        return "same_day_search"
    return "semantic_search"


def metadata_search_node(state: SearchState) -> dict:
    """纯 metadata 搜索（不经过语义向量）"""
    df = state.get("date_filter", {})
    where = _build_where(df)
    if not where or not _collection:
        return {"hits": []}
    try:
        results = _collection.get(where=where, limit=state.get("top_k", 8) * 2,
                                   include=["documents", "metadatas"])
        hits = [{"text": results["documents"][i], "meta": results["metadatas"][i], "score": None}
                for i in range(len(results["documents"]))]
        hits.sort(key=lambda h: h["meta"].get("created_at", ""), reverse=True)
        return {"hits": hits}
    except Exception as e:
        return {"hits": [], "error": str(e)}


def _build_where(date_dict: dict) -> Optional[dict]:
    conds = []
    for k, v in date_dict.items():
        if k == "year": conds.append({"year": {"$eq": int(v)}})
        elif k == "month": conds.append({"month": {"$eq": int(v)}})
        elif k == "day": conds.append({"day": {"$eq": int(v)}})
    if not conds: return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def semantic_search_node(state: SearchState) -> dict:
    """语义搜索"""
    text = state.get("semantic_text") or state.get("query", "")
    df = state.get("date_filter", {})
    where = _build_where(df)

    if not text or not _model:
        return {"hits": []}

    try:
        q_vec = _model.encode([text], normalize_embeddings=True).tolist()
        results = _collection.query(query_embeddings=q_vec, n_results=state.get("top_k", 8),
                                     where=where, include=["documents", "metadatas", "distances"])
        if not results["documents"] or not results["documents"][0]:
            return {"hits": []}
        hits = [{"text": results["documents"][0][i], "meta": results["metadatas"][0][i],
                 "score": round(1 - results["distances"][0][i], 4)}
                for i in range(len(results["documents"][0]))]
        return {"hits": hits}
    except Exception as e:
        # 降级：去掉 where 重试
        try:
            q_vec = _model.encode([text], normalize_embeddings=True).tolist()
            results = _collection.query(query_embeddings=q_vec, n_results=state.get("top_k", 8),
                                         include=["documents", "metadatas", "distances"])
            hits = [{"text": results["documents"][0][i], "meta": results["metadatas"][0][i],
                     "score": round(1 - results["distances"][0][i], 4)}
                    for i in range(len(results["documents"][0]))]
            return {"hits": hits}
        except:
            return {"hits": [], "error": str(e)}


def same_day_search_node(state: SearchState) -> dict:
    """去年今日搜索 — 跨年同日对比"""
    today_obj = date.today()
    query = state.get("query", "")
    qt = state.get("query_type", "")
    years_back = 2 if qt == "same_day_n_years_ago" else 1
    target_year = today_obj.year - years_back
    tm, td = today_obj.month, today_obj.day

    all_hits = []

    # 当年同日
    if target_year >= 2024:
        try:
            w = _build_where({"year": today_obj.year, "month": tm, "day": td})
            if w:
                r = _collection.get(where=w, limit=20, include=["documents", "metadatas"])
                all_hits.extend([{"text": r["documents"][i], "meta": r["metadatas"][i], "score": None}
                                 for i in range(len(r["documents"]))])
        except: pass

    # 目标年同日
    try:
        w = _build_where({"year": target_year, "month": tm, "day": td})
        if w:
            r = _collection.get(where=w, limit=20, include=["documents", "metadatas"])
            all_hits.extend([{"text": r["documents"][i], "meta": r["metadatas"][i], "score": None}
                             for i in range(len(r["documents"]))])
    except: pass

    # 跨年同日补充
    if HAS_TIME_FILTER:
        try:
            by_y = query_same_day_across_years(_collection, tm, td, top_k_per_year=5)
            for y, entries in by_y.items():
                for e in entries:
                    # 标准化字段名：time_filter 返回 'metadata'，内部用 'meta'
                    if 'meta' not in e and 'metadata' in e:
                        e['meta'] = e.pop('metadata')
                    if e not in all_hits: all_hits.append(e)
        except: pass

    all_hits.sort(key=lambda h: h["meta"].get("created_at", ""), reverse=True)
    return {"hits": all_hits[:40]}


def merge_results_node(state: SearchState) -> dict:
    """结果去重排序（当前无多路输入，为图预留）"""
    hits = state.get("hits", [])
    seen = set()
    deduped = []
    for h in hits:
        # 标准化字段名
        if 'meta' not in h and 'metadata' in h:
            h['meta'] = h.pop('metadata')
        tid = h.get("meta", {}).get("id", "") or str(id(h))
        if tid and tid not in seen:
            seen.add(tid)
            deduped.append(h)
    return {"hits": deduped}


def generate_report_node(state: SearchState) -> dict:
    """生成记忆报告 HTML"""
    query = state.get("query", "")
    hits = state.get("hits", [])
    now = datetime.now()

    parts = []
    parts.append(f"""
    <div style="margin-bottom:16px">
      <h2 style="margin:0 0 4px;font-size:20px">📖 记忆报告</h2>
      <p style="margin:0;color:#86868b;font-size:13px">
        查询："{query}" · {now.strftime('%Y-%m-%d %H:%M')}
        {f' · 🔍 {len(hits)} 条结果' if hits else ''}
      </p>
    </div>
    <hr style="border:none;border-top:0.5px solid #e5e5e7;margin:0 0 16px">""")

    if not hits:
        parts.append('<div style="text-align:center;padding:40px;color:#86868b;font-size:14px">未找到相关微博</div>')
        return {"report_html": "".join(parts)}

    # 去年今日区块
    qt = state.get("query_type", "")
    if qt in ("same_day_last_year", "same_day_n_years_ago"):
        parts.append("""
        <div style="background:#E1F5EE;border-radius:12px;padding:16px;margin-bottom:16px">
          <h3 style="margin:0 0 8px;font-size:15px;color:#0F6E56">📅 去年今日</h3>
          <table style="width:100%;border-collapse:collapse;font-size:13px">""")
        by_year = {}
        for h in hits:
            y = h["meta"].get("year")
            if y: by_year.setdefault(y, []).append(h)
        for y in sorted(by_year.keys(), reverse=True):
            texts = "<br>".join(f'<span style="color:#515154">· {e["text"][:80]}</span>' for e in by_year[y][:3])
            parts.append(f'<tr><td style="padding:6px 8px;font-weight:500;color:#0F6E56;width:60px;vertical-align:top">{y}年</td><td style="padding:6px 8px">{texts}</td></tr>')
        parts.append("</table></div>")

    # 结果卡片
    parts.append('<h3 style="font-size:15px;margin:0 0 10px">🔍 相关微博</h3>')
    for i, h in enumerate(hits):
        meta = h["meta"]
        dt = meta.get("datetime_str") or meta.get("date_str") or ""
        sb = ""
        if h["score"]:
            c = "#34c759" if h["score"] > 0.7 else "#ff9500" if h["score"] > 0.5 else "#ff3b30"
            sb = f'<span style="font-size:11px;color:{c}">相似度 {h["score"]:.2f}</span>'
        topics = "".join(f'<span style="display:inline-block;background:#f0edff;color:#5e3bd1;font-size:11px;padding:1px 6px;border-radius:4px;margin:0 2px">#{t}#</span>'
                         for t in (meta.get("topics","").split("|") if meta.get("topics") else []) if t)
        rg = f'<span style="display:inline-block;background:#f5f5f7;font-size:11px;padding:1px 6px;border-radius:4px;margin:0 2px">📍{meta.get("region","")}</span>' if meta.get("region") else ""
        parts.append(f"""
        <div style="background:#fff;border:0.5px solid #e5e5e7;border-radius:12px;padding:12px 14px;margin-bottom:10px">
          <div style="display:flex;gap:6px;align-items:center;margin-bottom:4px;flex-wrap:wrap">
            <span style="font-size:12px;color:#86868b">{dt}</span>{sb}{topics}{rg}
          </div>
          <div style="font-size:14px;line-height:1.7">{h["text"][:200]}{"…" if len(h["text"])>200 else ""}</div>
        </div>""")

    return {"report_html": "".join(parts)}


def critic_review_node(state: SearchState) -> dict:
    """Critic 审核 — 检查报告忠实性"""
    hits = state.get("hits", [])
    query = state.get("query", "")
    report_html = state.get("report_html", "")

    if not HAS_CRITIC or not hits:
        return {"critic_score": 1.0, "critic_report": "Critic 模块未加载"}

    try:
        src = [h["text"] for h in hits[:5]]
        ans = f"关于「{query}」的微博回忆：{hits[0]['text'][:100]}"
        if len(hits) > 1: ans += f" 还有{hits[1]['text'][:60]}"

        embed_fn = _embed_fn or (lambda t: _model.encode([t], normalize_embeddings=True).tolist()[0])
        report = generate_critic_report(ans, src, embed_fn, query)

        sm = re.search(r'忠实性评分[：:]\s*([\d.]+)', report)
        score = float(sm.group(1)) if sm else 1.0

        return {"critic_score": score, "critic_report": report}
    except Exception as e:
        return {"critic_score": 1.0, "critic_report": f"审核失败: {e}"}


def decide_retry(state: SearchState) -> Literal["research", "finalize"]:
    """条件边：Critic 分数低时重试"""
    score = state.get("critic_score", 1.0)
    retries = state.get("retry_count", 0)
    max_r = state.get("max_retries", 2)
    if score < 0.5 and retries < max_r:
        return "research"
    return "finalize"


def research_node(state: SearchState) -> dict:
    """重试：扩大检索范围 + 增加结果数"""
    retry = state.get("retry_count", 0) + 1
    top_k = state.get("top_k", 8) + 4 * retry
    return {"retry_count": retry, "top_k": top_k}


def finalize_node(state: SearchState) -> dict:
    """终节点 — 将 critic 报告附加到 HTML"""
    html = state.get("report_html", "")
    cs = state.get("critic_score", 1.0)
    cr = state.get("critic_report", "")

    if cr and "Critic 模块" not in cr and cs < 1.0:
        risk = "高" if cs < 0.4 else "中" if cs < 0.6 else "低"
        rc = "#ff3b30" if risk == "高" else "#ff9500" if risk == "中" else "#34c759"
        html += f"""
        <div style="background:#FAECE7;border-radius:12px;padding:16px;margin-top:16px">
          <h3 style="margin:0 0 8px;font-size:15px;color:#993C1D">✅ LangGraph Critic 审核</h3>
          <div style="display:flex;gap:16px;margin-bottom:10px">
            <div style="background:#fff;border-radius:8px;padding:8px 12px;text-align:center;min-width:80px">
              <div style="font-size:18px;font-weight:600;color:#1d1d1f">{cs:.2f}</div>
              <div style="font-size:11px;color:#86868b">忠实性</div>
            </div>
            <div style="background:#fff;border-radius:8px;padding:8px 12px;text-align:center;min-width:80px">
              <div style="font-size:18px;font-weight:600;color:{rc}">{risk}</div>
              <div style="font-size:11px;color:#86868b">幻觉风险</div>
            </div>
            {f'<div style="background:#fff;border-radius:8px;padding:8px 12px;text-align:center;min-width:80px"><div style="font-size:18px;font-weight:600;color:#1d1d1f">{state.get("retry_count",0)}</div><div style="font-size:11px;color:#86868b">重试次数</div></div>' if state.get("retry_count",0) > 0 else ""}
          </div>
          <div style="font-size:12px;color:#515154;line-height:1.6">{cr[:200]}</div>
        </div>"""

    return {"report_html": html}


# ─── 构建图 ───────────────────────────────────────────────────────────────────
def build_search_graph(collection, model, embed_fn=None):
    """
    构建完整的 LangGraph 搜索管线

    参数:
      collection: ChromaDB collection
      model: SentenceTransformer 模型
      embed_fn: embedding 函数（可选，默认用 model.encode）

    返回: Compilable Graph
    """
    global _collection, _model, _embed_fn
    _collection = collection
    _model = model
    _embed_fn = embed_fn

    if not HAS_LANGGRAPH:
        raise ImportError("请安装 langgraph：pip install langgraph")

    builder = StateGraph(SearchState)

    builder.add_node("parse_query", parse_query_node)
    builder.add_node("metadata_search", metadata_search_node)
    builder.add_node("semantic_search", semantic_search_node)
    builder.add_node("same_day_search", same_day_search_node)
    builder.add_node("merge_results", merge_results_node)
    builder.add_node("generate_report", generate_report_node)
    builder.add_node("critic_review", critic_review_node)
    builder.add_node("research", research_node)
    builder.add_node("finalize", finalize_node)

    builder.set_entry_point("parse_query")
    builder.add_conditional_edges("parse_query", route_search, {
        "metadata_search": "metadata_search",
        "semantic_search": "semantic_search",
        "same_day_search": "same_day_search",
    })
    builder.add_edge("metadata_search", "merge_results")
    builder.add_edge("semantic_search", "merge_results")
    builder.add_edge("same_day_search", "merge_results")
    builder.add_edge("merge_results", "generate_report")
    builder.add_edge("generate_report", "critic_review")
    builder.add_conditional_edges("critic_review", decide_retry, {
        "research": "research",
        "finalize": "finalize",
    })
    builder.add_edge("research", "semantic_search")
    builder.add_edge("finalize", END)

    return builder.compile()


def search_via_graph(collection, model, query: str, top_k: int = 8) -> str:
    """
    快捷函数：用 LangGraph 搜索并返回 HTML 报告

    用法:
      html = search_via_graph(collection, model, "去年今日")
    """
    graph = build_search_graph(collection, model)
    result = graph.invoke({
        "query": query,
        "top_k": top_k,
        "retry_count": 0,
        "max_retries": 2,
        "hits": [],
        "report_html": "",
        "critic_score": 1.0,
        "critic_report": "",
        "error": None,
    })
    return result.get("report_html", "<div>搜索失败</div>")

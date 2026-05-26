"""
Weibo RAG — Gradio Web 界面
============================
集成语义搜索、时间过滤（去年今日）、Critic 审核。
输出结构化的「记忆报告」。

v2 hardening:
  - 日期短语与语义查询分离，纯日期查询走 metadata 过滤
  - 支持 "2025年5月"、"2025年5月全部微博" 等年月级查询
  - 去年今日跨年对比
"""

import sys, json, re
from pathlib import Path
from datetime import datetime, date

try:
    import gradio as gr
except ImportError:
    print("请安装 gradio：pip install gradio")
    sys.exit(1)
try:
    import chromadb
except ImportError:
    print("请安装 chromadb：pip install chromadb")
    sys.exit(1)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("请安装 sentence-transformers：pip install sentence-transformers")
    sys.exit(1)

# ─── 可选模块 ────────────────────────────────────────────────────────────────
try:
    from time_filter import parse_date_query, query_same_day_across_years, has_time_intent
    HAS_TIME_FILTER = True
except ImportError:
    HAS_TIME_FILTER = False
try:
    from critic import generate_critic_report, quick_verify
    HAS_CRITIC = True
except ImportError:
    HAS_CRITIC = False

# ─── 路径 ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DB_DIR = BASE / "db" / "chroma"
COLLECTION_NAME = "weibo_posts"
collection = None
model = None


# ─── 引擎 ─────────────────────────────────────────────────────────────────────
def load_engine():
    print("[engine] 加载 ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        c = client.get_collection(COLLECTION_NAME)
        print(f"  ✅ 向量库：{c.count()} 条")
    except Exception as e:
        print(f"  ❌ 加载失败：{e}")
        return None, None
    print("[engine] 加载 BAAI/bge-m3 模型...")
    try:
        m = SentenceTransformer("BAAI/bge-m3")
        print("  ✅ 模型就绪")
    except Exception as e:
        print(f"  ❌ 模型加载失败：{e}")
        return None, None
    return c, m


# ─── 搜索原语 ────────────────────────────────────────────────────────────────
def build_where(year=None, month=None, day=None, hour=None, topic=None):
    conds = []
    if year: conds.append({"year": {"$eq": int(year)}})
    if month: conds.append({"month": {"$eq": int(month)}})
    if day: conds.append({"day": {"$eq": int(day)}})
    if hour is not None: conds.append({"hour": {"$eq": int(hour)}})
    if topic: conds.append({"topics": {"$contains": topic}})
    if not conds: return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def semantic_search(collection, model, query, where=None, top_k=8):
    q_vec = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=q_vec, n_results=top_k, where=where,
                               include=["documents", "metadatas", "distances"])
    return [{"text": results["documents"][0][i], "meta": results["metadatas"][0][i],
             "score": round(1 - results["distances"][0][i], 4)}
            for i in range(len(results["documents"][0]))]


def get_by_date_exact(collection, year, month, day, limit=50):
    where = build_where(year=year, month=month, day=day)
    if not where: return []
    results = collection.get(where=where, limit=limit, include=["documents", "metadatas"])
    hits = [{"text": results["documents"][i], "meta": results["metadatas"][i], "score": None}
            for i in range(len(results["documents"]))]
    hits.sort(key=lambda h: h["meta"].get("created_at", ""))
    return hits


def get_by_where(collection, where, limit=50):
    if not where: return []
    results = collection.get(where=where, limit=limit, include=["documents", "metadatas"])
    hits = [{"text": results["documents"][i], "meta": results["metadatas"][i], "score": None}
            for i in range(len(results["documents"]))]
    hits.sort(key=lambda h: h["meta"].get("created_at", ""), reverse=True)
    return hits


# ─── 核心：日期与语义分离 ─────────────────────────────────────────────────────
def _extract_date(query_str: str):
    """
    从查询中提取日期信息，返回 (where_dict, semantic_text)
    where_dict: 可用于 ChromaDB 过滤的字典 {"year": ..., "month": ..., "day": ...}
    semantic_text: 剥离日期后的纯语义文本
    无法提取时返回 (None, query_str)
    """
    if not HAS_TIME_FILTER or not query_str:
        return None, query_str

    time_info = parse_date_query(query_str)
    if not time_info:
        return None, query_str

    w = {}
    remaining = query_str

    t = time_info.get("type")
    if t in ("same_day_last_year", "same_day_n_years_ago"):
        remaining = re.sub(r'(去年|前年|\d+\s*年前的)\s*今日?', '', remaining).strip()
        remaining = re.sub(r'\d+\s*年前的今天', '', remaining).strip()
        return time_info, remaining

    if t == "exact_date":
        if time_info.get("year"): w["year"] = time_info["year"]
        if time_info.get("month"): w["month"] = time_info["month"]
        if time_info.get("day"): w["day"] = time_info["day"]
        remaining = re.sub(r'\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日?', '', remaining).strip()

    elif t == "month_in_year":
        if time_info.get("year"): w["year"] = time_info["year"]
        if time_info.get("month"): w["month"] = time_info["month"]
        remaining = re.sub(r'\d{4}\s*年\s*\d{1,2}\s*月', '', remaining).strip()

    elif t == "year":
        if time_info.get("year"): w["year"] = time_info["year"]
        remaining = re.sub(r'\d{4}\s*年', '', remaining).strip()

    elif t in ("month", "last_month_same_day", "days_ago"):
        if time_info.get("year"): w["year"] = time_info["year"]
        if time_info.get("month"): w["month"] = time_info["month"]
        if time_info.get("day"): w["day"] = time_info["day"]
        remaining = ""

    # 清理残留：全部微博、的微博 等
    remaining = re.sub(r'^[的]+', '', remaining).strip()
    remaining = re.sub(r'(全部|所有)?微博?$', '', remaining).strip()

    return w, remaining


def search_weibo(query, use_today, year_filter, month_filter, topic_filter, top_k):
    """全线搜索入口 — 纯日期分流 + 语义搜索 + 跨年对比"""
    global collection, model
    if collection is None or model is None:
        collection, model = load_engine()
        if collection is None:
            return "<div style='text-align:center;padding:40px;color:#ff3b30'>引擎加载失败</div>"

    if not query and not year_filter and not month_filter and not topic_filter:
        return "<div style='text-align:center;padding:40px;color:#86868b'>输入问题开始搜索</div>"

    # ── 第1步：从查询文本中提取日期 ────────────────────────────────────────
    date_info, semantic_text = _extract_date(query)

    # ── 第2步：合并过滤条件（查询提取 + UI 筛选栏） ─────────────────────────
    where_dict = {}
    if isinstance(date_info, dict) and "year" in date_info:
        where_dict.update(date_info)
    # UI 筛选栏覆盖查询中的
    if year_filter: where_dict["year"] = int(year_filter)
    if month_filter: where_dict["month"] = int(month_filter)
    if topic_filter: where_dict["topics"] = topic_filter

    where = build_where(**where_dict) if where_dict else None
    is_pure_date = (isinstance(date_info, dict) and not semantic_text) or (
        where_dict and not semantic_text
    )

    # ── 第3步：路由 ────────────────────────────────────────────────────────

    # 3a. 去年今日 系列
    if isinstance(date_info, dict) and date_info.get("type") in ("same_day_last_year", "same_day_n_years_ago"):
        today_obj = date.today()
        target_year = today_obj.year - date_info.get("years", 1)
        tm, td = today_obj.month, today_obj.day
        current = get_by_date_exact(collection, today_obj.year, tm, td)
        past = get_by_date_exact(collection, target_year, tm, td)
        all_hits = current + past
        all_hits.sort(key=lambda h: h["meta"].get("created_at", ""), reverse=True)
        # 如果有语义文本
        if semantic_text and len(semantic_text) >= 2:
            yr_where = build_where(year=target_year, month=tm, day=td)
            sem = semantic_search(collection, model, semantic_text, where=yr_where, top_k=top_k)
            all_hits = sem + [h for h in all_hits if h not in sem]
        try:
            by_y = query_same_day_across_years(collection, tm, td, top_k_per_year=5)
            for y, entries in by_y.items():
                for e in entries:
                    if e not in all_hits:
                        all_hits.append(e)
        except Exception:
            pass
        return build_report(query, all_hits[:top_k*2], collection, model, date_info)

    # 3b. 纯日期查询 — 不经过语义搜索，直接 metadata 过滤
    if is_pure_date:
        try:
            hits = get_by_where(collection, where, limit=top_k*2)
        except Exception as e:
            # ChromaDB 内部错误降级：缩小范围
            print(f"[warn] get_by_where 失败: {e}，尝试缩小范围")
            try:
                where = build_where(**{k: where_dict[k] for k in list(where_dict.keys())[:2]})
                hits = get_by_where(collection, where, limit=min(top_k, 8))
            except:
                hits = []
        return build_report(query, hits, collection, model, date_info if isinstance(date_info, dict) else None)

    # 3c. 语义搜索（可能带时间过滤）
    search_text = semantic_text or query
    if search_text:
        try:
            hits = semantic_search(collection, model, search_text, where=where, top_k=top_k)
        except Exception as e:
            print(f"[warn] semantic_search 失败: {e}")
            try:
                hits = semantic_search(collection, model, search_text, where=None, top_k=top_k)
            except:
                hits = get_by_where(collection, where, limit=top_k*2) if where else []
    else:
        hits = get_by_where(collection, where, limit=top_k)

    return build_report(query, hits, collection, model, date_info if isinstance(date_info, dict) else None)


# ─── 报告生成 ─────────────────────────────────────────────────────────────────
def build_report(query, hits, collection, model, time_info=None):
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
    <hr style="border:none;border-top:0.5px solid #e5e5e7;margin:0 0 16px">
    """)
    if not hits:
        parts.append('<div style="text-align:center;padding:40px;color:#86868b;font-size:14px">未找到相关微博</div>')
        return "".join(parts)

    # 去年今日区块
    if time_info and time_info.get("type") in ("same_day_last_year", "same_day_n_years_ago"):
        year_label = time_info.get("description", "去年今日")
        parts.append(f"""
        <div style="background:#E1F5EE;border-radius:12px;padding:16px;margin-bottom:16px">
          <h3 style="margin:0 0 8px;font-size:15px;color:#0F6E56">📅 {year_label}</h3>
          <table style="width:100%;border-collapse:collapse;font-size:13px">""")
        by_year = {}
        for h in hits:
            y = h["meta"].get("year")
            if y: by_year.setdefault(y, []).append(h)
        for y in sorted(by_year.keys(), reverse=True):
            texts = "<br>".join(f'<span style="color:#515154">· {e["text"][:80]}</span>' for e in by_year[y][:3])
            parts.append(f'<tr><td style="padding:6px 8px;font-weight:500;color:#0F6E56;width:60px;vertical-align:top">{y}年</td><td style="padding:6px 8px">{texts}</td></tr>')
        parts.append("</table></div>")

    # 搜索结果
    parts.append('<h3 style="font-size:15px;margin:0 0 10px">🔍 相关微博</h3>')
    for i, h in enumerate(hits):
        meta = h["meta"]
        dt = meta.get("datetime_str") or meta.get("date_str") or ""
        score_badge = ""
        if h["score"]:
            color = "#34c759" if h["score"] > 0.7 else "#ff9500" if h["score"] > 0.5 else "#ff3b30"
            score_badge = f'<span style="font-size:11px;color:{color}">相似度 {h["score"]:.2f}</span>'
        topics = "".join(f'<span style="display:inline-block;background:#f0edff;color:#5e3bd1;font-size:11px;padding:1px 6px;border-radius:4px;margin:0 2px">#{t}#</span>'
                         for t in (meta.get("topics","").split("|") if meta.get("topics") else []) if t)
        region = f'<span style="display:inline-block;background:#f5f5f7;font-size:11px;padding:1px 6px;border-radius:4px;margin:0 2px">📍{meta.get("region","")}</span>' if meta.get("region") else ""
        parts.append(f"""
        <div style="background:#fff;border:0.5px solid #e5e5e7;border-radius:12px;padding:12px 14px;margin-bottom:10px">
          <div style="display:flex;gap:6px;align-items:center;margin-bottom:4px;flex-wrap:wrap">
            <span style="font-size:12px;color:#86868b">{dt}</span>{score_badge}{topics}{region}
          </div>
          <div style="font-size:14px;line-height:1.7">{h["text"][:200]}{"…" if len(h["text"])>200 else ""}</div>
        </div>""")

    # Critic
    if HAS_CRITIC and hits:
        try:
            src = [h["text"] for h in hits[:5]]
            ans = f"关于「{query}」的微博回忆：{hits[0]['text'][:100]}"
            if len(hits) > 1: ans += f" 还有{hits[1]['text'][:60]}"
            report = generate_critic_report(ans, src, lambda t: model.encode([t], normalize_embeddings=True).tolist()[0], query)
            sm = re.search(r'忠实性评分[：:]\s*([\d.]+)', report)
            score_val = sm.group(1) if sm else "?"
            risk = "低"
            for rk, rc in [("高","#ff3b30"),("中","#ff9500"),("低","#34c759")]:
                if f"幻觉风险{''}   {rk}" in report or f"幻觉风险：{rk}" in report:
                    risk, risk_color = rk, rc; break
            else:
                risk_color = "#34c759"
            parts.append(f"""
            <div style="background:#FAECE7;border-radius:12px;padding:16px;margin-top:16px">
              <h3 style="margin:0 0 8px;font-size:15px;color:#993C1D">✅ Critic 审核</h3>
              <div style="display:flex;gap:16px;margin-bottom:10px;flex-wrap:wrap">
                <div style="background:#fff;border-radius:8px;padding:8px 12px;text-align:center;min-width:80px">
                  <div style="font-size:18px;font-weight:600;color:#1d1d1f">{score_val}</div>
                  <div style="font-size:11px;color:#86868b">忠实性</div>
                </div>
                <div style="background:#fff;border-radius:8px;padding:8px 12px;text-align:center;min-width:80px">
                  <div style="font-size:18px;font-weight:600;color:{risk_color}">{risk}</div>
                  <div style="font-size:11px;color:#86868b">幻觉风险</div>
                </div>
              </div>
              <div style="font-size:12px;color:#515154;line-height:1.6">{report[:300]}</div>
            </div>""")
        except:
            pass
    return "".join(parts)


# ─── UI ───────────────────────────────────────────────────────────────────────
def launch():
    global collection, model
    collection, model = load_engine()
    if collection is None:
        print("❌ 引擎加载失败，请先运行 run_pipeline.py")
        return
    with gr.Blocks(title="微博记忆库", theme=gr.themes.Soft()) as demo:
        gr.HTML(f"""
        <div style="text-align:center;padding:16px 0 4px">
          <h1 style="font-size:28px;margin:0">🧠 微博记忆库</h1>
          <p style="color:#86868b;margin:0;font-size:14px">输入问题 · AI 从你的微博中找记忆 · 支持「去年今日」「2025年5月」</p>
        </div>""")
        with gr.Row(equal_height=True):
            q = gr.Textbox(label="搜索", placeholder='例如：「找工作时的焦虑」「去年今日」「2025年5月的微博」', scale=4, lines=1)
            btn = gr.Button("🔍 搜索", variant="primary", scale=1, min_width=100)
        with gr.Accordion("⚙️ 筛选（可选）", open=False):
            with gr.Row():
                y_i = gr.Number(label="年份", value=None, minimum=2009, maximum=2030, precision=0)
                m_i = gr.Number(label="月份", value=None, minimum=1, maximum=12, precision=0)
            with gr.Row():
                t_i = gr.Textbox(label="话题", placeholder="如：香港、生活")
                k_i = gr.Slider(label="结果数", minimum=3, maximum=20, value=8, step=1)
        out = gr.HTML("<div style='text-align:center;padding:40px;color:#86868b'>输入问题开始搜索</div>")
        gr.Markdown(f"✅ 向量库 {collection.count()} 条 | bge-m3 | Critic {'✅' if HAS_CRITIC else '❌'}")

        fn = lambda *a: search_weibo(*a)
        btn.click(fn=fn, inputs=[q, gr.State(True), y_i, m_i, t_i, k_i], outputs=[out])
        q.submit(fn=fn, inputs=[q, gr.State(True), y_i, m_i, t_i, k_i], outputs=[out])
        gr.HTML("<footer style='text-align:center;padding:16px;font-size:11px;color:#c7c7cc'>本地离线 · ChromaDB + BAAI/bge-m3 · MIT</footer>")

    import webbrowser
    try: webbrowser.open("http://localhost:7860")
    except: pass
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, quiet=True)

if __name__ == "__main__":
    launch()

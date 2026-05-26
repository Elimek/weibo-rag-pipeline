"""
Weibo RAG — Gradio Web 界面
============================
集成语义搜索、时间过滤（去年今日）、Critic 审核。
输出结构化的「记忆报告」。

使用：
  python gradio_app.py
  浏览器自动打开 http://localhost:7860
"""

import sys
import json
import re
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

# ─── 尝试加载新模块 ───────────────────────────────────────────────────────────
try:
    from time_filter import parse_date_query, query_same_day_across_years, has_time_intent
    HAS_TIME_FILTER = True
except ImportError:
    HAS_TIME_FILTER = False
    print("[info] time_filter.py 未加载，时间过滤功能受限")

try:
    from critic import generate_critic_report, quick_verify
    HAS_CRITIC = True
except ImportError:
    HAS_CRITIC = False
    print("[info] critic.py 未加载，Critic 审核功能受限")

# ─── 路径 ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DB_DIR = BASE / "db" / "chroma"
COLLECTION_NAME = "weibo_posts"

# ─── 全局引擎 ────────────────────────────────────────────────────────────────
collection = None
model = None


# ─── 引擎加载 ─────────────────────────────────────────────────────────────────
def load_engine():
    """加载向量库和 embedding 模型"""
    print("[engine] 加载 ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        print(f"  ✅ 向量库：{count} 条")
    except Exception as e:
        print(f"  ❌ 加载失败：{e}")
        return None, None

    print("[engine] 加载 BAAI/bge-m3 模型（首次较慢）...")
    try:
        model = SentenceTransformer("BAAI/bge-m3")
        print("  ✅ 模型就绪")
    except Exception as e:
        print(f"  ❌ 模型加载失败：{e}")
        return None, None

    return collection, model


def embed_fn_wrapper(model, text: str):
    """embedding 函数封装"""
    return model.encode([text], normalize_embeddings=True).tolist()[0]


# ─── 搜索逻辑 ─────────────────────────────────────────────────────────────────
def build_where(year=None, month=None, day=None, hour=None, topic=None):
    """构建 ChromaDB 过滤条件"""
    conditions = []
    if year:
        conditions.append({"year": {"$eq": int(year)}})
    if month:
        conditions.append({"month": {"$eq": int(month)}})
    if day:
        conditions.append({"day": {"$eq": int(day)}})
    if hour is not None:
        conditions.append({"hour": {"$eq": int(hour)}})
    if topic:
        conditions.append({"topics": {"$contains": topic}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def semantic_search(collection, model, query, where=None, top_k=8):
    """向量语义搜索"""
    q_vec = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=q_vec,
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for i in range(len(results["documents"][0])):
        hits.append({
            "text": results["documents"][0][i],
            "meta": results["metadatas"][0][i],
            "score": round(1 - results["distances"][0][i], 4),
        })
    return hits


def get_by_date(collection, year, month, day, limit=50):
    """按精确日期查询"""
    where = build_where(year=year, month=month, day=day)
    if not where:
        return []
    results = collection.get(where=where, limit=limit, include=["documents", "metadatas"])
    hits = []
    for i in range(len(results["documents"])):
        hits.append({
            "text": results["documents"][i],
            "meta": results["metadatas"][i],
            "score": None,
        })
    hits.sort(key=lambda h: h["meta"].get("created_at", ""))
    return hits


# ─── 报告生成 ─────────────────────────────────────────────────────────────────
def build_report(query, hits, collection, model, time_info=None):
    """构建结构化记忆报告 HTML"""
    now = datetime.now()
    parts = []

    # ── 头部 ──
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

    # ── 去年今日区块 ──
    if time_info and time_info.get("type") in ("same_day_last_year", "same_day_n_years_ago"):
        year_label = time_info.get("description", "去年今日")
        parts.append(f"""
        <div style="background:#E1F5EE;border-radius:12px;padding:16px;margin-bottom:16px">
          <h3 style="margin:0 0 8px;font-size:15px;color:#0F6E56">📅 {year_label}</h3>
          <table style="width:100%;border-collapse:collapse;font-size:13px">
        """)
        # Group hits by year
        by_year = {}
        for h in hits:
            y = h["meta"].get("year")
            if y:
                by_year.setdefault(y, []).append(h)
        for y in sorted(by_year.keys(), reverse=True):
            entries = by_year[y]
            texts = "<br>".join(
                f'<span style="color:#515154">· {e["text"][:80]}</span>'
                for e in entries[:3]
            )
            parts.append(f"""
            <tr>
              <td style="padding:6px 8px;font-weight:500;color:#0F6E56;width:60px;vertical-align:top">{y}年</td>
              <td style="padding:6px 8px">{texts}</td>
            </tr>
            """)
        parts.append("</table></div>")

    # ── 搜索结果 ──
    parts.append('<h3 style="font-size:15px;margin:0 0 10px">🔍 相关微博</h3>')
    for i, h in enumerate(hits):
        meta = h["meta"]
        dt = meta.get("datetime_str") or meta.get("date_str") or ""
        score_badge = ""
        if h["score"]:
            color = "#34c759" if h["score"] > 0.7 else "#ff9500" if h["score"] > 0.5 else "#ff3b30"
            score_badge = f'<span style="font-size:11px;color:{color}">相似度 {h["score"]:.2f}</span>'
        topics_list = meta.get("topics", "").split("|") if meta.get("topics") else []
        topic_tags = "".join(
            f'<span style="display:inline-block;background:#f0edff;color:#5e3bd1;font-size:11px;padding:1px 6px;border-radius:4px;margin:0 2px">#{t}#</span>'
            for t in topics_list if t
        )
        region = meta.get("region", "")
        region_tag = f'<span style="display:inline-block;background:#f5f5f7;font-size:11px;padding:1px 6px;border-radius:4px;margin:0 2px">📍{region}</span>' if region else ""

        parts.append(f"""
        <div style="background:#fff;border:0.5px solid #e5e5e7;border-radius:12px;padding:12px 14px;margin-bottom:10px">
          <div style="display:flex;gap:6px;align-items:center;margin-bottom:4px;flex-wrap:wrap">
            <span style="font-size:12px;color:#86868b">{dt}</span>
            {score_badge}
            {topic_tags}
            {region_tag}
          </div>
          <div style="font-size:14px;line-height:1.7">{h["text"][:200]}{'…' if len(h['text']) > 200 else ''}</div>
        </div>
        """)

    # ── Critic 审核区块 ──
    if HAS_CRITIC and hits:
        try:
            source_texts = [h["text"] for h in hits[:5]]
            # Generate a simple answer summary from the top result
            top_texts = "\n".join(f"- {h['text'][:80]}" for h in hits[:3])
            answer_summary = f"关于「{query}」的微博回忆：{hits[0]['text'][:100]}"
            if len(hits) > 1:
                answer_summary += f" 还有{hits[1]['text'][:60]}"

            embed_fn = lambda t: embed_fn_wrapper(model, t)
            report = generate_critic_report(answer_summary, source_texts, embed_fn, query)

            # Parse score from report for badge
            score_match = re.search(r'忠实性评分[：:]\s*([\d.]+)', report)
            score_val = score_match.group(1) if score_match else "?"
            risk = "低" if "幻觉风险" in report and "低" in report.split("幻觉风险")[-1][:5] else "中" if "幻觉风险" in report and "中" in report.split("幻觉风险")[-1][:5] else "高"
            risk_color = "#34c759" if risk == "低" else "#ff9500" if risk == "中" else "#ff3b30"

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
              <div style="font-size:12px;color:#515154;line-height:1.6">
                {report.replace('##', '<br>').replace('**', '').replace('###', '<br><b>').replace('\n\n', '<br>')[:300]}
              </div>
            </div>
            """)
        except Exception as e:
            pass  # critic failed, skip

    return "".join(parts)


# ─── Gradio 接口 ──────────────────────────────────────────────────────────────
def search_weibo(query, use_today, year_filter, month_filter, topic_filter, top_k):
    """Gradio 搜索函数"""
    global collection, model
    if collection is None or model is None:
        collection, model = load_engine()
        if collection is None:
            return "<div style='text-align:center;padding:40px;color:#ff3b30'>引擎加载失败，请先运行 run_pipeline.py</div>"

    if not query and not year_filter and not month_filter and not topic_filter:
        return "<div style='text-align:center;padding:40px;color:#86868b'>输入问题开始搜索</div>"

    # 解析时间意图
    time_info = None
    if HAS_TIME_FILTER and query:
        time_info = parse_date_query(query)

    # 构建过滤条件
    where = build_where(
        year=year_filter if year_filter else None,
        month=month_filter if month_filter else None,
        topic=topic_filter if topic_filter else None,
    )

    # 特殊处理：去年今日
    if time_info and time_info.get("type") in ("same_day_last_year", "same_day_n_years_ago"):
        today = date.today()
        target_year = today.year - time_info.get("years", 1)
        target_month = today.month
        target_day = today.day
        hits = get_by_date(collection, target_year, target_month, target_day)

        # Also get current year for comparison
        current_hits = get_by_date(collection, today.year, today.month, today.day)
        all_hits = current_hits + hits
        all_hits.sort(key=lambda h: h["meta"].get("created_at", ""), reverse=True)

        if query and len(query) > 10:
            hits2 = semantic_search(collection, model, query, where=build_where(
                year=target_year, month=target_month, day=target_day
            ), top_k=top_k)
            all_hits = hits2 + [h for h in all_hits if h not in hits2]
            all_hits = all_hits[:top_k * 2]

        # Also get all same-day across years
        by_year_hits = query_same_day_across_years(collection, target_month, target_day, top_k_per_year=5)
        for y, entries in by_year_hits.items():
            for e in entries:
                if e not in all_hits:
                    all_hits.append(e)
        all_hits = all_hits[:top_k * 2]

        return build_report(query, all_hits, collection, model, time_info)

    # 普通语义搜索
    if query:
        hits = semantic_search(collection, model, query, where=where, top_k=top_k)
    else:
        hits = []
        if where:
            results = collection.get(where=where, limit=top_k, include=["documents", "metadatas"])
            for i in range(len(results["documents"])):
                hits.append({
                    "text": results["documents"][i],
                    "meta": results["metadatas"][i],
                    "score": None,
                })
            hits.sort(key=lambda h: h["meta"].get("created_at", ""), reverse=True)

    return build_report(query, hits, collection, model)


# ─── 启动 ─────────────────────────────────────────────────────────────────────
def launch():
    global collection, model
    collection, model = load_engine()
    if collection is None:
        print("\n❌ 引擎加载失败，请确保已运行 agent_3_embed.py 完成入库")
        return

    with gr.Blocks(
        title="微博记忆库",
        theme=gr.themes.Soft(),
        css="""
        .app-header { text-align:center; padding:20px 0 8px }
        .app-header h1 { font-size:28px; margin:0 }
        .app-header p { color:#86868b; margin:0; font-size:14px }
        footer { text-align:center; padding:20px; font-size:11px; color:#c7c7cc }
        """
    ) as demo:
        gr.HTML("""
        <div class="app-header">
          <h1>🧠 微博记忆库</h1>
          <p>输入任何问题，AI 从你的微博中找出相关记忆</p>
        </div>
        """)

        with gr.Row(equal_height=True):
            query_input = gr.Textbox(
                label="搜索微博",
                placeholder='例如：「找工作时的焦虑」「去年今日」「2024年夏天的心情」',
                scale=4,
                lines=1,
            )
            search_btn = gr.Button("🔍 搜索", variant="primary", scale=1, min_width=100)

        with gr.Accordion("⚙️ 筛选条件（可选）", open=False):
            with gr.Row():
                year_input = gr.Number(label="年份", value=None, minimum=2009, maximum=2030, precision=0)
                month_input = gr.Number(label="月份", value=None, minimum=1, maximum=12, precision=0)
            with gr.Row():
                topic_input = gr.Textbox(label="话题（如：香港、生活、求职）", placeholder="留空不限")
                top_k_input = gr.Slider(label="结果数量", minimum=3, maximum=20, value=8, step=1)

        output_html = gr.HTML(
            value="<div style='text-align:center;padding:40px;color:#86868b;font-size:14px'>输入问题开始搜索</div>"
        )

        # 状态提示
        status = gr.Markdown(f"✅ 向量库已加载：{collection.count()} 条微博 | Embedding: BAAI/bge-m3 | Critic: {'✅' if HAS_CRITIC else '❌未加载'}")

        # 事件绑定
        search_btn.click(
            fn=search_weibo,
            inputs=[query_input, gr.State(True), year_input, month_input, topic_input, top_k_input],
            outputs=[output_html],
        )
        query_input.submit(
            fn=search_weibo,
            inputs=[query_input, gr.State(True), year_input, month_input, topic_input, top_k_input],
            outputs=[output_html],
        )

        gr.HTML("<footer>所有数据本地存储 · ChromaDB + BAAI/bge-m3 · MIT License</footer>")

    print("\n🌐 浏览器自动打开中...")
    import webbrowser
    try:
        webbrowser.open("http://localhost:7860")
    except Exception:
        pass

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        quiet=True,
    )


if __name__ == "__main__":
    launch()

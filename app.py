"""
Weibo RAG — Web UI
==================
零依赖 Web 界面。运行后浏览器自动打开，全程图形化操作。

使用：
  python app.py

然后浏览器访问 http://localhost:8787
"""

import http.server
import json
import urllib.parse
import webbrowser
import os
import sys
from pathlib import Path

# ── 后端引擎 ──────────────────────────────────────────────────────────────────
DB_DIR = Path(__file__).parent / "db" / "chroma"
COLLECTION_NAME = "weibo_posts"

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False


def load_engine():
    """加载向量库和模型"""
    if not HAS_CHROMA:
        return None, "缺少依赖：pip install chromadb"
    if not DB_DIR.exists():
        return None, f"向量库不存在：{DB_DIR}\n请先运行 python run_pipeline.py"

    try:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        return None, f"加载向量库失败：{e}"

    embed_fn = None
    if HAS_MODEL:
        try:
            model = SentenceTransformer("BAAI/bge-m3")
            embed_fn = lambda text: model.encode([text], normalize_embeddings=True).tolist()[0]
        except Exception:
            pass

    return collection, embed_fn


def search(collection, embed_fn, query, year=None, month=None, day=None,
           hour=None, topic=None, top_k=10):
    """执行搜索"""
    # 构建过滤条件
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

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    # 语义搜索
    if embed_fn and query:
        q_vec = embed_fn(query)
        results = collection.query(
            query_embeddings=[q_vec],
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

    # 纯时间过滤（无 query）
    elif where:
        results = collection.get(where=where, limit=top_k, include=["documents", "metadatas"])
        hits = []
        for i in range(len(results["documents"])):
            hits.append({
                "text": results["documents"][i],
                "meta": results["metadatas"][i],
                "score": None,
            })
        hits.sort(key=lambda h: h["meta"].get("created_at", ""))
        return hits

    return []


# ── Web 服务器 ────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>微博记忆库</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f0f0f2;color:#1d1d1f;min-height:100vh}
.header{background:#fff;border-bottom:1px solid #e5e5e7;padding:14px 20px;position:sticky;top:0;z-index:10}
.header h1{font-size:18px;font-weight:600}
.header span{font-size:13px;color:#86868b;font-weight:400}
.container{max-width:720px;margin:0 auto;padding:20px 16px}

/* 搜索区 */
.search-box{background:#fff;border-radius:14px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.06);margin-bottom:16px}
.search-row{display:flex;gap:8px;margin-bottom:10px}
.search-row input{flex:1;padding:10px 14px;border:0.5px solid #d2d2d7;border-radius:10px;font-size:15px;outline:none;transition:border .2s}
.search-row input:focus{border-color:#007aff}
.search-row button{padding:10px 20px;background:#007aff;color:#fff;border:none;border-radius:10px;font-size:15px;cursor:pointer;transition:opacity .2s;white-space:nowrap}
.search-row button:hover{opacity:.85}
.search-row button:disabled{background:#ccc;cursor:not-allowed}

/* 过滤栏 */
.filters{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:4px}
.filter-tag{display:inline-flex;align-items:center;gap:4px;background:#f5f5f7;border:0.5px solid #e5e5e7;border-radius:8px;padding:4px 10px;font-size:12px;color:#515154;cursor:pointer;transition:all .2s}
.filter-tag.active{background:#e6f0ff;border-color:#007aff;color:#007aff}
.filter-tag input{display:none}
.filter-date{font-size:12px;padding:4px 8px;border:0.5px solid #e5e5e7;border-radius:8px;outline:none;color:#515154;background:#f5f5f7}
.filter-date:focus{border-color:#007aff}

/* 结果列表 */
.result-card{background:#fff;border-radius:12px;padding:14px 16px;margin-bottom:10px;box-shadow:0 1px 2px rgba(0,0,0,0.04);transition:transform .15s}
.result-card:hover{transform:translateY(-1px)}
.result-meta{display:flex;gap:8px;align-items:center;margin-bottom:6px;flex-wrap:wrap}
.result-time{font-size:12px;color:#86868b}
.result-tag{font-size:11px;background:#f0edff;color:#5e3bd1;padding:1px 6px;border-radius:4px}
.result-score{font-size:11px;color:#86868b}
.result-text{font-size:14px;line-height:1.7;word-break:break-word}
.result-text .highlight{background:#fff3b0;padding:0 2px}

/* 统计 */
.stats{font-size:12px;color:#86868b;text-align:center;padding:8px 0}
.loading{text-align:center;padding:20px;color:#86868b}
.empty{text-align:center;padding:40px;color:#86868b;font-size:14px;line-height:1.8}

/* 引导（首次使用无向量库时） */
.guide{background:#fff;border-radius:14px;padding:24px;box-shadow:0 1px 3px rgba(0,0,0,0.06);margin-bottom:16px}
.guide h2{font-size:16px;font-weight:600;margin-bottom:12px}
.guide ol{padding-left:18px;font-size:14px;line-height:2;color:#515154}
.guide code{background:#f5f5f7;padding:1px 6px;border-radius:4px;font-size:12px}
footer{text-align:center;padding:24px;font-size:11px;color:#c7c7cc}
</style>
</head>
<body>

<div class="header">
  <h1>🧠 微博记忆库 <span>本地离线 · 语义搜索</span></h1>
</div>

<div class="container">
  <div class="search-box">
    <div class="search-row">
      <input id="query" type="text" placeholder="搜微博内容… 比如「找工作时的焦虑」" onkeydown="if(event.key==='Enter')search()">
      <button id="searchBtn" onclick="search()">搜索</button>
    </div>
    <div class="filters">
      <input type="date" id="filterDate" class="filter-date" title="锁定到某天">
      <input type="number" id="filterHour" class="filter-date" placeholder="小时" min="0" max="23" style="width:60px" title="小时 (0-23)">
      <input type="number" id="filterYear" class="filter-date" placeholder="年" min="2009" style="width:70px" title="年份">
      <input type="number" id="filterMonth" class="filter-date" placeholder="月" min="1" max="12" style="width:60px" title="月份">
      <span class="filter-tag" onclick="this.classList.toggle('active');search()">🏷️ <input type="text" id="filterTopic" placeholder="话题" style="border:none;background:transparent;width:60px;font-size:12px;outline:none" oninput="debounceSearch()"></span>
      <span class="filter-tag" onclick="clearFilters()" style="color:#86868b">✕ 清除</span>
    </div>
  </div>

  <div id="results"></div>
  <div id="stats" class="stats"></div>
</div>

<footer>所有数据本地存储 · ChromaDB + BAAI/bge-m3</footer>

<script>
let debounceTimer;

function debounceSearch(){
  clearTimeout(debounceTimer);
  debounceTimer=setTimeout(search,500);
}

function getFilters(){
  const d=document.getElementById('filterDate').value;
  const h=document.getElementById('filterHour').value;
  const y=document.getElementById('filterYear').value;
  const m=document.getElementById('filterMonth').value;
  const t=document.getElementById('filterTopic').value;
  const filters={};
  if(d) filters.date=d;
  if(h) filters.hour=h;
  if(y) filters.year=y;
  if(m) filters.month=m;
  if(t) filters.topic=t;
  return filters;
}

function clearFilters(){
  document.getElementById('filterDate').value='';
  document.getElementById('filterHour').value='';
  document.getElementById('filterYear').value='';
  document.getElementById('filterMonth').value='';
  document.getElementById('filterTopic').value='';
  search();
}

function search(){
  const q=document.getElementById('query').value.trim();
  const filters=getFilters();
  const btn=document.getElementById('searchBtn');
  btn.disabled=true;
  btn.textContent='搜索中…';
  document.getElementById('results').innerHTML='<div class="loading">⏳ 搜索中…</div>';
  document.getElementById('stats').textContent='';

  fetch('/api/search',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({query:q,...filters})
  }).then(r=>r.json()).then(d=>{
    renderResults(d);
  }).catch(e=>{
    document.getElementById('results').innerHTML='<div class="empty">❌ 搜索出错：'+e.message+'</div>';
  }).finally(()=>{
    btn.disabled=false;
    btn.textContent='搜索';
  });
}

function renderResults(data){
  const el=document.getElementById('results');
  const st=document.getElementById('stats');
  if(data.error){
    el.innerHTML='<div class="guide"><h2>⚙️ 系统就绪，但还差一步</h2><p style="color:#515154;line-height:1.8">'+data.error.replace(/\n/g,'<br>')+'</p></div>';
    return;
  }
  if(data.hits.length===0){
    el.innerHTML='<div class="empty">没找到相关微博<br>试试其他关键词，或清除过滤条件</div>';
    st.textContent='';
    return;
  }
  const hits=data.hits;
  st.textContent='找到 '+hits.length+' 条结果';
  let html='';
  hits.forEach((h,i)=>{
    const meta=h.meta;
    const dt=meta.datetime_str||meta.date_str||meta.created_at||'';
    const topics=meta.topics?meta.topics.split('|').filter(Boolean):[];
    const score=h.score?'相似度 '+h.score:'';
    html+='<div class="result-card">';
    html+='<div class="result-meta">';
    html+='<span class="result-time">'+dt+'</span>';
    if(score) html+='<span class="result-score">'+score+'</span>';
    topics.forEach(t=>{html+='<span class="result-tag">#'+t+'#</span>'});
    if(meta.region) html+='<span class="result-tag">📍'+meta.region+'</span>';
    html+='</div>';
    html+='<div class="result-text">'+escapeHtml(h.text)+'</div>';
    html+='</div>';
  });
  el.innerHTML=html;
}

function escapeHtml(t){
  const d=document.createElement('div');
  d.textContent=t;
  return d.innerHTML;
}
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/search":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            params = json.loads(body)

            query = params.get("query", "")
            year = params.get("year") or None
            month = params.get("month") or None
            day = params.get("day") or None
            hour = params.get("hour") if params.get("hour") != "" else None
            topic = params.get("topic") or None

            # 如果填了 date 字段，解析年/月/日
            if params.get("date"):
                parts = params["date"].split("-")
                if len(parts) >= 1:
                    year = year or parts[0]
                if len(parts) >= 2:
                    month = month or parts[1]
                if len(parts) >= 3:
                    day = day or parts[2]

            try:
                hits = search(collection, embed_fn, query, year, month, day, hour, topic)
                resp = {"hits": hits}
            except Exception as e:
                resp = {"error": str(e)}

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(resp, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # 静默日志


def main():
    global collection, embed_fn

    PORT = 8787

    print("🧠 微博记忆库 — 启动中...")
    collection, err = load_engine()

    if err:
        print(f"  ⚠️  {err}")
        embed_fn = None
        collection = None
    else:
        count = collection.count()
        print(f"  ✅ 向量库已加载：{count} 条记录")
        print(f"  ✅ Embedding 模型就绪")

    server = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    url = f"http://localhost:{PORT}"

    print(f"\n  🌐 浏览器打开：{url}")
    print(f"  📋 按 Ctrl+C 停止服务\n")

    # 自动打开浏览器
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")


if __name__ == "__main__":
    main()

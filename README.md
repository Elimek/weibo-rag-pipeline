# Weibo RAG Pipeline

将微博历史构建为个人 RAG 知识库的完整 pipeline。

支持全量抓取、自动清洗分块、向量化存储、语义搜索 + 精确时间定位查询。

## 架构

```
Agent 1 (抓取) → Agent 2 (清洗) → Agent 3 (入库) → Agent 4 (查询)
   weibo.com      文本清洗 +        ChromaDB         交互终端
   桌面 API        分块合并          bge-m3          + 命令行
```

**数据流**：`raw/` → `processed/` → `db/chroma/` → 用户查询

## 快速开始

### 1. 安装

```bash
pip install -r requirements.txt
# sentence-transformers > 3.0 可能有 torchcodec 依赖问题，降级：
# pip install "sentence-transformers<3.0"
```

### 2. 获取 Cookie

```bash
Brave / Chrome → 打开 https://weibo.com → 登录
F12 → Application → Cookies → weibo.com
复制 SUB 和 SUBP 的值
```

填入 `agent_1_scrape.py` 顶部的 `MY_COOKIES` 配置区。

### 3. 一键运行

```bash
python run_pipeline.py
```

或分步运行：

```bash
python agent_1_scrape.py          # 全量抓取博客（每页20条，支持深分页）
python agent_2_clean.py           # 清洗 + 5分钟窗口合并
python agent_3_embed.py --mode local  # 本地 bge-m3 入库
```

### 4. 查询

```bash
# 交互模式（推荐）
python agent_4_query.py -i

# 语义搜索
python agent_4_query.py --query "找工作时的焦虑"

# 精确时间查询
python agent_4_query.py --date 2024-06-15
python agent_4_query.py --date 2024-06-15 --hour 22
python agent_4_query.py --year 2024 --month 3

# 话题过滤
python agent_4_query.py --topic "生活"
```

交互模式下支持组合过滤：

```
🔍 > :date 2024-06-15       锁定到某天
🔍 > :hour 22               锁定到小时
🔍 > :topic 生活             话题过滤
🔍 > :clear                 清除筛选
🔍 > 心情                   在筛选范围内语义搜索
🔍 > :quit                  退出
```

## Embedding 模式

| 模式 | 命令 | 速度 | 费用 | 中文质量 |
|------|------|------|------|---------|
| local | `--mode local` | 慢 (GTX1050 ~30min/3k条) | 免费 | 最好 |
| openai | `--mode openai` | 快 | ~$0.5/万条 | 很好 |
| ollama | `--mode ollama` | 中 | 免费 | 好 |

## 项目结构

```
weibo-rag/
├── agent_1_scrape.py      # 数据抓取（weibo.com 桌面 API）
├── agent_2_clean.py       # 清洗与分块
├── agent_3_embed.py       # 向量嵌入与 ChromaDB 入库
├── agent_4_query.py       # 交互式查询终端
├── run_pipeline.py        # 一键编排
├── requirements.txt
├── 查询.bat               # Windows 一键启动
├── 查询手册.html           # 可视化查询指南
└── .gitignore
```

## 数据隐私

- 所有数据本地存储，不上传任何服务器
- `raw/`、`processed/`、`db/` 目录已加入 `.gitignore`
- Cookie 需用户手动填入，不随代码分发

## 要求

- Python >= 3.10
- 依赖见 `requirements.txt`
- 本地模式需 4GB+ VRAM（GTX 1050 可跑）或 CPU 模式（慢约 5 倍）
- Windows / macOS / Linux 均可

## 许可

MIT

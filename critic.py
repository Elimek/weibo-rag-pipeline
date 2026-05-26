"""
Critic — 忠实性与覆盖率审核模块
=================================
在 RAG 系统生成回答后，检查回答是否忠实于原始微博内容。

用法：
    from critic import compute_faithfulness, compute_coverage, generate_critic_report

    embed_fn = get_embedder()           # 从 agent_4 获取
    report = generate_critic_report(answer, source_texts, embed_fn, query="我的问题")
"""

import re
import numpy as np
from typing import Callable


# ─── 工具函数 ─────────────────────────────────────────────────────────────────


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    try:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except Exception:
        return 0.0


def _split_sentences(text: str) -> list[str]:
    """将文本切分为句子，支持中英文句号、感叹号、问号"""
    if not text:
        return []
    parts = re.split(r'[。！？\.!\?]+', text)
    return [s.strip() for s in parts if s.strip()]


# 简易中文停用词表
_STOP_WORDS = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
    '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
    '会', '着', '没有', '看', '好', '自己', '这', '他', '她', '它',
    '们', '那', '啊', '吧', '吗', '呢', '嗯', '哦', '啦', '呀',
    '什么', '怎么', '哪', '为什么', '因为', '所以', '但是', '而且',
    '虽然', '如果', '然后', '可以', '已经', '还是', '只是', '不过',
    '这个', '那个', '这些', '那些', '非常', '比较', '一直', '真的',
    '不是', '就是', '但是', '还有', '以及', '或者', '关于',
    '又', '还', '更', '最', '太', '够', '像', '比', '跟',
    '被', '把', '从', '对', '为', '以', '与', '将', '让',
    '来', '去', '进', '出', '过', '起', '开', '回', '到', '往',
    'weibo', '微博', '转发', '评论',
}


def _extract_keywords(text: str) -> set[str]:
    """简易关键词提取：按标点和空白分词，过滤单字和停用词"""
    tokens = re.split(
        # 标点及空白分词；连字符放末尾避免被解释为范围
        r'[\s,，。、！？：；""''（）()【】《》<>/\\|·…—+=＝~～@#$^&*-]+',
        text,
    )
    keywords = set()
    for token in tokens:
        token = token.strip()
        if len(token) >= 2 and token not in _STOP_WORDS and not token.isdigit():
            keywords.add(token)
    return keywords


# ─── 核心函数 ─────────────────────────────────────────────────────────────────


def compute_faithfulness(
    answer: str,
    source_texts: list[str],
    embed_fn: Callable[[str], list[float]],
) -> dict:
    """
    计算回答对源文本的忠实性评分。

    流程：将回答拆分为句子 → 对每个句子和所有源文本做向量嵌入
          → 计算余弦相似度 → 判断忠实性。

    Args:
        answer: 生成的回答文本。
        source_texts: 源微博文本列表。
        embed_fn: 嵌入函数，接收 str，返回 list[float]。

    Returns:
        dict: {
            "overall_score": float,         # 0~1，忠实句子占比
            "hallucination_risk": str,      # "low"|"medium"|"high"
            "sentence_scores": list[dict],  # 各句明细
            "unverified_claims": list[str], # 相似度 < 0.40 的句子
            "warnings": list[str],
        }
    """
    result: dict = {
        "overall_score": 0.0,
        "hallucination_risk": "high",
        "sentence_scores": [],
        "unverified_claims": [],
        "warnings": [],
    }

    if not answer or not source_texts:
        result["warnings"].append("回答或源文本为空，无法审核")
        return result

    try:
        sentences = _split_sentences(answer)
        if not sentences:
            result["warnings"].append("回答中未识别出有效句子")
            return result

        # 源文本向量化
        source_vecs = []
        for src in source_texts:
            if src.strip():
                source_vecs.append(np.array(embed_fn(src), dtype=np.float64))

        if not source_vecs:
            result["warnings"].append("所有源文本嵌入失败")
            return result

        source_vecs = np.array(source_vecs)  # (n_sources, dim)

        # 逐句审核
        faithful_count = 0
        unverified = []

        for sent in sentences:
            try:
                sent_vec = np.array(embed_fn(sent), dtype=np.float64)
            except Exception:
                result["sentence_scores"].append({
                    "sentence": sent,
                    "max_similarity": 0.0,
                    "source_idx": -1,
                    "faithful": False,
                })
                unverified.append(sent)
                continue

            similarities = [_cosine_similarity(sent_vec, sv) for sv in source_vecs]
            max_sim = float(max(similarities))
            max_idx = int(np.argmax(similarities))

            is_faithful = max_sim >= 0.55
            result["sentence_scores"].append({
                "sentence": sent,
                "max_similarity": round(max_sim, 4),
                "source_idx": max_idx,
                "faithful": is_faithful,
            })

            if is_faithful:
                faithful_count += 1
            if max_sim < 0.40:
                unverified.append(sent)

        total = len(sentences)
        result["overall_score"] = round(faithful_count / total, 4)

        ratio = faithful_count / total
        if ratio > 0.80:
            result["hallucination_risk"] = "low"
        elif ratio > 0.60:
            result["hallucination_risk"] = "medium"
            result["warnings"].append(
                f"部分句子缺乏源文本支持 ({total - faithful_count}/{total})"
            )
        else:
            result["hallucination_risk"] = "high"
            result["warnings"].append(
                f"大量句子缺乏源文本支持 ({total - faithful_count}/{total})"
            )

        result["unverified_claims"] = unverified

    except Exception as e:
        result["warnings"].append(f"忠实性审核异常：{e}")

    return result


def compute_coverage(
    source_texts: list[str],
    answer: str,
    embed_fn: Callable[[str], list[float]],
) -> dict:
    """
    计算回答对源文本关键点的覆盖程度。

    使用关键词重叠率来估算覆盖率。embed_fn 参数保留以供后续语义扩展。

    Args:
        source_texts: 源微博文本列表。
        answer: 生成的回答文本。
        embed_fn: 嵌入函数（当前实现仅用于兼容接口，未使用语义）。

    Returns:
        dict: {
            "coverage_score": float,      # 0~1
            "covered_topics": list[str],  # 已覆盖的关键词
            "missed_topics": list[str],   # 遗漏的关键词
        }
    """
    result: dict = {
        "coverage_score": 0.0,
        "covered_topics": [],
        "missed_topics": [],
    }

    if not source_texts or not answer:
        return result

    try:
        source_keywords: set[str] = set()
        for src in source_texts:
            source_keywords.update(_extract_keywords(src))

        answer_keywords = _extract_keywords(answer)

        if not source_keywords:
            result["coverage_score"] = 1.0
            return result

        covered = source_keywords & answer_keywords
        missed = source_keywords - answer_keywords

        result["coverage_score"] = round(len(covered) / len(source_keywords), 4)
        result["covered_topics"] = sorted(covered)[:20]
        result["missed_topics"] = sorted(missed)[:20]

    except Exception:
        pass

    return result


def generate_critic_report(
    answer: str,
    source_texts: list[str],
    embed_fn: Callable[[str], list[float]],
    query: str = "",
) -> str:
    """
    结合忠实性与覆盖率，生成可读的 Markdown 审核报告。

    Args:
        answer: 生成的回答文本。
        source_texts: 源微博文本列表。
        embed_fn: 嵌入函数。
        query: 原始查询（可选，会显示在报告开头）。

    Returns:
        str: Markdown 格式的审核报告。
    """
    try:
        faith = compute_faithfulness(answer, source_texts, embed_fn)
        coverage = compute_coverage(source_texts, answer, embed_fn)
    except Exception as e:
        return f"## ❌ Critic 审核报告\n\n生成报告时出错：{e}"

    risk_cn = {"low": "低", "medium": "中", "high": "高"}

    lines = [
        "## ✅ Critic 审核报告",
        "",
    ]
    if query:
        lines.append(f"**查询**: {query}")
        lines.append("")

    lines += [
        f"**忠实性评分**: {faith['overall_score']:.2f}/1.0",
        f"**幻觉风险**: {risk_cn.get(faith['hallucination_risk'], '未知')}",
        f"**内容覆盖率**: {coverage['coverage_score']:.2f}/1.0",
        "",
        "### 各句审核",
    ]

    for item in faith["sentence_scores"]:
        sent = item["sentence"]
        sim = item["max_similarity"]
        if item["faithful"]:
            lines.append(f"- ✅ \"{sent}\" — 有原文支持 (相似度 {sim:.2f})")
        elif sim >= 0.40:
            lines.append(f"- ⚠️ \"{sent}\" — 部分偏离原文 (相似度 {sim:.2f})")
        else:
            lines.append(f"- ❌ \"{sent}\" — 无原文支持 (相似度 {sim:.2f})")

    lines.append("")

    if faith["unverified_claims"]:
        lines.append("### 未验证的陈述")
        for claim in faith["unverified_claims"]:
            lines.append(f"- \"{claim}\" — 微博中未找到相关记录")
        lines.append("")

    if coverage["missed_topics"]:
        lines.append("### 未覆盖的关键点")
        lines.append(f"- {'、'.join(coverage['missed_topics'][:10])}")
        lines.append("")

    lines.append("### 总结")
    risk = faith["hallucination_risk"]
    cov = coverage["coverage_score"]
    if risk == "low":
        if cov < 0.5:
            lines.append("回答忠实于原始微博，但内容覆盖率较低，可能遗漏了部分要点。")
        else:
            lines.append("回答基本忠实于原始微博，覆盖了主要关键点。")
    elif risk == "medium":
        lines.append("回答整体可信，但部分内容与原文存在偏差，建议核实。")
    else:
        lines.append("回答存在较大幻觉风险，建议重新生成或补充源信息。")

    return "\n".join(lines)


def quick_verify(
    query: str,
    answer: str,
    source_texts: list[str],
    embed_fn: Callable[[str], list[float]],
) -> str:
    """
    快速判断回答是否忠实，返回一行结论。

    Args:
        query: 原始查询（保留参数，当前未直接使用）。
        answer: 生成的回答。
        source_texts: 源微博文本列表。
        embed_fn: 嵌入函数。

    Returns:
        str: "✅ 回答忠实" / "⚠️ 部分偏离" / "❌ 存在幻觉"
    """
    try:
        faith = compute_faithfulness(answer, source_texts, embed_fn)
        risk = faith["hallucination_risk"]
        if risk == "low":
            return "✅ 回答忠实"
        elif risk == "medium":
            return "⚠️ 部分偏离"
        else:
            return "❌ 存在幻觉"
    except Exception:
        return "❌ 存在幻觉"

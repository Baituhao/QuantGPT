"""Compose final alpha-idea markdown from extracted body + image captions.

For each <source_id>:
  - read body.txt (cleaned article text with [[FIG:filename]] markers)
  - read captions.jsonl (VLM output for each image)
  - send to DeepSeek with a strict template prompt
  - LLM returns markdown that follows the agreed structure
  - copy 3-6 key images (formula/code/table/chart) to docs/knowledge/alpha-idea/images/<source_id>/
  - write docs/knowledge/alpha-idea/<source_id>-<slug>.md

Idempotent: skips a source_id if its md already exists (use --force to rewrite).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_ROOT = ROOT / ".tmp_alpha_idea_extracted"
OUT_DIR = ROOT / "docs" / "knowledge" / "alpha-idea"
IMG_OUT_ROOT = OUT_DIR / "images"


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


load_env(ROOT / ".env")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
# Use deepseek-chat by default — cheaper and faster than reasoner for this task.
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_COMPOSE_MODEL", "deepseek-chat")


SYSTEM_PROMPT = """你是 WorldQuant BRAIN 量化研究助理,精通因子设计、Fast Expression 算子语义。任务:把一篇博客原文(评论已混入)+ 截图 OCR 内容,整理成结构化的 alpha-idea 知识库文档。

你必须**只输出 markdown 全文**,从 frontmatter 第一行 `---` 开始,不要任何前后说明、不要 markdown 围栏包裹。

## 输出模板(严格遵守)

```
---
title: <从原文提炼的简洁标题,中文优先>
region: <CHN | USA | GLB | ASI | General>
delay: <0 或 1,从 source_id 推断;General 系列填 null>
source: <source_id>
source_paper: <若原文标注了论文/研报名/作者,填这里;没有就省略此字段>
tags: [<3-8 个英文标签,如 turnover, reversal, intraday, value, sentiment 等>]
operators: [<出现的 Fast Expression 算子,如 rank, ts_mean, group_neutralize 等>]
datasets: [<pv | fund | analyst | options | news | sentiment 等;不确定时省略>]
fields: [<出现的字段名,如 close, volume, cap, returns 等>]
---

## 核心思想

<3-5 句话讲清这篇文章的 alpha 是什么、解决什么问题、为何有效。**不堆术语**,直接说人话。>

## Alpha 公式

<把所有 Fast Expression 代码片段全部抽取出来,按文章演进顺序列出。每段:
1. 一行小标题(如「Step 1 — 原始版本」)
2. ```代码块``` 包裹的 Fast Expression(从 OCR 抽出的)
3. 一句关键指标(Sharpe / Returns / Robust Universe Sharpe 等,有就写)>

<若文章里有数学公式(论文截图),用 $$ ... $$ 写 LaTeX。>

<图片引用格式:`![描述](images/<source_id>/NN-name.png)`,只引用最关键的 1-4 张:
  - 论文公式/因子定义表
  - 关键 Fast Expression 截图(若 OCR 文本不够清晰)
  - 最终 PnL 曲线
  - 分布/对比图(如 power(rank,n) 曲线对比)
  路径用 source_id 子目录,文件名用 01-xxx.png 02-xxx.png 这种;NN 由 captioner 选定。>

## 构建逻辑

<3-6 条 bullet,讲清作者每一步为什么这么改进、决策依据、数据/逻辑链。>

## 经验提示

<这部分最重要:从原文+评论里提炼可复用的 know-how。包括但不限于:
- 单位陷阱 / 数据陷阱
- 中性化策略选择
- 解 Robust Universe / SC FAIL 的招式
- 跨 universe 适用性
- 评论区楼主自答的诊断方法
若没有,可省略此节。>

## 原文要点

<3-6 条 bullet,提炼论文/原文最核心的发现:
- 因子族构成
- 回归/IC 等学术验证结果
- 行业差异 / 市值差异 / 时间衰减等关键观察
- 数据集>
```

## 处理规则

- **去掉所有评论里的客套话**(感谢、求教、不会用、表情符号等)
- **保留楼主在评论区的自答**(诊断、修复、扩展实验) → 进「经验提示」节
- **保留高赞用户的实质性补充**(指出问题、贡献新发现) → 同上
- **数学公式优先用 LaTeX,代码用代码块**;图片只在文字无法表达时引用
- **OCR 文本可能有错别字**,凭你对 Fast Expression 算子的知识修正(如 group_neutalize → group_neutralize)
- **Settings 字段对比同代码不同表现** — 如果两张代码截图 Fast Expression 文本相同,但 key_metrics 里 Sharpe/Turnover 等不同,那一定是 BRAIN Settings 面板的 NEUTRALIZATION/DECAY/UNIVERSE 等参数不同。在每段代码块下方加一行 `Settings: NEUTRALIZATION=Industry, DECAY=5, ...` 把差异写出来,**绝不要写两段一模一样的代码**
- **若文章主要是文字总结(如 General 系列),没有具体 Alpha 公式**,Alpha 公式节可只列出公式族/算子框架,或者写「本文为综述,无具体表达式」
- **delay 从 source_id 拿:CHN-D0-* → 0(日内), CHN-D1-* → 1(日间), 没有 D{n} 的 → null**
- **简洁优先**:整篇控制在 100-300 行 markdown 内
"""


def call_deepseek(user_payload: str, retries: int = 2, timeout: int = 180) -> str:
    body = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ],
        "temperature": 0.2,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{DEEPSEEK_BASE}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_KEY}"},
    )
    last = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError) as e:
            last = e
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"deepseek call failed: {last}")


def select_key_images(captions: list[dict], max_n: int = 6) -> list[dict]:
    """Pick the most informative images: formulas/charts/tables, drop irrelevant/screenshot_misc."""
    priority = {"formula": 0, "code_screenshot": 1, "chart": 2, "table": 3, "text_block": 4, "screenshot_misc": 5, "irrelevant": 9, "error": 9}
    ranked = sorted(captions, key=lambda c: (priority.get(c.get("kind"), 6), c.get("filename", "")))
    pick: list[dict] = []
    seen_kinds: dict[str, int] = {}
    for c in ranked:
        k = c.get("kind", "")
        if k in ("irrelevant", "error"):
            continue
        # Cap each kind: at most 2 code_screenshots, 2 charts, 2 tables
        cap = {"code_screenshot": 2, "chart": 2, "table": 2, "screenshot_misc": 1, "text_block": 1, "formula": 3}.get(k, 1)
        if seen_kinds.get(k, 0) >= cap:
            continue
        pick.append(c)
        seen_kinds[k] = seen_kinds.get(k, 0) + 1
        if len(pick) >= max_n:
            break
    return pick


def slugify_title(s: str) -> str:
    s = re.sub(r"[<>:\"/\\|?*\[\]【】（）()]", "", s)
    s = re.sub(r"\s+", "-", s.strip())
    return s[:80]


def compose_one(source_id: str, work_dir: Path, force: bool = False) -> Path | None:
    body_file = work_dir / "body.txt"
    captions_file = work_dir / "captions.jsonl"
    meta_file = work_dir / "meta.json"
    if not (body_file.exists() and captions_file.exists() and meta_file.exists()):
        print(f"  [{source_id}] missing inputs, skip")
        return None

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    body = body_file.read_text(encoding="utf-8")
    captions = [json.loads(l) for l in captions_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Filter image references in body to only those whose kind != irrelevant/error
    keep_files = {c["filename"] for c in captions if c.get("kind") not in ("irrelevant", "error")}
    body_clean = re.sub(r"\[\[FIG:([^\]]+)\]\]", lambda m: f"[[FIG:{m.group(1)}]]" if m.group(1) in keep_files else "", body)

    # Pick key images and rename them
    picked = select_key_images(captions, max_n=6)
    img_subdir = IMG_OUT_ROOT / source_id
    img_subdir.mkdir(parents=True, exist_ok=True)
    rename_map: dict[str, str] = {}
    for i, c in enumerate(picked, 1):
        old = work_dir / "images" / c["filename"]
        if not old.exists():
            continue
        kind = c.get("kind", "img")
        new_name = f"{i:02d}-{kind}{old.suffix}"
        new_path = img_subdir / new_name
        if not new_path.exists():
            shutil.copy2(old, new_path)
        rename_map[c["filename"]] = new_name

    # Build user payload for the LLM
    payload = {
        "source_id": source_id,
        "region": meta.get("region"),
        "delay_token": meta.get("delay_token"),
        "delay": meta.get("delay"),
        "original_filename": meta.get("original_filename"),
        "body": body_clean,
        "image_renames": rename_map,    # original captioner filename → new path filename in images/<source_id>/
        "captions": [
            {
                "filename": c["filename"],
                "renamed_to": rename_map.get(c["filename"]),
                "kind": c.get("kind"),
                "summary": c.get("summary"),
                "text": c.get("text"),
                "key_metrics": c.get("key_metrics"),
            }
            for c in captions
            if c.get("kind") not in ("irrelevant", "error")
        ],
    }
    user_msg = (
        f"以下是 source_id={source_id} 的所有素材。\n"
        f"body 是去过滤的原文(含 [[FIG:原文件名]] 占位标记),\n"
        f"captions 是每张图的 VLM 识别结果(text 字段是 OCR 出的原文/代码,summary 是中文摘要)。\n"
        f"image_renames 告诉你保留下来的关键图,在最终 md 里要写成 `![alt](images/{source_id}/<renamed_to>)`,\n"
        f"未在 image_renames 里的图直接忽略。\n\n"
        f"---\nMATERIAL:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    # Output target file
    title_for_slug = meta.get("original_filename", source_id)
    # strip prefix like CHN-D1-1- and common suffixes
    title_for_slug = re.sub(r"^[A-Za-z]+(?:-D[0-9])?-\d+-", "", title_for_slug)
    title_for_slug = title_for_slug.replace(".mhtml", "").replace(" – WorldQuant BRAIN", "").replace(" - WorldQuant BRAIN", "")
    title_for_slug = title_for_slug.replace("【Alpha灵感】", "").replace("【alpha灵感】", "")
    title_for_slug = title_for_slug.replace("【Alpha Idea】", "").replace("[Alpha Idea]", "")
    title_for_slug = title_for_slug.replace("【General Tech】", "").replace("【Alpha 灵感】", "")
    title_for_slug = title_for_slug.strip(" -·")
    out_path = OUT_DIR / f"{source_id}-{slugify_title(title_for_slug)}.md"

    if out_path.exists() and not force:
        print(f"  [{source_id}] exists, skip ({out_path.name})")
        return out_path

    md = call_deepseek(user_msg)
    md = md.strip()
    # Strip stray code-fence wrapping if model added them
    if md.startswith("```"):
        md = re.sub(r"^```(?:markdown|md)?\s*\n", "", md)
        md = re.sub(r"\n```\s*$", "", md)

    out_path.write_text(md, encoding="utf-8")
    print(f"  [{source_id}] wrote {out_path.name}  ({len(md)} chars, {len(picked)} imgs)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=EXTRACTED_ROOT)
    ap.add_argument("--only", type=str, default=None, help="source_id substring filter")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    if not DEEPSEEK_KEY:
        print("[ERR] DEEPSEEK_API_KEY not set", file=sys.stderr); sys.exit(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_OUT_ROOT.mkdir(parents=True, exist_ok=True)

    dirs = sorted(d for d in args.root.iterdir() if d.is_dir())
    if args.only:
        dirs = [d for d in dirs if args.only in d.name]
    if args.limit:
        dirs = dirs[: args.limit]

    print(f"DeepSeek model: {DEEPSEEK_MODEL}  workers={args.workers}")
    print(f"Composing {len(dirs)} alpha-idea articles → {OUT_DIR}")

    if args.workers <= 1:
        for d in dirs:
            try:
                compose_one(d.name, d, force=args.force)
            except Exception as e:
                print(f"  [{d.name}] ERROR: {e}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(compose_one, d.name, d, args.force): d for d in dirs}
            for fut in as_completed(futs):
                d = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"  [{d.name}] ERROR: {e}")
    print("Done.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()

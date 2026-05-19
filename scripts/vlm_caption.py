"""Caption alpha-idea images via OpenAI-compatible VLM endpoint (Gemini-3.1-Pro).

Reads images from <extracted_root>/<source_id>/images/*, asks the VLM to:
  1) classify each image (formula | code_screenshot | table | chart | screenshot_misc | irrelevant)
  2) extract textual content / Fast Expression code if applicable
  3) summarize key data points if it's a chart/table

Caches per-image results to <source_id>/captions.jsonl (one line per image).
Skips already-captioned images on rerun.

Reads env from .env via simple parser (no external dep).

Usage:
  python scripts/vlm_caption.py --only CHN-D1-1
  python scripts/vlm_caption.py --root .tmp_alpha_idea_extracted --workers 4
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent.parent


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

API_KEY = os.environ.get("VLM_API_KEY", "")
BASE_URL = os.environ.get("VLM_BASE_URL", "").rstrip("/")
MODEL = os.environ.get("VLM_MODEL", "Gemini-3.1-Pro")

PROMPT = """你是 WorldQuant BRAIN Alpha 研究助理。我会给你一张文章里的截图,你必须**只输出一个 JSON 对象**(无 markdown 围栏、无前后文字)。

JSON schema:
{
  "kind": "formula | code_screenshot | table | chart | text_block | screenshot_misc | irrelevant",
  "text": "<原图里出现的所有文字/代码/数学公式;Fast Expression 代码用一行写,数学公式用 LaTeX>",
  "summary": "<一句中文概述截图传达的信息;若是图表说明轴/曲线/关键数值;若是 PnL/Sharpe 截图列出关键指标>",
  "key_metrics": {"<指标名>": "<值>"}
}

判定规则:
- "code_screenshot": Fast Expression / Python 代码,或 BRAIN 编辑器界面带表达式
- "formula": 论文公式或数学符号截图
- "table": 表格(含数值、行业列、列表等)
- "chart": PnL/收益曲线/分布图
- "text_block": 论文截图段落
- "screenshot_misc": BRAIN 设置面板、界面截图等
- "irrelevant": 头像、徽章、装饰图、空白
- text 字段务必包含原图里能识别的全部文字(中英数字符号);若 kind=irrelevant,text 留空字符串
- 输出必须是合法 JSON,字符串里的双引号要转义"""


def encode_image(p: Path) -> tuple[str, str]:
    data = p.read_bytes()
    mime = "image/png"
    suf = p.suffix.lower()
    if suf in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suf == ".webp":
        mime = "image/webp"
    elif suf == ".svg":
        mime = "image/svg+xml"
    elif suf == ".gif":
        mime = "image/gif"
    return mime, base64.b64encode(data).decode("ascii")


def call_vlm(img_path: Path, retries: int = 2, timeout: int = 90) -> dict:
    mime, b64 = encode_image(img_path)
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ],
        "temperature": 0.0,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    last_err = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
            content = resp_json["choices"][0]["message"]["content"].strip()
            # tolerate fenced output even though we asked for none
            if content.startswith("```"):
                content = content.strip("`")
                if content.lower().startswith("json"):
                    content = content[4:]
                content = content.strip("`").strip()
            return json.loads(content)
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"VLM call failed after retries: {last_err}")


def caption_one_dir(work_dir: Path, force: bool = False, workers: int = 1) -> int:
    img_dir = work_dir / "images"
    if not img_dir.exists():
        return 0
    cache_file = work_dir / "captions.jsonl"
    cached: dict[str, dict] = {}
    if cache_file.exists() and not force:
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                cached[rec["filename"]] = rec
            except Exception:
                pass

    images = sorted(img_dir.glob("*"))
    todo = [p for p in images if p.name not in cached]
    if not todo:
        print(f"  [{work_dir.name}] all {len(images)} captions cached")
        return 0

    print(f"  [{work_dir.name}] captioning {len(todo)} images (cached {len(cached)})", flush=True)

    def _do_one(p: Path) -> dict:
        try:
            result = call_vlm(p)
            return {"filename": p.name, **result}
        except Exception as e:
            return {"filename": p.name, "kind": "error", "text": "", "summary": str(e)[:200]}

    with cache_file.open("a", encoding="utf-8") as fout:
        if workers <= 1:
            for p in todo:
                rec = _do_one(p)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"    {p.name}: {rec.get('kind','?')}  {(rec.get('summary') or '')[:60]}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                fut_to_path = {ex.submit(_do_one, p): p for p in todo}
                for fut in as_completed(fut_to_path):
                    rec = fut.result()
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    print(f"    {rec['filename']}: {rec.get('kind','?')}  {(rec.get('summary') or '')[:60]}", flush=True)
    return len(todo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / ".tmp_alpha_idea_extracted")
    ap.add_argument("--only", type=str, default=None, help="source_id substring filter (e.g. CHN-D1-1)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not API_KEY or not BASE_URL:
        print("[ERR] VLM_API_KEY / VLM_BASE_URL not set in .env", file=sys.stderr)
        sys.exit(2)

    dirs = sorted(d for d in args.root.iterdir() if d.is_dir() and (d / "images").exists())
    if args.only:
        dirs = [d for d in dirs if args.only in d.name]
    if args.limit:
        dirs = dirs[: args.limit]

    print(f"VLM model: {MODEL} @ {BASE_URL}")
    print(f"Captioning {len(dirs)} extracted directories (workers={args.workers})", flush=True)
    total = 0
    for d in dirs:
        total += caption_one_dir(d, force=args.force, workers=args.workers)
    print(f"\nDone. Captioned {total} images this run.")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()

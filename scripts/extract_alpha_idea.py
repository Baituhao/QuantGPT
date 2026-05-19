"""Extract Alpha Idea content from WQB-ALPHA-IDEA/*.mhtml.

Splits each mhtml into:
  - <slug>.text.txt    — cleaned plain text (article body + comments, noise stripped)
  - <slug>.imgs/NN_*.png — images >5KB only (drops avatars/badges)
  - <slug>.meta.json   — meta: title, region, difficulty, source_id, image map (filename → pos in body)

Designed as the first stage of the alpha-idea knowledge base build.
"""
from __future__ import annotations

import argparse
import email
import hashlib
import json
import re
from email import policy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "WQB-ALPHA-IDEA"
DEFAULT_OUT = ROOT / ".tmp_alpha_idea_extracted"

NOISE_PATTERNS = [
    r"Skip to main content",
    r"BRAIN platform Community FAQ Submit a request",
    r"My activities My profile Sign out",
    r"English \(United States\)\s*简体中文",
    r"Menu\s*-->",
    r"Simulate Alphas Competitions Data Community Submit a request",
    r"WorldQuant BRAIN Community",
    r"中文论坛",
    r"Follow Followed by \d+ people",
    r"Sort by Date Votes",
    r"Post is closed for comments\.",
    r"Didn'?t find what you were looking for\?",
    r"New post",
    r"©\s*\d{4}\s*WorldQuant BRAIN",
    r"All Rights Reserved\.",
    r"Related to Alpha Idea",
]
NOISE_RE = re.compile("|".join(NOISE_PATTERNS))

REGION_RE = re.compile(r"^(CHN|USA|GLB|ASI|General)-", re.I)
DELAY_RE = re.compile(r"-(D[0-9])-", re.I)
# Match either {Region}-D{n}-{idx} or {Region}-{idx} (e.g. General-1)
SOURCE_ID_RE = re.compile(r"^([A-Za-z]+(?:-D[0-9])?-\d+)", re.I)

MIN_IMG_BYTES = 5_000  # skip avatars/badges


def slugify(name: str) -> str:
    """Make a filesystem-safe slug from the original mhtml filename (without extension)."""
    base = Path(name).stem
    base = base.replace(" – WorldQuant BRAIN", "").replace(" - WorldQuant BRAIN", "")
    base = base.replace("【Alpha灵感】", "").replace("【alpha灵感】", "")
    base = base.replace("【Alpha Idea】", "").replace("[Alpha Idea]", "")
    base = base.replace("【General Tech】", "").replace("【Alpha 灵感】", "")
    base = re.sub(r"\s+", " ", base).strip(" -·")
    return base


def parse_filename_meta(name: str) -> dict:
    base = Path(name).stem
    m_region = REGION_RE.search(base)
    m_delay = DELAY_RE.search(base)
    m_src = SOURCE_ID_RE.search(base)
    delay_token = m_delay.group(1).upper() if m_delay else ""
    delay = int(delay_token[1:]) if delay_token and delay_token[1:].isdigit() else None
    return {
        "region": m_region.group(1).upper() if m_region else "UNK",
        "delay": delay,            # 0 = intraday, 1 = T+1 (BRAIN DELAY field)
        "delay_token": delay_token,
        "source_id": m_src.group(1).upper() if m_src else base[:30],
        "original_filename": name,
    }


def extract_main_html(msg: email.message.Message) -> str:
    parts = [p for p in msg.walk() if p.get_content_type() == "text/html"]
    if not parts:
        return ""
    parts.sort(key=lambda p: len(p.get_payload(decode=True) or b""), reverse=True)
    raw = parts[0].get_payload(decode=True) or b""
    return raw.decode("utf-8", errors="replace")


def html_to_text_with_image_markers(html: str) -> tuple[str, list[str]]:
    """Strip tags but keep <img src> as markers like [[IMG:src]] so we can map them to position later."""
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.S | re.I)
    img_srcs: list[str] = []

    def img_repl(m: re.Match) -> str:
        src = re.search(r'src="([^"]+)"', m.group(0))
        if src:
            idx = len(img_srcs)
            img_srcs.append(src.group(1))
            return f" [[IMG:{idx}]] "
        return " "

    html = re.sub(r"<img[^>]*>", img_repl, html, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text, img_srcs


def trim_chrome(text: str) -> str:
    """Strip Zendesk navigation chrome from the start/end."""
    # Cut everything before the first occurrence of typical body markers.
    cut = text
    # remove known noise lines
    cut = NOISE_RE.sub(" ", cut)
    # collapse `--> ` artifacts
    cut = re.sub(r"\s*-->\s*", " ", cut)
    cut = re.sub(r"\s+", " ", cut).strip()
    return cut


def save_images(msg: email.message.Message, img_dir: Path) -> dict[str, str]:
    """Save images >MIN_IMG_BYTES and return src → saved-relpath map."""
    img_dir.mkdir(parents=True, exist_ok=True)
    by_loc: dict[str, str] = {}
    seq = 0
    for p in msg.walk():
        ct = p.get_content_type()
        if not ct.startswith("image/"):
            continue
        payload = p.get_payload(decode=True)
        if not payload or len(payload) < MIN_IMG_BYTES:
            continue
        loc = (p.get("Content-Location") or p.get("Content-ID") or "").strip("<>")
        if not loc:
            loc = f"unknown_{seq}"
        ext = ct.split("/")[-1].split("+")[0]
        if ext == "svg+xml":
            ext = "svg"
        # use a short hash of loc + sequence to avoid collisions and keep filenames short
        h = hashlib.md5(loc.encode("utf-8")).hexdigest()[:8]
        out_name = f"{seq:02d}_{h}.{ext}"
        (img_dir / out_name).write_bytes(payload)
        by_loc[loc] = out_name
        seq += 1
    return by_loc


def process_one(mhtml_path: Path, out_root: Path) -> dict:
    slug = slugify(mhtml_path.name)
    meta = parse_filename_meta(mhtml_path.name)
    work_dir = out_root / meta["source_id"]
    work_dir.mkdir(parents=True, exist_ok=True)

    with mhtml_path.open("rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    html = extract_main_html(msg)
    text, img_srcs = html_to_text_with_image_markers(html)
    text = trim_chrome(text)

    img_dir = work_dir / "images"
    src_to_file = save_images(msg, img_dir)

    # Map each [[IMG:N]] marker → either saved file (if size OK) or DROPPED.
    img_index = []
    for i, src in enumerate(img_srcs):
        saved = src_to_file.get(src)
        img_index.append({"idx": i, "src": src, "saved": saved})

    # Replace markers with a clearer reference for big imgs, drop small ones
    def marker_repl(m: re.Match) -> str:
        idx = int(m.group(1))
        info = img_index[idx]
        if info["saved"]:
            return f" [[FIG:{info['saved']}]] "
        return " "

    text = re.sub(r"\[\[IMG:(\d+)\]\]", marker_repl, text)
    text = re.sub(r"\s+", " ", text).strip()

    (work_dir / "body.txt").write_text(text, encoding="utf-8")
    meta_out = {
        **meta,
        "slug": slug,
        "n_images_kept": sum(1 for i in img_index if i["saved"]),
        "n_images_dropped": sum(1 for i in img_index if not i["saved"]),
        "images": [i for i in img_index if i["saved"]],
    }
    (work_dir / "meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", type=str, default=None, help="substring filter on filename")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.src.glob("*.mhtml"))
    if args.only:
        files = [f for f in files if args.only in f.name]
    if args.limit:
        files = files[: args.limit]

    print(f"Processing {len(files)} mhtml files → {args.out}")
    summary = []
    for f in files:
        try:
            meta = process_one(f, args.out)
            summary.append(meta)
            print(f"  OK  {meta['source_id']}  imgs_kept={meta['n_images_kept']} dropped={meta['n_images_dropped']}  {meta['slug']}")
        except Exception as e:
            print(f"  ERR {f.name}: {e}")
    (args.out / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote summary: {args.out / '_summary.json'}")


if __name__ == "__main__":
    import sys, io
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()

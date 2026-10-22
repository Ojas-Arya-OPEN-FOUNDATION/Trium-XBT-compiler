#!/usr/bin/env python3
"""Build the Next-Gen Foundation Compiler guide.

Reads docs/chapters/NN_*.md in order and produces:
  docs/Next_Gen_Foundation_Compiler_Guide.md   (single Markdown file, with TOC)
  docs/Next_Gen_Foundation_Compiler_Guide.html (single self-contained HTML page;
                                                diagrams render via Mermaid CDN)

Before building it VALIDATES the chapters:
  * every file path written in `backticks` that looks like a repo path must exist
  * mermaid fences must be balanced and every chapter must have a title
  * reports word counts, diagram counts and code-block counts

Usage:  python3 docs/build_docs.py [--strict]
        --strict  exit with status 1 if any referenced path is missing
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
CHAPTERS = DOCS / "chapters"
OUT_MD = DOCS / "Next_Gen_Foundation_Compiler_Guide.md"
OUT_HTML = DOCS / "Next_Gen_Foundation_Compiler_Guide.html"
ROOT_MD = ROOT / "NEXT_GEN_FOUNDATION_COMPILER_COMPLETE_GUIDE.md"
ROOT_HTML = ROOT / "NEXT_GEN_FOUNDATION_COMPILER_COMPLETE_GUIDE.html"
ROOT_TRIUM_MD = ROOT / "TRIUM_XBT_FOUNDATIONAL_COMPILER_COMPLETE_GUIDE.md"
ROOT_TRIUM_HTML = ROOT / "TRIUM_XBT_FOUNDATIONAL_COMPILER_COMPLETE_GUIDE.html"

TITLE = "Trium-XBT Foundational Compiler: The Complete Guide"
SUBTITLE = "From zero knowledge to rebuilding the whole project"

PATH_RE = re.compile(r"`([A-Za-z0-9_./\-]+\.(?:hpp|cpp|h|py|sh|td|mlir|bt|test|lean|metal|comp|mm|rs|toml|md|txt|dylib))`")
FENCE_RE = re.compile(r"^```(\w*)\s*$")
HEADING_RE = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")


def slugify(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_]+", "-", s).strip("-")


def load_chapters() -> list[tuple[Path, str]]:
    files = sorted(CHAPTERS.glob("[0-9][0-9]_*.md"))
    if not files:
        sys.exit(f"error: no chapters found in {CHAPTERS}")
    return [(f, f.read_text(encoding="utf-8")) for f in files]


def validate(chapters) -> tuple[list[str], dict]:
    problems: list[str] = []
    stats = {"words": 0, "mermaid": 0, "codeblocks": 0, "paths": 0}
    # generated/optional artefacts that legitimately may not exist in a clean checkout
    optional = {"libbt_runtime.dylib", "bt-compiler", "bt-opt"}
    for path, text in chapters:
        lines = text.splitlines()
        in_fence, fence_lang = False, ""
        if not any(HEADING_RE.match(l) and l.startswith("# ") for l in lines):
            problems.append(f"{path.name}: missing a top-level '# Title'")
        for n, line in enumerate(lines, 1):
            m = FENCE_RE.match(line.strip())
            if m:
                if not in_fence:
                    in_fence, fence_lang = True, m.group(1)
                    stats["codeblocks"] += 1
                    if fence_lang == "mermaid":
                        stats["mermaid"] += 1
                else:
                    in_fence = False
        if in_fence:
            problems.append(f"{path.name}: unbalanced ``` fence")
        stats["words"] += len(re.findall(r"\w+", text))
        for ref in set(PATH_RE.findall(text)):
            stats["paths"] += 1
            if ref in optional:
                continue
            if not (ROOT / ref).exists() and not any(ROOT.glob(f"**/{ref}")):
                problems.append(f"{path.name}: referenced path does not exist: {ref}")
    return problems, stats


def collect_headings(md: str):
    out, in_fence = [], False
    for line in md.splitlines():
        if FENCE_RE.match(line.strip()):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING_RE.match(line)
        if m and len(m.group(1)) <= 2:
            out.append((len(m.group(1)), m.group(2)))
    return out


def build_markdown(chapters) -> str:
    # Map chapter file names to their top-level heading anchor slugs
    chapter_slug_map: dict[str, str] = {}
    for path, text in chapters:
        for line in text.splitlines():
            m = HEADING_RE.match(line)
            if m and len(m.group(1)) == 1:
                slug = slugify(m.group(2))
                chapter_slug_map[path.name] = slug
                chapter_slug_map[f"./{path.name}"] = slug
                break

    # Resolve inter-chapter links like (01_number_systems.md) -> (#chapter-1-number-systems-from-zero)
    processed_chapters = []
    for path, text in chapters:
        c_text = text.strip()
        for c_file, slug in chapter_slug_map.items():
            c_text = c_text.replace(f"({c_file})", f"(#{slug})")
        processed_chapters.append((path, c_text))

    body = "\n\n---\n\n".join(text for _, text in processed_chapters)
    toc_lines = ["## Table of Contents", ""]
    seen: dict[str, int] = {}
    for level, title in collect_headings(body):
        slug = slugify(title)
        seen[slug] = seen.get(slug, 0) + 1
        if seen[slug] > 1:
            slug = f"{slug}-{seen[slug] - 1}"
        indent = "  " * (level - 1)
        toc_lines.append(f"{indent}- [{title}](#{slug})")
    header = f"# {TITLE}\n\n*{SUBTITLE}*\n\n"
    return header + "\n".join(toc_lines) + "\n\n---\n\n" + body + "\n"


def md_to_html_body(md: str) -> str:
    """Tiny dependency-free Markdown renderer (headings, lists, tables, code, emphasis, links)."""
    out: list[str] = []
    lines = md.splitlines()
    i, in_list = 0, None
    seen: dict[str, int] = {}

    def inline(t: str) -> str:
        t = html.escape(t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", t)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', t)
        return t

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(f"</{in_list}>")
            in_list = None

    while i < len(lines):
        line = lines[i]
        fm = FENCE_RE.match(line.strip())
        if fm:
            close_list()
            lang, buf = fm.group(1), []
            i += 1
            while i < len(lines) and not FENCE_RE.match(lines[i].strip()):
                buf.append(lines[i])
                i += 1
            code = "\n".join(buf)
            if lang == "mermaid":
                out.append(f'<pre class="mermaid">{html.escape(code)}</pre>')
            else:
                out.append(f'<pre><code class="lang-{lang}">{html.escape(code)}</code></pre>')
            i += 1
            continue
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:\-|]+\|\s*$", lines[i + 1]):
            close_list()
            head = [c.strip() for c in line.strip().strip("|").split("|")]
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>")
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</tbody></table>")
            continue
        hm = HEADING_RE.match(line)
        if hm:
            close_list()
            lvl, txt = len(hm.group(1)), hm.group(2)
            slug = slugify(txt)
            seen[slug] = seen.get(slug, 0) + 1
            if seen[slug] > 1:
                slug = f"{slug}-{seen[slug] - 1}"
            out.append(f'<h{lvl} id="{slug}">{inline(txt)}</h{lvl}>')
            i += 1
            continue
        lm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if lm:
            kind = "ol" if lm.group(2)[0].isdigit() else "ul"
            if in_list != kind:
                close_list()
                out.append(f"<{kind}>")
                in_list = kind
            out.append(f"<li>{inline(lm.group(3))}</li>")
            i += 1
            continue
        if line.strip() == "---":
            close_list()
            out.append("<hr>")
        elif line.startswith(">"):
            close_list()
            out.append(f"<blockquote>{inline(line.lstrip('> '))}</blockquote>")
        elif line.strip() == "":
            close_list()
        else:
            close_list()
            out.append(f"<p>{inline(line)}</p>")
        i += 1
    close_list()
    return "\n".join(out)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:900px;margin:2rem auto;padding:0 1.2rem;line-height:1.65;color:#1f2328}}
 h1,h2,h3{{line-height:1.25}} h1{{border-bottom:2px solid #d0d7de;padding-bottom:.3rem}}
 h2{{border-bottom:1px solid #d8dee4;padding-bottom:.2rem;margin-top:2.2rem}}
 code{{background:#f3f4f6;padding:.1rem .35rem;border-radius:4px;font-size:.92em}}
 pre{{background:#0d1117;color:#e6edf3;padding:1rem;border-radius:8px;overflow:auto}}
 pre code{{background:none;color:inherit;padding:0}}
 pre.mermaid{{background:#fff;color:#000;text-align:center;border:1px solid #d0d7de}}
 table{{border-collapse:collapse;width:100%;margin:1rem 0}} th,td{{border:1px solid #d0d7de;padding:.45rem .7rem;text-align:left}}
 th{{background:#f6f8fa}} blockquote{{border-left:4px solid #8250df;margin:1rem 0;padding:.2rem 1rem;background:#f6f0ff}}
 a{{color:#0969da}}
</style></head><body>
{body}
<script type="module">
 import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
 mermaid.initialize({{startOnLoad:true}});
</script>
</body></html>
"""


def main() -> int:
    strict = "--strict" in sys.argv
    chapters = load_chapters()
    problems, stats = validate(chapters)
    md = build_markdown(chapters)
    html_content = HTML_TEMPLATE.format(title=html.escape(TITLE), body=md_to_html_body(md))
    OUT_MD.write_text(md, encoding="utf-8")
    OUT_HTML.write_text(html_content, encoding="utf-8")
    ROOT_MD.write_text(md, encoding="utf-8")
    ROOT_HTML.write_text(html_content, encoding="utf-8")
    ROOT_TRIUM_MD.write_text(md, encoding="utf-8")
    ROOT_TRIUM_HTML.write_text(html_content, encoding="utf-8")

    print(f"Chapters : {len(chapters)}")
    for p, t in chapters:
        words = len(re.findall(r"\w+", t))
        print(f"  - {p.name}: {words:,} words")
    print(f"Total    : {stats['words']:,} words, {stats['mermaid']} mermaid diagrams, "
          f"{stats['codeblocks']} code blocks, {stats['paths']} file references checked")
    print(f"Wrote    : {OUT_MD.relative_to(ROOT)}  ({OUT_MD.stat().st_size // 1024} KiB)")
    print(f"Wrote    : {OUT_HTML.relative_to(ROOT)}  ({OUT_HTML.stat().st_size // 1024} KiB)")
    print(f"Wrote    : {ROOT_MD.name}  ({ROOT_MD.stat().st_size // 1024} KiB) [Project Root Single File]")
    print(f"Wrote    : {ROOT_HTML.name}  ({ROOT_HTML.stat().st_size // 1024} KiB) [Project Root Single File]")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  ! {p}")
        return 1 if strict else 0
    print("Validation: OK (all referenced paths exist, fences balanced)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

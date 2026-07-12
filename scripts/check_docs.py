#!/usr/bin/env python3
"""Fast repository-local checks for documentation structure and links."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
NAV_TARGET = re.compile(r":\s+([^\s#]+\.md)\s*$")
WEB_SCHEMES = ("http://", "https://", "mailto:", "chatgpt-conversation://")
BEGINNER_GUARDRAILS = {
    "index.md": (
        "jump to mixture",
        "](05-moe/",
        "sparse feed-forward",
        "backpropagation",
        "kv cache",
    ),
    "start-here.md": (
        "logits",
        "softmax",
        "backpropagation",
        "causal mask",
        "kv cache",
        "sparse feed-forward",
    ),
    "01-foundations/00-before-the-jargon.md": (
        "logits",
        "softmax",
        "backpropagation",
        "causal mask",
        "kv cache",
        "sparse feed-forward",
    ),
}


def nav_targets() -> list[Path]:
    targets = []
    for line in (ROOT / "mkdocs.yml").read_text().splitlines():
        match = NAV_TARGET.search(line)
        if match:
            targets.append(DOCS / match.group(1))
    return targets


def local_link_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
    if not target or target.startswith(("#", *WEB_SCHEMES)):
        return None
    target = target.split("#", 1)[0].split("?", 1)[0]
    if not target:
        return None
    return (source.parent / target).resolve()


def main() -> int:
    errors: list[str] = []
    markdown_files = sorted(DOCS.rglob("*.md")) + [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]

    for target in nav_targets():
        if not target.is_file():
            errors.append(f"mkdocs nav target does not exist: {target.relative_to(ROOT)}")

    home = (DOCS / "index.md").read_text().lower()
    if "[begin with lesson 0](01-foundations/00-before-the-jargon.md)" not in home:
        errors.append("homepage must make lesson 0 the primary starting action")

    for relative_path, banned_phrases in BEGINNER_GUARDRAILS.items():
        beginner_text = (DOCS / relative_path).read_text().lower()
        for phrase in banned_phrases:
            if phrase in beginner_text:
                errors.append(
                    f"advanced term or personalized shortcut in beginner path: "
                    f"{relative_path}: {phrase}"
                )

    for source in markdown_files:
        text = source.read_text()
        relative = source.relative_to(ROOT)
        lines = text.splitlines()
        fence_lines = [line for line in lines if line.startswith("```")]
        if len(fence_lines) % 2:
            errors.append(f"unbalanced fenced code block: {relative}")
        if "```mermaid\n```" in text:
            errors.append(f"empty Mermaid block: {relative}")
        if re.search(r"turn\d+(?:search|fetch|view)\d+", text):
            errors.append(f"internal web citation leaked into prose: {relative}")
        prose_lines: list[str] = []
        inside_fence = False
        for line in lines:
            if line.startswith("```"):
                inside_fence = not inside_fence
                continue
            if not inside_fence:
                prose_lines.append(line)
        for raw_target in MARKDOWN_LINK.findall("\n".join(prose_lines)):
            target = local_link_target(source, raw_target)
            if target is not None and not target.exists():
                errors.append(f"broken local link in {relative}: {raw_target}")

    if errors:
        print("documentation checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        f"checked {len(markdown_files)} Markdown files, "
        f"{len(nav_targets())} navigation targets, and local links"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

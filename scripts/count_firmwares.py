#!/usr/bin/env python3
import pathlib
import json
import re

PATTERNS = ["*.bin", "*.rom", "*.hex", "*.img", "*.fw", "*.dump"]

EXCLUDE_DIRS = {".git", ".github", "scripts", "__pycache__"}

repo_root = pathlib.Path(__file__).resolve().parent.parent

def is_firmware_file(p: pathlib.Path) -> bool:
    if not p.is_file():
        return False
    if p.parent.name in EXCLUDE_DIRS:
        return False
    if p.suffix in {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".gitignore"}:
        return False
    for pat in PATTERNS:
        if p.match(pat):
            return True
    return False

firmware_files = []
for path in repo_root.rglob("*"):
    parts = set(part for part in path.parts)
    if parts & EXCLUDE_DIRS:
        continue
    if is_firmware_file(path):
        firmware_files.append(path)

count = len(firmware_files)

badge_path = repo_root / ".github" / "firmware-count-badge.json"
badge_path.parent.mkdir(parents=True, exist_ok=True)

badge_obj = {
    "schemaVersion": 1,
    "label": "firmwares",
    "message": str(count),
    "color": "blue" if count > 0 else "lightgrey"
}

existing = None
if badge_path.exists():
    try:
        existing = json.loads(badge_path.read_text())
    except Exception:
        pass

if existing != badge_obj:
    badge_path.write_text(json.dumps(badge_obj, indent=2) + "\n")

readme_path = repo_root / "README.md"
if readme_path.exists():
    original = readme_path.read_text(encoding="utf-8")
    badge_markdown = "![Firmwares](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/1vers1on/kittyrips/main/.github/firmware-count-badge.json)"
    if original.startswith("# "):
        lines = original.splitlines()
        if lines and lines[0].startswith("# "):
            already = any("img.shields.io/endpoint?url=" in l for l in lines[1:4])
            if not already:
                lines.insert(1, badge_markdown)
            else:
                for i,l in enumerate(lines[:6]):
                    if "img.shields.io/endpoint?url=" in l:
                        lines[i] = badge_markdown
                        break
            updated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
            if updated != original:
                readme_path.write_text(updated, encoding="utf-8")

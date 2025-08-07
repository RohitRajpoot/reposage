import os
import csv

import re
from textwrap import wrap

def split_markdown(md_text, max_chars=600):
    """
    Yields successive (heading, block_text) chunks where each block is
    the heading plus its following content, wrapped to ≤ max_chars.
    """
    heading = "README"
    buffer  = []

    def flush():
        if buffer:
            block = " ".join(buffer).strip()
            for part in wrap(block, max_chars):
                yield heading, part.strip()
            buffer.clear()

    for line in md_text.splitlines():
        h_match = re.match(r'^(#{1,3})\s+(.*)', line)
        if h_match:
            # emit whatever we’ve buffered under the previous heading
            yield from flush()
            # start a new heading context
            heading = h_match.group(2)
        elif line.strip():          # non-blank line
            buffer.append(line.strip())
        else:
            # blank line = boundary
            yield from flush()

    yield from flush()



# 1. Point this at the folder containing your .md files
ROOT_DIR = "."   # or "." if your README.md is at repo root
OUT_CSV  = "data/classification.csv"

# 2. Collect all paragraphs
rows = []
for dirpath, _, filenames in os.walk(ROOT_DIR):
    for fn in filenames:
        if not fn.lower().endswith(".md"):
            continue
        path = os.path.join(dirpath, fn)
        text = open(path, encoding="utf-8").read()
        # split on blank lines
        for heading, block in split_markdown(text, max_chars=600):
            # you can choose to store heading if you like, or just the text
            rows.append({"text": block, "category": ""})

# 3. Write out a CSV with an empty category column
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text","category"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Extracted {len(rows)} paragraphs → {OUT_CSV}")

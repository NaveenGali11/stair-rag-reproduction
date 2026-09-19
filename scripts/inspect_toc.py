from pathlib import Path

import pymupdf

pdf_path = Path("data/raw/whole-child.pdf")

with pymupdf.open(pdf_path) as document:
    toc = document.get_toc(simple=True)

    print(f"PDF pages: {document.page_count}")
    print(f"ToC entries: {len(toc)}")
    print(f"Maximum depth: {max(level for level, _, _ in toc)}")
    print("\nFirst 80 ToC entries:\n")

    for level, title, page in toc[:80]:
        indent = "  " * (level - 1)
        print(f"{indent}- {title} [PDF page {page}]")
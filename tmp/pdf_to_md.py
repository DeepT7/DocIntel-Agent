"""Temporary helper: convert PDF(s) to Markdown with PyMuPDF4LLM and save .md files.

Usage:
    python pdf_to_md.py data/docs/sample.pdf
    python pdf_to_md.py data/docs                 # every *.pdf in the folder
    python pdf_to_md.py data/docs -o data/markdown --pages 0 1 2

Requires: pip install pymupdf4llm
"""

import argparse
from pathlib import Path


def convert(pdf: Path, out_dir: Path, pages: list[int] | None = None) -> Path:
    import pymupdf4llm

    # pages: 0-based indices, or None for the whole document.
    markdown = pymupdf4llm.to_markdown(str(pdf), pages=pages)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf.stem}.md"
    out_path.write_text(markdown, encoding="utf-8")
    return out_path


def collect_pdfs(inputs: list[str]) -> list[Path]:
    pdfs: list[Path] = []
    for item in inputs:
        # Tolerate Windows-style separators pasted into a Linux shell.
        path = Path(item.replace("\\", "/"))
        if path.is_dir():
            pdfs.extend(sorted(path.rglob("*.pdf")))
        elif path.suffix.lower() == ".pdf":
            pdfs.append(path)
        else:
            print(f"skip (not a pdf): {path}")
    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDF(s) to Markdown files.")
    parser.add_argument("inputs", nargs="+", help="PDF files and/or directories")
    parser.add_argument("-o", "--out", default="data/markdown", help="output directory")
    parser.add_argument(
        "--pages",
        type=int,
        nargs="*",
        default=None,
        help="0-based page indices to convert (default: all pages)",
    )
    args = parser.parse_args()

    pdfs = collect_pdfs(args.inputs)
    if not pdfs:
        raise SystemExit("No PDF files found.")

    out_dir = Path(args.out)
    for pdf in pdfs:
        try:
            out_path = convert(pdf, out_dir, pages=args.pages)
        except ImportError:
            raise SystemExit("pymupdf4llm is not installed. Run: pip install pymupdf4llm")
        print(f"{pdf} -> {out_path}")


if __name__ == "__main__":
    main()

"""Tests for canonical Markdown extraction (no Qdrant/LLM required)."""

import tempfile
import unittest
from pathlib import Path

from app.ingestion.extract import canonical_markdown, extract_blocks

SAMPLE = """# Title

Intro paragraph with a <!-- hidden note --> comment.

| Name | Value |
| --- | --- |
| alpha | 1 |

```python
# this is a code comment, not a heading
print("ok")
```

<!-- Start of picture text -->
Axis label
<!-- End of picture text -->

![chart](image1.png)

Table 1: Results
"""


class CanonicalMarkdownTest(unittest.TestCase):
    def test_markdown_is_normalised_and_split_into_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.md"
            path.write_text(SAMPLE, encoding="utf-8")

            markdown = canonical_markdown(path)
            self.assertIn("Intro paragraph with a  comment.", markdown)
            self.assertNotIn("hidden note", markdown)
            self.assertNotIn("![chart]", markdown)
            self.assertIn("<!-- Start of picture text -->", markdown)

            blocks = extract_blocks(path)
            self.assertEqual(blocks[0], {"type": "heading", "level": 1, "text": "Title"})
            headings = [block["text"] for block in blocks if block["type"] == "heading"]
            self.assertEqual(headings, ["Title"])
            self.assertIn(
                {"type": "table", "rows": [["Name", "Value"], ["alpha", "1"]]},
                blocks,
            )
            self.assertIn({"type": "picture", "text": "Axis label"}, blocks)
            body = "\n".join(block["text"] for block in blocks if block["type"] == "text")
            self.assertIn('print("ok")', body)
            self.assertNotIn("hidden note", body)

    def test_unsupported_extension_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "main.py"
            path.write_text("print('x')", encoding="utf-8")
            with self.assertRaises(ValueError):
                canonical_markdown(path)


if __name__ == "__main__":
    unittest.main()

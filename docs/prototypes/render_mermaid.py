"""Build a preview HTML page from the ```mermaid blocks of a markdown file."""
import html
import re
import sys
from pathlib import Path

HEAD = """<!doctype html>
<html><head><meta charset="utf-8"><title>Diagram preview</title>
<style>body{font-family:sans-serif;margin:24px;background:#fff}
h2{margin-top:40px} pre.mermaid{background:#fff;overflow:auto}</style></head>
<body>
<p><i>Click a diagram to switch between fit-to-width and full size.</i></p>
"""

TAIL = """
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
mermaid.initialize({startOnLoad: false, theme: "default", flowchart: {htmlLabels: true}});
await mermaid.run();
for (const svg of document.querySelectorAll("pre.mermaid svg")) {
  const w = svg.viewBox.baseVal.width;
  svg.style.cursor = "zoom-in";
  svg.addEventListener("click", () => {
    const full = svg.dataset.full === "1";
    svg.dataset.full = full ? "0" : "1";
    svg.style.maxWidth = full ? w + "px" : "none";
    svg.style.width = full ? "100%" : w + "px";
    svg.style.cursor = full ? "zoom-in" : "zoom-out";
  });
}
document.body.dataset.rendered = "1";
</script>
</body></html>
"""

src, out = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text(encoding="utf-8")
parts = []
for m in re.finditer(r"^#{2,3} (.+?)$|^```mermaid\n(.*?)^```", text, re.S | re.M):
    if m.group(1):
        parts.append(f"<h2>{html.escape(m.group(1))}</h2>")
    else:
        parts.append(f'<pre class="mermaid">\n{html.escape(m.group(2))}</pre>')
out.write_text(HEAD + "\n".join(parts) + TAIL, encoding="utf-8")
print(out)

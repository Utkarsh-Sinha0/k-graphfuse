"""Render the Markdown research manuscript to a linked, paginated PDF."""

from __future__ import annotations

import hashlib
import html
import re
import tempfile
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit
from xml.sax.saxutils import escape, quoteattr

import markdown
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab import Version as reportlab_version
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "research" / "paper.md"
PDF = ROOT / "docs" / "research" / "k-graphfuse-paper.pdf"
RECORD = ROOT / "research" / "PDF_RENDERING.md"
DOCS = ROOT / "docs"
MKDOCS = ROOT / "mkdocs.yml"
EXPECTED_FIGURES = (
    "precision_percent_comparison.png",
    "native_win_coverage.png",
    "graphfuse_pipeline.png",
)
INK = colors.HexColor("#17243A")
MUTED = colors.HexColor("#5C6878")
BLUE = colors.HexColor("#244E78")
PALE = colors.HexColor("#EEF3F8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain_text(markup: str) -> str:
    class Text(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.parts: list[str] = []

        def handle_data(self, value: str) -> None:
            self.parts.append(value)

    parser = Text()
    parser.feed(markup)
    return " ".join("".join(parser.parts).split())


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).replace("\u200b", "").lower().split())


def strip_initial_frontmatter(source: str) -> str:
    lines = source.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return source
    for index, line in enumerate(lines[1:], 1):
        if line.strip() in {"---", "..."}:
            return "".join(lines[index + 1:])
    raise ValueError("Unclosed initial YAML frontmatter")


assert strip_initial_frontmatter("---\ntitle: test\n---\n# body\n") == "# body\n"


def public_site_url() -> str:
    for line in MKDOCS.read_text(encoding="utf-8").splitlines():
        if line.startswith("site_url:"):
            return line.split(":", 1)[1].strip().strip("\"'").rstrip("/") + "/"
    raise ValueError("mkdocs.yml is missing site_url")


def register_fonts() -> None:
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    fonts = {
        "KGSerif": "DejaVuSerif.ttf",
        "KGSerif-Bold": "DejaVuSerif-Bold.ttf",
        "KGSerif-Italic": "DejaVuSerif-Italic.ttf",
        "KGSerif-BoldItalic": "DejaVuSerif-BoldItalic.ttf",
        "KGSans": "DejaVuSans.ttf",
        "KGSans-Bold": "DejaVuSans-Bold.ttf",
        "KGMono": "DejaVuSansMono.ttf",
    }
    for name, filename in fonts.items():
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily(
        "KGSerif", normal="KGSerif", bold="KGSerif-Bold",
        italic="KGSerif-Italic", boldItalic="KGSerif-BoldItalic",
    )
    pdfmetrics.registerFontFamily(
        "KGSans", normal="KGSans", bold="KGSans-Bold",
        italic="KGSans", boldItalic="KGSans-Bold",
    )


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["BodyText"]
    return {
        "body": ParagraphStyle(
            "KGBody", parent=base, fontName="KGSerif", fontSize=9.2,
            leading=13.2, textColor=INK, spaceAfter=6, alignment=TA_LEFT,
            allowWidows=0, allowOrphans=0,
        ),
        "title": ParagraphStyle(
            "KGTitle", fontName="KGSans-Bold", fontSize=20, leading=25,
            textColor=INK, alignment=TA_CENTER, spaceBefore=6, spaceAfter=13,
        ),
        "author": ParagraphStyle(
            "KGAuthor", fontName="KGSerif", fontSize=11.5, leading=15,
            textColor=INK, alignment=TA_CENTER, spaceAfter=4,
        ),
        "status": ParagraphStyle(
            "KGStatus", fontName="KGSerif-Italic", fontSize=8.4, leading=11,
            textColor=MUTED, alignment=TA_CENTER, spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "KGSection", fontName="KGSans-Bold", fontSize=14, leading=18,
            textColor=BLUE, spaceBefore=13, spaceAfter=6, keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "KGSubsection", fontName="KGSans-Bold", fontSize=10.8, leading=14,
            textColor=INK, spaceBefore=9, spaceAfter=4, keepWithNext=True,
        ),
        "h4": ParagraphStyle(
            "KGMinorHeading", fontName="KGSans-Bold", fontSize=9.5, leading=12,
            textColor=INK, spaceBefore=7, spaceAfter=3, keepWithNext=True,
        ),
        "quote": ParagraphStyle(
            "KGQuote", fontName="KGSerif-Italic", fontSize=9, leading=13,
            textColor=MUTED, leftIndent=14, rightIndent=12, spaceAfter=7,
        ),
        "list": ParagraphStyle(
            "KGList", fontName="KGSerif", fontSize=9.1, leading=12.4,
            textColor=INK, leftIndent=13, firstLineIndent=0, spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "KGCode", fontName="KGMono", fontSize=7.7, leading=10,
            textColor=INK, backColor=PALE, borderColor=colors.HexColor("#D8E1EA"),
            borderWidth=0.4, borderPadding=5, leftIndent=3, rightIndent=3,
            spaceBefore=3, spaceAfter=7, wordWrap="CJK",
        ),
        "equation": ParagraphStyle(
            "KGEquationFallback", fontName="KGMono", fontSize=8.2, leading=11,
            textColor=INK, backColor=PALE, borderColor=colors.HexColor("#D8E1EA"),
            borderWidth=0.4, borderPadding=5, leftIndent=5, rightIndent=5,
            spaceBefore=4, spaceAfter=7, wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "KGTable", fontName="KGSans", fontSize=6.9, leading=9,
            textColor=INK, spaceAfter=0, wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "KGTableHeader", fontName="KGSans-Bold", fontSize=7, leading=9,
            textColor=colors.white, spaceAfter=0, wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "KGCaption", fontName="KGSerif-Italic", fontSize=8.2, leading=11,
            textColor=MUTED, alignment=TA_CENTER, spaceBefore=2, spaceAfter=5,
            keepWithNext=True,
        ),
    }


class InlineMarkup(HTMLParser):
    """Translate Markdown inline HTML to ReportLab's small XML dialect."""

    TAGS = {"strong": "b", "b": "b", "em": "i", "i": "i", "del": "strike", "s": "strike", "sup": "super", "sub": "sub"}

    def __init__(self, resolve_link=None) -> None:
        super().__init__()
        self.out: list[str] = []
        self.resolve_link = resolve_link

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = dict(attrs)
        if tag in self.TAGS:
            self.out.append(f"<{self.TAGS[tag]}>")
        elif tag == "code" or tag == "kbd":
            self.out.append('<font name="KGMono" size="8.2">')
        elif tag == "a":
            raw_href = attrs_map.get("href") or ""
            href = html.escape(self.resolve_link(raw_href) if self.resolve_link else raw_href, quote=True)
            self.out.append(f'<a href="{href}" color="#244E78">')
        elif tag == "img":
            src = quoteattr(attrs_map.get("src") or "")
            width, height = attrs_map.get("width", "0"), attrs_map.get("height", "0")
            self.out.append(f'<img src={src} width="{width}" height="{height}" valign="middle"/>')
        elif tag == "br":
            self.out.append("<br/>")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.TAGS:
            self.out.append(f"</{self.TAGS[tag]}>")
        elif tag == "code" or tag == "kbd":
            self.out.append("</font>")
        elif tag == "a":
            self.out.append("</a>")

    def handle_data(self, value: str) -> None:
        if len(value) > 28 and not any(ch.isspace() for ch in value):
            value = re.sub(r"(?<=[/_?&=.-])", "\u200b", value)
        self.out.append(escape(value))


def inline_markup(source: str, resolve_link=None) -> str:
    parser = InlineMarkup(resolve_link)
    parser.feed(source)
    return "".join(parser.out)


def normalize_mathtext(formula: str) -> str:
    formula = re.sub(r"\\mathsf\s+([A-Za-z])", lambda match: r"\mathsf{" + match.group(1) + "}", formula)
    formula = re.sub(r"\\le(?![A-Za-z])", lambda _: r"\leq", formula)
    return re.sub(r"\\ge(?![A-Za-z])", lambda _: r"\geq", formula)


def render_math(formula: str, path: Path, *, display: bool) -> tuple[float, float]:
    lines = [normalize_mathtext(line.strip()) for line in formula.splitlines() if line.strip()] if display else [normalize_mathtext(formula.strip())]
    size, dpi = (12 if display else 9.5), 240
    line_paths = [path.with_name(f"{path.stem}-line-{index}.png") for index in range(len(lines))]
    for line, line_path in zip(lines, line_paths):
        figure = Figure(figsize=(8, 1), dpi=dpi)
        figure.patch.set_alpha(0)
        canvas = FigureCanvasAgg(figure)
        with matplotlib.rc_context({"mathtext.fontset": "dejavuserif", "font.family": "DejaVu Serif"}):
            text = figure.text(0.5, 0.5, f"${line}$", ha="center", va="center", fontsize=size)
            canvas.draw()
            bounds = text.get_window_extent(canvas.get_renderer()).expanded(1.06, 1.28)
            bounds = bounds.transformed(figure.dpi_scale_trans.inverted())
            figure.savefig(line_path, dpi=dpi, transparent=True, bbox_inches=bounds, pad_inches=0.025)
    images = [PILImage.open(line_path).convert("RGBA") for line_path in line_paths]
    pad, gap = 4, 3 if display else 0
    width = max(image.width for image in images) + 2 * pad
    height = sum(image.height for image in images) + gap * (len(images) - 1) + 2 * pad
    combined = PILImage.new("RGBA", (width, height), (255, 255, 255, 0))
    y = pad
    for image in images:
        combined.alpha_composite(image, ((width - image.width) // 2, y))
        y += image.height + gap
        image.close()
    combined.save(path)
    for line_path in line_paths:
        line_path.unlink()
    return width * 72 / dpi, height * 72 / dpi


class PaperHTML(HTMLParser):
    def __init__(self, root: Path, work: Path, styles: dict[str, ParagraphStyle]) -> None:
        super().__init__()
        self.root, self.work, self.styles = root, work, styles
        self.site_url = public_site_url()
        source_route = SOURCE.relative_to(DOCS).with_suffix("").as_posix()
        if source_route == "index":
            source_route = ""
        elif source_route.endswith("/index"):
            source_route = source_route[:-6]
        self.paper_url = urljoin(self.site_url, quote(source_route.rstrip("/") + "/", safe="/._~-"))
        self.story: list[object] = []
        self.mode: str | None = None
        self.buffer: list[str] = []
        self.table_rows: list[list[str]] | None = None
        self.row: list[str] = []
        self.tables: list[list[list[str]]] = []
        self.images: dict[str, tuple[str, str]] = {}
        self.maths: dict[str, tuple[str, bool]] = {}
        self.list_stack: list[tuple[str, int]] = []
        self.list_marker = "•"
        self.quote_depth = 0
        self.heading_seen = False
        self.meta_seen = False
        self.math_rendered = 0
        self.math_fallbacks: list[str] = []
        self.figure_hashes: dict[str, str] = {}
        self.relative_links: dict[str, str] = {}
        self.rendered_formulas: set[str] = set()

    def resolve_link(self, href: str) -> str:
        parsed = urlsplit(href)
        if parsed.scheme or parsed.netloc:
            return href
        if not parsed.path:
            resolved = urljoin(self.paper_url, href)
            self.relative_links[href] = resolved
            return resolved

        path = unquote(parsed.path)
        target = (SOURCE.parent / path).resolve()
        if Path(path).name == PDF.name:
            target = PDF
        if (not target.is_file() and target != PDF) or not target.is_relative_to(DOCS):
            raise ValueError(f"Relative paper link has no published documentation target: {href}")
        relative = target.relative_to(DOCS)
        if target.suffix.lower() == ".md":
            route = relative.with_suffix("").as_posix()
            if route == "index":
                route = ""
            elif route.endswith("/index"):
                route = route[:-6]
            public_path = quote(route, safe="/._~-") + "/"
        else:
            public_path = quote(relative.as_posix(), safe="/._~-" )
        destination = urljoin(self.site_url, public_path)
        destination_parts = urlsplit(destination)
        resolved = urlunsplit((destination_parts.scheme, destination_parts.netloc, destination_parts.path, parsed.query, parsed.fragment))
        self.relative_links[href] = resolved
        return resolved

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "table":
            self.table_rows = []
            return
        if tag == "tr" and self.table_rows is not None:
            self.row = []
            return
        if tag in {"td", "th"} and self.table_rows is not None:
            self.mode, self.buffer = tag, []
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "li"}:
            if self.mode is None:
                self.mode, self.buffer = tag, []
                if tag == "li":
                    depth = max(0, len(self.list_stack) - 1)
                    kind, number = self.list_stack[-1] if self.list_stack else ("ul", 0)
                    self.list_marker = f"{number}." if kind == "ol" else "•"
                    if self.list_stack and kind == "ol":
                        self.list_stack[-1] = (kind, number + 1)
                    self._list_depth = depth
            return
        if tag in {"ul", "ol"}:
            start = values.get("start")
            self.list_stack.append((tag, int(start or 1) if tag == "ol" else 0))
            return
        if tag == "blockquote":
            self.quote_depth += 1
            return
        if tag == "hr":
            self.story.extend([Spacer(1, 3), HRFlowable(width="100%", thickness=0.55, color=colors.HexColor("#CDD7E2")), Spacer(1, 6)])
            return
        if tag == "img":
            token = f"KGFIGURETOKEN{len(self.images)}END"
            self.images[token] = (values.get("src") or "", values.get("alt") or "")
            if self.mode is not None:
                self.buffer.append(token)
            else:
                self._add_image(token)
            return
        if self.mode is not None:
            self.buffer.append(self._opening_inline(tag, values))

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.mode == tag and self.table_rows is not None:
            self.row.append("".join(self.buffer))
            self.mode, self.buffer = None, []
            return
        if tag == "tr" and self.table_rows is not None:
            self.table_rows.append(self.row)
            return
        if tag == "table" and self.table_rows is not None:
            self._add_table(self.table_rows)
            self.tables.append(self.table_rows)
            self.table_rows = None
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "li"} and self.mode == tag:
            self._finish_block(tag, "".join(self.buffer))
            self.mode, self.buffer = None, []
            return
        if tag in {"ul", "ol"} and self.list_stack:
            self.list_stack.pop()
            return
        if tag == "blockquote":
            self.quote_depth = max(0, self.quote_depth - 1)
            return
        if self.mode is not None:
            self.buffer.append(self._closing_inline(tag))

    def _opening_inline(self, tag: str, attrs: dict[str, str | None]) -> str:
        mapping = {"strong": "b", "b": "b", "em": "i", "i": "i", "del": "strike", "s": "strike", "sup": "super", "sub": "sub"}
        if tag in mapping:
            return f"<{mapping[tag]}>"
        if tag in {"code", "kbd"}:
            return '<font name="KGMono" size="8.2">'
        if tag == "a":
            href = html.escape(self.resolve_link(attrs.get("href") or ""), quote=True)
            return f'<a href="{href}" color="#244E78">'
        if tag == "br":
            return "<br/>"
        return ""

    @staticmethod
    def _closing_inline(tag: str) -> str:
        mapping = {"strong": "b", "b": "b", "em": "i", "i": "i", "del": "strike", "s": "strike", "sup": "super", "sub": "sub"}
        if tag in mapping:
            return f"</{mapping[tag]}>"
        if tag in {"code", "kbd"}:
            return "</font>"
        if tag == "a":
            return "</a>"
        return ""

    def handle_data(self, value: str) -> None:
        if self.mode in {"pre", "td", "th", "p", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            if self.mode == "pre":
                self.buffer.append(value)
            else:
                self.buffer.append(escape(value))

    def _finish_block(self, tag: str, markup: str) -> None:
        if tag == "pre":
            escaped = "<br/>".join(escape(line) for line in html.unescape(markup).splitlines())
            self.story.append(Paragraph(f'<font name="KGMono" size="7.7">{escaped}</font>', self.styles["code"]))
            return
        if tag == "li":
            style = ParagraphStyle(f"KGList{len(self.story)}", parent=self.styles["list"], leftIndent=13 + 12 * getattr(self, "_list_depth", 0))
            self.story.append(Paragraph(self._render_inline_tokens(markup, inline=True), style, bulletText=self.list_marker))
            return
        if tag.startswith("h"):
            level = int(tag[1])
            if level == 1 and not self.heading_seen:
                self.heading_seen = True
                self.story.append(Paragraph(self._render_inline_tokens(markup, inline=True), self.styles["title"]))
            else:
                style = self.styles["h2"] if level <= 2 else self.styles["h3"] if level == 3 else self.styles["h4"]
                self.story.append(Paragraph(self._render_inline_tokens(markup, inline=True), style))
            return
        text = plain_text(markup)
        style = self.styles["quote"] if self.quote_depth else self.styles["body"]
        if self.heading_seen and not self.meta_seen and text == "Utkarsh Sinha":
            style, self.meta_seen = self.styles["author"], True
        elif "Technical research manuscript / preprint draft" in text:
            style = self.styles["status"]
        elif text.startswith("Keywords:"):
            style = ParagraphStyle("KGKeywords", parent=self.styles["body"], fontSize=8.5, textColor=MUTED)
        elif text.startswith("Figure "):
            style = self.styles["caption"]
        for flowable in self._paragraph_parts(markup, style):
            self.story.append(flowable)

    def _paragraph_parts(self, markup: str, style: ParagraphStyle) -> list[object]:
        pattern = r"(KGMATHBLOCK\d+TOKEN|KGFIGURETOKEN\d+END)"
        parts = re.split(pattern, markup)
        result: list[object] = []
        for part in parts:
            if not part:
                continue
            if part in self.images:
                result.append(self._image_flowable(part))
            elif part in self.maths:
                formula, is_block = self.maths[part]
                try:
                    path = self.work / f"{part}.png"
                    width, height = render_math(formula, path, display=is_block)
                    self.math_rendered += 1
                    self.rendered_formulas.add(formula.strip())
                    if is_block:
                        scale = min(1, 500 / max(width, 1), 150 / max(height, 1))
                        result.append(Image(str(path), width=width * scale, height=height * scale, hAlign="CENTER"))
                    else:
                        scale = min(1, 190 / max(width, 1), 17 / max(height, 1))
                        result.append(Paragraph(f'<img src="{html.escape(str(path), quote=True)}" width="{width * scale:.2f}" height="{height * scale:.2f}" valign="middle"/>', style))
                except Exception as exc:
                    self.math_fallbacks.append(formula)
                    exact = r"\[" + formula + r"\]" if is_block else r"\(" + formula + r"\)"
                    fallback = f'<font name="KGMono" size="8.0">{escape(exact).replace(chr(10), "<br/>")}</font>'
                    result.append(Paragraph(fallback, self.styles["equation" if is_block else "body"]))
            else:
                rendered = inline_markup(self._render_inline_tokens(part, inline=True), self.resolve_link)
                if rendered.strip():
                    result.append(Paragraph(rendered, style))
        return result

    def _render_inline_tokens(self, markup: str, *, inline: bool) -> str:
        pattern = r"(KGMATH(?:BLOCK|INLINE)\d+TOKEN|KGFIGURETOKEN\d+END)"

        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            if token in self.images:
                path, width, height = self._image_data(token)
                return f'<img src="{html.escape(str(path), quote=True)}" width="{width:.2f}" height="{height:.2f}" valign="middle"/>'
            formula, is_block = self.maths[token]
            try:
                path = self.work / f"{token}.png"
                width, height = render_math(formula, path, display=is_block)
                self.math_rendered += 1
                self.rendered_formulas.add(formula.strip())
                if inline and not is_block:
                    scale = min(1, 185 / max(width, 1), 16 / max(height, 1))
                    return f'<img src="{html.escape(str(path), quote=True)}" width="{width * scale:.2f}" height="{height * scale:.2f}" valign="middle"/>'
                self.math_fallbacks.append(formula)
            except Exception:
                self.math_fallbacks.append(formula)
            exact = r"\[" + formula + r"\]" if is_block else r"\(" + formula + r"\)"
            return f'<font name="KGMono" size="8.0">{escape(exact)}</font>'

        return re.sub(pattern, replace, markup)

    def _image_data(self, token: str) -> tuple[Path, float, float]:
        source, alt = self.images[token]
        path = (SOURCE.parent / source).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise FileNotFoundError(f"Figure unavailable: {alt or source}")
        with PILImage.open(path) as image:
            pixel_width, pixel_height = image.size
        max_width, max_height = 510.0, 270.0
        scale = min(max_width / pixel_width, max_height / pixel_height)
        self.figure_hashes[path.name] = digest(path)
        return path, pixel_width * scale, pixel_height * scale

    def _image_flowable(self, token: str) -> object:
        path, width, height = self._image_data(token)
        return Image(str(path), width=width, height=height, hAlign="CENTER")

    def _add_image(self, token: str) -> None:
        self.story.append(self._image_flowable(token))

    def _add_table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        cells = [[Paragraph(inline_markup(cell, self.resolve_link), self.styles["table_header"] if row_index == 0 else self.styles["table"])
                  for cell in row] for row_index, row in enumerate(rows)]
        column_count = max(map(len, rows))
        widths: list[float] = []
        for column in range(column_count):
            longest = max((len(plain_text(row[column])) for row in rows if column < len(row)), default=8)
            widths.append(max(4, min(32, longest)) ** 0.8)
        available = letter[0] - 94
        total = sum(widths) or 1
        col_widths = [available * width / total for width in widths]
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD5E0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ]
        if column_count > 2:
            style.append(("ALIGN", (2, 1), (-1, -1), "RIGHT"))
        self.story.extend([LongTable(cells, colWidths=col_widths, repeatRows=1, hAlign="LEFT", splitByRow=1, style=TableStyle(style)), Spacer(1, 8)])


def preprocess_math(source: str, parser: PaperHTML) -> str:
    def substitute(pattern: str, block: bool, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            token = f"KGMATH{'BLOCK' if block else 'INLINE'}{len(parser.maths)}TOKEN"
            parser.maths[token] = (match.group(1), block)
            return token
        return re.sub(pattern, replace, text, flags=re.DOTALL)

    source = substitute(r"\\\[(.*?)\\\]", True, source)
    return substitute(r"\\\((.*?)\\\)", False, source)


def parse_result_table(tables: list[list[list[str]]]) -> list[list[str]]:
    for rows in tables:
        if rows and rows[0] and plain_text(rows[0][0]).strip().lower() == "workload":
            data = [[plain_text(cell) for cell in row] for row in rows[1:]]
            if len(data) != 6:
                raise AssertionError(f"Expected six result rows, found {len(data)}")
            return data
    raise AssertionError("The six-row workload results table is missing")


def validate_pdf(
    path: Path,
    result_rows: list[list[str]],
    expected_links: set[str],
    rendered_formulas: set[str],
) -> tuple[int, int]:
    import pymupdf

    doc = pymupdf.open(path)
    text = normalized(" ".join(page.get_text("text") for page in doc))
    required = (
        "technical research manuscript / preprint draft",
        "not peer-reviewed",
        "does not compare these polynomial cases with native cugraph",
        "one deliberately asymmetric sparse family",
        "all six selected profiles passed",
        "59/89 attempts when failures count as non-wins",
        "26/89 attempts) also had aggregate and maximum-relative errors no worse",
        "4 k cases lacked accepted accuracy",
        "first 16 rows",
        "source/product_v1",
        "19.98%–73.26% lower pooled-median latency",
        "24.98%–273.95% higher derived reciprocal-latency rate",
        "the latter is computed from latency ratios, not measured throughput",
        "floating checks were made at",
        "and both interval endpoints",
        "their timings replay",
        "sensitivity accuracy checks additionally include both interval endpoints",
    )
    missing = [phrase for phrase in required if normalized(phrase) not in text]
    if missing:
        raise AssertionError(f"PDF text misses required scope/status phrases: {missing}")
    if r"\theta=0" not in rendered_formulas:
        raise AssertionError("The PDF does not render the stated theta=0 timing scope")
    if "description: unreviewed technical manuscript" in text:
        raise AssertionError("YAML frontmatter leaked into the PDF body")
    for index, row in enumerate(result_rows, 1):
        missing_cells = [cell for cell in row if normalized(cell) and normalized(cell) not in text]
        if missing_cells:
            raise AssertionError(f"PDF text misses result row {index} cells: {missing_cells}")
    link_count = sum(len(page.get_links()) for page in doc)
    if link_count < 8:
        raise AssertionError(f"Expected at least eight reference links, found {link_count}")
    link_targets = {link.get("uri") for page in doc for link in page.get_links() if link.get("uri")}
    missing_links = expected_links - link_targets
    if missing_links:
        raise AssertionError(f"PDF does not contain the expected public links: {sorted(missing_links)}")
    for page_number, page in enumerate(doc, 1):
        bounds = page.rect
        if any(word[0] < 0 or word[1] < 0 or word[2] > bounds.width or word[3] > bounds.height for word in page.get_text("words")):
            raise AssertionError(f"Text is cropped outside PDF page {page_number}")
        if any(
            rect.x0 < 0 or rect.y0 < 0 or rect.x1 > bounds.width or rect.y1 > bounds.height
            for item in page.get_images(full=True)
            for rect in page.get_image_rects(item[0])
        ):
            raise AssertionError(f"An image is cropped outside PDF page {page_number}")
    return len(doc), link_count


def write_record(parser: PaperHTML, page_count: int, link_count: int) -> None:
    figures = "\n".join(
        f"- `docs/assets/{name}` — SHA-256 `{parser.figure_hashes[name]}`"
        for name in EXPECTED_FIGURES
    )
    fallback_list = "None." if not parser.math_fallbacks else "; ".join(
        f"`{' '.join(item.split())}`" for item in parser.math_fallbacks
    )
    report = f"""# PDF rendering record

`docs/research/k-graphfuse-paper.pdf` is rendered from `docs/research/paper.md` by `benchmarks/render_research_paper.py` with Python-Markdown {markdown.__version__}, Matplotlib {matplotlib.__version__} MathText, ReportLab {reportlab_version}, and the bundled DejaVu fonts. It is an independent Markdown typesetting of the same technical manuscript; it is not compiled from the standalone LaTeX source.

- Source SHA-256: `{digest(SOURCE)}`
- PDF SHA-256: `{digest(PDF)}`
- PDF pages: {page_count}
- Reference link annotations: {link_count}
- Relative Markdown links use the public site URL from `mkdocs.yml`:
{chr(10).join(f"  - `{source}` → `{target}`" for source, target in parser.relative_links.items())}
- Initial YAML frontmatter is omitted from page content and retained in the source hash.
- Result table: {len(parse_result_table(parser.tables))} data rows; all row cells verified in extracted PDF text
- Equations rendered with MathText: {parser.math_rendered}; monospaced fallback formulas: {len(parser.math_fallbacks)}
- Fallback formulas (whitespace compacted here; exact source formulas remain visible in monospaced PDF text): {fallback_list}
- Renderer font resources: DejaVu Serif, DejaVu Sans, DejaVu Sans Mono from Matplotlib's installed font bundle

Included figure inputs:
{figures}

Run from any working directory with `python benchmarks/render_research_paper.py` using the repository's installed work environment. The script reads the Markdown and local PNG assets only, rejects missing or out-of-repository figures, checks the six result rows and scope/status text in the finished PDF, and writes this provenance record. ReportLab's invariant PDF mode makes metadata timestamps stable; the manuscript's own date remains visible in the document.
"""
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(report, encoding="utf-8", newline="\n")


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Manuscript source not found: {SOURCE.relative_to(ROOT)}")
    for filename in EXPECTED_FIGURES:
        path = ROOT / "docs" / "assets" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Expected figure not found: docs/assets/{filename}")
    source_text = SOURCE.read_text(encoding="utf-8-sig")
    register_fonts()
    styles = make_styles()
    PDF.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kg-paper-math-") as temporary:
        work = Path(temporary)
        parser = PaperHTML(ROOT, work, styles)
        prepared = preprocess_math(strip_initial_frontmatter(source_text), parser)
        rendered_html = markdown.markdown(prepared, extensions=["tables", "fenced_code", "sane_lists"])
        parser.feed(rendered_html)
        parser.close()
        result_rows = parse_result_table(parser.tables)
        expected_links = {
            "k-graphfuse-paper.pdf": urljoin(parser.site_url, "research/k-graphfuse-paper.pdf"),
            "../benchmarks/metric-definitions.md": urljoin(parser.site_url, "benchmarks/metric-definitions/"),
        }
        if any(parser.relative_links.get(source) != target for source, target in expected_links.items()):
            raise AssertionError(f"Public PDF/metric links were not resolved as expected: {parser.relative_links}")
        if set(EXPECTED_FIGURES) - parser.figure_hashes.keys():
            raise AssertionError(f"Expected figures not present in manuscript: {sorted(set(EXPECTED_FIGURES) - parser.figure_hashes.keys())}")

        def decorate(canvas, document) -> None:
            canvas.saveState()
            canvas.setTitle("K GraphFuse: Similarity-Aware Sparse Polynomial Evaluation with Selective GPU Compensation")
            canvas.setAuthor("Utkarsh Sinha")
            canvas.setSubject("Technical research manuscript / preprint draft; not peer-reviewed")
            canvas.setKeywords("K GraphFuse, sparse graph filters, matrix polynomials, numerical analysis, CUDA")
            canvas.setCreator("K GraphFuse Markdown research-paper renderer")
            canvas.setFont("KGSans", 7.4)
            canvas.setFillColor(MUTED)
            canvas.drawString(47, letter[1] - 30, "K GraphFuse  |  Technical research manuscript")
            canvas.setStrokeColor(colors.HexColor("#CDD7E2"))
            canvas.setLineWidth(0.4)
            canvas.line(47, letter[1] - 37, letter[0] - 47, letter[1] - 37)
            canvas.line(47, 35, letter[0] - 47, 35)
            canvas.drawString(47, 23, "10 October 2026  |  Preprint draft")
            canvas.drawRightString(letter[0] - 47, 23, f"Page {document.page}")
            canvas.restoreState()

        class StableDoc(SimpleDocTemplate):
            def build(self, flowables, **kwargs):
                def stable_canvas(*args, **options):
                    options["invariant"] = 1
                    return Canvas(*args, **options)

                return super().build(flowables, canvasmaker=stable_canvas, **kwargs)

        doc = StableDoc(
            str(PDF), pagesize=letter, rightMargin=47, leftMargin=47,
            topMargin=54, bottomMargin=48, title="K GraphFuse research manuscript",
            author="Utkarsh Sinha", subject="Technical preprint draft; not peer-reviewed",
        )
        doc.build(parser.story, onFirstPage=decorate, onLaterPages=decorate)
        pages, links = validate_pdf(PDF, result_rows, set(expected_links.values()), parser.rendered_formulas)
        write_record(parser, pages, links)
        print(f"Rendered {PDF.relative_to(ROOT)}: {pages} pages; {len(result_rows)} benchmark rows; "
              f"{parser.math_rendered} equations rendered, {len(parser.math_fallbacks)} monospaced fallbacks; {links} links.")


if __name__ == "__main__":
    main()

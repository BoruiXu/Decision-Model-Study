"""A small stdlib Markdown to HTML converter for the dashboard's run-report page.

Supports #-#### headings, paragraphs, - / * lists, pipe tables, fenced code blocks,
`code`, **bold** and [text](target) links. All text is escaped first; a link target
is kept only when it is a relative path or http(s)://, otherwise only its text shows.
"""
import html
import re

_HEADING = re.compile(r"^(#{1,4})\s+(.*)$")
_ITEM = re.compile(r"^[-*]\s+(.*)$")
_SEP_CELL = re.compile(r"^:?-+:?$")
_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
# Neither part may contain "[", so each match attempt stops at the next "[" (linear time).
_LINK = re.compile(r"\[([^\[\]]*)\]\(([^)\s\[]*)\)")


def _heading_text(body):
    """Trim an optional closing # run, only when it follows whitespace (so `C#` keeps its #)."""
    body = body.rstrip()
    t = body.rstrip("#")
    if t != body and (not t or t[-1].isspace()):
        return t.rstrip()
    return body


def _safe_target(target):
    t = target.strip()
    if "\\" in t:
        return False
    if t.startswith("//"):
        return False
    if t.lower().startswith(("http://", "https://")):
        return True
    return ":" not in t.split("/")[0] and ":" not in t.split("?")[0].split("#")[0]


def _link(m):
    text, target = m.group(1), m.group(2)
    text = _BOLD.sub(r"<b>\1</b>", text)
    if _safe_target(html.unescape(target)):
        return '<a href="%s">%s</a>' % (target, text)
    return text


def _inline(text):
    text = html.escape(text, quote=True)
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for p in parts:
        if len(p) > 1 and p.startswith("`") and p.endswith("`"):
            out.append("<code>%s</code>" % p[1:-1])
        else:
            pos = 0
            for m in _LINK.finditer(p):
                out.append(_BOLD.sub(r"<b>\1</b>", p[pos:m.start()]))
                out.append(_link(m))
                pos = m.end()
            out.append(_BOLD.sub(r"<b>\1</b>", p[pos:]))
    return "".join(out)


def _cells(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells = re.split(r"(?<!\\)\|", s)
    return [c.strip().replace("\\|", "|") for c in cells]


def _is_row(line):
    return line.strip().startswith("|")


def _is_sep(line):
    cells = _cells(line)
    return bool(cells) and all(_SEP_CELL.match(c) for c in cells)


def _block_start(line):
    s = line.strip()
    return (not s or s.startswith("```") or _HEADING.match(s) or _ITEM.match(s) or _is_row(s))


def to_html(text):
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
        elif s.startswith("```"):
            i += 1
            code = []
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>%s</code></pre>" % html.escape("\n".join(code)))
        elif _HEADING.match(s):
            m = _HEADING.match(s)
            k = len(m.group(1))
            out.append("<h%d>%s</h%d>" % (k, _inline(_heading_text(m.group(2))), k))
            i += 1
        elif _ITEM.match(s):
            items = []
            while i < n and _ITEM.match(lines[i].strip()):
                items.append("<li>%s</li>" % _inline(_ITEM.match(lines[i].strip()).group(1)))
                i += 1
            out.append("<ul>%s</ul>" % "".join(items))
        elif _is_row(s):
            rows = []
            while i < n and _is_row(lines[i]):
                rows.append(lines[i])
                i += 1
            head = rows[0]
            body = rows[1:]
            if body and _is_sep(body[0]):
                body = body[1:]
            t = '<div class="tablewrap"><table><thead><tr>%s</tr></thead>' % "".join("<th>%s</th>" % _inline(c) for c in _cells(head))
            t += "<tbody>%s</tbody></table></div>" % "".join(
                "<tr>%s</tr>" % "".join("<td>%s</td>" % _inline(c) for c in _cells(r)) for r in body)
            out.append(t)
        else:
            para = []
            while i < n and not _block_start(lines[i]):
                para.append(lines[i].strip())
                i += 1
            out.append("<p>%s</p>" % _inline(" ".join(para)))
    return "\n".join(out) + ("\n" if out else "")

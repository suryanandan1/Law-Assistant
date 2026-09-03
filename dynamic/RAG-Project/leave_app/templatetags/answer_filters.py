import html
import re
from django import template
from django.utils.safestring import mark_safe

register = template.Library()


def _is_table_separator(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    return bool(cells) and all(re.fullmatch(r':?-{3,}:?', c or '') for c in cells)


def _split_table_row(line: str):
    return [html.escape(c.strip()) for c in line.strip().strip('|').split('|')]


def _render_table(lines):
    if len(lines) < 2 or not _is_table_separator(lines[1]):
        return None

    headers = _split_table_row(lines[0])
    rows = [_split_table_row(row) for row in lines[2:] if '|' in row]

    table = ['<div class="table-scroll"><table class="answer-table">']
    table.append('<thead><tr>')
    for header in headers:
        table.append(f'<th>{header}</th>')
    table.append('</tr></thead>')

    table.append('<tbody>')
    for row in rows:
        table.append('<tr>')
        for i in range(len(headers)):
            value = row[i] if i < len(row) else ''
            table.append(f'<td>{value}</td>')
        table.append('</tr>')
    table.append('</tbody></table></div>')
    return ''.join(table)


@register.filter(name='format_answer')
def format_answer(value):
    """Render simple markdown-like LLM answers, especially pipe tables, as clean HTML."""
    if not value:
        return ''

    text = str(value).replace('**', '')
    lines = text.splitlines()
    html_parts = []
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            i += 1
            continue

        # Markdown table block
        if '|' in line and i + 1 < len(lines) and _is_table_separator(lines[i + 1]):
            table_lines = [line, lines[i + 1]]
            i += 2
            while i < len(lines) and '|' in lines[i] and lines[i].strip():
                table_lines.append(lines[i])
                i += 1
            table_html = _render_table(table_lines)
            if table_html:
                html_parts.append(table_html)
            continue

        # Markdown headings like ### Summary
        heading_match = re.match(r'^(#{1,4})\s+(.*)$', line.strip())
        if heading_match:
            level = min(len(heading_match.group(1)) + 2, 6)
            content = html.escape(heading_match.group(2))
            html_parts.append(f'<h{level} class="answer-heading">{content}</h{level}>')
            i += 1
            continue

        # Bullet lines
        if line.strip().startswith(('- ', '* ')):
            items = []
            while i < len(lines) and lines[i].strip().startswith(('- ', '* ')):
                item = html.escape(lines[i].strip()[2:])
                items.append(f'<li>{item}</li>')
                i += 1
            html_parts.append('<ul class="answer-list">' + ''.join(items) + '</ul>')
            continue

        html_parts.append(f'<p>{html.escape(line)}</p>')
        i += 1

    return mark_safe('\n'.join(html_parts))

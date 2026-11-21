export type TableAlignment = 'left' | 'center' | 'right';

export type MarkdownBlock =
  | { type: 'heading'; level: 1 | 2 | 3 | 4; text: string }
  | { type: 'table'; headers: string[]; alignments: TableAlignment[]; rows: string[][] }
  | { type: 'ascii'; label: string; content: string }
  | { type: 'code'; language: string; content: string }
  | { type: 'list'; ordered: boolean; items: Array<{ text: string; checked?: boolean }> }
  | { type: 'blockquote'; lines: string[] }
  | { type: 'hr' }
  | { type: 'paragraph'; text: string };

export type InlineToken =
  | { type: 'text'; value: string }
  | { type: 'bold'; value: string }
  | { type: 'italic'; value: string }
  | { type: 'boldItalic'; value: string }
  | { type: 'strike'; value: string }
  | { type: 'code'; value: string }
  | { type: 'link'; label: string; href: string };

const ASCII_FENCE_LANGS = new Set([
  'ascii',
  'chart',
  'diagram',
  'art',
  'tree',
  'graph',
  'plot',
  'box',
  'txt',
  'text',
  'plaintext',
]);

const UNICODE_ASCII_ART_RE = /[┌┐└┘├┤┬┴┼─│═║╔╗╚╝╠╣╦╩╬█▓▒░▇▆▅▄▃▂▁■▪▬▲▼►◄]/;
const ASCII_BORDER_LINE_RE = /^\s*(\+[-=+]{2,}\+|\|.*\|)\s*$/;
const ASCII_BAR_LINE_RE = /^\s*[\w .:/()%-]{1,28}\s*[|│:]\s*[█▓▒░#=*■▪-]{3,}/;
const ASCII_TREE_LINE_RE = /^\s*(?:│\s*)*(?:├──|└──|\|--|\+--)/;

export function looksLikeAsciiVisual(content: string): boolean {
  if (UNICODE_ASCII_ART_RE.test(content)) return true;
  const lines = content.split('\n').filter((l) => l.trim().length > 0);
  if (lines.length < 2) return false;
  let visualLines = 0;
  for (const line of lines) {
    if (
      ASCII_BORDER_LINE_RE.test(line) ||
      ASCII_BAR_LINE_RE.test(line) ||
      ASCII_TREE_LINE_RE.test(line)
    ) {
      visualLines += 1;
    }
  }
  return visualLines >= 2;
}

function isAsciiArtLine(line: string): boolean {
  if (!line.trim()) return false;
  return (
    UNICODE_ASCII_ART_RE.test(line) ||
    ASCII_BORDER_LINE_RE.test(line) ||
    ASCII_BAR_LINE_RE.test(line) ||
    ASCII_TREE_LINE_RE.test(line)
  );
}

function splitTableRow(line: string): string[] {
  const trimmed = line.trim();
  const stripped = trimmed.replace(/^\|/, '').replace(/\|$/, '');
  const cells: string[] = [];
  let current = '';
  for (let i = 0; i < stripped.length; i += 1) {
    if (stripped[i] === '\\' && stripped[i + 1] === '|') {
      current += '|';
      i += 1;
    } else if (stripped[i] === '|') {
      cells.push(current.trim());
      current = '';
    } else {
      current += stripped[i];
    }
  }
  cells.push(current.trim());
  return cells;
}

function parseTableSeparator(line: string): TableAlignment[] | null {
  const trimmed = line.trim();
  if (!trimmed.includes('|') && !/^:?-+:?$/.test(trimmed)) return null;
  const cells = splitTableRow(trimmed);
  if (cells.length < 2) return null;
  const alignments: TableAlignment[] = [];
  for (const cell of cells) {
    const clean = cell.trim();
    if (!/^:?-{2,}:?$/.test(clean)) return null;
    if (clean.startsWith(':') && clean.endsWith(':')) {
      alignments.push('center');
    } else if (clean.endsWith(':')) {
      alignments.push('right');
    } else {
      alignments.push('left');
    }
  }
  return alignments;
}

function isPotentialTableStart(lines: string[], index: number): TableAlignment[] | null {
  if (index + 1 >= lines.length) return null;
  const headerLine = lines[index];
  const sepLine = lines[index + 1];
  if (!headerLine.includes('|')) return null;
  const alignments = parseTableSeparator(sepLine);
  if (!alignments) return null;
  const headers = splitTableRow(headerLine);
  if (headers.length < 2) return null;
  return alignments;
}

export function parseMarkdownBlocks(raw: string): MarkdownBlock[] {
  const normalized = raw.replace(/\r\n/g, '\n');
  const lines = normalized.split('\n');
  const blocks: MarkdownBlock[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i += 1;
      continue;
    }

    // 1. Fenced code / ASCII chart block
    const fenceMatch = trimmed.match(/^```([\w.+-]*)\s*$/);
    if (fenceMatch) {
      const lang = (fenceMatch[1] || '').toLowerCase();
      const bodyLines: string[] = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        bodyLines.push(lines[i]);
        i += 1;
      }
      if (i < lines.length && lines[i].trim().startsWith('```')) {
        i += 1;
      }
      const content = bodyLines.join('\n');
      if (ASCII_FENCE_LANGS.has(lang) || (!lang && looksLikeAsciiVisual(content))) {
        blocks.push({
          type: 'ascii',
          label: lang && lang !== 'txt' && lang !== 'text' ? lang.toUpperCase() : 'ASCII Chart / Diagram',
          content,
        });
      } else {
        blocks.push({
          type: 'code',
          language: lang || 'code',
          content,
        });
      }
      continue;
    }

    // 2. Horizontal rule
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
      blocks.push({ type: 'hr' });
      i += 1;
      continue;
    }

    // 3. Heading (# through ####)
    const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length as 1 | 2 | 3 | 4;
      blocks.push({ type: 'heading', level, text: headingMatch[2].trim() });
      i += 1;
      continue;
    }

    // 4. GFM Pipe Data Table
    const tableAlignments = isPotentialTableStart(lines, i);
    if (tableAlignments) {
      const headers = splitTableRow(lines[i]);
      const colCount = headers.length;
      const alignments: TableAlignment[] = Array.from(
        { length: colCount },
        (_, idx) => tableAlignments[idx] || 'left',
      );
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i].trim().includes('|') && !lines[i].trim().startsWith('```')) {
        const rowCells = splitTableRow(lines[i]);
        const normalizedRow = Array.from({ length: colCount }, (_, idx) => rowCells[idx] ?? '');
        rows.push(normalizedRow);
        i += 1;
      }
      blocks.push({ type: 'table', headers, alignments, rows });
      continue;
    }

    // 5. Unfenced ASCII chart / diagram / tree block (2+ consecutive visual lines)
    if (isAsciiArtLine(line) && i + 1 < lines.length && isAsciiArtLine(lines[i + 1])) {
      const artLines: string[] = [];
      while (i < lines.length && (isAsciiArtLine(lines[i]) || (lines[i].trim() === '' && i + 1 < lines.length && isAsciiArtLine(lines[i + 1])))) {
        artLines.push(lines[i]);
        i += 1;
      }
      blocks.push({
        type: 'ascii',
        label: 'ASCII Chart / Diagram',
        content: artLines.join('\n'),
      });
      continue;
    }

    // 6. Blockquote
    if (trimmed.startsWith('>')) {
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith('>')) {
        quoteLines.push(lines[i].trim().replace(/^>\s?/, ''));
        i += 1;
      }
      blocks.push({ type: 'blockquote', lines: quoteLines });
      continue;
    }

    // 7. Unordered / Checklist / Ordered Lists
    const unorderedMatch = line.match(/^\s*[-*•]\s+(.+)$/);
    const orderedMatch = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unorderedMatch || orderedMatch) {
      const ordered = Boolean(orderedMatch);
      const items: Array<{ text: string; checked?: boolean }> = [];
      while (i < lines.length) {
        const current = lines[i];
        const m = ordered
          ? current.match(/^\s*\d+[.)]\s+(.+)$/)
          : current.match(/^\s*[-*•]\s+(.+)$/);
        if (!m) break;
        const body = m[1].trim();
        const taskMatch = !ordered ? body.match(/^\[( |x|X)\]\s+(.+)$/) : null;
        if (taskMatch) {
          items.push({
            checked: taskMatch[1].toLowerCase() === 'x',
            text: taskMatch[2].trim(),
          });
        } else {
          items.push({ text: body });
        }
        i += 1;
      }
      blocks.push({ type: 'list', ordered, items });
      continue;
    }

    // 8. Paragraph (collect until blank line or next structural block)
    const paraLines: string[] = [];
    while (i < lines.length) {
      const cur = lines[i];
      const curTrim = cur.trim();
      if (!curTrim) break;
      if (
        curTrim.startsWith('```') ||
        /^#{1,4}\s+/.test(curTrim) ||
        /^(-{3,}|\*{3,}|_{3,})$/.test(curTrim) ||
        curTrim.startsWith('>') ||
        /^\s*[-*•]\s+/.test(cur) ||
        /^\s*\d+[.)]\s+/.test(cur) ||
        isPotentialTableStart(lines, i) ||
        (isAsciiArtLine(cur) && i + 1 < lines.length && isAsciiArtLine(lines[i + 1]))
      ) {
        break;
      }
      paraLines.push(cur);
      i += 1;
    }
    if (paraLines.length > 0) {
      blocks.push({ type: 'paragraph', text: paraLines.join('\n') });
    } else {
      // Safety guard against non-advancing lines
      blocks.push({ type: 'paragraph', text: line });
      i += 1;
    }
  }

  return blocks;
}

const INLINE_PATTERN =
  /(`[^`\n]+`|\*\*\*(?!\s)(?:[^*\n]*[^\s*])\*\*\*|\*\*(?!\s)(?:[^*\n]*[^\s*])\*\*|__(?!\s)(?:[^_\n]*[^\s_])__|~~(?!\s)(?:[^~\n]*[^\s~])~~|\*(?!\s)(?:[^*\n]*[^\s*])\*|(?<![a-zA-Z0-9])_(?!\s)(?:[^_\n]*[^\s_])_(?![a-zA-Z0-9])|\[[^\]\n]+\]\((?:https?:\/\/|\/)[^)\s]+\))/g;

export function parseInlineMarkdown(text: string): InlineToken[] {
  const tokens: InlineToken[] = [];
  let lastIndex = 0;
  INLINE_PATTERN.lastIndex = 0;

  let match: RegExpExecArray | null;
  while ((match = INLINE_PATTERN.exec(text)) !== null) {
    if (match.index > lastIndex) {
      tokens.push({ type: 'text', value: text.slice(lastIndex, match.index) });
    }
    const raw = match[0];
    if (raw.startsWith('`') && raw.endsWith('`')) {
      tokens.push({ type: 'code', value: raw.slice(1, -1) });
    } else if (raw.startsWith('***') && raw.endsWith('***')) {
      tokens.push({ type: 'boldItalic', value: raw.slice(3, -3) });
    } else if (
      (raw.startsWith('**') && raw.endsWith('**')) ||
      (raw.startsWith('__') && raw.endsWith('__'))
    ) {
      tokens.push({ type: 'bold', value: raw.slice(2, -2) });
    } else if (raw.startsWith('~~') && raw.endsWith('~~')) {
      tokens.push({ type: 'strike', value: raw.slice(2, -2) });
    } else if (
      (raw.startsWith('*') && raw.endsWith('*')) ||
      (raw.startsWith('_') && raw.endsWith('_'))
    ) {
      tokens.push({ type: 'italic', value: raw.slice(1, -1) });
    } else if (raw.startsWith('[')) {
      const linkMatch = raw.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      if (linkMatch) {
        tokens.push({ type: 'link', label: linkMatch[1], href: linkMatch[2] });
      } else {
        tokens.push({ type: 'text', value: raw });
      }
    } else {
      tokens.push({ type: 'text', value: raw });
    }
    lastIndex = INLINE_PATTERN.lastIndex;
  }

  if (lastIndex < text.length) {
    tokens.push({ type: 'text', value: text.slice(lastIndex) });
  }

  return tokens;
}

export function tableToCsv(headers: string[], rows: string[][]): string {
  const escapeCell = (val: string) => {
    const clean = val.replace(/\*\*/g, '').replace(/`/g, '');
    if (/[",\n]/.test(clean)) {
      return `"${clean.replace(/"/g, '""')}"`;
    }
    return clean;
  };
  return [
    headers.map(escapeCell).join(','),
    ...rows.map((r) => r.map(escapeCell).join(',')),
  ].join('\n');
}

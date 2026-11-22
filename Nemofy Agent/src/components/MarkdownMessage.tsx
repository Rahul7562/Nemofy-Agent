import { BarChart3, Check, CheckSquare, Copy, Square, Table as TableIcon } from 'lucide-react';
import { Fragment, useState } from 'react';
import {
  type InlineToken,
  type MarkdownBlock,
  type TableAlignment,
  parseInlineMarkdown,
  parseMarkdownBlocks,
  tableToCsv,
} from '../lib/markdownRenderer.ts';

function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // ignore clipboard denial in headless contexts
    }
  }

  return (
    <button
      type="button"
      onClick={() => void handleCopy()}
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium text-[var(--ink-soft)] hover:text-[var(--ink)] hover:bg-[var(--paper)] transition-colors cursor-pointer"
    >
      {copied ? <Check className="w-3 h-3 text-[var(--jade)]" /> : <Copy className="w-3 h-3" />}
      <span>{copied ? 'Copied' : label}</span>
    </button>
  );
}

function renderInlineTokens(tokens: InlineToken[], isUser: boolean) {
  return tokens.map((tok, idx) => {
    switch (tok.type) {
      case 'bold':
        return (
          <strong key={idx} className="font-semibold">
            {tok.value}
          </strong>
        );
      case 'italic':
        return (
          <em key={idx} className="italic">
            {tok.value}
          </em>
        );
      case 'boldItalic':
        return (
          <strong key={idx} className="font-semibold italic">
            {tok.value}
          </strong>
        );
      case 'strike':
        return (
          <del key={idx} className="line-through opacity-75">
            {tok.value}
          </del>
        );
      case 'code':
        return (
          <code
            key={idx}
            className={`px-1.5 py-0.5 rounded font-mono text-[12px] ${
              isUser
                ? 'bg-white/20 text-white'
                : 'bg-[var(--paper-sunk)] border border-[var(--rule)] text-[var(--signal)]'
            }`}
          >
            {tok.value}
          </code>
        );
      case 'link': {
        const isExternal = tok.href.startsWith('http') || tok.href.startsWith('//');
        return (
          <a
            key={idx}
            href={tok.href}
            target={isExternal ? '_blank' : undefined}
            rel={isExternal ? 'noopener noreferrer' : undefined}
            className={`underline underline-offset-2 ${
              isUser ? 'text-white hover:text-white/80' : 'text-[var(--signal)] hover:opacity-80'
            }`}
          >
            {tok.label}
          </a>
        );
      }
      default:
        return <Fragment key={idx}>{tok.value}</Fragment>;
    }
  });
}

function InlineText({ text, isUser }: { text: string; isUser: boolean }) {
  const lines = text.split('\n');
  return (
    <>
      {lines.map((line, i) => (
        <Fragment key={i}>
          {renderInlineTokens(parseInlineMarkdown(line), isUser)}
          {i < lines.length - 1 && <br />}
        </Fragment>
      ))}
    </>
  );
}

function alignClass(alignment: TableAlignment): string {
  if (alignment === 'center') return 'text-center';
  if (alignment === 'right') return 'text-right tabular-nums';
  return 'text-left';
}

function renderBlock(block: MarkdownBlock, idx: number, isUser: boolean) {
  switch (block.type) {
    case 'heading': {
      const sizeClass =
        block.level === 1
          ? 'text-lg font-semibold mt-4 mb-2'
          : block.level === 2
            ? 'text-base font-semibold mt-3.5 mb-1.5'
            : block.level === 3
              ? 'text-sm font-semibold mt-3 mb-1'
              : 'text-xs font-semibold uppercase tracking-wide mt-2.5 mb-1';
      return (
        <div
          key={idx}
          className={`font-display tracking-tight ${sizeClass} ${
            isUser ? 'text-white' : 'text-[var(--ink)]'
          }`}
        >
          <InlineText text={block.text} isUser={isUser} />
        </div>
      );
    }

    case 'paragraph':
      return (
        <p key={idx} className="leading-relaxed my-2 first:mt-0 last:mb-0">
          <InlineText text={block.text} isUser={isUser} />
        </p>
      );

    case 'hr':
      return (
        <hr
          key={idx}
          className={`my-3.5 border-t ${isUser ? 'border-white/25' : 'border-[var(--rule)]'}`}
        />
      );

    case 'blockquote':
      return (
        <blockquote
          key={idx}
          className={`my-2.5 pl-3.5 py-1 border-l-2 text-[13px] ${
            isUser
              ? 'border-white/50 text-white/90'
              : 'border-[var(--signal)]/50 bg-[var(--paper-sunk)]/60 rounded-r-lg pr-3 text-[var(--ink-soft)]'
          }`}
        >
          {block.lines.map((line, i) => (
            <p key={i} className="my-0.5 leading-relaxed">
              <InlineText text={line} isUser={isUser} />
            </p>
          ))}
        </blockquote>
      );

    case 'list': {
      const ListTag = block.ordered ? 'ol' : 'ul';
      const hasCheckboxes = block.items.some((item) => item.checked !== undefined);
      return (
        <ListTag
          key={idx}
          className={`my-2 space-y-1.5 ${
            hasCheckboxes
              ? 'list-none pl-0'
              : block.ordered
                ? 'list-decimal pl-5'
                : 'list-disc pl-5'
          }`}
        >
          {block.items.map((item, i) => (
            <li key={i} className="leading-relaxed">
              {item.checked !== undefined ? (
                <span className="inline-flex items-start gap-2">
                  {item.checked ? (
                    <CheckSquare className="w-4 h-4 text-[var(--jade)] shrink-0 mt-0.5" />
                  ) : (
                    <Square className="w-4 h-4 text-[var(--ink-faint)] shrink-0 mt-0.5" />
                  )}
                  <span className={item.checked ? 'line-through opacity-75' : ''}>
                    <InlineText text={item.text} isUser={isUser} />
                  </span>
                </span>
              ) : (
                <InlineText text={item.text} isUser={isUser} />
              )}
            </li>
          ))}
        </ListTag>
      );
    }

    case 'table':
      return (
        <div
          key={idx}
          className="my-3 rounded-lg border border-[var(--rule)] bg-[var(--paper)] overflow-hidden"
        >
          <div className="px-3 py-1.5 bg-[var(--paper-sunk)] border-b border-[var(--rule)] flex items-center justify-between gap-2">
            <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-[var(--ink-soft)]">
              <TableIcon className="w-3.5 h-3.5 text-[var(--signal)]" />
              <span>
                Data Table ({block.rows.length} {block.rows.length === 1 ? 'row' : 'rows'})
              </span>
            </span>
            <CopyButton text={tableToCsv(block.headers, block.rows)} label="Copy CSV" />
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="bg-[var(--paper-sunk)]/60 border-b border-[var(--rule)]">
                  {block.headers.map((header, hIdx) => (
                    <th
                      key={hIdx}
                      className={`px-3 py-2 font-semibold text-[var(--ink)] whitespace-nowrap ${alignClass(
                        block.alignments[hIdx] || 'left',
                      )}`}
                    >
                      <InlineText text={header} isUser={false} />
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--rule)]">
                {block.rows.map((row, rIdx) => (
                  <tr
                    key={rIdx}
                    className="hover:bg-[var(--paper-sunk)]/45 transition-colors"
                  >
                    {row.map((cell, cIdx) => (
                      <td
                        key={cIdx}
                        className={`px-3 py-2 text-[var(--ink-soft)] align-top ${alignClass(
                          block.alignments[cIdx] || 'left',
                        )}`}
                      >
                        <InlineText text={cell} isUser={false} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );

    case 'ascii':
      return (
        <div
          key={idx}
          className="my-3 rounded-lg border border-[var(--rule)] bg-[var(--paper-sunk)] overflow-hidden"
        >
          <div className="px-3 py-1.5 border-b border-[var(--rule)] flex items-center justify-between gap-2">
            <span className="inline-flex items-center gap-1.5 text-[11px] font-mono font-medium text-[var(--ink-soft)]">
              <BarChart3 className="w-3.5 h-3.5 text-[var(--signal)]" />
              <span>{block.label}</span>
            </span>
            <CopyButton text={block.content} />
          </div>
          <pre className="p-3.5 text-[12px] font-mono text-[var(--ink)] overflow-x-auto whitespace-pre leading-[1.35] tabular-nums">
            <code>{block.content}</code>
          </pre>
        </div>
      );

    case 'code':
      return (
        <div
          key={idx}
          className="my-3 rounded-lg border border-[var(--rule)] bg-[var(--paper-sunk)] overflow-hidden"
        >
          <div className="px-3 py-1.5 border-b border-[var(--rule)] flex items-center justify-between gap-2">
            <span className="text-[11px] font-mono text-[var(--ink-faint)] uppercase">
              {block.language}
            </span>
            <CopyButton text={block.content} />
          </div>
          <pre className="p-3.5 text-xs font-mono text-[var(--ink)] overflow-x-auto leading-relaxed">
            <code>{block.content}</code>
          </pre>
        </div>
      );
  }
}

export function MarkdownMessage({ content, isUser = false }: { content: string; isUser?: boolean }) {
  const blocks = parseMarkdownBlocks(content);
  return <div className="text-sm break-words">{blocks.map((b, i) => renderBlock(b, i, isUser))}</div>;
}

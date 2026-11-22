import assert from 'node:assert/strict';
import test from 'node:test';
import {
  looksLikeAsciiVisual,
  parseInlineMarkdown,
  parseMarkdownBlocks,
  tableToCsv,
} from './markdownRenderer.ts';

test('parseInlineMarkdown converts bold, italic, code, strike, and links cleanly', () => {
  const tokens = parseInlineMarkdown(
    'Hello **Nemo**, use `python3` with *clean* ~~old~~ [Nemofy](https://nemofy.tech) and snake_case_var.',
  );
  assert.ok(tokens.some((t) => t.type === 'bold' && t.value === 'Nemo'));
  assert.ok(tokens.some((t) => t.type === 'code' && t.value === 'python3'));
  assert.ok(tokens.some((t) => t.type === 'italic' && t.value === 'clean'));
  assert.ok(tokens.some((t) => t.type === 'strike' && t.value === 'old'));
  assert.ok(
    tokens.some((t) => t.type === 'link' && t.label === 'Nemofy' && t.href === 'https://nemofy.tech'),
  );
  // snake_case_var must remain plain text, not italicized
  assert.ok(tokens.some((t) => t.type === 'text' && t.value.includes('snake_case_var')));

  // Math multiplication asterisks must not swallow following **bold** spans
  const mathTokens = parseInlineMarkdown(
    'What is 17 * 23? Show the answer in **bold** and *a* note.',
  );
  assert.ok(mathTokens.some((t) => t.type === 'text' && t.value.includes('17 * 23')));
  assert.ok(mathTokens.some((t) => t.type === 'bold' && t.value === 'bold'));
  assert.ok(mathTokens.some((t) => t.type === 'italic' && t.value === 'a'));
});

test('parseMarkdownBlocks parses GFM data tables with alignments and CSV export', () => {
  const md = [
    '## Model Benchmark',
    '',
    '| Provider | Model | Latency (ms) |',
    '|:---------|:-----:|-------------:|',
    '| **River AI** | `DeepSeek-V4.1` | 420 |',
    '| OpenAI | `gpt-4o-mini` | 380 |',
  ].join('\n');

  const blocks = parseMarkdownBlocks(md);
  assert.equal(blocks.length, 2);
  assert.equal(blocks[0].type, 'heading');
  assert.equal(blocks[1].type, 'table');
  if (blocks[1].type === 'table') {
    assert.deepEqual(blocks[1].headers, ['Provider', 'Model', 'Latency (ms)']);
    assert.deepEqual(blocks[1].alignments, ['left', 'center', 'right']);
    assert.equal(blocks[1].rows.length, 2);
    const csv = tableToCsv(blocks[1].headers, blocks[1].rows);
    assert.match(csv, /Provider,Model,Latency \(ms\)/);
    assert.match(csv, /River AI,DeepSeek-V4.1,420/);
  }
});

test('parseMarkdownBlocks detects fenced and unfenced ASCII charts and checklists', () => {
  const md = [
    '```ascii',
    'Q1 | ████████ 40%',
    'Q2 | ██████████████ 70%',
    '```',
    '',
    'CPU  | ██████████ 50%',
    'MEM  | ████████████████ 80%',
    '',
    '- [x] Checked item',
    '- [ ] Pending item',
  ].join('\n');

  const blocks = parseMarkdownBlocks(md);
  assert.equal(blocks[0].type, 'ascii');
  assert.equal(blocks[1].type, 'ascii');
  assert.equal(blocks[2].type, 'list');
  if (blocks[2].type === 'list') {
    assert.equal(blocks[2].items[0].checked, true);
    assert.equal(blocks[2].items[1].checked, false);
  }
  assert.equal(looksLikeAsciiVisual('┌────┐\n│ OK │\n└────┘'), true);
});

test('parseInlineMarkdown rejects protocol-relative external URLs while keeping relative paths and https links (BUG-20)', () => {
  const protoRel = parseInlineMarkdown('Click [phish](//evil.example/login) safely.');
  assert.equal(
    protoRel.some((t) => t.type === 'link'),
    false,
  );

  const okLinks = parseInlineMarkdown('See [docs](/docs/guide) and [site](https://nemofy.tech).');
  assert.ok(okLinks.some((t) => t.type === 'link' && t.href === '/docs/guide'));
  assert.ok(okLinks.some((t) => t.type === 'link' && t.href === 'https://nemofy.tech'));
});

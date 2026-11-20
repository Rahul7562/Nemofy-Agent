import assert from 'node:assert/strict';
import test from 'node:test';
import {
  countWorkspaceItems,
  detectFileLanguage,
  filterWorkspaceTree,
  formatBytes,
  isValidWorkspacePath,
  normalizeWorkspacePath,
} from './filesystemUtils.ts';
import type { WorkspaceNode } from '../types/agent.ts';

test('normalizeWorkspacePath and isValidWorkspacePath reject path traversal', () => {
  assert.equal(normalizeWorkspacePath('/workspace/reports/summary.md'), 'reports/summary.md');
  assert.equal(isValidWorkspacePath('reports/summary.md'), true);
  assert.equal(isValidWorkspacePath('../etc/passwd'), false);
  assert.equal(isValidWorkspacePath(''), false);
});

test('filterWorkspaceTree preserves matching folders and nested files', () => {
  const tree: WorkspaceNode[] = [
    {
      name: 'scripts',
      path: 'scripts',
      type: 'folder',
      children: [
        { name: 'analyze.py', path: 'scripts/analyze.py', type: 'file', size: 120 },
        { name: 'notes.txt', path: 'scripts/notes.txt', type: 'file', size: 42 },
      ],
    },
    { name: 'README.md', path: 'README.md', type: 'file', size: 512 },
  ];
  const filtered = filterWorkspaceTree(tree, 'analyze');
  assert.equal(filtered.length, 1);
  assert.equal(filtered[0].name, 'scripts');
  assert.equal(filtered[0].children?.length, 1);
  assert.equal(filtered[0].children?.[0].name, 'analyze.py');

  const counts = countWorkspaceItems(tree);
  assert.deepEqual(counts, { files: 3, folders: 1 });
  assert.equal(detectFileLanguage('scripts/analyze.py'), 'Python');
  assert.equal(formatBytes(2048), '2.0 KB');
});

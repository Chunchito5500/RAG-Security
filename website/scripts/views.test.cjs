const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');

// Render the real presentation components without introducing a browser dependency.
const root = path.resolve(__dirname, '..');
const resolve = Module._resolveFilename;
Module._resolveFilename = function(request, parent, ...rest) {
  return resolve.call(this, request.startsWith('@/') ? path.join(root, request.slice(2)) : request, parent, ...rest);
};
for (const extension of ['.ts', '.tsx']) require.extensions[extension] = (module, filename) => {
  const output = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2017 },
  });
  module._compile(output.outputText, filename);
};
const { RetrievalEventDetail } = require('../components/retrieval-event-detail.tsx');
const { DocumentDetails } = require('../components/document-details.tsx');
const { ActivityTable } = require('../components/activity-table.tsx');
const { EvaluationResults } = require('../components/evaluation-workspace.tsx');
const { LoadState } = require('../components/retrieval-ui.tsx');
const { readableReason, isQuarantined, eventTime } = require('../lib/retrieval/presentation.ts');
const render = (component, props) => renderToStaticMarkup(React.createElement(component, props));

const event = {
  id: 'event-1', query: 'documents', occurredAt: '', mode: 'security', origin: 'backend',
  answer: { status: 'unavailable', text: null, citationIds: [] },
  documents: [
    { id: 'r1', documentId: 'DOC-WARN', title: null, trustScore: 0.7, status: 'flagged', decision: 'allowed', flagReason: 'Moderate trust score', excerpt: 'Eligible source passage.' },
    { id: 'r2', documentId: 'DOC-BLOCK', title: null, trustScore: 0.95, status: 'blocked', decision: 'excluded', quarantineStatus: 'flagged', flagReason: 'HIGH:(developer|system) (message|prompt)', excerpt: 'EXCLUDED_PAYLOAD_MUST_NOT_RENDER' },
  ],
};

test('retrieval view keeps warning eligibility separate from high-score quarantine', () => {
  const html = render(RetrievalEventDetail, { event });
  assert.ok(html.includes('Eligible source passage.'));
  assert.ok(!html.includes('EXCLUDED_PAYLOAD_MUST_NOT_RENDER'));
  assert.ok(html.includes('0.95'));
  assert.ok(html.includes('Allowed into context'));
  assert.ok(html.includes('Excluded from context'));
  assert.ok(html.includes('High-severity instruction override pattern detected.'));
  assert.ok(html.includes('Technical detection details'));
  assert.ok(html.includes('HIGH:(developer|system) (message|prompt)'));
  assert.ok(html.includes('DOC-WARN') && !html.includes('Untitled document'));
  const sequence = ['Retrieval summary', 'Trusted passages', 'Retrieved documents', 'Security decisions', 'Retrieval trace'].map(title => html.indexOf(`>${title}</h2>`));
  assert.ok(sequence.every((position, index) => position >= 0 && (!index || position > sequence[index - 1])));
});

test('historical metadata and empty retrieval do not imply sources or answers', () => {
  const historical = { ...event, mode: 'legacy', documents: [{ id: 'missing', documentId: 'doc7', trustScore: null, status: 'unknown', decision: 'allowed', excerpt: null }] };
  const html = render(RetrievalEventDetail, { event: historical });
  assert.ok(html.includes('doc7'));
  assert.ok(html.includes('Source text was not recorded'));
  assert.ok(html.includes('Time not recorded'));
  assert.ok(!html.includes('Untitled document'));
  assert.ok(render(RetrievalEventDetail, { event: { ...event, documents: [] } }).includes('No documents were retrieved'));
  assert.ok(eventTime('2026-01-01 12:00:00').includes('timezone not recorded'));
});

test('document details never infer quarantine or severity from a trust score', () => {
  const document = { document_id: 'DOC-ZERO', filename: 'notes.txt', trust_score: 0, decision: 'warn', security_status: 'flagged', quarantine_status: 'none', flag_reason: 'Moderate trust score' };
  assert.equal(isQuarantined(document), false);
  const html = render(DocumentDetails, { document });
  assert.ok(html.includes('Not quarantined'));
  assert.ok(html.includes('Eligible with warning'));
  assert.ok(html.includes('0.00'));
  assert.equal(readableReason(null), 'No reason recorded.');
});

test('activity links select the actual event and show separate blocked and allowed counts', () => {
  const html = render(ActivityTable, { events: [event] });
  assert.ok(html.includes('/activity?event=event-1'));
  assert.ok(html.includes('Blocked sources'));
  assert.ok(html.includes('Allowed'));
  assert.ok(render(ActivityTable, { events: [] }).includes('No retrieval events yet'));
});

test('evaluation preserves non-perfect results and unknown values', () => {
  const result = { source: 'saved.json', timestamp: null, total_documents_tested: 2, correctly_classified: 1, accuracy: 50, false_positives: 1, missed_malicious_documents: 0, quarantine_exclusion_passed: true, all_low_trust_test_passed: null,
    individual_test_results: [{ filename: 'sample.txt', expected: 'benign', trust_score: null, decision: 'quarantine', correct: false, entered_context: null }] };
  const html = render(EvaluationResults, { result });
  assert.ok(html.includes('50%'));
  assert.ok(!html.includes('100%'));
  assert.ok(html.includes('Not recorded'));
  assert.ok(html.includes('Passed'));
  assert.ok(html.includes('sample.txt'));
});

test('loading and connection errors remain explicit and offer retry', () => {
  assert.ok(render(LoadState, { loading: true, error: '', refresh() {} }).includes('role="status"'));
  const html = render(LoadState, { loading: false, error: 'Cannot reach the backend.', refresh() {} });
  assert.ok(html.includes('role="alert"'));
  assert.ok(html.includes('Cannot reach the backend.'));
  assert.ok(html.includes('Retry'));
});

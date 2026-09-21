const assert = require("node:assert/strict");
const { test, after } = require("node:test");
const { execFileSync } = require("node:child_process");
const { mkdtempSync, rmSync } = require("node:fs");
const { tmpdir } = require("node:os");
const path = require("node:path");

// Compile only the data boundary to a temporary directory using the installed
// TypeScript compiler. No test dependencies or generated project files needed.
const output = mkdtempSync(path.join(tmpdir(), "retrieval-tests-"));
after(() => rmSync(output, { recursive: true, force: true }));
execFileSync(process.execPath, [require.resolve("typescript/bin/tsc"),
  "lib/retrieval/demo-source.ts", "lib/retrieval/selectors.ts",
  "--outDir", output, "--module", "commonjs", "--moduleResolution", "node",
  "--target", "ES2017", "--strict", "--skipLibCheck",
], { cwd: path.resolve(__dirname, ".."), stdio: "pipe" });
const { createDemoEvent, createDemoDataSource, runDemoScenario } = require(path.join(output, "demo-source.js"));
const { demoScenarios } = require(path.join(output, "mock-scenarios.js"));
const { isFlaggedEvent, formatTrustScore, documentTitle } = require(path.join(output, "selectors.js"));
const create = (scenario, mode = "strict") => createDemoEvent(scenario, mode, "test-event", "2026-04-16T15:00:00Z");

test("normal retrieval is trusted, allowed, cited, and absent from flagged review", () => {
  const event = create("incident");
  assert.equal(isFlaggedEvent(event), false);
  assert.ok(event.documents.every(d => d.status === "trusted" && d.decision === "allowed"));
  assert.equal(event.answer.status, "available");
  assert.equal(event.answer.citationIds.length, 2);
});

test("mixed retrieval preserves flagged and blocked evidence for review", () => {
  const event = create("retention");
  assert.equal(isFlaggedEvent(event), true);
  assert.ok(event.documents.some(d => d.status === "trusted" && d.decision === "allowed"));
  assert.ok(event.documents.some(d => d.status === "flagged" && d.decision === "excluded"));
  assert.ok(event.documents.some(d => d.status === "blocked" && d.flagReason && d.documentId && d.source.id));
});

test("mode changes inclusion without hiding warnings or mutating fixture history", () => {
  const snapshot = JSON.stringify(demoScenarios);
  const strict = create("retention");
  const balanced = create("retention", "balanced");
  const observed = create("retention", "observe");
  assert.equal(balanced.documents.find(d => d.trustScore === 0.65).decision, "allowed");
  assert.equal(balanced.documents.find(d => d.trustScore === 0.12).decision, "excluded");
  assert.ok(observed.documents.every(d => d.decision === "allowed"));
  assert.equal(isFlaggedEvent(observed), true);
  assert.equal(strict.documents.find(d => d.trustScore === 0.65).decision, "excluded");
  assert.equal(JSON.stringify(demoScenarios), snapshot);
});

test("all-low-trust evidence withholds the answer; observation does not invent one", () => {
  const blocked = create("all-low");
  assert.ok(blocked.documents.every(d => d.decision === "excluded"));
  assert.equal(blocked.answer.status, "withheld");
  assert.deepEqual(blocked.answer.citationIds, []);
  const observed = create("all-low", "observe");
  assert.equal(observed.answer.status, "unavailable");
  assert.match(observed.answer.text, /No answer is available/);
});

test("missing and invalid metadata produce explicit fallbacks, including score zero", () => {
  const document = create("incomplete").documents[0];
  assert.equal(documentTitle(document), "Untitled document");
  assert.equal(formatTrustScore(document.trustScore), "Not available");
  assert.equal(document.flagReason, null);
  for (const score of [-1, 1.01, NaN, Infinity, undefined]) assert.equal(formatTrustScore(score), "Not available");
  assert.equal(formatTrustScore(0), "0.00");
  assert.equal(formatTrustScore(1), "1.00");
  assert.equal(create("incomplete", "balanced").documents[0].decision, "excluded");
});

test("all fixtures and modes maintain stable references and valid citation decisions", () => {
  for (const scenario of demoScenarios) for (const mode of ["strict", "balanced", "observe"]) {
    const event = create(scenario.id, mode);
    assert.equal(new Set(event.documents.map(d => d.id)).size, event.documents.length);
    for (const id of event.answer.citationIds) assert.ok(event.documents.some(d => d.id === id && d.decision === "allowed"));
    for (const document of event.documents) {
      assert.ok(document.trustScore === null || (document.trustScore >= 0 && document.trustScore <= 1));
      if (document.status === "blocked") assert.equal(document.decision, "excluded");
    }
  }
  assert.deepEqual(create("no-matches").documents, []);
  assert.equal(create("no-matches").answer.status, "unavailable");
});

test("data source supports empty, no-flag, failure, recovery, and new distinct events", async () => {
  assert.deepEqual(await createDemoDataSource("empty").listEvents(), []);
  const trusted = await createDemoDataSource("trusted-only").listEvents();
  assert.ok(trusted.length > 0 && trusted.every(event => !isFlaggedEvent(event)));
  await assert.rejects(createDemoDataSource("error").listEvents(), /Retrieval history could not be loaded/);
  const recovered = await createDemoDataSource("scenarios").listEvents();
  assert.equal(recovered.length, demoScenarios.length);
  const first = await runDemoScenario("retention", "strict");
  const second = await runDemoScenario("retention", "observe");
  assert.notEqual(first.id, second.id);
  assert.equal(first.query, second.query);
  assert.throws(() => create("missing"), /unavailable/);
});

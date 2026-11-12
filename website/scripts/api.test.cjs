const assert = require("node:assert/strict");
const { test, after } = require("node:test");
const { execFileSync } = require("node:child_process");
const { mkdtempSync, rmSync } = require("node:fs");
const { tmpdir } = require("node:os");
const path = require("node:path");

const output = mkdtempSync(path.join(tmpdir(), "retrieval-api-tests-"));
after(() => rmSync(output, { recursive: true, force: true }));
execFileSync(process.execPath, [require.resolve("typescript/bin/tsc"), "lib/retrieval/api-source.ts",
  "--outDir", output, "--module", "commonjs", "--moduleResolution", "node",
  "--target", "ES2017", "--strict", "--skipLibCheck",
], { cwd: path.resolve(__dirname, ".."), stdio: "pipe" });
const { api, toRetrievalEvent } = require(path.join(output, "api-source.js"));

test("API adapter preserves backend decisions without generating an answer", () => {
  const event = toRetrievalEvent({ retrieval_event_id: "test", origin: "file_pipeline", query: "test",
    timestamp: null, retrieved_documents: [
      { document_id: "a", trust_score: 0.7, security_status: "flagged", inclusion: "allowed", excerpt: "Real passage" },
      { document_id: "b", trust_score: 0.99, security_status: "blocked", inclusion: "excluded", flag_reason: "Injection" },
      { document_id: null, trust_score: 5, security_status: null, inclusion: null },
    ],
  });
  assert.equal(event.documents[0].decision, "allowed");
  assert.equal(event.documents[1].decision, "excluded");
  assert.equal(event.documents[1].trustScore, 0.99);
  assert.equal(event.documents[2].trustScore, null);
  assert.equal(event.documents[2].status, "unknown");
  assert.equal(event.documents[2].source, null);
  assert.equal(event.answer.status, "unavailable");
  assert.equal(event.answer.text, null);
  assert.equal(event.mode, "security");
  assert.equal(new Set(event.documents.map(d => d.id)).size, 3);
});

test("API client exposes network, HTTP, malformed response, and empty states", async (t) => {
  t.mock.method(globalThis, "fetch", async () => { throw new TypeError("fetch failed"); });
  await assert.rejects(api.listEvents(), /Cannot reach the backend/);
  globalThis.fetch = async () => new Response(JSON.stringify({ error: "Document is invalid" }), { status: 422 });
  await assert.rejects(api.query("test", 3), /Document is invalid/);
  globalThis.fetch = async () => new Response("invalid JSON");
  await assert.rejects(api.listEvents(), /invalid response/);
  globalThis.fetch = async () => new Response(JSON.stringify({ data: [] }));
  assert.deepEqual(await api.listEvents(), []);
  globalThis.fetch = async () => new Response(JSON.stringify({ data: [{ query: "missing id" }] }));
  await assert.rejects(api.listEvents(), /invalid retrieval event/);
});

test("query and upload send user input to the API", async (t) => {
  const calls = [];
  t.mock.method(globalThis, "fetch", async (url, options) => {
    calls.push({ url, options });
    return new Response(JSON.stringify({ data: { retrieval_event_id: "real", retrieved_documents: [] } }));
  });
  await api.query("arbitrary user question", 5);
  assert.deepEqual(JSON.parse(calls[0].options.body), { query: "arbitrary user question", top_k: 5 });
  const file = new File(["hello"], "notes with spaces.txt");
  await api.upload(file);
  assert.match(calls[1].url, /filename=notes%20with%20spaces.txt/);
  assert.equal(calls[1].options.body, file);
});

test("simultaneous reads share a request but refresh gets fresh backend data", async (t) => {
  let calls = 0;
  let complete;
  t.mock.method(globalThis, "fetch", () => {
    calls += 1;
    return new Promise(resolve => { complete = () => resolve(new Response(JSON.stringify({ data: [] }))); });
  });
  const first = api.listEvents();
  const second = api.listEvents();
  assert.equal(calls, 1);
  complete();
  await Promise.all([first, second]);
  const refreshed = api.listEvents();
  assert.equal(calls, 2);
  complete();
  await refreshed;
});

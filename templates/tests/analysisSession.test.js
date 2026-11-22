import test from "node:test";
import assert from "node:assert/strict";
import { loadPredictionModel } from "../src/analysisSession.js";

const response = (status, data) => ({ status, ok: status >= 200 && status < 300, json: async () => data });

test("model loads automatically with bounded exponential backoff", async () => {
  const replies = [response(202, {status: "training", retryable: true}), response(503, {}),
    response(200, {status: "ready", features: [{name: "age"}]})];
  const delays = [];
  const data = await loadPredictionModel(async () => replies.shift(), {
    signal: new AbortController().signal, wait: async ms => delays.push(ms),
  });
  assert.equal(data.status, "ready");
  assert.deepEqual(delays, [300, 600]);
});

test("pending review stops backoff and approval loads immediately", async () => {
  const opts = {signal: new AbortController().signal, wait: async () => assert.fail("unexpected retry")};
  const pending = await loadPredictionModel(async () => response(409, {status: "pending_review"}), opts);
  assert.equal(pending.status, "pending_review");
  const approved = await loadPredictionModel(async () => response(200, {status: "ready"}), opts);
  assert.equal(approved.status, "ready");
});

test("dataset switch discards responses from the old request", async () => {
  const controller = new AbortController();
  const obsolete = await loadPredictionModel(async () => {
    controller.abort();
    return response(200, {status: "ready", target: "old"});
  }, {signal: controller.signal, wait: async () => {}});
  assert.equal(obsolete, null);
});

test("unavailable artifacts stop, transient errors exhaust the attempt budget", async () => {
  let calls = 0;
  const signal = new AbortController().signal;
  const unavailable = await loadPredictionModel(async () => response(409, {status: "artifact_unavailable"}), {
    signal, wait: async () => assert.fail("artifact should not retry"),
  });
  assert.equal(unavailable.status, "artifact_unavailable");
  const failure = await loadPredictionModel(async () => { calls++; return response(503, {}); }, {
    signal, wait: async () => {}, maxAttempts: 3,
  });
  assert.equal(failure.status, "unavailable");
  assert.equal(calls, 3);
});

/**
 * Q1-Q9 전체 평가: 검색 → Ollama 답변 → Ollama 판정
 * IPEX-LLM Ollama가 localhost:11434에서 실행 중이어야 한다.
 */
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { pipeline, env } from "@huggingface/transformers";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
env.cacheDir = path.join(os.homedir(), ".cache", "huggingface-transformers");

const VECTOR_TOP = 5;
const BM25_TOP = 3;
const MODEL = "qwen2.5:3b";

const docs = JSON.parse(fs.readFileSync(path.join(ROOT, "app", "public", "ismsp-docs.json"), "utf-8"));
const evalSet = JSON.parse(fs.readFileSync(path.join(ROOT, "eval", "questions.json"), "utf-8"));

const extractor = await pipeline("feature-extraction", "onnx-community/embeddinggemma-300m-ONNX", { dtype: "q4" });

const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

// --- BM25 ---
function tokenize(text) {
  const items = text.match(/\d{1,2}(?:\.\d{1,2}){1,2}/g) || [];
  const rest = text.replace(/\d{1,2}(?:\.\d{1,2}){1,2}/g, " ")
    .toLowerCase().split(/[^a-z0-9가-힣]+/).filter(t => t.length > 1);
  return [...items, ...rest];
}

const docTokens = docs.map(d => tokenize(d.text));
const avgDl = docTokens.reduce((s, t) => s + t.length, 0) / docTokens.length;
const df = {};
for (const toks of docTokens) {
  for (const t of new Set(toks)) df[t] = (df[t] || 0) + 1;
}
const N = docs.length;

function bm25Score(queryTokens, docIdx) {
  const toks = docTokens[docIdx];
  const tf = {};
  for (const t of toks) tf[t] = (tf[t] || 0) + 1;
  let score = 0;
  const k1 = 1.2, b = 0.75;
  for (const qt of queryTokens) {
    if (!tf[qt]) continue;
    const idf = Math.log((N - (df[qt] || 0) + 0.5) / ((df[qt] || 0) + 0.5) + 1);
    score += idf * (tf[qt] * (k1 + 1)) / (tf[qt] + k1 * (1 - b + b * toks.length / avgDl));
  }
  return score;
}

// --- Retrieval ---
async function retrieve(query) {
  const res = await extractor(query, { pooling: "mean", normalize: true });
  const qv = Array.from(res.data);

  const vecScored = docs.map((d, i) => ({ idx: i, score: dot(qv, d.vector) }))
    .sort((a, b) => b.score - a.score).slice(0, VECTOR_TOP);

  const qTokens = tokenize(query);
  const bm25Scored = docs.map((d, i) => ({ idx: i, score: bm25Score(qTokens, i) }))
    .sort((a, b) => b.score - a.score);

  const seen = new Set(vecScored.map(v => v.idx));
  const bm25Picks = [];
  for (const b of bm25Scored) {
    if (seen.has(b.idx) || b.score <= 0) continue;
    bm25Picks.push(b);
    seen.add(b.idx);
    if (bm25Picks.length >= BM25_TOP) break;
  }

  const hits = [
    ...vecScored.map(v => ({ chunk: docs[v.idx], score: v.score, method: "vector" })),
    ...bm25Picks.map(b => ({ chunk: docs[b.idx], score: b.score, method: "bm25" })),
  ];
  return hits;
}

// --- Prompt ---
function buildPrompt(question, hits) {
  const ctx = hits.map(h => `[${h.chunk.id}] ${h.chunk.text}`).join("\n\n");
  return `아래 근거만으로 질문에 답하세요. 근거에 없는 내용은 지어내지 마세요.\n답변에 사용한 근거의 [ID]를 반드시 표시하세요.\n\n--- 근거 ---\n${ctx}\n--- 끝 ---\n\n질문: ${question}`;
}

// --- Ollama API ---
async function ollamaChat(messages) {
  const res = await fetch("http://localhost:11434/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: MODEL, messages, stream: false, think: false }),
  });
  if (!res.ok) throw new Error(`ollama ${res.status}`);
  const j = await res.json();
  return {
    text: j.message?.content || "",
    evalCount: j.eval_count || 0,
    evalDuration: j.eval_duration || 0,
  };
}

async function ollamaJudge(prompt) {
  const res = await fetch("http://localhost:11434/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      stream: false, think: false, format: "json",
      options: { temperature: 0 },
    }),
  });
  if (!res.ok) throw new Error(`ollama judge ${res.status}`);
  const j = await res.json();
  return j.message?.content || "";
}

// --- Judge rubrics (simplified) ---
async function judgeAnswer(question, sources, answer) {
  const rubrics = [
    { name: "근거충실성", prompt: `질문: ${question}\n근거:\n${sources}\n답변: ${answer}\n\n이 답변이 주어진 근거에만 기반하는지 평가하세요. JSON으로 {"score": 0~100, "comment": "한줄평"} 형태로 답하세요.` },
    { name: "환각통제", prompt: `질문: ${question}\n근거:\n${sources}\n답변: ${answer}\n\n이 답변에 근거에 없는 내용(환각)이 포함되어 있는지 평가하세요. 환각이 없으면 100점, 있으면 감점. JSON으로 {"score": 0~100, "comment": "한줄평"} 형태로 답하세요.` },
    { name: "출처표시", prompt: `질문: ${question}\n답변: ${answer}\n\n이 답변에 출처 ID(예: [IC-1.1.1], [AR-32])가 올바르게 표시되어 있는지 평가하세요. JSON으로 {"score": 0~100, "comment": "한줄평"} 형태로 답하세요.` },
  ];

  const results = [];
  for (const r of rubrics) {
    try {
      const raw = await ollamaJudge(r.prompt);
      const parsed = JSON.parse(raw);
      results.push({ name: r.name, score: parsed.score ?? 0, comment: parsed.comment ?? "" });
    } catch {
      results.push({ name: r.name, score: -1, comment: "parse error" });
    }
  }
  const avg = results.filter(r => r.score >= 0).reduce((s, r) => s + r.score, 0) / results.filter(r => r.score >= 0).length;
  return { rubrics: results, score: Math.round(avg) };
}

// --- Main ---
console.log("=== ISMS-P 챗봇 Q1-Q9 전체 평가 ===");
console.log(`모델: ${MODEL} (IPEX-LLM)\n`);

const allResults = [];

for (const q of evalSet.questions) {
  console.log(`\n[${q.qid}] "${q.text}"`);
  console.log(`  유형: ${q.type}${q.subtype ? ` · ${q.subtype}` : ""}`);

  const hits = await retrieve(q.text);
  const hitIds = hits.map(h => `${h.chunk.id}(${h.method[0]})`).join(", ");
  console.log(`  검색: ${hitIds}`);

  const expectedHit = q.expect_chunks.filter(id => hits.some(h => h.chunk.id === id));
  console.log(`  기대 근거 적중: ${expectedHit.length}/${q.expect_chunks.length}`);

  const sysMsg = "당신은 ISMS-P 인증기준 안내 도우미입니다. 「정보보호 및 개인정보보호 관리체계 인증 등에 관한 고시」 원문에 근거한 내용만 답하고, 자료에 없는 정보는 '제가 가진 자료에는 없습니다'라고 답합니다. 근거 조각의 [ID]를 답에 표시합니다.";
  const userMsg = buildPrompt(q.text, hits);

  const t0 = Date.now();
  const ans = await ollamaChat([
    { role: "system", content: sysMsg },
    { role: "user", content: userMsg },
  ]);
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  const tps = ans.evalDuration > 0 ? (ans.evalCount / (ans.evalDuration / 1e9)).toFixed(1) : "?";
  console.log(`  답변 (${elapsed}s, ${tps} tok/s): ${ans.text.slice(0, 200)}...`);

  const sources = hits.map(h => `[${h.chunk.id}] ${h.chunk.text}`).join("\n");
  const judge = await judgeAnswer(q.text, sources, ans.text);
  console.log(`  판정: ${judge.score}점 — ${judge.rubrics.map(r => `${r.name} ${r.score}`).join(" · ")}`);

  const hasCitation = /\[(?:IC|AR)-[\d.]+(?:-\d+)?\]/.test(ans.text);
  const hasRefusal = /자료에.*없|답할 수 없|제공하지 않|범위.*벗어|도움.*드리기 어렵/i.test(ans.text);

  allResults.push({
    qid: q.qid, type: q.type, question: q.text,
    expectedChunks: q.expect_chunks, hitChunks: expectedHit,
    cited: hasCitation, refusal: hasRefusal,
    score: judge.score, rubrics: judge.rubrics,
    answer: ans.text.slice(0, 500),
  });
}

console.log("\n" + "=".repeat(60));
console.log("=== 요약 ===\n");

const cited = allResults.filter(r => ["정상", "경계"].includes(r.type) && r.cited).length;
const refusalOk = allResults.filter(r => r.type === "무근거" && r.refusal).length;
const grounded = allResults.filter(r => r.score >= 70).length;

console.log(`R1 출처 인용 (정상+경계 중): ${cited}/6`);
console.log(`R2 정당한 거부 (무근거 중): ${refusalOk}/3`);
console.log(`R3 근거성 (70점 이상): ${grounded}/9`);
console.log(`R4 평균 점수: ${Math.round(allResults.reduce((s, r) => s + r.score, 0) / allResults.length)}점`);

const outputPath = path.join(ROOT, "eval", "results-ipex.json");
fs.writeFileSync(outputPath, JSON.stringify(allResults, null, 2));
console.log(`\n상세 결과: ${outputPath}`);

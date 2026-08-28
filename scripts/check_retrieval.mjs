/**
 * 고정 질문 세트로 검색을 스팟체크한다.
 *
 * 생성 명령이 성공했다는 메시지만으로 벡터스토어가 옳다고 볼 수 없다.
 * 실제 질문에서 기대한 청크가 위로 올라오는지, 자료 밖 질문에서 억지로
 * 관련 청크가 정답처럼 보이지는 않는지를 눈으로 확인한다.
 *
 * 여기서는 벡터(코사인) 검색만 본다. BM25 보강은 앱에서 붙는다.
 */
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { pipeline, env } from "@huggingface/transformers";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
env.cacheDir = path.join(os.homedir(), ".cache", "huggingface-transformers");

const WEAK = 0.55; // 약한 근거 임계값 — 차단선이 아니라 말투를 바꾸는 신호
const TOPN = 3;

const docs = JSON.parse(fs.readFileSync(path.join(ROOT, "app", "public", "ismsp-docs.json"), "utf-8"));
const set = JSON.parse(fs.readFileSync(path.join(ROOT, "eval", "questions.json"), "utf-8"));

// 문서 벡터와 같은 조건. 다른 조건으로 질의를 만들면 비교할 자리를 잃는다.
const extractor = await pipeline("feature-extraction", "onnx-community/embeddinggemma-300m-ONNX", { dtype: "q4" });

const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

let pass = 0;
const failures = [];

for (const q of set.questions) {
  const res = await extractor(q.text, { pooling: "mean", normalize: true });
  const qv = Array.from(res.data);

  const ranked = docs
    .map((d) => ({ id: d.id, section: d.section, score: dot(qv, d.vector) }))
    .sort((a, b) => b.score - a.score);
  const top = ranked.slice(0, TOPN);
  const best = top[0].score;

  console.log(`\n[${q.qid}] ${q.type}${q.subtype ? ` · ${q.subtype}` : ""}  "${q.text}"`);
  for (const [i, r] of top.entries()) {
    const mark = q.expect_chunks.includes(r.id) ? " ←기대" : "";
    console.log(`   ${i + 1}. ${r.score.toFixed(3)}  ${r.id}  ${r.section.split(" > ").pop()}${mark}`);
  }

  if (q.type === "무근거" && q.subtype === "자료 밖") {
    const ok = best < WEAK;
    console.log(`   → 최고 유사도 ${best.toFixed(3)} ${ok ? `< ${WEAK} · 약한 근거로 표시됨 (기대대로)` : `≥ ${WEAK} · 근거가 있는 것처럼 보임 (확인 필요)`}`);
    ok ? pass++ : failures.push(`${q.qid} 자료 밖 질문인데 유사도 ${best.toFixed(3)}`);
    continue;
  }

  if (q.expect_chunks.length === 0) {
    console.log(`   → 기대 근거 없음. 최고 유사도 ${best.toFixed(3)}`);
    pass++;
    continue;
  }

  const hit = q.expect_chunks.filter((id) => top.some((t) => t.id === id));
  const ok = hit.length > 0;
  console.log(`   → 기대 ${q.expect_chunks.length}개 중 top-${TOPN} 안에 ${hit.length}개 ${ok ? "" : "✗"}`);
  ok ? pass++ : failures.push(`${q.qid} 기대 청크가 top-${TOPN}에 없음 (기대: ${q.expect_chunks.join(", ")})`);
}

console.log(`\n${"=".repeat(60)}`);
console.log(`통과 ${pass}/${set.questions.length}`);
if (failures.length) {
  console.log("\n확인할 것:");
  for (const f of failures) console.log(`  - ${f}`);
  console.log("\n임계값부터 만지지 말고, 질문과 문서가 같은 임베딩 경로를 지났는지 →");
  console.log("청크의 사실 경계가 적절한지 → BM25가 정확한 표기를 보충하는지 순으로 본다.");
}

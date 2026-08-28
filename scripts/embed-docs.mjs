/**
 * docs/chunks.json 의 청크에 768차원 벡터를 붙여 정적 벡터스토어를 만든다.
 *
 * 문서 벡터와 질의 벡터는 반드시 같은 공간이어야 하므로,
 * 브라우저 질의 임베딩과 아래 조건을 똑같이 맞춘다.
 *   모델      onnx-community/embeddinggemma-300m-ONNX
 *   dtype     q4
 *   pooling   mean
 *   normalize true  (정규화하면 내적이 곧 코사인 유사도가 된다)
 *
 * 출력: app/public/ismsp-docs.json
 */
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { pipeline, env } from "@huggingface/transformers";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

// 모델 캐시는 저장소 밖 홈 폴더에 둔다. 저장소를 통째로 옮겨도 캐시가 따라다니지 않게.
env.cacheDir = path.join(os.homedir(), ".cache", "huggingface-transformers");

const MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX";
const DIM = 768;

const IN = path.join(ROOT, "docs", "chunks.json");
const OUT_DIR = path.join(ROOT, "app", "public");
const OUT = path.join(OUT_DIR, "ismsp-docs.json");

const chunks = JSON.parse(fs.readFileSync(IN, "utf-8"));
console.log(`청크 ${chunks.length}개 로드: ${IN}`);
console.log(`모델 캐시: ${env.cacheDir}`);
console.log("모델 로드 중... (첫 실행은 내려받느라 오래 걸립니다)");

const extractor = await pipeline("feature-extraction", MODEL_ID, { dtype: "q4" });
console.log("모델 준비 완료. 임베딩 시작.");

const out = [];
for (let i = 0; i < chunks.length; i++) {
  const c = chunks[i];
  const res = await extractor(c.text, { pooling: "mean", normalize: true });
  const vector = Array.from(res.data);

  if (vector.length !== DIM) {
    throw new Error(`${c.id}: 차원이 ${vector.length} — ${DIM}이어야 합니다. 모델/설정을 확인하세요.`);
  }
  out.push({ id: c.id, text: c.text, url: c.url, section: c.section, vector });

  if ((i + 1) % 20 === 0 || i === chunks.length - 1) {
    console.log(`  ${i + 1}/${chunks.length}`);
  }
}

// 검증: 전 벡터가 768차원이고, 정규화되어 길이가 1인지 확인한다.
const badDim = out.filter((c) => c.vector.length !== DIM).map((c) => c.id);
const badNorm = out
  .map((c) => [c.id, Math.hypot(...c.vector)])
  .filter(([, n]) => Math.abs(n - 1) > 1e-3);

if (badDim.length) throw new Error(`차원 불량: ${badDim.join(", ")}`);
if (badNorm.length) {
  throw new Error(`정규화 불량: ${badNorm.map(([id, n]) => `${id}(${n.toFixed(4)})`).join(", ")}`);
}

fs.mkdirSync(OUT_DIR, { recursive: true });
fs.writeFileSync(OUT, JSON.stringify(out), "utf-8");

const mb = (fs.statSync(OUT).size / 1024 / 1024).toFixed(2);
console.log(`\n검증 통과: ${out.length}개 전부 ${DIM}차원, L2 정규화됨`);
console.log(`저장: ${OUT} (${mb} MB)`);

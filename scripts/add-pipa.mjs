// 개인정보 보호법 → 청크 변환 스크립트
// 사용법: node scripts/add-pipa.mjs
import { readFileSync, writeFileSync } from "fs";
import { join } from "path";

const TEMP = process.env.TEMP || "/tmp";
const raw = JSON.parse(readFileSync(join(TEMP, "pipa.json"), "utf8"));
const arts = raw.법령.조문.조문단위;

const LAW_NAME = "개인정보 보호법";
const LAW_URL = "https://www.law.go.kr/법령/개인정보보호법";

// 장 제목 추적
let currentChapter = "";

function buildText(art) {
  let text = (art.조문내용 || "").replace(/<개정[^>]*>/g, "").trim();

  // 항 처리
  const paragraphs = art.항;
  if (paragraphs) {
    const items = Array.isArray(paragraphs) ? paragraphs : [paragraphs];
    for (const p of items) {
      // 항 자체의 내용
      if (p.항내용) {
        text += " " + p.항내용.replace(/<개정[^>]*>/g, "").replace(/<신설[^>]*>/g, "").trim();
      }
      // 호 처리
      const hos = p.호;
      if (hos) {
        const hoList = Array.isArray(hos) ? hos : [hos];
        for (const h of hoList) {
          const hoText = (h.호내용 || "").replace(/<개정[^>]*>/g, "").replace(/<신설[^>]*>/g, "").trim();
          if (hoText) text += " " + hoText;
          // 목 처리
          const moks = h.목;
          if (moks) {
            const mokList = Array.isArray(moks) ? moks : [moks];
            for (const m of mokList) {
              const mokText = (m.목내용 || "").replace(/<개정[^>]*>/g, "").replace(/<신설[^>]*>/g, "").trim();
              if (mokText) text += " " + mokText;
            }
          }
        }
      }
    }
  }
  return text.replace(/\s+/g, " ").trim();
}

const chunks = [];

for (const art of arts) {
  if (art.조문여부 === "전문") {
    // 장/절 제목
    const title = (art.조문내용 || "").trim();
    if (title.includes("장")) currentChapter = title.replace(/\s+/g, " ").trim();
    continue;
  }

  const num = art.조문번호 || "";
  const text = buildText(art);
  if (!text || text.length < 20) continue;

  // 조문 제목 추출: "제N조(제목)" 패턴
  const titleMatch = text.match(/제\d+조(?:의\d+)?\([^)]+\)/);
  const artTitle = titleMatch ? titleMatch[0] : `제${num}조`;

  const id = `PIPA-${num}`;
  const section = `${LAW_NAME} > ${currentChapter} > ${artTitle}`.replace(/\s+/g, " ");

  chunks.push({
    id,
    text: `[${section}] ${text}`,
    url: LAW_URL,
    section,
  });
}

console.log(`생성된 청크: ${chunks.length}개`);

// 기존 chunks.json에 병합
const existingPath = join("docs", "chunks.json");
const existing = JSON.parse(readFileSync(existingPath, "utf8"));
// 기존 PIPA- 청크 제거 (재실행 시 중복 방지)
const filtered = existing.filter(c => !c.id.startsWith("PIPA-"));
const merged = [...filtered, ...chunks];
writeFileSync(existingPath, JSON.stringify(merged, null, 2), "utf8");
console.log(`docs/chunks.json: ${filtered.length} 기존 + ${chunks.length} 신규 = ${merged.length} 총`);

// 샘플 출력
for (const c of chunks.slice(0, 3)) {
  console.log(`\n[${c.id}] ${c.text.substring(0, 150)}...`);
}
// 유출 관련 조문 확인
const leak = chunks.filter(c => c.text.includes("유출") || c.text.includes("침해"));
console.log(`\n유출·침해 관련 조문: ${leak.map(c => c.id).join(", ")}`);

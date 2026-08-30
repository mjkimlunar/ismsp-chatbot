import type { Retrieved } from "./rag";

// 기관·법률 등 고유명사 패턴 (한국어 조직명에 흔한 접미사)
const ORG_SUFFIX = /[가-힣]{2,}(?:위원회|보호원|진흥원|연구원|부|처|청|국|원|센터|공단|재단|협회|학회)/g;
const LAW_PATTERN = /「[^」]+」/g;

/** 답변에서 출처 자료에 없는 고유명사를 찾아 경고를 덧붙인다 */
export function postCheck(answer: string, hits: Retrieved[]): string {
  const srcText = hits.map((h) => h.chunk.text).join(" ");

  const unverified: string[] = [];

  // 기관명 검사
  for (const m of answer.matchAll(ORG_SUFFIX)) {
    const name = m[0];
    if (!srcText.includes(name)) unverified.push(name);
  }

  // 법률명 검사 (「」로 감싼 이름)
  for (const m of answer.matchAll(LAW_PATTERN)) {
    const name = m[0];
    if (!srcText.includes(name)) unverified.push(name);
  }

  if (unverified.length === 0) return answer;

  const unique = [...new Set(unverified)];
  const warning = `\n\n⚠️ 주의: ${unique.join(", ")} — 출처 자료에 없는 내용입니다. 정확성을 직접 확인하세요.`;
  return answer + warning;
}

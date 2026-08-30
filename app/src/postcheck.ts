import type { Retrieved } from "./rag";

// 기관명 패턴: 짧은 접미사(부,처,청,국,원)는 오탐이 많으므로
// 긴 접미사(위원회, 센터 등)와 3글자 이상 조합만 검사
const ORG_PATTERNS = [
  /[가-힣]{2,}위원회/g,
  /[가-힣]{2,}보호원/g,
  /[가-힣]{2,}진흥원/g,
  /[가-힣]{2,}연구원/g,
  /[가-힣]{2,}센터/g,
  /[가-힣]{2,}공단/g,
  /[가-힣]{2,}재단/g,
  /[가-힣]{2,}협회/g,
  /[가-힣]{3,}부(?=[^터\가-힣]|$)/g,  // "~으로부터" 오탐 방지
  /[가-힣]{3,}처(?=[^리\가-힣]|$)/g,  // "~절차" 오탐 방지
  /[가-힣]{3,}청(?=[^구\가-힣]|$)/g,  // "~신청" 오탐 방지
];
const LAW_PATTERN = /「[^」]+」/g;

/** 답변에서 출처 자료에 없는 고유명사를 찾아 경고를 덧붙인다 */
export function postCheck(answer: string, hits: Retrieved[]): string {
  const srcText = hits.map((h) => h.chunk.text).join(" ");

  const unverified: string[] = [];

  for (const pat of ORG_PATTERNS) {
    pat.lastIndex = 0;
    for (const m of answer.matchAll(pat)) {
      const name = m[0];
      if (!srcText.includes(name)) unverified.push(name);
    }
  }

  for (const m of answer.matchAll(LAW_PATTERN)) {
    const name = m[0];
    if (!srcText.includes(name)) unverified.push(name);
  }

  if (unverified.length === 0) return answer;

  const unique = [...new Set(unverified)];
  const warning = `\n\n⚠️ 주의: ${unique.join(", ")} — 출처 자료에 없는 내용입니다. 정확성을 직접 확인하세요.`;
  return answer + warning;
}

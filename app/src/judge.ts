// ISMS-P 인증기준 안내 챗봇 — LLM-as-a-Judge 공통 로직
// 루브릭별 독립 채점(병렬) → 평균. 판정 엔진(ollama/gemini)은 call 함수만 제공한다.

export interface RubricScore {
  id: "grounded" | "noHalluc" | "cited";
  name: string;         // UI 표기명
  score: number;        // 0-100
  comment: string;      // 한 문장 평어
}

export interface JudgeResult {
  grounded: boolean;      // 답변이 근거 조각에 기반하는가 (루브릭 점수 70 이상)
  noHalluc: boolean;      // 자료에 없는 내용을 지어내지 않았는가 (동일)
  cited: boolean;         // [ID] 근거 표시를 했는가 (동일)
  refusal: boolean;       // 자료에 없을 때 없다고 답했는가 (해당 시)
  score: number;          // 0-100 = 루브릭 3개 점수의 평균
  comment: string;        // 최저점 루브릭의 평어 (가장 약한 축 안내)
  rubrics: RubricScore[]; // 루브릭별 개별 점수
}

/** 채점 루브릭 — 각각 별도 호출로 0-100점을 매긴다. refusal은 상황 플래그라 평균에서 제외. */
export const RUBRICS: { id: RubricScore["id"]; name: string; criterion: string }[] = [
  {
    id: "grounded",
    name: "근거 충실성",
    criterion:
      "답변의 모든 사실 주장이 [근거자료]에서 나왔는가. 근거와 무관하거나 모순되는 주장이 섞일수록 감점.",
  },
  {
    id: "noHalluc",
    name: "환각 통제",
    criterion:
      "[근거자료]에 없는 정보(날짜·숫자·이름·규칙)를 지어내지 않았는가. 지어낸 내용이 하나라도 있으면 0에 가깝게.",
  },
  {
    id: "cited",
    name: "출처 표시",
    criterion:
      "답변 안에 근거 조각의 [ID] 표시가 있는가. 주장마다 표시했으면 100, 일부만이면 그 비율, 없으면 0.",
  },
];

/** 단일 루브릭 채점 프롬프트 — 이 기준만 보고 점수를 매기게 다른 기준은 명시적으로 배제한다. */
export function buildRubricPrompt(
  rubric: (typeof RUBRICS)[number],
  question: string,
  sources: string,
  answer: string,
): string {
  return [
    "당신은 RAG 챗봇 답변의 평가자입니다. 아래 [질문], [근거자료], [답변]을 읽고 다음 기준 하나만으로 채점합니다.",
    `기준 (${rubric.name}): ${rubric.criterion}`,
    "이 기준 외의 다른 품질(문체, 완결성 등)은 보지 않습니다.",
    "score: 0-100 정수, comment: 한 문장 평어(한국어)",
    '출력 형식: {"score":0,"comment":"..."} — JSON 외 텍스트 금지.',
    "",
    `[질문] ${question}`,
    "",
    `[근거자료] ${sources}`,
    "",
    `[답변] ${answer}`,
  ].join("\n");
}

/** 거부 여부 조사 프롬프트 — 점수가 아니라 상황 플래그만 판정한다. */
export function buildRefusalPrompt(question: string, sources: string, answer: string): string {
  return [
    "아래 [질문], [근거자료], [답변]을 읽고 판정합니다.",
    "refusal: [근거자료]에 답이 없어서 답변이 '없다/찾을 수 없다'고 답한 경우 true, 그 외 false.",
    '출력 형식: {"refusal":false} — JSON 외 텍스트 금지.',
    "",
    `[질문] ${question}`,
    "",
    `[근거자료] ${sources}`,
    "",
    `[답변] ${answer}`,
  ].join("\n");
}

/** 모델이 5점 만점으로 오해하는 경우 방어 — 0~5 범위면 100점 척도로 환산 */
function to100(raw: unknown): number {
  let s = typeof raw === "number" ? raw : 0;
  if (s <= 5) s = (s / 5) * 100;
  return Math.max(0, Math.min(100, Math.round(s)));
}

function parseJson(text: string): Record<string, unknown> {
  const m = text.match(/\{[\s\S]*\}/);
  if (!m) throw new Error("평가 JSON 파싱 실패");
  return JSON.parse(m[0]) as Record<string, unknown>;
}

/**
 * 루브릭별 병렬 채점 — 채점 루브릭 3개 + 거부 조사 1개를 Promise.all로 동시에 돌리고,
 * 최종 점수는 루브릭 3개의 산술 평균. call은 프롬프트를 받아 모델 텍스트를 반환하는 엔진 함수.
 */
export async function judgeAll(
  question: string,
  sources: string,
  answer: string,
  call: (prompt: string) => Promise<string>,
): Promise<JudgeResult> {
  const rubricJobs = RUBRICS.map(async (r) => {
    const j = parseJson(await call(buildRubricPrompt(r, question, sources, answer)));
    return { id: r.id, name: r.name, score: to100(j.score), comment: String(j.comment ?? "") } as RubricScore;
  });
  const refusalJob = (async () => {
    const j = parseJson(await call(buildRefusalPrompt(question, sources, answer)));
    return j.refusal === true;
  })();

  const [rubrics, refusal] = await Promise.all([Promise.all(rubricJobs), refusalJob]);

  const score = Math.round(rubrics.reduce((a, r) => a + r.score, 0) / rubrics.length);
  const weakest = [...rubrics].sort((a, b) => a.score - b.score)[0];
  return {
    grounded: rubrics.find((r) => r.id === "grounded")!.score >= 70,
    noHalluc: rubrics.find((r) => r.id === "noHalluc")!.score >= 70,
    cited: rubrics.find((r) => r.id === "cited")!.score >= 70,
    refusal,
    score,
    comment: weakest.comment,
    rubrics,
  };
}

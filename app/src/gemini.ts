// ISMS-P 인증기준 안내 챗봇 — Gemini API 폴백 클라이언트 (사용자 API 키, 브라우저 직접 호출)
import { judgeAll, type JudgeResult } from "./judge";

export type { JudgeResult };
export interface GeminiMsg { role: "user" | "model"; text: string }

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** 429(분당 요청 한도)면 기다렸다 다시 시도한다.
 *  무료 등급은 모델별 분당 한도가 낮아, 연속 질문이나 답변+판정 두 번 호출만으로도 걸린다.
 *  응답 본문의 "retry in 12.3s" 힌트를 우선 쓰고, 없으면 2초씩 늘려 간다. */
async function fetchWithRetry(
  url: string,
  init: RequestInit,
  label: string,
  tries = 3,
): Promise<Response> {
  for (let i = 0; ; i++) {
    const res = await fetch(url, init);
    if (res.status !== 429 || i >= tries - 1) {
      if (!res.ok) throw Object.assign(new Error(`${label} ${res.status}`), { status: res.status });
      return res;
    }
    // 본문을 읽어 서버가 알려 준 대기 시간을 쓴다 (읽어도 재시도에는 지장 없음)
    const body = await res.text().catch(() => "");
    const hinted = Number(body.match(/retry in ([\d.]+)s/i)?.[1]);
    const waitMs = Number.isFinite(hinted) ? Math.ceil(hinted * 1000) + 500 : (i + 1) * 2000;
    if (init.signal?.aborted) throw new Error(`${label} aborted`);
    await sleep(Math.min(waitMs, 35000));
  }
}

/** SSE 스트리밍 generateContent. 키는 사용자가 UI에서 입력해 로컬스토리지에 저장. */
export async function geminiStream(
  msgs: GeminiMsg[],
  apiKey: string,
  onToken: (t: string) => void,
  signal?: AbortSignal,
  model = "gemini-3.5-flash",
): Promise<string> {
  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse&key=${apiKey}`;
  const res = await fetchWithRetry(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: msgs.map((m) => ({
        role: m.role === "user" ? "user" : "model",
        parts: [{ text: m.text }],
      })),
    }),
    signal,
  }, "gemini");
  const reader = res.body!.getReader();
  const dec = new TextDecoder();
  let full = "";
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6).trim();
      if (!payload || payload === "[DONE]") continue;
      try {
        const j = JSON.parse(payload);
        const parts = j.candidates?.[0]?.content?.parts ?? [];
        for (const p of parts) {
          if (typeof p.text === "string" && p.text && !p.thoughtSignature) {
            full += p.text;
            onToken(p.text);
          }
        }
      } catch { /* 불완전 라인 */ }
    }
  }
  return full;
}

/** LLM-as-a-Judge(Gemini): 한 턴(질문·근거·답변)을 루브릭별로 병렬 채점해 평균.
 *  기준·병렬·집계는 judge.ts 공통 로직 사용. */
export async function judgeTurn(
  question: string,
  sources: string,
  answer: string,
  apiKey: string,
  // 판정은 답변과 다른 모델로 한다. 같은 모델이 자기 답을 채점하면 독립 심사가 아니고,
  // 무료 등급 쿼터도 답변 호출과 같은 버킷을 나눠 쓰게 되어 429가 빨리 온다.
  model = "gemini-3.5-flash-lite",
): Promise<JudgeResult> {
  const call = async (prompt: string): Promise<string> => {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
    const res = await fetchWithRetry(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] }),
    }, "judge");
    const j = await res.json();
    return j.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  };
  return judgeAll(question, sources, answer, call);
}

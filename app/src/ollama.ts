// ISMS-P 인증기준 안내 챗봇 — ollama 스트리밍 클라이언트
// 브라우저가 직접 localhost:11434 호출 (OLLAMA_ORIGINS 설정 필요)

import { judgeAll, type JudgeResult } from "./judge";

export interface ChatMsg {
  role: "system" | "user" | "assistant";
  content: string;
}

export type onToken = (text: string) => void;

/** 스트리밍 채팅. fetch + ReadableStream으로 토큰 단위 콜백 */
export async function chatStream(
  messages: ChatMsg[],
  onToken: onToken,
  model = "qwen2.5:3b",
  signal?: AbortSignal,
): Promise<string> {
  const res = await fetch("http://localhost:11434/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, messages, stream: true, think: false }),
    signal,
  });
  if (!res.ok) {
    throw Object.assign(new Error(`ollama ${res.status}`), { status: res.status });
  }
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let full = "";
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      const s = line.trim();
      if (!s) continue;
      try {
        const j = JSON.parse(s);
        const piece: string = j.message?.content ?? "";
        if (piece) {
          full += piece;
          onToken(piece);
        }
      } catch {
        // 불완전한 줄은 다음 청크에서
      }
    }
  }
  return full;
}

/** LLM-as-a-Judge(로컬): 답변에 쓴 qwen이 같은 모델로 자기 답을 평가.
 *  judge.ts가 루브릭별로 프롬프트를 병렬로 보내고 평균을 낸다 (API 키 불필요). */
export async function judgeWithOllama(
  question: string,
  sources: string,
  answer: string,
  model = "qwen2.5:3b",
): Promise<JudgeResult> {
  const call = async (prompt: string): Promise<string> => {
    const res = await fetch("http://localhost:11434/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: prompt }],
        stream: false,
        think: false,
        format: "json",
        options: { temperature: 0 },
      }),
    });
    if (!res.ok) {
      throw Object.assign(new Error(`ollama judge ${res.status}`), { status: res.status });
    }
    const j = await res.json();
    return j.message?.content ?? "";
  };
  return judgeAll(question, sources, answer, call);
}

/** ollama 서버 상태 확인 (페이지 로드 시) */
export async function pingOllama(): Promise<boolean> {
  try {
    const res = await fetch("http://localhost:11434/api/tags", { method: "GET" });
    return res.ok;
  } catch {
    return false;
  }
}

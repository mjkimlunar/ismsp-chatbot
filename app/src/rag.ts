// ISMS-P 인증기준 안내 챗봇 — RAG 유틸리티
// 임베딩: embeddinggemma-300m (model_no_gather_q4 변형 — 브라우저 WASM ORT 호환)
//   - transformers.js pipeline()은 q4/q8 기본 파일을 골라 GatherBlockQuantized
//     미지원으로 실패하므로, 토크나이저만 transformers.js로 쓰고
//     ORT 세션은 no_gather_q4 파일로 직접 만든다 (2026-08 헤드리스 검증).
// 검색: ismsp-docs.json 정적 벡터스토어와 코사인 유사도 top-k

import { AutoTokenizer, type PreTrainedTokenizer } from "@huggingface/transformers";

const MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX";
const HF_ONNX = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx`;
// transformers.js 4.2.0이 쓰는 것과 같은 onnxruntime-web 빌드 (검증된 조합)
const ORT_URL =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/ort.webgpu.bundle.min.mjs";

interface OrtSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, { dims: number[]; data: Float32Array }>>;
}

export interface DocChunk {
  id: string;
  text: string;
  url: string;
  section: string;
  vector: number[];
}

let session: OrtSession | null = null;
let tokenizer: PreTrainedTokenizer | null = null;
let ready: Promise<void> | null = null;

/** 임베딩 모델 내려받기 진행률 (첫 방문 1회, 이후 브라우저 캐시 — cached: 캐시 히트) */
export type EmbedProgress = { pct: number; file: string; cached?: boolean };
let progressCb: ((p: EmbedProgress) => void) | null = null;
export function onEmbedProgress(cb: (p: EmbedProgress) => void) {
  progressCb = cb;
}

/** 모델 캐시 보유 여부만 확인한다(다운로드 없음) — 첫 화면에서 "캐시된 모델" 표시용 */
export async function peekModelCache(): Promise<boolean> {
  try {
    const c = await caches.open(MODEL_CACHE);
    return (await c.match(`${HF_ONNX}/model_no_gather_q4.onnx_data`)) !== undefined;
  } catch {
    return false;
  }
}

// 모델 파일 캐시 — HF resolve URL은 요청마다 서명이 다른 CDN 주소로 리다이렉트되어
// HTTP 캐시가 히트하지 않는다. Cache Storage에 직접 보관해 재방문 시 재다운로드를 막는다.
const MODEL_CACHE = "ismsp-embed-v1";

async function fetchWithProgress(url: string, file: string): Promise<Uint8Array> {
  let cache: Cache | null = null;
  try {
    cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(url);
    if (hit) {
      progressCb?.({ pct: 100, file, cached: true });
      return new Uint8Array(await hit.arrayBuffer());
    }
  } catch {
    cache = null; // 캐시 API를 쓸 수 없는 환경 — 그냥 내려받는다
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`모델 파일 내려받기 실패 (${res.status}): ${file}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  let lastPct = -1;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    const pct = Math.round((got / total) * 100);
    if (pct !== lastPct) {
      lastPct = pct;
      progressCb?.({ pct, file });
    }
  }
  const out = new Uint8Array(got);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  if (cache) await putWithRetry(cache, url, out);
  return out;
}

/** 캐시 저장 — Chrome이 간헐히 Unexpected internal error를 내는 경우가 있어 1회 재시도한다 */
async function putWithRetry(cache: Cache, url: string, body: Uint8Array<ArrayBuffer>): Promise<void> {
  for (let i = 0; i < 2; i++) {
    try {
      await cache.put(url, new Response(body));
      return;
    } catch (e) {
      console.warn(`임베딩 모델 캐시 저장 실패 (${i + 1}/2) — ${url}:`, e);
      await new Promise((r) => setTimeout(r, 400));
    }
  }
}

/** 토크나이저 + ORT 세션 준비 (첫 호출 시 모델 다운로드, 이후 Cache Storage 재사용) */
function ensureReady(): Promise<void> {
  if (session && tokenizer) return Promise.resolve();
  if (!ready) {
    ready = (async () => {
      // 디스크 여유가 부족할 때 브라우저가 이 사이트의 저장소를 임의로 비우지 않게 한다
      navigator.storage?.persist?.().catch(() => undefined);
      const ort = (await import(/* @vite-ignore */ ORT_URL)) as {
        InferenceSession: { create(
          buf: Uint8Array,
          opts: { executionProviders: string[]; externalData: { path: string; data: Uint8Array }[] },
        ): Promise<OrtSession> };
      };
      const core = await fetchWithProgress(`${HF_ONNX}/model_no_gather_q4.onnx`, "model_no_gather_q4.onnx");
      const data = await fetchWithProgress(`${HF_ONNX}/model_no_gather_q4.onnx_data`, "model_no_gather_q4.onnx_data");
      session = await ort.InferenceSession.create(core, {
        executionProviders: ["wasm"],
        externalData: [{ path: "model_no_gather_q4.onnx_data", data }],
      });
      tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
    })().catch((e) => {
      ready = null; // 실패 시 다음 질문에서 재시도 가능
      throw e;
    });
  }
  return ready;
}

/** 문장 → 768차원 벡터 (mean pooling + L2 정규화 — 벡터스토어 생성 방식과 동일) */
export async function embed(text: string): Promise<number[]> {
  await ensureReady();
  const { input_ids, attention_mask } = await tokenizer!(text);
  const out = await session!.run({ input_ids, attention_mask });
  const hs = out.last_hidden_state;
  const [, seq, hid] = hs.dims;
  const am = attention_mask.data as ArrayLike<bigint> | ArrayLike<number>;
  const acc = new Float64Array(hid);
  let cnt = 0;
  for (let s = 0; s < seq; s++) {
    const w = Number(am[s]);
    cnt += w;
    if (!w) continue;
    for (let h = 0; h < hid; h++) acc[h] += hs.data[s * hid + h];
  }
  let norm = 0;
  for (let h = 0; h < hid; h++) {
    acc[h] /= cnt;
    norm += acc[h] * acc[h];
  }
  norm = Math.sqrt(norm);
  const vec = new Array<number>(hid);
  for (let h = 0; h < hid; h++) vec[h] = acc[h] / norm;
  return vec;
}

let corpus: DocChunk[] | null = null;

export async function loadCorpus(): Promise<DocChunk[]> {
  if (corpus) return corpus;
  const res = await fetch(`${import.meta.env.BASE_URL}ismsp-docs.json`);
  if (!res.ok) throw new Error(`docs 로드 실패: ${res.status}`);
  corpus = (await res.json()) as DocChunk[];
  return corpus;
}

export interface Retrieved {
  chunk: DocChunk;
  score: number;
  method: "vector" | "bm25";
}

// 단어 검색용 불용어 — 조사·접속사·군더더기 표현 (2글자 이상만 걸러낸다)
const STOPWORDS = new Set([
  "에서", "에게", "한테", "부터", "까지", "처럼", "같이", "마다", "보다", "라는",
  "무엇", "언제", "어디", "누구", "어떤", "어떻게", "왜요", "인가요", "나요",
  "있는", "없는", "하는", "했던", "하는지", "인지", "니까", "이며", "하고",
  "주세요", "알려줘", "알려주세요", "가르쳐", "가르쳐줘", "말해줘", "해줘",
  "해주세요", "해주실", "그리고", "그래서", "하지만", "그런데", "근데", "the", "is", "what", "when", "where", "how", "about", "please", "tell",
]);

// ── 검색 설정 ─────────────────────────────────────────────────────────────
//  실험은 여기 값 하나씩만 바꾸고 eval/rubric.md의 지표로 비교한다.
export const SEARCH = {
  /** 벡터 유사도로 뽑는 개수 */
  vectorTop: 5,
  /** BM25로 보강하는 개수 (벡터 결과와 중복 제외) */
  bm25Top: 3,
  /** 이 값 아래면 약한 근거로 보고 프롬프트를 보수화한다 */
  weakThreshold: 0.55,
  /** 항목번호(1.1.1, 2.10.3)를 한 토큰으로 살릴지 */
  keepItemNumbers: true,
};

// 인증기준 항목번호 — 이 도메인의 고유명사. 1.1.1 처럼 점으로 이어진 숫자.
const ITEM_NO = /\d{1,2}(?:\.\d{1,2}){1,2}/g;

/** 질문에서 검색어 뽑기 — 소문자 통일, 1글자·불용어 제거
 *
 *  기본 분해 규칙은 점을 구분자로 보기 때문에 "1.1.1"이 ["1","1","1"]로 흩어지고
 *  1글자 필터에 전부 걸려 사라진다. 모두콘은 ID(MC-001)가 본문에 안 쓰여 문제가
 *  없었지만, 이 도메인은 항목번호가 곧 사용자가 쓰는 검색어다. 그래서 먼저 떼어 둔다.
 */
function queryTerms(q: string): string[] {
  const lower = q.toLowerCase();
  const nums = SEARCH.keepItemNumbers ? (lower.match(ITEM_NO) ?? []) : [];
  const rest = (SEARCH.keepItemNumbers ? lower.replace(ITEM_NO, " ") : lower)
    .split(/[^가-힣a-z0-9]+/)
    .filter((t) => t.length >= 2 && !STOPWORDS.has(t));
  return [...nums, ...rest];
}

// ── BM25 단어 검색 ────────────────────────────────────────────────────────
//  tf(빈도)·IDF(희귀도)·문서 길이 정규화를 갖춘 표준 단어 검색 점수(k1=1.5, b=0.75).
//  tf는 토큰 '포함' 관계로 세서 조사가 붙은 형태("인증기준을")도 "인증기준" 검색어에
//  적중시킨다. 말뭉치가 119조각이라 질문마다 그 자리에서 계산한다(별도 색인 없음).
const BM25_K1 = 1.5;
const BM25_B = 0.75;

/** BM25 원점수 — 문서별 점수(정규화 전). 점수 0 = 적중 없음 */
function bm25(docs: DocChunk[], terms: string[]): { chunk: DocChunk; score: number }[] {
  if (!terms.length) return [];
  const toks = docs.map((d) => queryTerms(d.text)); // 문서 토큰화도 질문과 같은 규칙
  const avgdl = toks.reduce((s, t) => s + t.length, 0) / docs.length;
  const df = new Map<string, number>(); // 검색어 → 그 검색어를 포함하는 문서 수
  for (const t of new Set(terms)) {
    df.set(t, toks.reduce((n, dt) => n + (dt.some((x) => x.includes(t)) ? 1 : 0), 0));
  }
  return docs.map((chunk, i) => {
    const dl = toks[i].length || 1;
    let score = 0;
    for (const [t, dfv] of df) {
      if (!dfv) continue;
      let tf = 0;
      for (const x of toks[i]) if (x.includes(t)) tf++;
      if (!tf) continue;
      const idf = Math.log((docs.length - dfv + 0.5) / (dfv + 0.5) + 1);
      score += (idf * tf * (BM25_K1 + 1)) / (tf + BM25_K1 * (1 - BM25_B + (BM25_B * dl) / avgdl));
    }
    return { chunk, score };
  });
}

/** 하이브리드 검색 — 벡터 상위 SEARCH.vectorTop개 + BM25 상위 SEARCH.bm25Top개(중복 제외).
 *  BM25가 몫을 못 채우면 벡터 다음 순위로 보충해 항상 k개를 돌려준다.
 *  BM25 점수는 표시를 위해 이번 질문의 최상위가 1이 되게 정규화한다.
 *
 *  k 기본값이 15가 아니라 8인 이유: 이 코퍼스의 청크는 평균 270자(최대 1,052자)라
 *  15개를 넣으면 ollama 기본 컨텍스트 4096 토큰을 넘겨 근거가 잘린다. */
export async function retrieve(question: string, k = SEARCH.vectorTop + SEARCH.bm25Top): Promise<Retrieved[]> {
  const [docs, q] = await Promise.all([loadCorpus(), embed(question)]);
  const vec = docs
    .map((chunk) => {
      let dot = 0;
      const v = chunk.vector;
      for (let i = 0; i < v.length; i++) dot += v[i] * q[i];
      return { chunk, score: dot, method: "vector" as const };
    })
    .sort((a, b) => b.score - a.score);
  const topVec = vec.slice(0, Math.min(SEARCH.vectorTop, k));
  const picked = new Set(topVec.map((r) => r.chunk.id));
  const scored = bm25(docs, queryTerms(question))
    .filter((r) => r.score > 0 && !picked.has(r.chunk.id))
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(0, k - topVec.length));
  const top = scored[0]?.score ?? 0;
  const lex = scored.map((r) => ({
    chunk: r.chunk,
    score: top > 0 ? r.score / top : 0,
    method: "bm25" as const,
  }));
  for (const r of lex) picked.add(r.chunk.id);
  const rest = vec.filter((r) => !picked.has(r.chunk.id)).slice(0, k - topVec.length - lex.length);
  return [...topVec, ...lex, ...rest];
}

/** RAG 시스템 지시 — 근거 원칙을 고정 */
export function buildPrompt(question: string, hits: Retrieved[]): string {
  const now = new Intl.DateTimeFormat("ko-KR", {
    timeZone: "Asia/Seoul",
    year: "numeric", month: "long", day: "numeric", weekday: "long",
    hour: "2-digit", minute: "2-digit", hour12: true,
  }).format(new Date());
  const context = hits
    .map((h) => `[${h.chunk.id} | ${h.chunk.section}] ${h.chunk.text}`)
    .join("\n\n");
  const best = hits[0]?.score ?? 0;
  const weakNote = best < SEARCH.weakThreshold
    ? "주의: 검색된 조각의 유사도가 낮습니다. 질문과 완전히 맞는 근거가 아닐 수 있으니, 근거에 있는 내용만 짧게 답하고 자료에 없는 부분은 없다고 말합니다."
    : "자료에 근거한 내용만 답하고, 자료에 없으면 없다고 말합니다.";
  return [
    "다음 자료는 「정보보호 및 개인정보보호 관리체계 인증 등에 관한 고시」(2024. 7. 24. 시행)에서 뽑은 조각입니다. IC-로 시작하는 조각은 별표 7의 인증기준 항목이고, AR-로 시작하는 조각은 고시 본문 조문입니다.",
    weakNote,
    "근거가 된 조각의 [ID]를 답 안에서 표시합니다.",
    // 이 챗봇은 원문이 무엇을 요구하는지만 답한다. 판단과 이행 방법은 PRD의 비목표다.
    "기본 원칙: 근거 조각에 해당 항목이나 조문이 있으면 그 원문 내용을 반드시 먼저 설명합니다. 항목번호를 묻는 질문('1.1.1이 뭐야?')에는 그 항목이 무엇을 요구하는지 원문 그대로 답합니다. 이때 거절하지 않습니다.",
    "그 다음, 질문이 아래에 해당하는 부분을 포함할 때만 그 부분에 한해 선을 긋습니다.",
    "1. 특정 조직이 인증 대상인지, 심사를 통과할지, 결함인지 같은 판단을 요구하는 부분 — 관련 원문은 제시하되 그 판단은 제공하지 않는다고 덧붙입니다.",
    "2. 어떻게 이행하는지, 어떤 문서·양식을 만들어야 하는지 묻는 부분 — 인증기준 원문을 설명한 뒤, 구체적인 이행 방법 해설은 이 자료에 없으므로 KISA가 배포하는 'ISMS-P 인증기준 안내서'를 확인하도록 덧붙입니다.",
    "즉 위 두 경우에도 답변 전체를 거절하지 않습니다. 자료에 있는 내용은 답하고, 자료 밖인 부분만 선을 긋습니다.",
    "신청 절차·수수료·서식 관련 답변에는, 서식과 수수료 기준은 개정될 수 있으니 KISA ISMS-P 누리집 자료실에서 최신본을 확인하라는 안내를 덧붙입니다.",
    `현재 시각은 ${now}(한국 표준시 KST)입니다. '지금', '올해', '다음 주' 같은 상대 표현은 이 시각을 기준으로 해석합니다.`,
    "",
    "[자료]",
    context,
    "",
    "[질문]",
    question,
  ].join("\n");
}

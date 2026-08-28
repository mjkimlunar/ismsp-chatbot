import { useState, useRef, useEffect, type CSSProperties } from "react";
import { retrieve, buildPrompt, loadCorpus, onEmbedProgress, peekModelCache, SEARCH, type Retrieved } from "./rag";
import { chatStream, pingOllama, judgeWithOllama, type ChatMsg } from "./ollama";
import { geminiStream, judgeTurn } from "./gemini";
import type { JudgeResult } from "./judge";
import "./App.css";

interface Turn {
  /** 말풍선 고유 번호. 스트리밍·판정 결과를 이 번호로 찾아 쓴다.
   *  배열 끝(length-1)에 쓰면, 늦게 끝난 판정이 그 사이 도착한 다른 답변을 덮어쓴다. */
  id: number;
  role: "user" | "assistant";
  content: string;
  sources?: Retrieved[];
  question?: string;
  judge?: JudgeResult;
  judgeBy?: "qwen3.5:2b" | "gemini-3.5-flash-lite";
  judgeError?: boolean;
  feedback?: "up" | "down";
}

type Phase = "idle" | "embed" | "search" | "stream" | "error-ollama";

// 파이프라인 단계 — 튜토리얼용 표시
const PHASE_LABEL: Record<Phase, string> = {
  idle: "",
  embed: "① 질문 임베딩 중 — 브라우저에서 질문을 벡터로 바꿉니다",
  search: `② 근거 검색 중 — 벡터 유사도 상위 ${SEARCH.vectorTop}개 + BM25 상위 ${SEARCH.bm25Top}개, 총 ${SEARCH.vectorTop + SEARCH.bm25Top}개를 찾습니다`,
  stream: "③ 답변 생성 중 — 찾은 근거를 붙여 모델이 답을 씁니다",
  "error-ollama": "연결 실패",
};

export default function App() {
  const turnSeq = useRef(0);
  const [turns, setTurns] = useState<Turn[]>([
    {
      id: 0,
      role: "assistant",
      content:
        "안녕하세요. ISMS-P 인증기준 안내 챗봇입니다. 인증기준 101개 항목과 신청·심사 절차를 고시 원문에 근거해 안내합니다. 다만 이행 방법과 통과 여부 판단은 제공하지 않습니다.",
    },
  ]);
  const [input, setInput] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [ollamaOk, setOllamaOk] = useState<boolean | null>(null);
  // 엔진 선택도 브라우저에 남긴다. 새로고침 때마다 로컬로 되돌아가면
  // Gemini로 쓰던 사람은 영문 모르고 몇 분씩 기다리게 된다.
  const [engine, setEngine] = useState<"local" | "gemini">(
    () => (localStorage.getItem("engine") === "gemini" ? "gemini" : "local"),
  );
  const [apiKey, setApiKey] = useState(() => localStorage.getItem("gemini_key") ?? "");
  const [showSource, setShowSource] = useState<Retrieved[] | null>(null);
  const [lastHits, setLastHits] = useState<Retrieved[] | null>(null);
  const [dlPct, setDlPct] = useState<number | null>(null);
  const [judgeBusy, setJudgeBusy] = useState(false);
  const [openSrc, setOpenSrc] = useState<Record<number, boolean>>({});
  const [hitsOpen, setHitsOpen] = useState(false);
  const [embedCached, setEmbedCached] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    pingOllama().then(setOllamaOk);
    loadCorpus().catch(() => undefined); // 프리로드
    peekModelCache().then(setEmbedCached); // 재방문이면 "캐시된 모델" 표시
    onEmbedProgress((p) => {
      if (p.cached) setEmbedCached(true);
      setDlPct(p.pct >= 100 ? null : p.pct);
    });
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, phase]);

  async function ask() {
    const q = input.trim();
    if (!q || phase !== "idle") return;
    setInput("");
    setTurns((t) => [...t, { id: ++turnSeq.current, role: "user", content: q }]);

    let streamStarted = false;
    try {
      setPhase("embed");
      await new Promise((r) => setTimeout(r, 350)); // 임베딩 단계를 눈으로 볼 수 있게 짧게 표시
      setPhase("search");
      const hits = await retrieve(q); // 개수는 rag.ts의 SEARCH 설정을 따른다
      setDlPct(null);
      setLastHits(hits);
      setHitsOpen(false);
      await new Promise((r) => setTimeout(r, 450)); // 검색 결과를 눈으로 볼 수 있게 표시
      const prompt = buildPrompt(q, hits);
      const messages: ChatMsg[] = [
        {
          role: "system",
          content:
            "당신은 ISMS-P 인증기준 안내 도우미입니다. 「정보보호 및 개인정보보호 관리체계 인증 등에 관한 고시」 원문에 근거한 내용만 답하고, 자료에 없는 정보는 '제가 가진 자료에는 없습니다'라고 답합니다. 근거 조각의 [ID]를 답에 표시합니다. 당신은 법률 자문을 제공하지 않으며, 특정 조직의 인증 대상 여부·심사 통과 가능성·결함 여부를 판단하지 않습니다. 인증기준을 어떻게 이행하는지에 관한 해설도 이 자료에는 없으므로 지어내지 말고 KISA 안내서를 확인하도록 안내합니다.",
        },
        { role: "user", content: prompt },
      ];

      const lastQ = q;
      const answerId = ++turnSeq.current; // 이 답변 칸의 주소. 이후 갱신은 전부 이 번호로만 한다.
      setTurns((t) => [...t, { id: answerId, role: "assistant", content: "", sources: hits, question: lastQ }]);
      setPhase("stream");
      streamStarted = true;
      abortRef.current = new AbortController();
      let acc = "";
      const onPiece = (piece: string) => {
        acc += piece;
        setTurns((t) => t.map((x) => (x.id === answerId ? { ...x, content: acc, sources: hits } : x)));
      };
      if (engine === "gemini") {
        await geminiStream(
          [
            { role: "user", text: messages[0].content },
            { role: "user", text: messages[1].content },
          ],
          apiKey,
          onPiece,
          abortRef.current.signal,
        );
      } else {
        await chatStream(messages, onPiece, "qwen3.5:2b", abortRef.current.signal);
      }
      setPhase("idle");
      // ④ LLM-as-a-Judge — 답변에 쓴 엔진과 같은 모델이 판정한다
      //    로컬 → qwen 자기평가(API 키 불필요, 독립 평가 아님)
      //    gemini → 답변은 gemini-3.5-flash, 판정은 gemini-3.5-flash-lite (다른 모델이 채점)
      setJudgeBusy(true);
      try {
        const src = hits.map((h) => `[${h.chunk.id}] ${h.chunk.text}`).join("\n");
        const by = engine === "gemini" && apiKey ? "gemini-3.5-flash-lite" as const : "qwen3.5:2b" as const;
        const verdict =
          engine === "gemini" && apiKey
            ? await judgeTurn(lastQ, src, acc, apiKey)
            : await judgeWithOllama(lastQ, src, acc);
        setTurns((t) => t.map((x) => (x.id === answerId ? { ...x, judge: verdict, judgeBy: by } : x)));
      } catch {
        // 판정 실패가 답변을 해치지 않게 배지만 단다
        setTurns((t) => t.map((x) => (x.id === answerId ? { ...x, judgeError: true } : x)));
      } finally {
        setJudgeBusy(false);
      }
    } catch (e: unknown) {
      console.error("챗봇 파이프라인 오류:", e);
      setDlPct(null);
      const msg = e instanceof Error ? e.message : String(e);
      if (!streamStarted) {
        // 검색/임베딩 단계 오류 — 원인을 채팅창에 그대로 보여준다 (ollama와 무관)
        setTurns((t) => [
          ...t.filter((x) => x.content !== ""),
          { id: ++turnSeq.current, role: "assistant", content: `⚠ 답변을 만들지 못했습니다 — ${msg}` },
        ]);
        setPhase("idle");
        return;
      }
      // 생성 단계 오류. 어느 엔진이 실패했는지에 따라 안내가 달라야 한다.
      // (Gemini 429를 "ollama를 실행하세요"로 안내하면 사용자가 엉뚱한 곳을 고치게 된다)
      let notice: string;
      if (engine === "gemini") {
        notice = /429/.test(msg)
          ? "⚠ Gemini API 요청 한도(무료 등급 분당 제한)에 걸렸습니다 — 잠시 뒤 다시 질문하시거나, 답변 엔진을 로컬 ollama로 바꿔 주세요."
          : `⚠ Gemini API 호출에 실패했습니다 — ${msg}. API 키와 네트워크를 확인해 주세요.`;
      } else {
        notice = "⚠ 로컬 모델(ollama)에 연결하지 못했습니다 — 페이지 위 안내를 따라 ollama를 실행·설정한 뒤 다시 질문해 주세요.";
        setPhase("error-ollama");
        setOllamaOk(false);
      }
      setTurns((t) => [
        ...t.filter((x) => x.content !== ""),
        { id: ++turnSeq.current, role: "assistant", content: notice },
      ]);
      setPhase("idle");
    }
  }

  function stop() {
    abortRef.current?.abort();
    setPhase("idle");
  }

  function pickEngine(v: "local" | "gemini") {
    setEngine(v);
    localStorage.setItem("engine", v);
  }

  function setFeedback(id: number, v: "up" | "down") {
    setTurns((t) => t.map((x) => (x.id === id ? { ...x, feedback: x.feedback === v ? undefined : v } : x)));
    // 피드백은 로컬에만 기록 (제출 없음 — 데모)
    console.log("feedback", { turn: id, value: v });
  }

  const nBm = lastHits ? lastHits.filter((h) => h.method === "bm25").length : 0;

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-inner">
          <p className="hero-badge">개인정보보호위원회·과학기술정보통신부 고시 · 2024. 7. 24. 시행</p>
          <h1>ISMS-P <span className="accent">인증기준 안내</span></h1>
          <p className="hero-sub">
            정보보호 및 개인정보보호 관리체계 인증기준 101개 항목과 신청·심사 절차를
            고시 원문에 근거해 안내합니다. 답변마다 근거가 된 항목번호와 원문 링크를
            함께 보여 주고, 자료에 없으면 없다고 답합니다.
          </p>
          <a className="hero-cta" href="#chat">챗봇으로 물어보기 ↓</a>
        </div>
      </header>

      <div className="factbar">
        <span>인증기준 <b>101개 항목</b></span>
        <span>신청·심사 조문 <b>18개</b></span>
        <span>출처 <b>국가법령정보센터</b></span>
        <span>고시 시행 <b>2024. 7. 24.</b></span>
      </div>

      <section className="engine">
        <div className="engine-row">
          <span>답변 엔진:</span>
          <label><input type="radio" checked={engine==="local"} onChange={()=>pickEngine("local")} /> 로컬 ollama (qwen3.5:2b)</label>
          <label><input type="radio" checked={engine==="gemini"} onChange={()=>pickEngine("gemini")} /> Gemini API</label>
        </div>
        {engine === "gemini" && (
          <div className="engine-row">
            <input
              type="password"
              placeholder="Gemini API 키 (브라우저에만 저장됩니다)"
              value={apiKey}
              onChange={(e) => { setApiKey(e.target.value); localStorage.setItem("gemini_key", e.target.value); }}
            />
          </div>
        )}
      </section>

      {ollamaOk === false && engine === "local" && (
        <div className="banner">
          <strong>로컬 모델(ollama)에 연결할 수 없습니다.</strong>
          <ol>
            <li>브라우저 확인 — Safari는 이 페이지(https)에서 로컬 ollama 호출을 차단하므로 Chrome·Edge를 사용하세요. Chrome에서 "로컬 네트워크" 접근 권한을 물으면 <strong>허용</strong>을 누릅니다.</li>
            <li><code>ollama serve</code> 실행 (또는 Ollama 앱 실행) · 모델 확인: <code>ollama pull qwen3.5:2b</code></li>
            <li>
              github.io에서 열었다면 CORS 허용 — 운영 체제별로 한 번만 설정하고 Ollama를 재시작합니다:
              <div className="os-guide">
                <div><strong>macOS</strong><code>launchctl setenv OLLAMA_ORIGINS "https://*.github.io"</code>입력 후 메뉴 막대의 Ollama 앱을 종료하고 다시 실행합니다.</div>
                <div><strong>Windows</strong>작업 표시줄에서 Ollama를 종료합니다. 설정에서 <code>환경 변수</code>를 검색해 <code>계정의 환경 변수 편집</code>을 열고 새 변수 <code>OLLAMA_ORIGINS</code>에 <code>https://*.github.io</code>를 넣은 뒤 Ollama를 다시 시작합니다.</div>
                <div><strong>Linux</strong><code>sudo systemctl edit ollama.service</code>를 열어 <code>[Service]</code> 아래에 <code>Environment="OLLAMA_ORIGINS=https://*.github.io"</code>를 추가하고 <code>sudo systemctl restart ollama</code>로 재시작합니다.</div>
              </div>
            </li>
          </ol>
          <button onClick={() => pingOllama().then(setOllamaOk)}>다시 확인</button>
        </div>
      )}

      <section className="info">
        <div className="card card-a">
          <h2>다루는 자료</h2>
          <p>
            「정보보호 및 개인정보보호 관리체계 인증 등에 관한 고시」(2024. 7. 24. 시행)의
            <strong> 별표 7 인증기준 101개 항목</strong>과 <strong>신청·심사 관련 조문 18개</strong>.
            출처는 국가법령정보센터이며, 답변의 출처 칩에서 원문으로 이동할 수 있습니다.
          </p>
        </div>
        <div className="card card-b">
          <h2>답하지 않는 것</h2>
          <p>
            법률 자문, 특정 조직의 인증 대상 여부·심사 통과 가능성·결함 판단은 하지 않습니다.
            인증기준을 <strong>어떻게 이행하는지</strong>도 답하지 않습니다 — 그 해설은 KISA
            안내서에 있고, 이 챗봇의 자료에는 없기 때문입니다.
          </p>
        </div>
        <div className="card card-c">
          <h2>사용 조건</h2>
          <p>
            답변은 <strong>여러분 컴퓨터의 ollama(qwen3.5:2b)</strong>가 만듭니다. 서버가 대신
            실행하지 않으므로 ollama가 실행 중이어야 합니다. 첫 방문에는 임베딩 모델
            약 200MB를 내려받습니다. Chrome·Edge 권장.
          </p>
        </div>
      </section>

      <section id="chat" className="chat">
        <h2>
          ISMS-P 인증기준 안내 챗봇
          <span className={`conn ${ollamaOk === true ? "ok" : ollamaOk === false ? "bad" : ""}`}>
            {engine === "gemini"
              ? "Gemini API"
              : ollamaOk === true
                ? "ollama 연결됨"
                : ollamaOk === false
                  ? "ollama 미연결"
                  : "연결 확인 중…"}
          </span>
        </h2>
        <div className="chat-log">
          {turns.map((t, i) => (
            <div key={t.id} className={`bubble ${t.role}`}>
              <div className="bubble-text">{t.content || (phase === "stream" && i === turns.length - 1 ? "…" : "")}</div>
              {t.role === "assistant" && t.question && (
                <div className="meta-row">
                  {t.judge ? (
                    <span className={`judge ${(t.judge.score ?? 0) >= 70 ? "ok" : "bad"}`}>
                      평가 {t.judge.score}점 (루브릭 평균) ·
                      {(t.judge.rubrics ?? []).map((r) => ` ${r.name} ${r.score}`).join(" ·")}
                      {t.judge.refusal ? " · 정당한 거부" : ""}
                      {t.judge.comment && <em> “{t.judge.comment}”</em>}
                      <span className="judge-by"> · 판정 {t.judgeBy === "gemini-3.5-flash-lite" ? "gemini-3.5-flash-lite" : "qwen3.5:2b 자기평가"}</span>
                    </span>
                  ) : t.judgeError ? (
                    <span className="judge fail">판정 실패 — 평가 모델이 결과를 만들지 못했습니다 (답변은 정상)</span>
                  ) : judgeBusy && i === turns.length - 1 ? (
                    <span className="judge">④ 판정 중… (LLM-as-a-Judge)</span>
                  ) : null}
                  <span className="feedback">
                    <button aria-label="좋아요" className={t.feedback === "up" ? "on" : ""} onClick={() => setFeedback(t.id, "up")}>👍</button>
                    <button aria-label="싫어요" className={t.feedback === "down" ? "on" : ""} onClick={() => setFeedback(t.id, "down")}>👎</button>
                  </span>
                </div>
              )}
              {t.sources && !(phase === "stream" && i === turns.length - 1) && (
                <div className="chips">
                  <button
                    className="chips-toggle"
                    onClick={() => setOpenSrc((m) => ({ ...m, [i]: !m[i] }))}
                  >
                    출처 {t.sources.length}개 {openSrc[i] ? "접기 ▴" : "펼쳐 보기 ▾"}
                  </button>
                  {t.sources[0].score < 0.55 && (
                    <span className="weak-badge">⚠ 최고 유사도 {(t.sources[0].score * 100).toFixed(1)}%</span>
                  )}
                  {openSrc[i] &&
                    t.sources.map((s) => (
                      <button
                        key={s.chunk.id}
                        className={`chip ${s.method === "bm25" ? "bm25" : "vec"}`}
                        onClick={() => setShowSource(t.sources!)}
                      >
                        {s.chunk.id} · {s.chunk.section} · {s.method === "bm25" ? "BM25" : "벡터"} {(s.score * 100).toFixed(0)}%
                      </button>
                    ))}
                </div>
              )}
            </div>
          ))}
          {(phase === "embed" || phase === "search") && (
            <div className="phase-box">
              <span className="spinner" />
              <span>
                {phase === "embed" && embedCached
                  ? "① 질문 임베딩 중 — 캐시된 모델 사용 (다운로드 없음)"
                  : PHASE_LABEL[phase]}
                {dlPct !== null && (
                  <div className="dl-progress">
                    임베딩 모델을 내려받는 중 {dlPct}% — 첫 방문 1회(약 200MB), 이후 브라우저에 캐시됩니다
                  </div>
                )}
              </span>
            </div>
          )}
          {lastHits && phase === "stream" && (
            <div className="hits-box">
              <div className="hits-title">
                <button className="chips-toggle" onClick={() => setHitsOpen((o) => !o)}>
                  ② 검색된 근거 {lastHits.length}개 — 벡터 {lastHits.length - nBm} · BM25 {nBm}{" "}
                  {hitsOpen ? "접기 ▴" : "펼쳐 보기 ▾"}
                </button>
                {lastHits[0].score < 0.55 && (
                  <span className="weak-badge"> ⚠ 최고 유사도 {(lastHits[0].score * 100).toFixed(1)}% — 근거가 약합니다</span>
                )}
              </div>
              {hitsOpen &&
                lastHits.map((h) => (
                  <div key={h.chunk.id} className={`hit-row ${h.method === "bm25" ? "bm25" : "vec"}`}>
                    <span className="hit-id">{h.chunk.id}</span>
                    <span className="hit-sec">{h.chunk.section}</span>
                    <span className="hit-score" style={{ "--w": `${Math.round(h.score * 100)}%` } as CSSProperties}>
                      {h.method === "bm25" ? "BM25" : "벡터"} {(h.score * 100).toFixed(1)}%
                    </span>
                    <span className="hit-text">{h.chunk.text.slice(0, 80)}…</span>
                  </div>
                ))}
              {phase === "stream" && (
                <div className="hits-title" style={{ marginTop: hitsOpen ? ".6rem" : undefined }}>③ 이 근거로 답변을 만듭니다…</div>
              )}
            </div>
          )}
          <div ref={bottomRef} />
        </div>
        <div className="chat-input">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
            placeholder="예: 1.1.1이 뭐야? / 인증서 유효기간은?"
            disabled={phase !== "idle"}
          />
          {phase === "stream" ? (
            <button onClick={stop} className="stop-btn">정지</button>
          ) : (
            <button onClick={ask} disabled={phase !== "idle" || !input.trim()}>
              보내기
            </button>
          )}
        </div>
      </section>

      {showSource && (
        <div className="modal" onClick={() => setShowSource(null)}>
          <div className="modal-body" onClick={(e) => e.stopPropagation()}>
            <h3>근거 조각</h3>
            {showSource.map((s) => (
              <div key={s.chunk.id} className="source-item">
                <div className="source-meta">
                  {s.chunk.id} · {s.chunk.section} · {s.method === "bm25" ? "BM25" : "벡터 유사도"} {(s.score * 100).toFixed(0)}%
                </div>
                <p>{s.chunk.text}</p>
                <a href={s.chunk.url} target="_blank" rel="noreferrer">원문 보기 →</a>
              </div>
            ))}
            <button onClick={() => setShowSource(null)}>닫기</button>
          </div>
        </div>
      )}

      <footer className="footer">
        <p>
          ISMS-P 인증기준 안내 챗봇 — 로컬 실행 데모. 자료: 국가법령정보센터 고시 원문(2024. 7. 24. 시행).
          모델: qwen3.5:2b (ollama) · 임베딩: embeddinggemma-300m (브라우저).
        </p>
      </footer>
    </div>
  );
}

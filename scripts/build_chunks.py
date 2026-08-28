"""고시 별표 7 인증기준 + 신청·심사 관련 조문을 RAG 청크로 만든다.

출력: docs/chunks.json  (id, text, url, section — vector는 임베딩 단계에서 붙인다)

ID 규칙
  IC-<인증기준번호>   예: IC-1.1.1   (Item of Criteria)
  AR-<조번호>         예: AR-19, AR-18-2  (Article)
한 코퍼스 안에서 이 규칙을 바꾸지 않는다.
"""
import html
import json
import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# 현행 고시: 개인정보보호위원회고시 제2024-8호 / 과학기술정보통신부고시 제2024-30호 (2024. 7. 24. 시행)
SOURCE_URL = "https://www.law.go.kr/LSW/admRulLsInfoP.do?admRulSeq=2100000244750"
NOTICE = "정보보호 및 개인정보보호 관리체계 인증 등에 관한 고시 (2024. 7. 24. 시행)"

# 1순위 사용자(인증 준비 실무자)의 질문이 닿는 조문만 넣는다. PRD §5.1 참조.
WANT_ARTICLES = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36}

MIN_LEN = 120


def strip_tags(s):
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    return re.sub(r"[\s 　]+", " ", s).strip()


def build_criteria():
    """별표 7 인증기준 101항목 → 청크."""
    items = json.loads((DOCS / "byl7_items.json").read_text(encoding="utf-8"))
    out = []
    for it in items:
        section = (
            f"별표 7 인증기준 > {it['area']} > "
            f"{it['mid_no']}. {it['mid_name']} > {it['id']} {it['name']}"
        )
        # 상세내용만으로는 주어와 맥락이 흐려지는 항목이 있어 분류 머리를 붙인다.
        text = (
            f"[{it['area']} > {it['mid_no']}. {it['mid_name']} > "
            f"{it['id']} {it['name']}] {it['detail']}"
        )
        out.append({
            "id": f"IC-{it['id']}",
            "text": text,
            "url": SOURCE_URL,
            "section": section,
        })
    return out


def build_articles():
    """고시 조문 → 청크."""
    raw = (DOCS / "고시본문.html").read_text(encoding="utf-8")
    blocks = re.findall(r'<div class="pgroup">(.*?)</div>', raw, flags=re.S)
    out, seen = [], set()
    for b in blocks:
        t = strip_tags(b)
        m = re.match(r"^제(\d+)조(?:의(\d+))?\s*\(([^)]+)\)", t)
        if not m:
            continue
        no = int(m.group(1))
        if no not in WANT_ARTICLES:
            continue
        key = m.group(0)
        if key in seen:
            continue
        seen.add(key)
        label = f"제{no}조" + (f"의{m.group(2)}" if m.group(2) else "")
        cid = f"AR-{no}" + (f"-{m.group(2)}" if m.group(2) else "")
        out.append({
            "id": cid,
            "text": f"[{NOTICE} {label}({m.group(3)})] {t}",
            "url": SOURCE_URL,
            "section": f"고시 본문 > {label}({m.group(3)})",
        })
    return out


criteria, articles = build_criteria(), build_articles()
chunks = criteria + articles

short = [c["id"] for c in chunks if len(c["text"]) < MIN_LEN]
dupes = [i for i in {c["id"] for c in chunks} if [c["id"] for c in chunks].count(i) > 1]

print(f"인증기준 청크: {len(criteria)}개")
print(f"조문 청크:     {len(articles)}개  ({', '.join(c['id'] for c in articles)})")
print(f"합계:          {len(chunks)}개")
print(f"중복 id:       {dupes if dupes else '없음'}")
print(f"{MIN_LEN}자 미만:    {len(short)}개  {short if short else ''}")
print(f"url 누락:      {[c['id'] for c in chunks if not c['url']]}")

out = DOCS / "chunks.json"
out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n저장: {out}")

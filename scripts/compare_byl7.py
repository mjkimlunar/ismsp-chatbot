"""별표 7 인증기준: 2023.10.5본(HWPX 추출)과 2024.7.24 현행본(PDF 추출)의 항목 비교."""
import re
import pathlib

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"

old_text = (DOCS / "별표7-인증기준-원문.txt").read_text(encoding="utf-8")
new_text = (DOCS / "별표7-인증기준-현행-20240724.txt").read_text(encoding="utf-8")

ID = re.compile(r"\b(\d{1,2}\.\d{1,2}\.\d{1,2})\b")


def ids_in(text):
    """등장 순서를 유지한 채 중복 없는 항목번호 목록을 만든다."""
    seen, out = set(), []
    for m in ID.finditer(text):
        v = m.group(1)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def names_old(text):
    """HWPX 추출본은 '1.1.1\n경영진의 참여\n...' 순서라 번호 다음 줄이 항목명."""
    lines = [ln.strip() for ln in text.splitlines()]
    out = {}
    for i, ln in enumerate(lines):
        if ID.fullmatch(ln) and i + 1 < len(lines):
            out.setdefault(ln, lines[i + 1])
    return out


def names_new(text):
    """PDF layout 추출본은 한 줄 안에 '1.1.1  경영진의 참여   상세...' 형태로 섞여 있다."""
    out = {}
    for m in re.finditer(r"(\d{1,2}\.\d{1,2}\.\d{1,2})\s+(\S[^\n]{0,24}?)(?:\s{2,}|$)", text):
        out.setdefault(m.group(1), m.group(2).strip())
    return out


old_ids, new_ids = ids_in(old_text), ids_in(new_text)
o, n = names_old(old_text), names_new(new_text)

print(f"2023본 항목수={len(old_ids)}  2024본 항목수={len(new_ids)}")
print(f"번호 집합 동일: {set(old_ids) == set(new_ids)}")

only_old = sorted(set(old_ids) - set(new_ids))
only_new = sorted(set(new_ids) - set(old_ids))
if only_old:
    print("2023본에만:", only_old)
if only_new:
    print("2024본에만:", only_new)

diff = [(k, o.get(k), n.get(k)) for k in sorted(set(o) & set(n)) if o.get(k) != n.get(k)]
print(f"\n항목명이 다른 번호: {len(diff)}건")
for k, a, b in diff[:15]:
    print(f"  {k}\n    2023: {a}\n    2024: {b}")

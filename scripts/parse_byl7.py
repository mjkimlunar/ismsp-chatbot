"""별표 7 인증기준 파싱 + 현행본 대조.

2023.10.5본 HWPX에서 뽑은 텍스트는 셀 순서가 보존돼 있어 구조 파싱이 가능하다.
2024.7.24 현행본은 PDF뿐이라 표 셀이 병합돼 파싱이 어렵다.
그래서 2023본을 파싱한 뒤, 각 항목의 상세내용이 현행본 PDF 텍스트 안에
그대로 존재하는지 대조해 '별표 7이 개정되었는가'를 판정한다.
"""
import json
import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

SRC = DOCS / "별표7-인증기준-원문.txt"          # 2023.10.5본 (HWPX 추출)
CUR = DOCS / "byl7_raw.txt"                      # 2024.7.24 현행본 (PDF 추출)
OUT = DOCS / "byl7_items.json"

MID = re.compile(r"^(\d{1,2}\.\d{1,2})\.?$")     # 중분류: 1.1
ITEM = re.compile(r"^(\d{1,2}\.\d{1,2}\.\d{1,2})$")  # 항목: 1.1.1
AREA = re.compile(r"^([가-다])\.\s*(.+)$")       # 대분류: 가. 관리체계 수립 및 운영
BLANK = {"", "　", "　"}


def norm(s):
    """공백·개행을 모두 없앤 비교용 문자열. PDF 줄바꿈 위치가 달라도 대조할 수 있다."""
    return re.sub(r"\s+", "", s)


def parse(text):
    lines = [ln.strip() for ln in text.splitlines()]
    items, area, mid_no, mid_name = [], None, None, None
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln in BLANK:
            i += 1
            continue

        m = AREA.match(ln)
        if m and len(ln) < 30:
            area = f"{m.group(1)}. {m.group(2)}"
            i += 1
            continue

        m = MID.match(ln)
        if m:
            mid_no = m.group(1)
            j = i + 1
            while j < len(lines) and lines[j] in BLANK:
                j += 1
            mid_name = lines[j] if j < len(lines) else ""
            i = j + 1
            continue

        m = ITEM.match(ln)
        if m:
            item_no = m.group(1)
            j = i + 1
            while j < len(lines) and lines[j] in BLANK:
                j += 1
            item_name = lines[j] if j < len(lines) else ""
            # 항목명 다음의 연속된 본문 줄을 상세내용으로 모은다
            k, detail = j + 1, []
            while k < len(lines):
                nxt = lines[k]
                if nxt in BLANK:
                    k += 1
                    continue
                if ITEM.match(nxt) or MID.match(nxt) or (AREA.match(nxt) and len(nxt) < 30):
                    break
                detail.append(nxt)
                k += 1
            items.append({
                "id": item_no,
                "area": area,
                "mid_no": mid_no,
                "mid_name": mid_name,
                "name": item_name,
                "detail": " ".join(detail).strip(),
            })
            i = k
            continue
        i += 1
    return items


items = parse(SRC.read_text(encoding="utf-8"))
cur = norm(CUR.read_text(encoding="utf-8"))

print(f"파싱된 항목: {len(items)}개")
by_area = {}
for it in items:
    by_area[it["area"]] = by_area.get(it["area"], 0) + 1
for k, v in by_area.items():
    print(f"  {k}: {v}개")

empty = [it["id"] for it in items if not it["detail"]]
if empty:
    print(f"\n상세내용 비어있음: {empty}")

missing = [it for it in items if norm(it["detail"]) not in cur]
print(f"\n현행본(2024.7.24)에 상세내용이 그대로 없는 항목: {len(missing)}건")
for it in missing[:20]:
    print(f"  {it['id']} {it['name']}: {it['detail'][:60]}...")

OUT.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n저장: {OUT}")

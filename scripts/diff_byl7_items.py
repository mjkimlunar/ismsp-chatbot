"""현행본과 불일치한 6개 항목이 실제 개정인지 추출 오류인지 가른다.

2023본 상세내용의 앞부분을 조금씩 줄여가며 현행본 텍스트에서 찾아,
어디까지 같고 어디부터 갈라지는지를 보여 준다.
"""
import json
import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

items = json.loads((DOCS / "byl7_items.json").read_text(encoding="utf-8"))
cur = re.sub(r"\s+", "", (DOCS / "byl7_raw.txt").read_text(encoding="utf-8"))

TARGETS = ["2.2.3", "3.1.4", "3.1.5", "3.1.6", "3.1.7", "3.4.2"]


def norm(s):
    return re.sub(r"\s+", "", s)


for it in items:
    if it["id"] not in TARGETS:
        continue
    old = norm(it["detail"])
    # 앞에서부터 몇 글자까지 현행본에 그대로 있는지 이분 탐색
    lo, hi = 0, len(old)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if old[:mid] in cur:
            lo = mid
        else:
            hi = mid - 1
    match_len = lo
    print(f"\n===== {it['id']} {it['name']} =====")
    print(f"2023본 길이 {len(old)}자 중 앞 {match_len}자까지 현행본과 동일")
    if match_len == 0:
        print("  (전혀 매칭되지 않음 — 항목 전체가 다르거나 추출 실패)")
        print(f"  2023본: {it['detail'][:120]}")
        continue
    print(f"  갈라지는 지점 직전: ...{old[max(0,match_len-40):match_len]}")
    print(f"  2023본 이후:      {old[match_len:match_len+70]}")
    pos = cur.find(old[:match_len])
    print(f"  현행본 이후:      {cur[pos+match_len:pos+match_len+70]}")

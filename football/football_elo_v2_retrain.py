"""足球 Elo 重訓研究（elo-v2 候選）。

同 tianxi-football/scripts/elo_s2.py 同一套主客獨立 Elo，只重新搵參數：
- 校準：2020/21 季或之前（逐場賽前預測，賽後先更新）
- 驗證：2021/22–2024/25 四季逐季比較現行 elo-s2（HFA60/K22/REG0.70）
- 閘門：四季 RPS 全部唔差過現行，且平均改善 ≥ 0.0005
只係研究輸出，唔郁凍結 S5 矩陣同 dual-v1；升唔升由用戶拍板。
用法：python3 football_elo_v2_retrain.py /tmp/fb/Matches.csv out.json
"""
import csv, json, math, sys, itertools
from collections import defaultdict

SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fb/Matches.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "football_elo_v2_results.json"
VAL = [2021, 2022, 2023, 2024]
FIVE = {"E0", "SP1", "I1", "D1", "F1"}

rows = []
with open(SRC, newline="") as f:
    for r in csv.DictReader(f):
        try:
            hg, ag = int(float(r["FTHome"])), int(float(r["FTAway"]))
        except (ValueError, TypeError):
            continue
        rows.append((r["MatchDate"], r["Division"], r["HomeTeam"], r["AwayTeam"], hg, ag))
rows.sort(key=lambda x: (x[0], x[1]))


def season_of(d):
    y, m = int(d[:4]), int(d[5:7])
    return y if m >= 7 else y - 1


def run(hfa, k0, reg, d0=0.30, d1=0.18):
    hr, ar = defaultdict(lambda: 1500.0), defaultdict(lambda: 1500.0)
    last = {}
    agg = defaultdict(lambda: [0.0, 0, 0, 0.0, 0, 0])  # rps,n,hit | five-league rps,n,hit
    for d, div, h, a, hg, ag in rows:
        s = season_of(d)
        for t in (h, a):
            if last.get(t) is not None and last[t] != s:
                hr[t] = 1500 + reg * (hr[t] - 1500)
                ar[t] = 1500 + reg * (ar[t] - 1500)
            last[t] = s
        eh = 1 / (1 + 10 ** (-((hr[h] + hfa) - ar[a]) / 400))
        ea = 1 - eh
        pd = max(0.10, d0 - d1 * abs(eh - ea))
        p = ((1 - pd) * eh, pd, (1 - pd) * ea)
        res = hg - ag
        act = 0 if res > 0 else (1 if res == 0 else 2)
        y = [0, 0, 0]; y[act] = 1
        rps = 0.5 * ((p[0] - y[0]) ** 2 + (p[0] + p[1] - y[0] - y[1]) ** 2)
        key = "cal" if s <= 2020 else (s if s in VAL else None)
        if key is not None:
            g = agg[key]
            hit = int(p.index(max(p)) == act)
            g[0] += rps; g[1] += 1; g[2] += hit
            if div in FIVE:
                g[3] += rps; g[4] += 1; g[5] += hit
        km = 1.0 if abs(res) <= 1 else (1.5 if abs(res) == 2 else (1.75 if abs(res) == 3 else 2.0))
        delta = k0 * km * ((1.0 if res > 0 else 0.5 if res == 0 else 0.0) - eh)
        hr[h] += delta
        ar[a] -= delta
    return {str(k): {"n": v[1], "rps": v[0] / v[1], "acc": v[2] / v[1],
                     "n5": v[4], "rps5": v[3] / max(1, v[4]), "acc5": v[5] / max(1, v[4])}
            for k, v in agg.items()}


base = run(60, 22, 0.70)
best, best_rps = None, base["cal"]["rps"]
for hfa, k0, reg in itertools.product([50, 60, 70, 80], [16, 20, 24, 28, 32], [0.6, 0.7, 0.8, 0.9]):
    r = run(hfa, k0, reg)
    print(hfa, k0, reg, round(r["cal"]["rps"], 5), flush=True)
    if r["cal"]["rps"] < best_rps:
        best_rps, best = r["cal"]["rps"], ((hfa, k0, reg), r)

if best is None:
    params, cand = (60, 22, 0.70), base
else:
    params, cand = best
seasons = []
for s in VAL:
    b, c = base[str(s)], cand[str(s)]
    seasons.append({"season": f"{str(s)[2:]}{str(s+1)[2:]}", "n": c["n"],
                    "rps_old": b["rps"], "rps_new": c["rps"], "acc_old": b["acc"], "acc_new": c["acc"],
                    "rps5_old": b["rps5"], "rps5_new": c["rps5"], "acc5_old": b["acc5"], "acc5_new": c["acc5"]})
mean_gain = sum(x["rps_old"] - x["rps_new"] for x in seasons) / len(seasons)
gate = all(x["rps_new"] <= x["rps_old"] for x in seasons) and mean_gain >= 0.0005
out = {
    "version": "elo-v2",
    "base_version": "elo-s2",
    "params_old": {"hfa": 60, "k0": 22, "reg": 0.70},
    "params_new": {"hfa": params[0], "k0": params[1], "reg": params[2]},
    "calibrated_on": "≤2020/21",
    "cal_rps_old": base["cal"]["rps"], "cal_rps_new": cand["cal"]["rps"],
    "validation": seasons,
    "mean_rps_gain": mean_gain,
    "gate_pass": gate,
}
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))

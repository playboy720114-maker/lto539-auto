 backtest.py
# 走動外推回測（不偷看未來）：
# 用最近150期學參數 → 在下一期驗證，整段歷史逐日外推。
# 產出：out/backtest_report.csv 與 out/backtest_summary.txt

from pathlib import Path
import itertools, collections
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIRS = [ROOT/"data", ROOT/"數據"]
OUT_DIRS  = [ROOT/"out", ROOT/"出去"]

# 抓 data/ 或 數據/ 最新檔
data_dir = next((d for d in DATA_DIRS if d.exists()), None)
assert data_dir is not None, "找不到資料夾 data/ 或 數據/"
files = sorted(list(data_dir.glob("*.csv"))+list(data_dir.glob("*.xlsx")), key=lambda p:p.stat().st_mtime, reverse=True)
assert files, "data/（或 數據/）沒有檔案"
src = files[0]

# 讀檔
if src.suffix.lower()==".csv":
    df = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
else:
    df = pd.read_excel(src, engine="openpyxl")

need = ["Date","N1","N2","N3","N4","N5"]
df.columns = [str(c).strip() for c in df.columns]
assert all(c in df.columns for c in need), f"CSV 欄位需包含：{need}"

df["Date"] = pd.to_datetime(df["Date"].astype(str).str.replace(r"[./]", "-", regex=True), errors="coerce")
for c in ["N1","N2","N3","N4","N5"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().astype({"N1":int,"N2":int,"N3":int,"N4":int,"N5":int})
df = df.sort_values("Date").reset_index(drop=True)

vals = df[["N1","N2","N3","N4","N5"]].values.tolist()
dates= df["Date"].dt.date.tolist()
ALL  = list(range(1,40))
N    = len(vals)

def window(end_idx,k):
    s = max(0, end_idx-k+1); return vals[s:end_idx+1]

def pick6_simple(end_idx):
    # 近60/100/150 期加權頻率（與 model.py 對齊）
    S, M, L = window(end_idx, min(60,end_idx+1)), window(end_idx, min(100,end_idx+1)), window(end_idx, min(150,end_idx+1))
    def cnt(block):
        c = collections.Counter(itertools.chain.from_iterable(block))
        return {i:c.get(i,0) for i in ALL}
    wS,wM,wL = 0.60,0.25,0.15
    fS,fM,fL = cnt(S), cnt(M), cnt(L)
    score = {n: wS*fS[n]+wM*fM[n]+wL*fL[n] for n in ALL}
    cand  = sorted(ALL, key=lambda x:(-score[x], x))
    # 輕量約束
    def ok(S):
        tails = collections.Counter([x%10 for x in S])
        tens  = collections.Counter([x//10 for x in S])
        if any(v>2 for v in tails.values()): return False
        if any(v>3 for v in tens.values()):  return False
        if max(S)-min(S) < 12:               return False
        return True
    best, bestV = None, -1
    for S in itertools.combinations(cand[:18], 6):
        if not ok(S): continue
        v = sum(score[x] for x in S) - 0.4*sum(1 for a,b in itertools.combinations(S,2) if abs(a-b)<=2)
        if v>bestV: bestV, best = v, tuple(sorted(S))
    if best is None: best = tuple(sorted(cand[:6]))
    tri = tuple(sorted(sorted(best, key=lambda x:(-score[x], x))[:3]))
    return best, tri

rows = []
start_idx = max(1, N-1-365*2)  # 回測範圍：近 1–2 年（可調整）
for i in range(start_idx, N-1):
    # 用截至 i 的資料挑號，驗證在 i+1
    main6, tri3 = pick6_simple(i)
    real_set = set(vals[i+1])
    hit_main = len(set(main6)&real_set)
    hit_tri  = len(set(tri3)&real_set)
    rows.append({
        "Date_pred_for": str(dates[i+1]),
        "Main6": " ".join(f"{x:02d}" for x in main6),
        "Tri3":  " ".join(f"{x:02d}" for x in tri3),
        "Real":  " ".join(f"{x:02d}" for x in sorted(real_set)),
        "HitMain": hit_main,
        "HitTri":  hit_tri,
        "Hit2+":   1 if hit_main>=2 else 0,
        "Hit3+":   1 if hit_main>=3 else 0,
        "HitTri>=1": 1 if hit_tri>=1 else 0
    })

bt = pd.DataFrame(rows)

# 輸出
out_dir = next((d for d in OUT_DIRS if d.exists()), ROOT/"out")
out_dir.mkdir(parents=True, exist_ok=True)
rep_csv = out_dir/"backtest_report.csv"
bt.to_csv(rep_csv, index=False, encoding="utf-8-sig")

# 摘要
def rate(col): return bt[col].mean()*100 if len(bt) else 0.0
summary = [
    f"樣本期數：{len(bt)}（{bt['Date_pred_for'].min()} ~ {bt['Date_pred_for'].max()}）",
    f"主力6碼 命中≥2：{rate('Hit2+'):.1f}%",
    f"主力6碼 命中≥3：{rate('Hit3+'):.1f}%",
    f"內三碼 命中≥1：{rate('HitTri>=1'):.1f}%"
]
(rep_txt := out_dir/"backtest_summary.txt").write_text("\n".join(summary), encoding="utf-8")
print("\n".join(summary))
print(f"已輸出：{rep_csv.name}、{rep_txt.name}")

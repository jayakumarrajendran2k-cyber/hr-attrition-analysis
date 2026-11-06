"""HR Analytics - Employee Attrition & Performance
Run: python hr_analysis.py  -> creates hr_attrition_data.csv + prints analysis and insights.
NOTE: dataset is synthetic (practice data). Swap in a real one (e.g. IBM HR Analytics on Kaggle) - same columns work."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42); n = 1470

# ---------- 1. Build dataset ----------
df = pd.DataFrame({"EmployeeID": range(1001, 1001+n)})
df["Age"] = np.clip(rng.normal(37, 9, n), 19, 60).astype(int)
df["Gender"] = rng.choice(["Male", "Female"], n, p=[.58, .42])
df["Department"] = rng.choice(["Sales", "IT", "HR", "Finance", "Operations"], n, p=[.28, .27, .07, .13, .25])
bump = df["Department"].map({"Sales": 0, "IT": 9000, "HR": -2000, "Finance": 6000, "Operations": -3000})
df["MonthlyIncome"] = (np.clip(16000 + df["Age"]*750 + bump + rng.normal(0, 9000, n), 18000, 150000) // 100 * 100).astype(int)
df["YearsAtCompany"] = np.minimum((df["Age"]-19) * rng.beta(1.2, 3, n) * 0.9, 30).astype(int)
df["YearsSinceLastPromotion"] = np.minimum(df["YearsAtCompany"], rng.integers(0, 9, n))
df["OverTime"] = rng.choice(["Yes", "No"], n, p=[.28, .72])
df["JobSatisfaction"] = rng.choice([1, 2, 3, 4], n, p=[.19, .20, .31, .30])
df["WorkLifeBalance"] = rng.choice([1, 2, 3, 4], n, p=[.06, .23, .52, .19])
df["PerformanceRating"] = rng.choice([2, 3, 4, 5], n, p=[.05, .62, .27, .06])
df["DistanceFromHome"] = rng.integers(1, 30, n)
z = (-3.55 + 1.35*(df.OverTime == "Yes") + 0.75*(df.JobSatisfaction <= 2) + 0.55*(df.YearsAtCompany <= 2)
     - 0.00002*(df.MonthlyIncome-50000) + 0.03*df.DistanceFromHome + 0.08*df.YearsSinceLastPromotion
     + 0.5*(df.WorkLifeBalance == 1) + 0.35*(df.Department == "Sales"))
df["Attrition"] = np.where(rng.random(n) < 1/(1+np.exp(-z)), "Yes", "No")
df.to_csv("hr_attrition_data.csv", index=False)

# ---------- 2. Analysis ----------
df["Left"] = (df.Attrition == "Yes").astype(int)
df["TenureBand"] = pd.cut(df.YearsAtCompany, [-1, 2, 5, 10, 99], labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"])
df["IncomeBand"] = pd.cut(df.MonthlyIncome, [0, 35000, 50000, 65000, 1e9], labels=["Below 35k", "35k-50k", "50k-65k", "65k+"])
df["AgeBand"] = pd.cut(df.Age, [0, 29, 39, 49, 99], labels=["Under 30", "30-39", "40-49", "50+"])
df["Satisfaction"] = df.JobSatisfaction.map({1: "Low", 2: "Medium", 3: "High", 4: "Very high"})
def rate(col, order=None):
    g = df.groupby(col, observed=True).Left.agg(["mean", "size"])
    if order: g = g.reindex(order)
    return [(str(i), round(r*100, 1), int(s)) for i, (r, s) in g.iterrows()]
R = {"Department": rate("Department"), "Overtime": rate("OverTime"), "Tenure": rate("TenureBand"),
     "Monthly income (₹)": rate("IncomeBand"), "Job satisfaction": rate("Satisfaction", ["Low", "Medium", "High", "Very high"]),
     "Age group": rate("AgeBand")}
total, left = len(df), int(df.Left.sum()); overall = round(left/total*100, 1)
perf = df.groupby("Attrition").PerformanceRating.mean().round(2).to_dict()
print(f"Headcount {total} | Leavers {left} | Attrition {overall}% | Avg rating {perf}")
for k, v in R.items(): print(k, v)

# ---------- 3. Insights ----------
def d(k, lbl): return next(r for r in R[k] if r[0] == lbl)[1]
top = max(R["Department"], key=lambda r: r[1])
ds = df.Satisfaction.isin(["Low","Medium"]); dsr = round(df[ds].Left.mean()*100,1); hsr = round(df[~ds].Left.mean()*100,1)
ins = [f"Overtime is the biggest driver: {d('Overtime','Yes')}% of overtime staff leave vs {d('Overtime','No')}% without it.",
       f"Early-tenure risk: {d('Tenure','0-2 yrs')}% of employees with 0-2 years leave, vs {d('Tenure','6-10 yrs')}% at 6-10 years.",
       f"Dissatisfied staff (low/medium satisfaction) leave at {dsr}% vs {hsr}% for satisfied staff.",
       f"{top[0]} has the highest department attrition at {top[1]}% (company average {overall}%).",
       f"Pay matters: {d('Monthly income (₹)','Below 35k')}% leave in the lowest income band vs {d('Monthly income (₹)','65k+')}% in the highest.",
       f"Average performance rating is almost the same for leavers ({perf['Yes']}) and stayers ({perf['No']}), so good performers are leaving too."]
rec = ["Cap overtime and track workload in high-overtime teams.", "Run a 90-day onboarding and mentoring plan for new joiners.",
       "Review pay and promotion cycles in the lowest income band.", "Run quarterly satisfaction pulse surveys and act on low scores."]
print("\nKEY INSIGHTS")
for i in ins: print("-", i)
print("\nRECOMMENDATIONS")
for r in rec: print("-", r)

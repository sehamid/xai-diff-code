"""
XAI-DIF - Explainable AI-Driven Decision Intelligence Framework
Evaluation pipeline for the paper's results (Sections 6.1-6.7).

Reproduces, from the public Enterprise Agile Sprint Delivery Risk dataset:
  - Predictive performance (RF / Gradient Boosting): precision, recall, F1, ROC-AUC
  - Confusion matrix
  - H3: multi-source vs planning-only (10-fold CV) with Wilcoxon p, Cliff's delta, bootstrap CI
  - Cycle-time regression (MAE, RMSE, R2)
  - H1: explanation faithfulness (local surrogate fidelity, rank agreement)
  - Decision-Intelligence recommendation relevance vs recorded failure cause
  - H2: decision-layer ablation
  - A worked example

Usage:
    python src/run_evaluation.py [path/to/enterprise_agile_sprint_risk.csv]

If no path is given it defaults to data/enterprise_agile_sprint_risk.csv
Outputs (results.json + figures) are written to ./outputs/
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from scipy.stats import spearmanr, wilcoxon

SEED = 42
np.random.seed(SEED)
NAVY, RED, GRAY = "#1F3864", "#C0392B", "#95A5A6"

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "enterprise_agile_sprint_risk.csv")
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
if not os.path.exists(DATA):
    sys.exit(f"Dataset not found at '{DATA}'. See data/README.md for the download link.")
df = pd.read_csv(DATA)

# ---- feature groups (multi-source mapping) ----
PLANNING = ["planned_story_points", "team_capacity_points", "scope_added_ratio", "scope_removed_ratio",
            "unplanned_work_ratio", "backlog_refinement_score", "previous_sprint_carryover_ratio"]
DEPS = ["dependency_count", "external_dependency_ratio", "blocked_items_count"]
QUALITY = ["technical_debt_index", "test_automation_coverage", "code_review_compliance", "production_incidents"]
TEAM = ["team_size", "remote_team_ratio", "avg_team_tenure_years", "team_stability_index", "team_member_change",
        "psychological_safety_score", "requirements_clarity_score", "historical_velocity_cv",
        "ceremony_hours_per_week", "holiday_days_in_sprint"]
CONTEXT_NUM = ["sprint_duration_days"]
DRIVER_FEATS = PLANNING + DEPS + QUALITY + TEAM          # controllable process factors for recommendations
# realized-outcome fields excluded to avoid target leakage:
LEAK = ["completed_story_points", "spillover_story_points", "escaped_defects", "median_cycle_time_days",
        "delivery_risk_probability", "sprint_outcome", "primary_failure_reason", "sprint_id", "team_id",
        "sprint_start_date"]

# label + one-hot context
y = (df["sprint_outcome"] == "Failed").astype(int)
ctx = pd.get_dummies(df[["industry", "region", "agile_method"]].astype(str))
Xnum = df[DRIVER_FEATS + CONTEXT_NUM].astype(float).copy()
X = pd.concat([Xnum, ctx.astype(float)], axis=1)
FEATS = list(X.columns)

R = {"n": int(len(df)), "fail_rate": round(float(y.mean()), 3), "n_features": len(FEATS)}
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

def ev(m, Xe, ye):
    p, pb = m.predict(Xe), m.predict_proba(Xe)[:, 1]
    return dict(precision=round(precision_score(ye, p), 3), recall=round(recall_score(ye, p), 3),
                f1=round(f1_score(ye, p), 3), roc_auc=round(roc_auc_score(ye, pb), 3))

rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.08, random_state=SEED).fit(Xtr, ytr)
R["clf"] = {"RandomForest": ev(rf, Xte, yte), "GradientBoosting": ev(gb, Xte, yte)}

# ---- confusion matrix (Section 6.1 / Figure 5) ----
pred = rf.predict(Xte)
cm = confusion_matrix(yte, pred)
tn, fp, fn, tp = cm.ravel()
R["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                  "predicted_failed": int(tp + fp), "actual_failed": int(fn + tp)}

# ---- H3: multi-source vs planning-only (10-fold CV) + Wilcoxon + Cliff's delta + bootstrap CI ----
skf = StratifiedKFold(10, shuffle=True, random_state=SEED)
multi = cross_val_score(RandomForestClassifier(n_estimators=250, random_state=SEED, n_jobs=-1), X, y, cv=skf, scoring="f1")
sing = cross_val_score(RandomForestClassifier(n_estimators=250, random_state=SEED, n_jobs=-1), df[PLANNING], y, cv=skf, scoring="f1")

def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((xi > yj) for xi in a for yj in b)
    lt = sum((xi < yj) for xi in a for yj in b)
    return (gt - lt) / (len(a) * len(b))

diff = multi - sing
rng = np.random.default_rng(SEED)
boot = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(5000)]
R["h3"] = {"multi_f1_mean": round(float(multi.mean()), 3), "multi_f1_sd": round(float(multi.std()), 3),
           "single_f1_mean": round(float(sing.mean()), 3), "single_f1_sd": round(float(sing.std()), 3),
           "wilcoxon_p": float(f"{wilcoxon(multi, sing).pvalue:.2e}"),
           "cliffs_delta": round(cliffs_delta(multi, sing), 3),
           "paired_diff_mean": round(float(diff.mean()), 4),
           "paired_diff_ci95": [round(float(np.percentile(boot, 2.5)), 4),
                                round(float(np.percentile(boot, 97.5)), 4)]}

# ---- cycle-time regression (Section 6.3) ----
yr = df["median_cycle_time_days"]
Xtr2, Xte2, rtr, rte = train_test_split(X, yr, test_size=0.25, random_state=SEED)
reg = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.06, random_state=SEED).fit(Xtr2, rtr)
rp = reg.predict(Xte2)
R["reg"] = {"target": "median_cycle_time_days", "MAE": round(mean_absolute_error(rte, rp), 2),
            "RMSE": round(mean_squared_error(rte, rp) ** 0.5, 2), "R2": round(r2_score(rte, rp), 3)}

# ---- global importance ----
perm = permutation_importance(rf, Xte, yte, n_repeats=8, random_state=SEED, n_jobs=-1)
gimp = pd.Series(perm.importances_mean, index=FEATS).sort_values(ascending=False)
R["top_features"] = [[k, round(float(v), 4)] for k, v in gimp.head(8).items()]

# ---- local drivers + explanation faithfulness (H1) ----
fstd = Xtr[DRIVER_FEATS].std().replace(0, 1); fmean = Xtr[DRIVER_FEATS].mean()
csign = {f: (np.sign(np.corrcoef(Xtr[f], ytr)[0, 1]) or 1.0) for f in DRIVER_FEATS}
gi = gimp.reindex(DRIVER_FEATS).fillna(0)

def drivers(inst, top=4):
    z = (inst[DRIVER_FEATS] - fmean) / fstd
    c = pd.Series({f: gi[f] * z[f] * csign[f] for f in DRIVER_FEATS}).sort_values(ascending=False)
    return c[c > 0].head(top)

def surrogate(inst, ns=600, kw=0.75, seed=0):
    r = np.random.default_rng(seed); Z = r.normal(0, 1, (ns, len(DRIVER_FEATS)))
    rows = []
    for zz in Z:
        b = inst.copy(); b[DRIVER_FEATS] = inst[DRIVER_FEATS].values + zz * fstd.values; rows.append(b)
    Xp = pd.DataFrame(rows, columns=FEATS)
    fp = rf.predict_proba(Xp)[:, 1]; wts = np.exp(-((Z ** 2).sum(1)) / (2 * kw ** 2))
    lin = LinearRegression().fit(Xp[DRIVER_FEATS], fp, sample_weight=wts)
    return lin.score(Xp[DRIVER_FEATS], fp, sample_weight=wts)

proba = rf.predict_proba(Xte)[:, 1]
flagged_top = Xte.index[np.argsort(-proba)[:40]]
fid = [surrogate(Xte.loc[i], seed=k) for k, i in enumerate(flagged_top)]
agr = []
for i in flagged_top:
    dr = drivers(Xte.loc[i], top=len(DRIVER_FEATS))
    rho, _ = spearmanr(np.abs(dr.reindex(DRIVER_FEATS).fillna(0).values), gi.reindex(DRIVER_FEATS).values)
    agr.append(rho)
R["faith"] = {"fidelity_mean": round(float(np.mean(fid)), 3), "fidelity_min": round(float(np.min(fid)), 3),
              "fidelity_max": round(float(np.max(fid)), 3), "agreement_mean": round(float(np.nanmean(agr)), 3)}

# ---- recommendation relevance vs recorded failure cause (Section 6.5) ----
REASON_FEATS = {
    "Unplanned work": ["unplanned_work_ratio", "blocked_items_count"],
    "Scope volatility": ["scope_added_ratio", "scope_removed_ratio", "requirements_clarity_score"],
    "Dependency delays": ["dependency_count", "external_dependency_ratio", "blocked_items_count"],
    "Team instability": ["team_member_change", "team_stability_index", "avg_team_tenure_years"],
    "Planning / clarity": ["requirements_clarity_score", "backlog_refinement_score", "previous_sprint_carryover_ratio"],
    "Quality / tech debt": ["technical_debt_index", "test_automation_coverage", "code_review_compliance", "production_incidents"]}
test_reason = df.loc[Xte.index, "primary_failure_reason"]
failed_idx = [i for i in Xte.index if isinstance(test_reason[i], str)]
K = 3
per = {r: {"hit": 0, "tot": 0} for r in REASON_FEATS}; hits = 0
for i in failed_idx:
    reason = test_reason[i]
    if reason not in REASON_FEATS:
        continue
    topk = set(drivers(Xte.loc[i], top=K).index)
    ok = len(topk & set(REASON_FEATS[reason])) > 0
    per[reason]["tot"] += 1; per[reason]["hit"] += int(ok); hits += int(ok)
tot = sum(v["tot"] for v in per.values())
R["reco_relevance"] = {"k": K, "overall": round(hits / tot, 3), "n_failed_eval": tot,
                       "per_reason": {r: round(v["hit"] / v["tot"], 3) for r, v in per.items() if v["tot"] > 0}}

# ---- PMBOK-grounded recommendation rules ----
RULES = {
    "unplanned_work_ratio": ("Risk, Scope", "Protect capacity; triage and defer non-critical intake."),
    "scope_added_ratio": ("Scope", "Freeze scope; route new requests through change control."),
    "scope_removed_ratio": ("Scope", "Improve upfront planning and estimation to curb scope churn."),
    "blocked_items_count": ("Risk, Schedule", "Assign owners to blockers; add a daily blocker review."),
    "dependency_count": ("Integration, Risk", "Map and de-risk dependencies; add integration checkpoints."),
    "external_dependency_ratio": ("Integration, Risk", "Establish SLAs with external teams; add buffers."),
    "technical_debt_index": ("Quality", "Allocate refactoring capacity; set a tech-debt budget."),
    "test_automation_coverage": ("Quality", "Invest in automated regression coverage."),
    "code_review_compliance": ("Quality", "Enforce code-review gates before merge."),
    "production_incidents": ("Quality, Risk", "Stabilize production; add monitoring and post-incident reviews."),
    "team_member_change": ("Resource", "Stabilize team membership; onboard and pair new members."),
    "team_stability_index": ("Resource", "Reduce churn; protect the core team for the sprint."),
    "avg_team_tenure_years": ("Resource", "Invest in ramp-up and knowledge sharing."),
    "psychological_safety_score": ("Stakeholder", "Hold retros; address team dynamics and safety."),
    "requirements_clarity_score": ("Scope", "Clarify acceptance criteria before commitment."),
    "backlog_refinement_score": ("Scope", "Strengthen backlog refinement and grooming."),
    "historical_velocity_cv": ("Schedule", "Reduce velocity variability; right-size commitments."),
    "previous_sprint_carryover_ratio": ("Schedule", "Reduce WIP and commit realistically."),
    "planned_story_points": ("Schedule", "Align commitment to proven capacity."),
    "team_capacity_points": ("Schedule", "Re-baseline commitment against available capacity."),
    "team_size": ("Resource", "Right-size the team to the workload."),
    "remote_team_ratio": ("Communication", "Strengthen async communication norms."),
    "ceremony_hours_per_week": ("Communication", "Streamline ceremonies to protect delivery time."),
    "holiday_days_in_sprint": ("Schedule", "Account for holidays in capacity planning.")}

# ---- worked example (Section 6.7) ----
ex = failed_idx[int(np.argmax(proba[[list(Xte.index).index(i) for i in failed_idx]]))]
dr = drivers(Xte.loc[ex]); mx = dr.max()
R["example"] = {"reason": test_reason[ex], "p_risk": round(float(rf.predict_proba(Xte.loc[ex].to_frame().T)[0, 1]), 2),
                "recs": [{"driver": f, "pmbok": RULES.get(f, ("General", "Investigate with the team."))[0],
                          "action": RULES.get(f, ("General", "Investigate with the team."))[1],
                          "priority": "High" if c >= 0.5 * mx else "Medium", "contribution": round(float(c), 3)}
                         for f, c in dr.items()]}

# ---- H2 ablation (on predicted-failed set = true + false positives, matching the confusion matrix) ----
thr = 0.5
safe = Xtr[ytr == 0][DRIVER_FEATS].median()
flagged = Xte.index[pred == 1]          # predicted-failed (TP + FP); equals Figure-5 positive column

def apply_recs(k):
    Xs = Xte.loc[flagged].copy()
    for i in Xs.index:
        for f in drivers(Xte.loc[i], top=k).index:
            Xs.loc[i, f] = Xs.loc[i, f] + 0.7 * (safe[f] - Xs.loc[i, f])
    return round(float((rf.predict_proba(Xs)[:, 1] < thr).mean()), 3)

R["ablation"] = {"flagged": int(len(flagged)), "prediction_only": 0.0,
                 "plus_top1": apply_recs(1), "plus_top2": apply_recs(2)}

# ---- write results ----
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(R, f, indent=1)

# ---- figures ----
def save(fig, n):
    fig.savefig(os.path.join(OUT, n), dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

fig = plt.figure(figsize=(7, 4.2)); gimp.head(12)[::-1].plot(kind="barh", color=NAVY)
plt.title("Global feature importance (permutation)"); plt.xlabel("mean importance"); save(fig, "global_importance.png")
fig = plt.figure(figsize=(4.6, 4.2)); fpr, tpr, _ = roc_curve(yte, rf.predict_proba(Xte)[:, 1])
plt.plot(fpr, tpr, color=NAVY, lw=2, label=f"RF (AUC={R['clf']['RandomForest']['roc_auc']})"); plt.plot([0, 1], [0, 1], "--", color=GRAY)
plt.xlabel("false positive rate"); plt.ylabel("true positive rate"); plt.title("ROC - sprint-failure classifier"); plt.legend(loc="lower right"); save(fig, "roc.png")
fig = plt.figure(figsize=(4.4, 3.6)); plt.bar(["Multi-source", "Planning only"], [R["h3"]["multi_f1_mean"], R["h3"]["single_f1_mean"]],
    yerr=[R["h3"]["multi_f1_sd"], R["h3"]["single_f1_sd"]], color=[NAVY, GRAY], capsize=5); plt.ylabel("F1 (10-fold CV)")
plt.title("H3: multi vs single-source"); save(fig, "h3_multi_vs_single.png")
fig = plt.figure(figsize=(4.6, 4.2)); plt.scatter(rte, rp, s=8, alpha=0.35, color=NAVY); lim = [rte.min(), rte.max()]
plt.plot(lim, lim, "r--", lw=1); plt.xlabel("actual cycle time (days)"); plt.ylabel("predicted"); plt.title(f"Cycle-time regression (R2={R['reg']['R2']})"); save(fig, "cycle_time_regression.png")
fig = plt.figure(figsize=(4.8, 3.6)); plt.bar(["Prediction\nonly", "+ top-1", "+ top-2"],
    [R["ablation"]["prediction_only"], R["ablation"]["plus_top1"], R["ablation"]["plus_top2"]], color=[GRAY, "#5B7BB4", NAVY])
plt.ylabel("share predicted-safe"); plt.title("Decision-layer ablation (H2)"); save(fig, "ablation.png")
pr = R["reco_relevance"]["per_reason"]
fig = plt.figure(figsize=(6.2, 3.4)); plt.barh(list(pr.keys())[::-1], list(pr.values())[::-1], color=NAVY)
plt.axvline(R["reco_relevance"]["overall"], color=RED, ls="--")
plt.xlabel(f"recommendation relevance @{R['reco_relevance']['k']}  (overall={R['reco_relevance']['overall']})"); plt.xlim(0, 1)
plt.title("Recommendations vs recorded failure cause"); save(fig, "recommendation_relevance.png")

print(json.dumps(R, indent=1))
print(f"\nResults and figures written to ./{OUT}/")

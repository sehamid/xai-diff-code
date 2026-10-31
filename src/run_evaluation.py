"""
XAI-DIF - Explainable AI-Driven Decision Intelligence Framework
Evaluation pipeline for the paper's results (Section 6), v1.1.

Reproduces, from the public Enterprise Agile Sprint Delivery Risk dataset:
  6.1  Predictive performance (Random Forest, Gradient Boosting, XGBoost) and confusion matrix
  6.2  Feature-group comparison (all signal groups vs planning-only), 10-fold CV
  6.3  Cycle-time regression
  6.4  Explanation faithfulness: LIME local-surrogate fit and SHAP-LIME rank agreement
  6.5  Recommendation relevance vs the recorded failure cause, with baselines
       (same global top-3 for every sprint; random 3 features)
  6.6  What-if sensitivity analysis of the decision layer (model-based, not an outcome study)
  6.7  Worked example

Local explanations are exact SHAP values (TreeExplainer) of the Random Forest.

Usage:
    python src/run_evaluation.py [path/to/enterprise_agile_sprint_risk.csv]
Outputs (results.json + figures) are written to ./outputs/
"""
import os, sys, json
from math import comb
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular
from joblib import Parallel, delayed
from xgboost import XGBClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from scipy.stats import spearmanr, wilcoxon

SEED = 42
np.random.seed(SEED)
NAVY, RED, GRAY, STEEL = "#1F3864", "#C0392B", "#95A5A6", "#5B7BB4"

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "enterprise_agile_sprint_risk.csv")
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
if not os.path.exists(DATA):
    sys.exit(f"Dataset not found at '{DATA}'. See data/README.md for the download link.")
df = pd.read_csv(DATA)

# ---- feature groups ----
PLANNING = ["planned_story_points", "team_capacity_points", "scope_added_ratio", "scope_removed_ratio",
            "unplanned_work_ratio", "backlog_refinement_score", "previous_sprint_carryover_ratio"]
DEPS = ["dependency_count", "external_dependency_ratio", "blocked_items_count"]
QUALITY = ["technical_debt_index", "test_automation_coverage", "code_review_compliance", "production_incidents"]
TEAM = ["team_size", "remote_team_ratio", "avg_team_tenure_years", "team_stability_index", "team_member_change",
        "psychological_safety_score", "requirements_clarity_score", "historical_velocity_cv",
        "ceremony_hours_per_week", "holiday_days_in_sprint"]
CONTEXT_NUM = ["sprint_duration_days"]
DRIVER_FEATS = PLANNING + DEPS + QUALITY + TEAM          # controllable process factors (24)
# realized-outcome fields are excluded to avoid target leakage:
LEAK = ["completed_story_points", "spillover_story_points", "escaped_defects", "median_cycle_time_days",
        "delivery_risk_probability", "sprint_outcome", "primary_failure_reason", "sprint_id", "team_id",
        "sprint_start_date"]

y = (df["sprint_outcome"] == "Failed").astype(int)
ctx = pd.get_dummies(df[["industry", "region", "agile_method"]].astype(str))
X = pd.concat([df[DRIVER_FEATS + CONTEXT_NUM].astype(float), ctx.astype(float)], axis=1)
FEATS = list(X.columns)
DIDX = [FEATS.index(f) for f in DRIVER_FEATS]

R = {"n": int(len(df)), "fail_rate": round(float(y.mean()), 3), "n_features": len(FEATS)}
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

def ev(m, Xe, ye):
    p, pb = m.predict(Xe), m.predict_proba(Xe)[:, 1]
    return dict(precision=round(precision_score(ye, p), 3), recall=round(recall_score(ye, p), 3),
                f1=round(f1_score(ye, p), 3), roc_auc=round(roc_auc_score(ye, pb), 3))

# ---- 6.1 predictive layer ----
rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.08, random_state=SEED).fit(Xtr, ytr)
xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                    eval_metric="logloss", random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
R["clf"] = {"RandomForest": ev(rf, Xte, yte), "GradientBoosting": ev(gb, Xte, yte), "XGBoost": ev(xgb, Xte, yte)}
pred = rf.predict(Xte)
proba = rf.predict_proba(Xte)[:, 1]
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
R["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                  "predicted_failed": int(tp + fp), "actual_failed": int(fn + tp)}

# ---- 6.2 feature-group comparison (all groups vs planning-only) ----
skf = StratifiedKFold(10, shuffle=True, random_state=SEED)
allg = cross_val_score(RandomForestClassifier(n_estimators=250, random_state=SEED, n_jobs=-1), X, y, cv=skf, scoring="f1")
plan = cross_val_score(RandomForestClassifier(n_estimators=250, random_state=SEED, n_jobs=-1), df[PLANNING], y, cv=skf, scoring="f1")
def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (sum(xi > yj for xi in a for yj in b) - sum(xi < yj for xi in a for yj in b)) / (len(a) * len(b))
diff = allg - plan
rng = np.random.default_rng(SEED)
boot = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(5000)]
R["h3"] = {"multi_f1_mean": round(float(allg.mean()), 3), "multi_f1_sd": round(float(allg.std()), 3),
           "single_f1_mean": round(float(plan.mean()), 3), "single_f1_sd": round(float(plan.std()), 3),
           "wilcoxon_p": float(f"{wilcoxon(allg, plan).pvalue:.2e}"),
           "cliffs_delta": round(cliffs_delta(allg, plan), 3),
           "paired_diff_mean": round(float(diff.mean()), 4),
           "paired_diff_ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]}

# ---- 6.3 cycle-time regression ----
yr = df["median_cycle_time_days"]
Xtr2, Xte2, rtr, rte = train_test_split(X, yr, test_size=0.25, random_state=SEED)
reg = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.06, random_state=SEED).fit(Xtr2, rtr)
rp = reg.predict(Xte2)
R["reg"] = {"target": "median_cycle_time_days", "MAE": round(mean_absolute_error(rte, rp), 2),
            "RMSE": round(mean_squared_error(rte, rp) ** 0.5, 2), "R2": round(r2_score(rte, rp), 3)}

# ---- global permutation importance (Figure 4) ----
perm = permutation_importance(rf, Xte, yte, n_repeats=8, random_state=SEED, n_jobs=-1)
gimp = pd.Series(perm.importances_mean, index=FEATS).sort_values(ascending=False)
R["top_features"] = [[k, round(float(v), 4)] for k, v in gimp.head(8).items()]

# ---- exact SHAP values for every failed or flagged test sprint ----
reason = df.loc[Xte.index, "primary_failure_reason"]
failed_idx = [i for i in Xte.index if isinstance(reason[i], str)]
flagged = list(Xte.index[pred == 1])
need = sorted(set(failed_idx) | set(flagged))
def _shap_chunk(rows):
    e = shap.TreeExplainer(rf)
    v = e.shap_values(Xte.loc[rows], check_additivity=False)
    v = np.asarray(v)
    return v[:, :, 1] if v.ndim == 3 else v[1]
chunks = np.array_split(np.array(need), max(1, os.cpu_count() or 1))
parts = Parallel(n_jobs=-1)(delayed(_shap_chunk)(list(c)) for c in chunks if len(c))
SV = pd.DataFrame(np.vstack(parts), index=[i for c in chunks if len(c) for i in c], columns=FEATS)
R["shap_rows_explained"] = int(len(SV))

def shap_drivers(i, k=3):
    s = SV.loc[i, DRIVER_FEATS].sort_values(ascending=False)
    return s[s > 0].head(k)

global_rank = SV[DRIVER_FEATS].abs().mean().sort_values(ascending=False)
GLOBAL3 = list(global_rank.head(3).index)
R["global_shap_top3"] = GLOBAL3

# ---- 6.4 faithfulness: LIME local fit and SHAP-LIME agreement (40 highest-risk sprints) ----
top40 = list(Xte.index[np.argsort(-proba)[:40]])
lexp = lime.lime_tabular.LimeTabularExplainer(Xtr.values, feature_names=FEATS, class_names=["ok", "failed"],
                                              mode="classification", random_state=SEED)
f_pp = lambda a: rf.predict_proba(pd.DataFrame(a, columns=FEATS))
lime_r2, agree = [], []
for i in top40:
    e = lexp.explain_instance(Xte.loc[i].values, f_pp, num_features=len(FEATS), num_samples=3000)
    lime_r2.append(e.score)
    w = dict(e.as_map()[1])
    lw = np.array([abs(w.get(j, 0.0)) for j in DIDX])
    sw = np.abs(SV.loc[i, DRIVER_FEATS].values) if i in SV.index else np.abs(np.asarray(_shap_chunk([i]))[0][DIDX])
    agree.append(spearmanr(sw, lw)[0])
R["faith"] = {"lime_r2_mean": round(float(np.mean(lime_r2)), 3), "lime_r2_min": round(float(np.min(lime_r2)), 3),
              "lime_r2_max": round(float(np.max(lime_r2)), 3),
              "shap_lime_spearman_mean": round(float(np.nanmean(agree)), 3),
              "shap_lime_spearman_sd": round(float(np.nanstd(agree)), 3), "n": len(top40)}

# ---- 6.5 recommendation relevance vs recorded failure cause, with baselines ----
REASON_FEATS = {
    "Unplanned work": ["unplanned_work_ratio", "blocked_items_count"],
    "Scope volatility": ["scope_added_ratio", "scope_removed_ratio", "requirements_clarity_score"],
    "Dependency delays": ["dependency_count", "external_dependency_ratio", "blocked_items_count"],
    "Team instability": ["team_member_change", "team_stability_index", "avg_team_tenure_years"],
    "Planning / clarity": ["requirements_clarity_score", "backlog_refinement_score", "previous_sprint_carryover_ratio"],
    "Quality / tech debt": ["technical_debt_index", "test_automation_coverage", "code_review_compliance", "production_incidents"]}
K = 3
nD = len(DRIVER_FEATS)
per = {r: {"n": 0, "shap": 0, "global": 0, "random": 0.0} for r in REASON_FEATS}
for i in failed_idx:
    r = reason[i]
    if r not in REASON_FEATS:
        continue
    tgt = set(REASON_FEATS[r])
    per[r]["n"] += 1
    per[r]["shap"] += int(len(set(shap_drivers(i, K).index) & tgt) > 0)
    per[r]["global"] += int(len(set(GLOBAL3) & tgt) > 0)
    per[r]["random"] += 1 - comb(nD - len(tgt), K) / comb(nD, K)
N = sum(v["n"] for v in per.values())
R["reco_relevance"] = {
    "k": K, "n_failed_eval": N,
    "overall": {"shap": round(sum(v["shap"] for v in per.values()) / N, 3),
                "global": round(sum(v["global"] for v in per.values()) / N, 3),
                "random": round(sum(v["random"] for v in per.values()) / N, 3)},
    "per_reason": {r: {"n": v["n"], "shap": round(v["shap"] / v["n"], 3), "global": round(v["global"] / v["n"], 3),
                       "random": round(v["random"] / v["n"], 3)} for r, v in per.items() if v["n"]}}
minor = [r for r in REASON_FEATS if not (set(GLOBAL3) & set(REASON_FEATS[r]))]
if minor:
    nm = sum(per[r]["n"] for r in minor)
    R["reco_relevance"]["causes_outside_global_top3"] = {
        "causes": minor, "n": nm,
        "shap": round(sum(per[r]["shap"] for r in minor) / nm, 3),
        "global": round(sum(per[r]["global"] for r in minor) / nm, 3)}

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

# ---- 6.7 worked example: highest-risk failed test sprint ----
ex = max(failed_idx, key=lambda i: proba[list(Xte.index).index(i)])
dr = shap_drivers(ex, 4); mx = dr.max()
R["example"] = {"reason": reason[ex], "p_risk": round(float(proba[list(Xte.index).index(ex)]), 2),
                "recs": [{"driver": f, "pmbok": RULES[f][0], "action": RULES[f][1],
                          "priority": "High" if c >= 0.5 * mx else "Medium", "shap": round(float(c), 3)}
                         for f, c in dr.items()]}

# ---- 6.6 what-if sensitivity analysis on the predicted-failed set ----
thr = 0.5
safe = Xtr[ytr == 0][DRIVER_FEATS].median()
def what_if(k, mode):
    Xs = Xte.loc[flagged].copy()
    for i in Xs.index:
        feats = list(shap_drivers(i, k).index) if mode == "shap" else GLOBAL3[:k]
        for f in feats:
            Xs.loc[i, f] = Xs.loc[i, f] + 0.7 * (safe[f] - Xs.loc[i, f])
    return round(float((rf.predict_proba(Xs)[:, 1] < thr).mean()), 3)
R["ablation"] = {"flagged": len(flagged), "prediction_only": 0.0,
                 "shap_top1": what_if(1, "shap"), "shap_top2": what_if(2, "shap"),
                 "global_top1": what_if(1, "global"), "global_top2": what_if(2, "global")}

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(R, f, indent=1)

# ---- figures ----
def save(fig, n):
    fig.savefig(os.path.join(OUT, n), dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

fig = plt.figure(figsize=(7, 4.2)); gimp.head(12)[::-1].plot(kind="barh", color=NAVY)
plt.title("Global feature importance (permutation)"); plt.xlabel("mean importance"); save(fig, "global_importance.png")
fig = plt.figure(figsize=(4.6, 4.2)); fpr, tpr, _ = roc_curve(yte, proba)
plt.plot(fpr, tpr, color=NAVY, lw=2, label=f"RF (AUC={R['clf']['RandomForest']['roc_auc']})"); plt.plot([0, 1], [0, 1], "--", color=GRAY)
plt.xlabel("false positive rate"); plt.ylabel("true positive rate"); plt.title("ROC - sprint-failure classifier"); plt.legend(loc="lower right"); save(fig, "roc.png")
fig = plt.figure(figsize=(4.4, 3.6)); plt.bar(["All signal groups", "Planning only"], [R["h3"]["multi_f1_mean"], R["h3"]["single_f1_mean"]],
    yerr=[R["h3"]["multi_f1_sd"], R["h3"]["single_f1_sd"]], color=[NAVY, GRAY], capsize=5); plt.ylabel("F1 (10-fold CV)")
plt.title("Feature-group comparison"); save(fig, "h3_multi_vs_single.png")
fig = plt.figure(figsize=(4.6, 4.2)); plt.scatter(rte, rp, s=8, alpha=0.35, color=NAVY); lim = [rte.min(), rte.max()]
plt.plot(lim, lim, "r--", lw=1); plt.xlabel("actual cycle time (days)"); plt.ylabel("predicted"); plt.title(f"Cycle-time regression (R2={R['reg']['R2']})"); save(fig, "cycle_time_regression.png")
A = R["ablation"]
fig, ax = plt.subplots(figsize=(5.2, 3.6)); xs = np.arange(2); w = 0.36
ax.bar(xs - w / 2, [A["shap_top1"], A["shap_top2"]], w, color=NAVY, label="Per-sprint SHAP drivers")
ax.bar(xs + w / 2, [A["global_top1"], A["global_top2"]], w, color=GRAY, label="Same global top drivers")
ax.set_xticks(xs); ax.set_xticklabels(["top-1 adjusted", "top-2 adjusted"]); ax.set_ylabel("share re-scored below 0.5")
ax.set_title("What-if sensitivity of the decision layer"); ax.legend(fontsize=8); save(fig, "ablation.png")
pr = R["reco_relevance"]["per_reason"]; names = list(pr.keys())[::-1]
fig, ax = plt.subplots(figsize=(6.4, 3.8)); yy = np.arange(len(names)); h = 0.38
ax.barh(yy + h / 2, [pr[n]["shap"] for n in names], h, color=NAVY, label="Per-sprint SHAP top-3")
ax.barh(yy - h / 2, [pr[n]["global"] for n in names], h, color=GRAY, label="Same global top-3 for all")
ax.set_yticks(yy); ax.set_yticklabels([f"{n} (n={pr[n]['n']})" for n in names], fontsize=8); ax.set_xlim(0, 1.05)
ax.set_xlabel("share of failed sprints whose top-3 includes the recorded cause"); ax.legend(fontsize=8, loc="lower right")
ax.set_title("Recommendations vs recorded failure cause"); save(fig, "recommendation_relevance.png")
ex_d = pd.Series({d["driver"]: d["shap"] for d in R["example"]["recs"]})
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.barh(list(ex_d.index)[::-1], list(ex_d.values)[::-1], color=[NAVY if d["priority"] == "High" else STEEL for d in R["example"]["recs"]][::-1])
ax.set_xlabel("SHAP value (contribution to failure probability)")
ax.set_title(f"Local SHAP explanation - worked example (recorded cause: {R['example']['reason']})", fontsize=9); save(fig, "local_explanation.png")

print(json.dumps(R, indent=1))
print(f"\nResults and figures written to ./{OUT}/")

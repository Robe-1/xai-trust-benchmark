#!/usr/bin/env python3
"""
xai_trust_benchmark.py
======================
Quantitative, statistically tested comparison of SHAP (TreeExplainer) and LIME
explanations for Random Forest and XGBoost classifiers on two public medical
tabular datasets (Breast Cancer Wisconsin Diagnostic; Pima Indians diabetes).

Sub-commands
------------
    python xai_trust_benchmark.py run      --out results     # run all experiments
    python xai_trust_benchmark.py analyze --out results     # tables + statistics
    python xai_trust_benchmark.py figures --out results     # figures
    python xai_trust_benchmark.py all     --out results     # everything
    add --quick for a ~2 minute smoke test (tiny settings, NOT for reporting)

What is computed (one block per reviewer request)
-------------------------------------------------
 * Diagnostic metrics with the DISEASE class as positive (malignant / diabetes):
   sensitivity = TP/(TP+FN), specificity = TN/(TN+FP), Wilson 95% CIs.
 * Fidelity: AOPC (cumulative deletion, baseline = training mean, probability of
   the ORIGINALLY PREDICTED class), a random-ordering control, and K sensitivity.
 * Stability: Pearson / Spearman / top-k Jaccard between the explanation of x and
   of perturbed copies of x. D independent Gaussian perturbations per instance and
   noise level; SHAP and LIME are scored on the SAME perturbed inputs.
 * LIME self-consistency (zero noise): LIME re-run on the UNCHANGED input with
   different random seeds ("noise level 0"), the floor for LIME stability.
 * LIME num_samples sensitivity: 1000 / 2000 / 5000 (5000 = LIME default = primary).
 * Sparsity: #features carrying 70/80/90 % of the absolute attribution mass and
   the share of mass in the top-k features.
 * Paired Wilcoxon signed-rank tests (unit of pairing = one test instance, the
   per-instance value is the mean over the D noise draws), rank-biserial effect
   size, bootstrap CI of the paired mean difference, Holm correction.
 * Robustness to the data split: repeated stratified 75/25 splits.
"""
import argparse, json, os, sys, time, zlib, warnings, pickle, platform
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# CONFIGURATION (everything that affects results lives here)
# ----------------------------------------------------------------------------
CFG = dict(
    seed=42,                       # primary split / model seed
    test_size=0.25,
    lime_ns_list=[1000, 2000, 5000],
    lime_ns_primary=5000,          # LIME's default num_samples
    noise_levels=[0.01, 0.05, 0.10],   # fraction of the TRAINING std of each feature
    n_draws=5,                     # independent perturbations per instance & level
    n_random_orders=20,            # random feature orderings per instance (control)
    K_main={"bc": 10, "pima": 5},
    K_list={"bc": [5, 10, 15, 30], "pima": [3, 5, 7]},
    topk_stab={"bc": 5, "pima": 3},    # k for top-k Jaccard
    sparsity_thresholds=[0.7, 0.8, 0.9],
    sparsity_main=0.8,
    n_repeats=10,                  # repeated stratified splits
    repeat_seeds=[101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    light=dict(ns=5000, n_draws=1, n_random_orders=10),   # per-split settings
    bootstrap=5000,
    rf=dict(n_estimators=200),
    xgb=dict(n_estimators=200),
    n_jobs=2,
    max_instances=None,
)
DATASETS = ["bc", "pima"]
PRETTY = {"bc": "Breast Cancer Wisconsin (Diagnostic)", "pima": "Pima Indians Diabetes (MASS)"}

MASS_URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MASS/{}.csv"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def seed_for(*keys):
    """Deterministic 31-bit seed from arbitrary keys (platform independent)."""
    return zlib.crc32("|".join(map(str, keys)).encode()) & 0x7FFFFFFF


# ----------------------------------------------------------------------------
# DATA   (label 1 = DISEASE in both datasets: malignant / diabetes)
# ----------------------------------------------------------------------------
def load_dataset(name):
    if name == "bc":
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        X = d.data.astype(float)
        y = (d.target == 0).astype(int)          # sklearn: 0 = malignant -> 1 = positive (disease)
        # Stratify on sklearn's ORIGINAL coding (benign = 1). Stratified splitting depends on the label coding,
        # so this reproduces exactly the train/test split used in the version of the paper that was reviewed.
        return X, y, list(d.feature_names), d.target.copy()
    if name == "pima":
        frames = []
        for part in ("Pima.tr", "Pima.te"):
            local = os.path.join(HERE, "data", part + ".csv")
            frames.append(pd.read_csv(local if os.path.exists(local) else MASS_URL.format(part)))
        d = pd.concat(frames, ignore_index=True)
        feats = ["npreg", "glu", "bp", "skin", "bmi", "ped", "age"]
        y = (d["type"] == "Yes").astype(int).to_numpy()
        return d[feats].to_numpy(float), y, feats, y.copy()
    raise ValueError(name)


# ----------------------------------------------------------------------------
# MODELS + DIAGNOSTIC METRICS
# ----------------------------------------------------------------------------
def make_models(seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
        "RF": RandomForestClassifier(random_state=seed, n_jobs=1, **CFG["rf"]),
        "XGB": XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=1, **CFG["xgb"]),
    }


def clinical_metrics(y_true, y_pred, p_pos):
    """Positive class = disease (label 1)."""
    from sklearn.metrics import roc_auc_score
    from statsmodels.stats.proportion import proportion_confint
    tn = int(((y_true == 0) & (y_pred == 0)).sum()); fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum()); tp = int(((y_true == 1) & (y_pred == 1)).sum())
    n = tn + fp + fn + tp
    ci = lambda k, m: tuple(float(v) for v in proportion_confint(k, m, method="wilson")) if m else (np.nan, np.nan)
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else np.nan
    return dict(TN=tn, FP=fp, FN=fn, TP=tp, n=n, accuracy=(tp + tn) / n, sensitivity=sens,
                specificity=spec, precision=prec, f1=f1, roc_auc=float(roc_auc_score(y_true, p_pos)),
                accuracy_ci=ci(tp + tn, n), sensitivity_ci=ci(tp, tp + fn), specificity_ci=ci(tn, tn + fp))


# ----------------------------------------------------------------------------
# EXPLAINERS
# ----------------------------------------------------------------------------
def shap_matrix(model, X):
    """Exact TreeSHAP attributions for the positive (disease) class."""
    import shap
    v = shap.TreeExplainer(model).shap_values(X)
    if isinstance(v, list):
        v = v[1]
    elif np.ndim(v) == 3:
        v = v[:, :, 1]
    return np.asarray(v, float)


def _lime_chunk(model, X_train, feats, rows, seeds, ns):
    from lime.lime_tabular import LimeTabularExplainer
    out = np.zeros(rows.shape)
    for j in range(len(rows)):
        ex = LimeTabularExplainer(X_train, feature_names=feats, class_names=["neg", "pos"],
                                  mode="classification", discretize_continuous=True,
                                  random_state=int(seeds[j]))
        e = ex.explain_instance(rows[j], model.predict_proba, num_features=rows.shape[1],
                                num_samples=ns, labels=(1,))
        for idx, w in e.as_map()[1]:
            out[j, idx] = w
    return out


def lime_matrix(model, X_train, feats, rows, seeds, ns, n_jobs):
    """LIME attributions (all features, class 1) for each row with its own seed."""
    from joblib import Parallel, delayed
    m = len(rows)
    chunks = [c for c in np.array_split(np.arange(m), max(1, min(m, n_jobs * 4))) if len(c)]
    if n_jobs == 1:
        parts = [_lime_chunk(model, X_train, feats, rows[c], seeds[c], ns) for c in chunks]
    else:
        parts = Parallel(n_jobs=n_jobs)(delayed(_lime_chunk)(model, X_train, feats, rows[c], seeds[c], ns)
                                        for c in chunks)
    return np.vstack(parts)


# ----------------------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------------------
def deletion_drops(model, X, orders, baseline, cls):
    """drops[i,k-1] = p_c(x_i) - p_c(x_i with its first k ranked features replaced by `baseline`),
    where c = class originally predicted for x_i.  Returns (n, n_features)."""
    n, f = X.shape
    rank = np.empty_like(orders)
    rank[np.arange(n)[:, None], orders] = np.arange(f)[None, :]
    masked = rank[:, None, :] < np.arange(f + 1)[None, :, None]            # (n, f+1, f)
    Xm = np.where(masked, baseline[None, None, :], X[:, None, :])
    P = model.predict_proba(Xm.reshape(-1, f)).reshape(n, f + 1, -1)
    pc = P[np.arange(n)[:, None], np.arange(f + 1)[None, :], cls[:, None]]
    return pc[:, [0]] - pc[:, 1:]


def aopc_from_drops(drops, K):
    return drops[:, :K].mean(1)


def stab_metrics(orig, new, k):
    from scipy.stats import rankdata

    def corr(a, b):
        a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
        den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
        with np.errstate(all="ignore"):
            return np.where(den > 0, (a * b).sum(1) / den, np.nan)

    pear = corr(orig, new)
    spear = corr(rankdata(orig, axis=1), rankdata(new, axis=1))
    top = lambda a: np.argsort(-np.abs(a), axis=1)[:, :k]
    to, tn = top(orig), top(new)
    jac = np.array([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(to, tn)])
    return pear, spear, jac


def sparsity_stats(E, thresholds, k):
    A = np.abs(E)
    tot = A.sum(1, keepdims=True)
    S = -np.sort(-A, axis=1)
    cs = np.cumsum(S, 1) / np.where(tot > 0, tot, 1)
    f = E.shape[1]
    out = {f"sp{int(t * 100)}": np.minimum((cs < t - 1e-12).sum(1) + 1, f) for t in thresholds}
    out["topk_share"] = S[:, :k].sum(1) / np.where(tot[:, 0] > 0, tot[:, 0], 1)
    return out


# ----------------------------------------------------------------------------
# ONE TRAIN/TEST SPLIT
# ----------------------------------------------------------------------------
def run_split(name, data, split_seed, mode, n_jobs):
    from sklearn.model_selection import train_test_split
    X, y, feats, strat = data
    t0 = time.time()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=CFG["test_size"], random_state=split_seed, stratify=strat)
    if CFG["max_instances"]:
        Xte, yte = Xte[:CFG["max_instances"]], yte[:CFG["max_instances"]]
    n, f = Xte.shape
    if mode == "full":
        ns_list, levels, D, R = CFG["lime_ns_list"], [0.0] + CFG["noise_levels"], CFG["n_draws"], CFG["n_random_orders"]
    else:
        ns_list, levels, D, R = [CFG["light"]["ns"]], list(CFG["noise_levels"]), CFG["light"]["n_draws"], CFG["light"]["n_random_orders"]
    K_main, topk = CFG["K_main"][name], CFG["topk_stab"][name]

    models = make_models(split_seed)
    for m in models.values():
        m.fit(Xtr, ytr)
    res = dict(meta=dict(dataset=name, split_seed=split_seed, n_train=len(Xtr), n_test=n, n_features=f,
                         features=feats, mode=mode, levels=levels, D=D, ns_list=ns_list, R=R,
                         K_main=K_main, topk=topk, prevalence_test=float(yte.mean())),
               y_test=yte, clf={}, models={})
    for mn, m in models.items():
        res["clf"][mn] = clinical_metrics(yte, m.predict(Xte), m.predict_proba(Xte)[:, 1])

    baseline, std = Xtr.mean(0), Xtr.std(0)
    lo, hi = Xtr.min(0), Xtr.max(0)
    rng = np.random.default_rng(seed_for("noise", name, split_seed))
    noisy = {(lv, d): (Xte if lv == 0 else np.clip(Xte + rng.standard_normal(Xte.shape) * std * lv, lo, hi))
             for lv in levels for d in range(D)}      # shared by SHAP and LIME, all models, all ns

    for mn in ("RF", "XGB"):
        model = models[mn]
        cls = model.predict_proba(Xte).argmax(1)
        out = dict(cls=cls, shap={}, lime={}, rand={}, stab={"shap": {}, "lime": {}})
        # --- SHAP -----------------------------------------------------------
        Es = shap_matrix(model, Xte)
        rerun_diff = float(np.abs(Es - shap_matrix(model, Xte)).max())      # determinism check
        out["shap"]["rerun_maxdiff"] = rerun_diff
        out["shap"]["drops"] = deletion_drops(model, Xte, np.argsort(-np.abs(Es), 1), baseline, cls)
        out["shap"]["sparsity"] = sparsity_stats(Es, CFG["sparsity_thresholds"], topk)
        out["shap"]["base"] = Es
        # random-ordering control
        rr = np.random.default_rng(seed_for("rand", name, split_seed, mn))
        dr = np.mean([deletion_drops(model, Xte, np.argsort(rr.random((n, f)), 1), baseline, cls)
                      for _ in range(R)], axis=0)
        out["rand"]["drops"] = dr
        # SHAP stability (same perturbed inputs as LIME)
        for lv in levels:
            per = [stab_metrics(Es, shap_matrix(model, noisy[(lv, d)]) if lv > 0 else Es.copy(), topk) for d in range(D)]
            out["stab"]["shap"][lv] = tuple(np.nanmean([p[i] for p in per], axis=0) for i in range(3))
        log(f"  [{name} split {split_seed}] {mn}: SHAP done (rerun max|diff| = {rerun_diff:.1e})")
        # --- LIME for each num_samples ---------------------------------------
        for ns in ns_list:
            tl = time.time()
            base_seeds = np.array([seed_for("lime", name, split_seed, mn, ns, "base", i) for i in range(n)])
            El = lime_matrix(model, Xtr, feats, Xte, base_seeds, ns, n_jobs)
            rows = np.vstack([noisy[(lv, d)] for lv in levels for d in range(D)])
            seeds = np.array([seed_for("lime", name, split_seed, mn, ns, "stab", lv, d, i)
                              for lv in levels for d in range(D) for i in range(n)])
            En = lime_matrix(model, Xtr, feats, rows, seeds, ns, n_jobs).reshape(len(levels), D, n, f)
            ent = dict(base=El,
                       drops=deletion_drops(model, Xte, np.argsort(-np.abs(El), 1), baseline, cls),
                       sparsity=sparsity_stats(El, CFG["sparsity_thresholds"], topk))
            out["lime"][ns] = ent
            out["stab"]["lime"][ns] = {}
            for li, lv in enumerate(levels):
                per = [stab_metrics(El, En[li, d], topk) for d in range(D)]
                out["stab"]["lime"][ns][lv] = tuple(np.nanmean([p[i] for p in per], axis=0) for i in range(3))
            log(f"  [{name} split {split_seed}] {mn}: LIME ns={ns} done in {time.time() - tl:.0f}s")
        res["models"][mn] = out
    log(f"[{name} split {split_seed}] finished in {time.time() - t0:.0f}s")
    return res


# ----------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------
def cmd_run(out, only=None, stages=("primary", "repeats")):
    os.makedirs(out, exist_ok=True)
    import shap, lime, xgboost, sklearn, scipy
    json.dump(dict(config=CFG, python=platform.python_version(),
                   versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                                 sklearn=sklearn.__version__, xgboost=xgboost.__version__, shap=shap.__version__,
                                 lime=getattr(lime, "__version__", "unknown"))),
              open(os.path.join(out, "run_config.json"), "w"), indent=2, default=str)
    for name in (only or DATASETS):
        data = load_dataset(name)
        log(f"=== {PRETTY[name]}: {data[0].shape[0]} rows, {data[0].shape[1]} features, prevalence {data[1].mean():.3f}")
        if "primary" in stages:
            p = os.path.join(out, f"raw_{name}_primary.pkl")
            res = run_split(name, data, CFG["seed"], "full", CFG["n_jobs"])
            pickle.dump(res, open(p, "wb"))
        if "repeats" in stages:
            reps = []
            for s in CFG["repeat_seeds"][:CFG["n_repeats"]]:
                reps.append(run_split(name, data, s, "light", CFG["n_jobs"]))
                pickle.dump(reps, open(os.path.join(out, f"raw_{name}_repeats.pkl"), "wb"))
    log("run finished")


# ----------------------------------------------------------------------------
# STATISTICS
# ----------------------------------------------------------------------------
def paired_test(a, b, rng, B):
    from scipy.stats import wilcoxon, rankdata
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]; d = a - b
    nz = d != 0
    out = dict(n=int(len(a)), mean_a=float(a.mean()), mean_b=float(b.mean()), mean_diff=float(d.mean()),
               frac_a_gt_b=float((d > 0).mean()))
    idx = rng.integers(0, len(d), (B, len(d)))
    bm = d[idx].mean(1)
    out["diff_ci_lo"], out["diff_ci_hi"] = [float(v) for v in np.percentile(bm, [2.5, 97.5])]
    if nz.sum() == 0:
        out.update(W=np.nan, p=1.0, r=0.0)
        return out
    W, p = wilcoxon(a, b)
    rk = rankdata(np.abs(d[nz])); wp = rk[d[nz] > 0].sum(); wn = rk[d[nz] < 0].sum()
    out.update(W=float(W), p=float(p), r=float((wp - wn) / (wp + wn)))
    return out


def holm(pvals):
    from statsmodels.stats.multitest import multipletests
    return multipletests(pvals, method="holm")[1]


def cmd_analyze(out):
    rng = np.random.default_rng(12345)
    B = CFG["bootstrap"]
    rows_primary, rows_sens, summ, selfc, clf_rows, per_inst = [], [], [], [], [], []
    ns_p = CFG["lime_ns_primary"]
    meta = {}
    for name in DATASETS:
        res = pickle.load(open(os.path.join(out, f"raw_{name}_primary.pkl"), "rb"))
        m = res["meta"]; meta[name] = m
        K, ths = m["K_main"], CFG["sparsity_thresholds"]
        sp_key = f"sp{int(CFG['sparsity_main'] * 100)}"
        levels = m["levels"]; nz_levels = [lv for lv in levels if lv > 0]
        ns_use = ns_p if ns_p in m["ns_list"] else m["ns_list"][-1]
        for mn, c in res["clf"].items():
            clf_rows.append(dict(dataset=name, model=mn, **{k: v for k, v in c.items() if not k.endswith("_ci")},
                                 **{k + "_lo": v[0] for k, v in c.items() if k.endswith("_ci")},
                                 **{k + "_hi": v[1] for k, v in c.items() if k.endswith("_ci")}))
        for mn in ("RF", "XGB"):
            r = res["models"][mn]
            aop = lambda drops, k=K: aopc_from_drops(drops, k)
            # ---- summary table -------------------------------------------------
            def add(method, ns, drops, sp, stab):
                row = dict(dataset=name, model=mn, method=method, lime_ns=ns,
                           aopc_mean=aop(drops).mean(), aopc_sd=aop(drops).std(ddof=1),
                           sp80_mean=sp[sp_key].mean(), sp80_sd=sp[sp_key].std(ddof=1),
                           topk_share_mean=sp["topk_share"].mean())
                for lv in levels:
                    for i, nm in enumerate(["pearson", "spearman", "top_k_jaccard"]):
                        v = stab[lv][i]
                        row[f"{nm}_{lv:g}_mean"] = float(np.nanmean(v)); row[f"{nm}_{lv:g}_sd"] = float(np.nanstd(v, ddof=1))
                summ.append(row)
            add("SHAP", "", r["shap"]["drops"], r["shap"]["sparsity"], r["stab"]["shap"])
            for ns in m["ns_list"]:
                add("LIME", ns, r["lime"][ns]["drops"], r["lime"][ns]["sparsity"], r["stab"]["lime"][ns])
            add("Random order", "", r["rand"]["drops"], r["shap"]["sparsity"], r["stab"]["shap"])
            summ[-1].update({k: np.nan for k in summ[-1] if k.startswith(("sp80", "topk", "pearson", "spearman", "top_k"))})
            # ---- LIME self-consistency ------------------------------------------
            for ns in m["ns_list"]:
                row = dict(dataset=name, model=mn, lime_ns=ns)
                for lv in levels:
                    for i, nm in enumerate(["pearson", "spearman", "top_k_jaccard"]):
                        row[f"{nm}_{lv:g}"] = float(np.nanmean(r["stab"]["lime"][ns][lv][i]))
                selfc.append(row)
            # ---- per-instance export (primary config) ----------------------------
            for method, dr, sp, st in [("SHAP", r["shap"]["drops"], r["shap"]["sparsity"], r["stab"]["shap"]),
                                       (f"LIME@{ns_use}", r["lime"][ns_use]["drops"], r["lime"][ns_use]["sparsity"], r["stab"]["lime"][ns_use])]:
                for i in range(m["n_test"]):
                    d = dict(dataset=name, model=mn, method=method, instance=i, y_true=int(res["y_test"][i]),
                             pred_class=int(r["cls"][i]), aopc=aop(dr)[i], sparsity80=int(sp[sp_key][i]))
                    for lv in nz_levels:
                        for j, nm in enumerate(["pearson", "spearman", "jaccard"]):
                            d[f"{nm}_{lv:g}"] = st[lv][j][i]
                    per_inst.append(d)
            # ---- PRIMARY tests (Holm family) -------------------------------------
            S, L = r["shap"], r["lime"][ns_use]
            T = lambda family, label, a, b, **kw: rows_primary.append(dict(
                dataset=name, model=mn, family=family, test=label, **paired_test(a, b, rng, B), **kw))
            T("fidelity", f"AOPC@K={K}: SHAP vs LIME", aop(S["drops"]), aop(L["drops"]))
            T("fidelity", f"AOPC@K={K}: SHAP vs random order", aop(S["drops"]), aop(r["rand"]["drops"]))
            T("fidelity", f"AOPC@K={K}: LIME vs random order", aop(L["drops"]), aop(r["rand"]["drops"]))
            T("sparsity", "Sparsity(80%): SHAP vs LIME", S["sparsity"][sp_key], L["sparsity"][sp_key])
            for lv in nz_levels:
                for j, nm in enumerate(["Pearson", "Spearman", "Top-k Jaccard"]):
                    T("stability", f"Stability {nm} @{lv:.0%}: SHAP vs LIME", r["stab"]["shap"][lv][j], r["stab"]["lime"][ns_use][lv][j])
            # ---- SENSITIVITY tests (own Holm family) -------------------------------
            Tn = lambda sens, label, a, b: rows_sens.append(dict(dataset=name, model=mn, sensitivity=sens, test=label, **paired_test(a, b, rng, B)))
            for ns in m["ns_list"]:
                Ln = r["lime"][ns]
                Tn("LIME num_samples", f"ns={ns} | AOPC@K={K}", aop(S["drops"]), aop(Ln["drops"]))
                Tn("LIME num_samples", f"ns={ns} | Sparsity(80%)", S["sparsity"][sp_key], Ln["sparsity"][sp_key])
                for lv in nz_levels:
                    Tn("LIME num_samples", f"ns={ns} | Pearson @{lv:.0%}", r["stab"]["shap"][lv][0], r["stab"]["lime"][ns][lv][0])
            for k in CFG["K_list"][name]:
                Tn("AOPC depth K", f"K={k} | AOPC SHAP vs LIME", aop(S["drops"], k), aop(L["drops"], k))
                Tn("AOPC depth K", f"K={k} | AOPC SHAP vs random", aop(S["drops"], k), aop(r["rand"]["drops"], k))
                Tn("AOPC depth K", f"K={k} | AOPC LIME vs random", aop(L["drops"], k), aop(r["rand"]["drops"], k))
            for t in ths:
                Tn("Sparsity threshold", f"{int(t * 100)}% mass | SHAP vs LIME", S["sparsity"][f"sp{int(t * 100)}"], L["sparsity"][f"sp{int(t * 100)}"])
            Tn("Sparsity measure", "Top-k mass share: SHAP vs LIME", S["sparsity"]["topk_share"], L["sparsity"]["topk_share"])
        # mean deletion curves (for figure)
    P = pd.DataFrame(rows_primary)
    P["p_holm"] = holm(P["p"].values)
    P["sig_holm"] = P["p_holm"] < 0.05
    Sn = pd.DataFrame(rows_sens)
    Sn["p_holm"] = holm(Sn["p"].values)
    pd.DataFrame(clf_rows).to_csv(os.path.join(out, "table_classification_primary.csv"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(out, "table_summary_primary.csv"), index=False)
    pd.DataFrame(selfc).to_csv(os.path.join(out, "table_lime_self_consistency.csv"), index=False)
    P.to_csv(os.path.join(out, "table_tests_primary_holm.csv"), index=False)
    Sn.to_csv(os.path.join(out, "table_tests_sensitivity_holm.csv"), index=False)
    pd.DataFrame(per_inst).to_csv(os.path.join(out, "per_instance_primary.csv"), index=False)
    # ---------------- repeated splits ------------------------------------------------
    rep_rows, rep_clf = [], []
    for name in DATASETS:
        rp = os.path.join(out, f"raw_{name}_repeats.pkl")
        reps = pickle.load(open(rp, "rb")) if os.path.exists(rp) else []
        if not reps:
            log(f"no repeated-split results for {name} (skipped)")
        for res in reps:
            m = res["meta"]; K = m["K_main"]; sp_key = f"sp{int(CFG['sparsity_main'] * 100)}"
            for mn, c in res["clf"].items():
                rep_clf.append(dict(dataset=name, split_seed=m["split_seed"], model=mn,
                                    **{k: v for k, v in c.items() if not k.endswith("_ci")}))
            ns = m["ns_list"][0]
            for mn in ("RF", "XGB"):
                r = res["models"][mn]; S, L = r["shap"], r["lime"][ns]
                row = dict(dataset=name, split_seed=m["split_seed"], model=mn,
                           aopc_shap=aopc_from_drops(S["drops"], K).mean(), aopc_lime=aopc_from_drops(L["drops"], K).mean(),
                           aopc_random=aopc_from_drops(r["rand"]["drops"], K).mean(),
                           sp80_shap=S["sparsity"][sp_key].mean(), sp80_lime=L["sparsity"][sp_key].mean())
                for lv in m["levels"]:
                    row[f"pearson_shap_{lv:g}"] = np.nanmean(r["stab"]["shap"][lv][0])
                    row[f"pearson_lime_{lv:g}"] = np.nanmean(r["stab"]["lime"][ns][lv][0])
                rep_rows.append(row)
    if not rep_rows:
        log("repeated-split tables skipped (no repeats available)")
    pd.DataFrame(rep_clf).to_csv(os.path.join(out, "table_repeated_splits_classification.csv"), index=False)
    pd.DataFrame(rep_rows).to_csv(os.path.join(out, "table_repeated_splits_explanations.csv"), index=False)
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "features"} for k, v in meta.items()},
              open(os.path.join(out, "meta_primary.json"), "w"), indent=2, default=str)
    log("analysis finished; tables written to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze", "figures", "all"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test settings (NOT for reporting)")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=None,
                    help="number of extra stratified splits (default 10; 0 = primary split only, much faster)")
    ap.add_argument("--only", nargs="+", choices=DATASETS, default=None, help="restrict `run` to these datasets")
    ap.add_argument("--stages", nargs="+", choices=["primary", "repeats"], default=["primary", "repeats"])
    a = ap.parse_args()
    if a.quick:
        CFG.update(lime_ns_list=[200, 400], lime_ns_primary=400, n_draws=2, n_random_orders=3, n_repeats=2,
                   max_instances=20, bootstrap=200)
        CFG["light"].update(ns=400, n_random_orders=3)
    if a.n_jobs:
        CFG["n_jobs"] = a.n_jobs
    if a.repeats is not None:
        CFG["n_repeats"] = a.repeats
    if a.cmd in ("run", "all"):
        cmd_run(a.out, a.only, tuple(a.stages))
    if a.cmd in ("analyze", "all"):
        cmd_analyze(a.out)
    if a.cmd in ("figures", "all"):
        from figures import make_figures
        make_figures(a.out)

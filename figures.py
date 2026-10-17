"""Figure generation for the SHAP-vs-LIME trust benchmark (reads files written by xai_trust_benchmark.py)."""
import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical slots 1-2 of the validated reference palette + neutral for the control
C_SHAP, C_LIME, C_RAND, C_INK, C_GRID = "#2a78d6", "#eb6834", "#6b6a66", "#0b0b0b", "#d9d8d3"
STYLES = [("--", "^", 0.55), (":", "D", 0.75), ("-", "s", 1.0)]
TITLES = {"bc": "Breast Cancer", "pima": "Pima Diabetes"}
DATASETS = ["bc", "pima"]
MODELS = ["RF", "XGB"]


def _cfg(out):
    return json.load(open(os.path.join(out, "run_config.json")))["config"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.linewidth": 0.6, "axes.edgecolor": "#444", "axes.spines.top": False,
    "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.4,
    "grid.color": C_GRID, "grid.linewidth": 0.5, "figure.dpi": 150, "savefig.dpi": 300,
})


def _load(out, name):
    return pickle.load(open(os.path.join(out, f"raw_{name}_primary.pkl"), "rb"))


def _panel_label(ax, text):
    ax.set_title(text, loc="left", fontweight="bold", color=C_INK, pad=3)


def fig_deletion(out):
    NS_PRIMARY = _cfg(out)["lime_ns_primary"]
    fig, axes2 = plt.subplots(2, 2, figsize=(3.45, 2.7), sharey=False)
    axes = axes2.ravel()
    for ax, (name, mn) in zip(axes, [(d, m) for d in DATASETS for m in MODELS]):
        res = _load(out, name); r = res["models"][mn]; K = res["meta"]["K_main"]
        ks = np.arange(0, res["meta"]["n_features"] + 1)
        curve = lambda dr: np.concatenate([[0], dr.mean(0)])
        ax.plot(ks, curve(r["shap"]["drops"]), color=C_SHAP, marker="o", ms=2.2, label="SHAP")
        ax.plot(ks, curve(r["lime"][NS_PRIMARY]["drops"]), color=C_LIME, marker="s", ms=2.2, ls="-", label="LIME")
        ax.plot(ks, curve(r["rand"]["drops"]), color=C_RAND, ls="--", label="Random order")
        ax.axvline(K, color=C_GRID, lw=0.9, zorder=0)
        ax.set_xlabel("# features removed (k)")
        ax.grid(axis="y")
        _panel_label(ax, f"({'abcd'[list(axes).index(ax)]}) {TITLES[name].split()[0]} – {mn}")
        if ax in (axes[0], axes[2]):
            ax.set_ylabel(r"Mean drop in $p_c$")
    axes[0].legend(loc="lower right", frameon=False, handlelength=1.8)
    fig.tight_layout(w_pad=0.8, h_pad=0.9)
    p = os.path.join(out, "fig_deletion_curves.png"); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_stability(out):
    cfg = _cfg(out); nss_all = cfg["lime_ns_list"]; ns_show = [nss_all[0], cfg["lime_ns_primary"]]
    summ = pd.read_csv(os.path.join(out, "table_summary_primary.csv"))
    fig, axes2 = plt.subplots(2, 2, figsize=(3.45, 2.8))
    axes = axes2.ravel()
    levels = [0, 0.01, 0.05, 0.1]
    for ax, (name, mn) in zip(axes, [(d, m) for d in DATASETS for m in MODELS]):
        s = summ[(summ.dataset == name) & (summ.model == mn)]
        sh = s[s.method == "SHAP"].iloc[0]
        xs = np.arange(4)
        ax.plot(xs, [sh[f"pearson_{lv:g}_mean"] for lv in levels], color=C_SHAP, marker="o", ms=2.6, label="SHAP")
        for ns in ns_show:
            l = s[(s.method == "LIME") & (s.lime_ns == ns)].iloc[0]
            ls, mk, al = STYLES[0] if ns == ns_show[0] else STYLES[2]
            ax.plot(xs, [l[f"pearson_{lv:g}_mean"] for lv in levels], color=C_LIME, ls=ls, marker=mk, ms=2.6,
                    alpha=al, label=f"LIME ({ns})")
        ax.set_xticks(xs); ax.set_xticklabels(["0%", "1%", "5%", "10%"])
        ax.set_xlabel("Input noise level")
        ax.grid(axis="y")
        _panel_label(ax, f"({'abcd'[list(axes).index(ax)]}) {TITLES[name].split()[0]} – {mn}")
        if ax in (axes[0], axes[2]):
            ax.set_ylabel("Pearson stability")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, handlelength=2.0, bbox_to_anchor=(0.5, -0.045))
    fig.tight_layout(w_pad=0.8, h_pad=0.9)
    p = os.path.join(out, "fig_stability.png"); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_sparsity_sens(out):
    cfg = _cfg(out); NS_PRIMARY = cfg["lime_ns_primary"]; nss = cfg["lime_ns_list"]
    summ = pd.read_csv(os.path.join(out, "table_summary_primary.csv"))
    sens = pd.read_csv(os.path.join(out, "table_tests_sensitivity_holm.csv"))
    fig = plt.figure(figsize=(3.45, 3.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    # (a) sparsity at 80 % mass
    ax = axes[0]
    groups = [(d, m) for d in DATASETS for m in MODELS]
    x = np.arange(len(groups)); w = 0.36
    for off, meth, col, hatch in [(-w / 2, "SHAP", C_SHAP, None), (w / 2, "LIME", C_LIME, "///")]:
        vals, errs = [], []
        for d, m in groups:
            s = summ[(summ.dataset == d) & (summ.model == m) & (summ.method == meth)]
            if meth == "LIME":
                s = s[s.lime_ns == NS_PRIMARY]
            vals.append(s.sp80_mean.iloc[0]); errs.append(s.sp80_sd.iloc[0])
        ax.bar(x + off, vals, w, yerr=errs, color=col, hatch=hatch, edgecolor="white", linewidth=0.4,
               error_kw=dict(lw=0.6, capsize=1.5), label=meth)
    ax.set_xticks(x); ax.set_xticklabels([f"{TITLES[d].split()[0]} {m}" for d, m in groups])
    ax.set_ylabel("# features for 80% of mass"); ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.2)
    _panel_label(ax, "(a) Sparsity (lower = more compact)")
    # (b)/(c) LIME num_samples sensitivity: stability @5% noise and AOPC
    for ax, key, ylabel, ttl in [(axes[1], "pearson_0.05_mean", "Pearson @5% noise", "(b) Stability vs samples"),
                                 (axes[2], "aopc_mean", "AOPC", "(c) AOPC vs samples")]:
        for (d, m), mk in zip(groups, ["o", "s", "^", "D"]):
            s = summ[(summ.dataset == d) & (summ.model == m)]
            lim = [s[(s.method == "LIME") & (s.lime_ns == n)][key].iloc[0] for n in nss]
            shp = s[s.method == "SHAP"][key].iloc[0]
            ax.plot(range(len(nss)), lim, color=C_LIME, marker=mk, ms=2.8, lw=1.0, label=f"{TITLES[d].split()[0]} {m}" if ax is axes[1] else None)
            ax.plot(range(len(nss)), [shp] * len(nss), color=C_SHAP, ls="-", lw=0.7, alpha=0.8)
        ax.set_xticks(range(len(nss))); ax.set_xticklabels(nss); ax.set_xlabel("LIME num_samples")
        ax.set_ylabel(ylabel); ax.grid(axis="y")
        _panel_label(ax, ttl)
    axes[1].legend(frameon=False, loc="lower right", fontsize=5, handlelength=1.2)
    fig.tight_layout(w_pad=0.7, h_pad=0.9)
    p = os.path.join(out, "fig_sparsity_and_lime_samples.png"); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def make_figures(out):
    paths = [fig_deletion(out), fig_stability(out), fig_sparsity_sens(out)]
    for p in paths:
        print("saved", p)


if __name__ == "__main__":
    import sys
    make_figures(sys.argv[1] if len(sys.argv) > 1 else "results")

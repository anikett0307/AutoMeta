# -*- coding: utf-8 -*-
"""
Make a self-contained HTML report for Steps 1-8 results.
- Reads artifacts from results/ (created by your training, evaluation, and CP scripts)
- Rebuilds plots if images are missing but CSVs/JSON exist
- Embeds images as base64 so the report is a single file

Usage (from project root or src/):
    python src/make_report.py
Optional:
    python src/make_report.py --title "AutoMeta Few-Shot Results" --episodes 1000
"""

import os, sys, io, base64, json, argparse, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

# Default artifact paths (created earlier in your pipeline)
METRICS_CSV       = os.path.join(RESULTS_DIR, "metrics_tr.csv")
METRICS_PNG       = os.path.join(RESULTS_DIR, "metrics_tr.png")
TEST_SUMMARY_JSON = os.path.join(RESULTS_DIR, "test_summary.json")
TEST_EPISODES_CSV = os.path.join(RESULTS_DIR, "test_episodes.csv")
TEST_CURVES_PNG   = os.path.join(RESULTS_DIR, "test_curves.png")
TEST_HISTS_PNG    = os.path.join(RESULTS_DIR, "test_hists.png")
TSNE_PNG          = os.path.join(RESULTS_DIR, "tsne_memory_bank.png")
CP_JSON           = os.path.join(RESULTS_DIR, "change_points.json")
DRIFT_CSV         = os.path.join(RESULTS_DIR, "drift_series.csv")
CP_PLOT_PNG       = os.path.join(RESULTS_DIR, "change_points_plot.png")

REPORT_HTML       = os.path.join(RESULTS_DIR, "report.html")


# ---------------------------
# Helpers
# ---------------------------
def _exists(p): 
    return p and os.path.isfile(p)

def _img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def _fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def _section(title, html_inner):
    return f"""
    <section>
      <h2>{title}</h2>
      {html_inner}
    </section>
    """

def _kv_table(d, keys=None, title_map=None):
    keys = keys or list(d.keys())
    title_map = title_map or {}
    rows = []
    for k in keys:
        if k not in d: 
            continue
        disp = title_map.get(k, k)
        val = d[k]
        if isinstance(val, float):
            val = f"{val:.4f}"
        rows.append(f"<tr><td>{disp}</td><td>{val}</td></tr>")
    if not rows:
        return ""
    return "<table class='kv'><tbody>" + "\n".join(rows) + "</tbody></table>"

def _maybe_rebuild_training_plot():
    if _exists(METRICS_PNG):
        return
    if not _exists(METRICS_CSV):
        return
    df = pd.read_csv(METRICS_CSV)
    plt.figure(figsize=(12,5))
    # Loss
    plt.subplot(1,2,1)
    plt.title("Training vs Validation Loss")
    plt.plot(df["epoch"], df["train_loss"], label="Train")
    if "val_loss" in df:
        plt.plot(df["epoch"], df["val_loss"], label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    # Acc
    plt.subplot(1,2,2)
    plt.title("Training vs Validation Accuracy")
    plt.plot(df["epoch"], df["train_acc"], label="Train")
    if "val_acc" in df:
        plt.plot(df["epoch"], df["val_acc"], label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend()
    plt.tight_layout()
    plt.savefig(METRICS_PNG, dpi=150)
    plt.close()

def _maybe_rebuild_test_plots():
    # Only build curves/hists if images missing but CSV exists
    if (_exists(TEST_CURVES_PNG) and _exists(TEST_HISTS_PNG)) or not _exists(TEST_EPISODES_CSV):
        return
    df = pd.read_csv(TEST_EPISODES_CSV)
    # Curves: rolling accuracy and adapt time
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title("Per-episode Accuracy (rolling mean)")
    y = df["acc"]
    # safe rolling window
    w = max(1, min(25, len(y)//20))
    y_roll = y.rolling(window=w, min_periods=1).mean()
    plt.plot(df.index+1, y_roll)
    plt.xlabel("Episode"); plt.ylabel("Accuracy")

    plt.subplot(1,2,2)
    plt.title("Adaptation Time (ms) (rolling mean)")
    t = df["adapt_ms"]
    t_roll = t.rolling(window=w, min_periods=1).mean()
    plt.plot(df.index+1, t_roll)
    plt.xlabel("Episode"); plt.ylabel("ms")
    plt.tight_layout()
    plt.savefig(TEST_CURVES_PNG, dpi=150); plt.close()

    # Hists
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title("Accuracy Histogram")
    plt.hist(df["acc"], bins=20)
    plt.xlabel("Accuracy"); plt.ylabel("Count")
    plt.subplot(1,2,2)
    plt.title("Adaptation Time Histogram (ms)")
    plt.hist(df["adapt_ms"], bins=20)
    plt.xlabel("ms"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(TEST_HISTS_PNG, dpi=150); plt.close()

def _maybe_build_cp_plot():
    if _exists(CP_PLOT_PNG):
        return
    if not (_exists(CP_JSON) and _exists(DRIFT_CSV)):
        return
    with open(CP_JSON, "r") as f:
        cp = json.load(f)
    drift = pd.read_csv(DRIFT_CSV)["drift"].values
    plt.figure(figsize=(12,4))
    plt.title("Embedding Drift with Detected Change-Points")
    plt.plot(np.arange(len(drift)), drift)
    for idx in cp.get("cp_drift", []):
        try:
            x = int(idx)
            plt.axvline(x=x, linestyle="--", linewidth=1)
        except:
            pass
    plt.xlabel("Episode Index"); plt.ylabel("Drift")
    plt.tight_layout(); plt.savefig(CP_PLOT_PNG, dpi=150); plt.close()


def build_report_html(title="AutoMeta Few-Shot Results", episodes_hint=None):
    # Try to rebuild missing plots
    _maybe_rebuild_training_plot()
    _maybe_rebuild_test_plots()
    _maybe_build_cp_plot()

    # Sections
    sections = []

    # Header metadata
    meta_html = _kv_table({
        "Generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Results folder": RESULTS_DIR,
        "Episodes (test)": episodes_hint if episodes_hint else "see summary",
    }, title_map={"Episodes (test)": "Meta-test Episodes"})
    sections.append(_section("Report Metadata", meta_html))

    # Training curves
    if _exists(METRICS_PNG):
        b64 = _img_to_base64(METRICS_PNG)
        sections.append(_section("Training Curves",
            f"<img src='data:image/png;base64,{b64}' alt='training curves'/>"))
    elif _exists(METRICS_CSV):
        sections.append(_section("Training Curves", "<p>Plot rebuilt from metrics_tr.csv (but image save failed).</p>"))
    else:
        sections.append(_section("Training Curves", "<p><em>No training metrics found.</em></p>"))

    # Test summary
    if _exists(TEST_SUMMARY_JSON):
        with open(TEST_SUMMARY_JSON, "r") as f:
            summ = json.load(f)
        keys = [
            "episodes", "n_way", "k_shot", "q_query",
            "accuracy_mean", "loss_mean",
            "adapt_ms_mean", "checkpoint", "split_root"
        ]
        tmap = {
            "episodes":"Episodes",
            "n_way":"N-way", "k_shot":"K-shot", "q_query":"Q/query",
            "accuracy_mean":"Accuracy (mean)",
            "loss_mean":"Loss (mean)",
            "adapt_ms_mean":"Adapt time (ms, mean)",
            "checkpoint":"Checkpoint",
            "split_root":"Test split root"
        }
        sections.append(_section("Meta-Test Summary", _kv_table(summ, keys, tmap)))
    else:
        sections.append(_section("Meta-Test Summary", "<p><em>No test summary found.</em></p>"))

    # Test plots (curves/hists)
    imgs = []
    if _exists(TEST_CURVES_PNG):
        imgs.append(("Per-episode Curves", _img_to_base64(TEST_CURVES_PNG)))
    if _exists(TEST_HISTS_PNG):
        imgs.append(("Episode Histograms", _img_to_base64(TEST_HISTS_PNG)))
    if imgs:
        html = "".join([f"<h3>{t}</h3><img src='data:image/png;base64,{b}'/>" for t,b in imgs])
        sections.append(_section("Meta-Test Visualizations", html))

    # Memory bank t-SNE
    if _exists(TSNE_PNG):
        sections.append(_section("Task Embedding t-SNE",
            f"<img src='data:image/png;base64,{_img_to_base64(TSNE_PNG)}'/>"))
    else:
        sections.append(_section("Task Embedding t-SNE", "<p><em>No t-SNE plot found.</em></p>"))

    # Change-point detection
    if _exists(CP_JSON):
        with open(CP_JSON, "r") as f:
            cp = json.load(f)
        cp_keys = []
        for k in ("cp_acc_rupt", "cp_loss_rupt", "cp_drift", "cp_loss_bocpd"):
            if k in cp: cp_keys.append(k)
        cp_html = _kv_table({k: len(cp.get(k, [])) for k in cp_keys},
                            title_map={
                                "cp_acc_rupt":"Ruptures (accuracy) – #CPs",
                                "cp_loss_rupt":"Ruptures (loss) – #CPs",
                                "cp_drift":"Ruptures (drift) – #CPs",
                                "cp_loss_bocpd":"BOCPD (loss) – #CPs",
                            })
        if _exists(CP_PLOT_PNG):
            cp_html += f"<img src='data:image/png;base64,{_img_to_base64(CP_PLOT_PNG)}'/>"
        sections.append(_section("Change-Point Detection", cp_html))
    else:
        sections.append(_section("Change-Point Detection", "<p><em>No change-point outputs found.</em></p>"))

    # Build final HTML
    styles = """
    <style>
      body { font-family: Segoe UI, Roboto, Arial, sans-serif; margin: 20px; line-height: 1.4; }
      h1 { margin-bottom: 0.2rem; }
      .subtitle { color: #555; margin-top: 0; }
      section { margin: 22px 0; }
      img { max-width: 100%; height: auto; border: 1px solid #ddd; padding: 6px; border-radius: 6px; }
      table.kv { border-collapse: collapse; }
      table.kv td { border: 1px solid #ddd; padding: 6px 10px; }
      table.kv td:first-child { font-weight: 600; background: #fafafa; }
    </style>
    """
    html = f"""
    <html>
    <head><meta charset="utf-8"><title>{title}</title>{styles}</head>
    <body>
      <h1>{title}</h1>
      <p class="subtitle">AutoMeta | Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
      {"".join(sections)}
    </body>
    </html>
    """
    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✓ Report saved -> {REPORT_HTML}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="AutoMeta Few-Shot Results")
    ap.add_argument("--episodes", type=int, help="(Optional) Meta-test episodes hint for the header.")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    build_report_html(title=args.title, episodes_hint=args.episodes)

if __name__ == "__main__":
    main()

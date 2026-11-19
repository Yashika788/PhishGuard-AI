
"""PhishGuard AI - Premium Cybersecurity Dashboard."""

from pathlib import Path
from datetime import datetime
import json
import textwrap
import math

import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from src.feature_extraction import extract_feature_matrix


# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "phishing_model.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
META_PATH = BASE_DIR / "models" / "metadata.json"

RESULTS_DIR = BASE_DIR / "results"
METRICS_PATH = RESULTS_DIR / "metrics.json"

CSS_PATH = BASE_DIR / "assets" / "style.css"


# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PhishGuard AI | Security Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 3. DISPLAY HELPERS
# =========================================================

def render_html(html):
    st.html(textwrap.dedent(html).strip())


def format_percent(value):
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


def format_threshold(value):
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def get_scan_counts(history):
    total = len(history)

    phishing = sum(
        bool(item.get("is_phishing", False))
        for item in history
    )

    legitimate = total - phishing

    return total, phishing, legitimate


def render_card(title, description, icon=""):
    render_html(
        f"""
        <div class="section-card">
            <h3>{icon} {title}</h3>
            <p>{description}</p>
        </div>
        """
    )


def show_class_metrics(metrics, class_key, display_name):

    st.markdown(f"#### {display_name}")

    class_metrics = metrics.get(class_key)

    if not isinstance(class_metrics, dict):
        st.info(f"{display_name} metrics unavailable.")
        return

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Precision",
        format_percent(class_metrics.get("precision"))
    )

    col2.metric(
        "Recall",
        format_percent(class_metrics.get("recall"))
    )

    col3.metric(
        "F1-score",
        format_percent(class_metrics.get("f1_score"))
    )

    support = class_metrics.get("support")

    try:
        support = f"{int(support):,}"
    except (TypeError, ValueError):
        support = "N/A"

    col4.metric("Support", support)


# =========================================================
# 4. NAVIGATION CALLBACKS
# =========================================================

def navigate_to(destination):
    st.session_state.navigation = destination


def clear_previous_result():
    st.session_state.prediction_result = None


# =========================================================
# 5. ANALYTICS DONUT CHART
# =========================================================

def create_analytics_donut(phishing_count, legitimate_count):

    total = phishing_count + legitimate_count

    if total == 0:
        return None

    phishing_color = "#FF5263"
    legitimate_color = "#19C99A"
    background_color = "#14243A"
    text_color = "#F0F4FA"

    fig, ax = plt.subplots(figsize=(7, 5))

    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    values = [phishing_count, legitimate_count]
    colors = [phishing_color, legitimate_color]

    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={
            "width": 0.34,
            "edgecolor": background_color,
            "linewidth": 3,
        },
    )

    ax.text(
        0,
        0.08,
        f"{total:,}",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color=text_color,
    )

    ax.text(
        0,
        -0.18,
        "TOTAL SCANS",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#AEBED0",
    )

    for wedge, count in zip(wedges, values):

        if count == 0:
            continue

        percentage = count / total * 100

        if percentage >= 5:
            angle = (wedge.theta1 + wedge.theta2) / 2

            x = 0.82 * math.cos(math.radians(angle))
            y = 0.82 * math.sin(math.radians(angle))

            ax.text(
                x,
                y,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
            )

    phishing_pct = phishing_count / total * 100
    legitimate_pct = legitimate_count / total * 100

    legend_labels = [
        f"Potential Phishing: {phishing_count:,} ({phishing_pct:.1f}%)",
        f"Legitimate: {legitimate_count:,} ({legitimate_pct:.1f}%)",
    ]

    ax.legend(
        wedges,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=1,
        frameon=False,
        fontsize=10,
        labelcolor=text_color,
    )

    ax.axis("equal")
    fig.tight_layout(pad=2)

    return fig


# =========================================================
# 6. LOAD CSS
# =========================================================

if CSS_PATH.exists():
    css_content = CSS_PATH.read_text(encoding="utf-8")
    st.html(f"<style>{css_content}</style>")
else:
    st.warning("Custom CSS file not found.")


# =========================================================
# 7. SESSION STATE
# =========================================================

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# =========================================================
# 8. LOAD MODEL AND SCALER
# =========================================================

@st.cache_resource
def load_artifacts():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)

    if META_PATH.exists():
        loaded_metadata = json.loads(
            META_PATH.read_text(encoding="utf-8")
        )
    else:
        loaded_metadata = {"threshold": 0.5}

    return loaded_model, loaded_scaler, loaded_metadata


try:
    model, scaler, metadata = load_artifacts()

except Exception as error:
    st.error(f"Unable to load ANN model: {error}")
    st.stop()


# =========================================================
# 9. SIDEBAR
# =========================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">🛡️</div>
            <div class="sidebar-name">PhishGuard AI</div>
            <div class="sidebar-tagline">
                CYBERSECURITY INTELLIGENCE
            </div>
        </div>
        """
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "URL Scanner",
            "Scan History",
            "Live Scan Analytics",
            "Model Evaluation",
        ],
        key="navigation",
    )

    st.divider()

    st.markdown("### System Status")

    render_html(
        """
        <span class="status-pill">
            ● ANN MODEL ACTIVE
        </span>
        """
    )

    st.write("")
    st.caption("Deep Learning | TensorFlow")
    st.caption("25 URL-based features")
    st.caption("URL-only analysis")

    st.divider()

    st.caption("PhishGuard AI v1.0")
    st.caption("Intelligent URL Screening")


# =========================================================
# 10. GLOBAL HERO HEADER
# =========================================================

render_html(
    """
    <div class="hero">

        <div class="hero-topline">
            <span class="hero-label">
                CYBERSECURITY • DEEP LEARNING
            </span>

            <span class="hero-indicator">
                ● SYSTEM ONLINE
            </span>
        </div>

        <div class="hero-title">
            🛡️ PhishGuard AI
        </div>

        <p class="hero-description">
            Intelligent Phishing URL Detection System
            powered by an Artificial Neural Network.
        </p>

        <div class="hero-bottom">
            <span>ARTIFICIAL NEURAL NETWORK</span>
            <span>25 URL FEATURES</span>
            <span>INTELLIGENT URL SCREENING</span>
        </div>

    </div>
    """
)

st.caption(
    "This application analyzes URL text only. "
    "It does not visit websites, verify website content, "
    "or guarantee that a URL is safe."
)


# =========================================================
# 11. OVERVIEW DASHBOARD
# =========================================================

if page == "Overview":

    st.title("Security Dashboard")

    st.caption(
        "Monitor URL screening activity and access your security tools."
    )

    history = st.session_state.scan_history

    total, phishing, legitimate = get_scan_counts(history)

    # TOP STATISTICS

    st.markdown("### Security Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🌐 Total URLs Scanned",
        f"{total:,}",
    )

    col2.metric(
        "⚠️ Potential Phishing",
        f"{phishing:,}",
    )

    col3.metric(
        "✅ Legitimate Predictions",
        f"{legitimate:,}",
    )

    st.divider()

    # QUICK ACCESS

    st.markdown("### Quick Access")

    st.caption(
        "Choose a tool to continue your security analysis."
    )

    quick_col1, quick_col2 = st.columns(2)

    with quick_col1:

        render_card(
            "URL Scanner",
            "Submit a URL and analyze its extracted features "
            "using the trained Artificial Neural Network.",
            "🔍",
        )

        st.button(
            "Open URL Scanner →",
            key="overview_scanner_button",
            use_container_width=True,
            on_click=navigate_to,
            args=("URL Scanner",),
        )

    with quick_col2:

        render_card(
            "Model Evaluation",
            "Explore classification metrics, training curves, "
            "validation performance and the confusion matrix.",
            "🧠",
        )

        st.button(
            "View Model Evaluation →",
            key="overview_evaluation_button",
            use_container_width=True,
            on_click=navigate_to,
            args=("Model Evaluation",),
        )

    st.divider()

    # LOWER DASHBOARD PANELS

    st.markdown("### Activity & Classification")

    left_panel, right_panel = st.columns(
        [1.25, 1],
        vertical_alignment="top",
    )

    with left_panel:

        render_html(
            """
            <div class="dashboard-panel">
                <div class="panel-heading">🕘 Recent Scans</div>
                <div class="panel-description">
                    Latest URL screening activity in this session.
                </div>
            </div>
            """
        )

        if history:

            recent_df = pd.DataFrame(history[-5:][::-1])

            st.dataframe(
                recent_df[["time", "url", "status", "score"]],
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No URLs scanned yet. Open URL Scanner to begin."
            )

            st.button(
                "Start Your First Scan →",
                key="first_scan_button",
                use_container_width=True,
                on_click=navigate_to,
                args=("URL Scanner",),
            )

    with right_panel:

        render_html(
            """
            <div class="dashboard-panel">
                <div class="panel-heading">📊 Classification Distribution</div>
                <div class="panel-description">
                    Breakdown of phishing and legitimate predictions.
                </div>
            </div>
            """
        )

        if total > 0:

            overview_fig = create_analytics_donut(
                phishing,
                legitimate,
            )

            if overview_fig is not None:
                st.pyplot(
                    overview_fig,
                    use_container_width=True,
                )
                plt.close(overview_fig)

        else:

            st.info(
                "Classification statistics will appear here "
                "after your first URL scan."
            )

    st.divider()

    # SYSTEM INFORMATION

    st.markdown("### Detection Engine")

    info1, info2, info3 = st.columns(3)

    with info1:
        render_card(
            "ANN Architecture",
            "Artificial Neural Network-based classification.",
            "🧠",
        )

    with info2:
        render_card(
            "Feature Extraction",
            "25 numerical features extracted from URL text.",
            "⚙️",
        )

    with info3:
        render_card(
            "Session Analytics",
            "Live statistics based on URLs scanned in this session.",
            "📈",
        )


# =========================================================
# 12. URL SCANNER
# =========================================================

elif page == "URL Scanner":

    st.title("🔍 Intelligent URL Scanner")

    st.caption(
        "Analyze suspicious URLs using the trained ANN model."
    )

    render_card(
        "SCAN ENGINE",
        "Enter a website URL to perform intelligent security analysis.",
        "🛡️",
    )

    url = st.text_input(
        "Website URL",
        placeholder="https://example.com",
        key="url_input",
        on_change=clear_previous_result,
    )

    if st.button(
        "🔎 ANALYZE URL",
        type="primary",
        use_container_width=True,
        disabled=not url.strip(),
    ):

        try:

            with st.spinner("Analyzing URL with ANN model..."):

                cleaned_url = url.strip()

                features = extract_feature_matrix([cleaned_url])

                if not isinstance(features, pd.DataFrame):
                    features = pd.DataFrame(features)

                if features.empty:
                    raise ValueError(
                        "Feature extraction returned no features."
                    )

                expected_features = getattr(
                    scaler,
                    "n_features_in_",
                    None,
                )

                if (
                    expected_features is not None
                    and features.shape[1] != expected_features
                ):
                    raise ValueError(
                        f"Expected {expected_features} features, "
                        f"received {features.shape[1]}."
                    )

                scaled_features = scaler.transform(features)

                prediction = model.predict(
                    scaled_features,
                    verbose=0,
                )

                probability = float(prediction.reshape(-1)[0])

                threshold = float(
                    metadata.get("threshold", 0.5)
                )

                is_phishing = probability >= threshold

                result = {
                    "url": cleaned_url,
                    "probability": probability,
                    "threshold": threshold,
                    "is_phishing": is_phishing,
                    "features": features.copy(),
                }

                st.session_state.prediction_result = result

                st.session_state.scan_history.append(
                    {
                        "time": datetime.now().strftime(
                            "%d-%m-%Y %H:%M:%S"
                        ),
                        "url": cleaned_url,
                        "status": (
                            "Potential Phishing"
                            if is_phishing
                            else "Legitimate"
                        ),
                        "score": f"{probability:.2%}",
                        "phishing_score": probability * 100,
                        "is_phishing": is_phishing,
                    }
                )

        except Exception as error:

            st.session_state.prediction_result = None
            st.error(f"URL analysis failed: {error}")

    result = st.session_state.prediction_result

    if result is not None:

        st.divider()
        st.subheader("📊 Analysis Results")

        st.markdown(f"**Analyzed URL:** `{result['url']}`")

        col1, col2 = st.columns(2)

        with col1:

            if result["is_phishing"]:

                render_html(
                    """
                    <div class="result-phishing">
                        <div class="card-eyebrow">
                            SECURITY WARNING
                        </div>
                        <h3>⚠️ Potential Phishing</h3>
                        <p>
                            The model detected phishing-related URL patterns.
                        </p>
                    </div>
                    """
                )

            else:

                render_html(
                    """
                    <div class="result-safe">
                        <div class="card-eyebrow">
                            SCAN RESULT
                        </div>
                        <h3>✅ Legitimate Prediction</h3>
                        <p>
                            No phishing pattern was detected by this model.
                        </p>
                    </div>
                    """
                )

        with col2:

            st.metric(
                "Model Phishing Score",
                f"{result['probability']:.2%}",
            )

            st.progress(
                min(max(result["probability"], 0.0), 1.0)
            )

        st.caption(
            f"Decision threshold: {result['threshold']:.2f}. "
            "This is a model output, not a verified probability of harm."
        )

        st.divider()

        st.subheader("📈 Individual URL Feature Analysis")

        feature_df = result["features"]

        if not feature_df.empty:

            feature_chart = pd.DataFrame(
                {
                    "Feature": feature_df.columns.astype(str),
                    "Value": feature_df.iloc[0].astype(float).values,
                }
            ).set_index("Feature")

            st.bar_chart(
                feature_chart,
                y_label="Extracted Feature Value",
            )

            with st.expander("View All Extracted Features"):

                st.dataframe(
                    feature_df,
                    use_container_width=True,
                    hide_index=True,
                )


# =========================================================
# 13. SCAN HISTORY
# =========================================================

elif page == "Scan History":

    st.title("🕘 URL Scan History")

    st.caption(
        "Review URLs analyzed during the current application session."
    )

    history = st.session_state.scan_history

    if history:

        st.metric(
            "Total Scans in This Session",
            f"{len(history):,}",
        )

        history_df = pd.DataFrame(history[::-1])

        st.dataframe(
            history_df[["time", "url", "status", "score"]],
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        if st.button("🗑️ Clear Scan History"):

            st.session_state.scan_history = []
            st.session_state.prediction_result = None

            st.rerun()

    else:

        st.info(
            "No scan history available. Analyze a URL to create your first record."
        )


# =========================================================
# 14. LIVE SCAN ANALYTICS
# =========================================================

elif page == "Live Scan Analytics":

    st.title("📊 Live Scan Analytics")

    st.caption(
        "Real-time classification statistics for URLs scanned "
        "during the current application session."
    )

    history = st.session_state.scan_history

    if history:

        live_df = pd.DataFrame(history)

        total, phishing, legitimate = get_scan_counts(history)

        col1, col2, col3 = st.columns(3)

        col1.metric("🌐 Total Scans", f"{total:,}")

        col2.metric(
            "⚠️ Potential Phishing",
            f"{phishing:,}",
            delta=f"{phishing / total * 100:.1f}% of scans",
            delta_color="inverse",
        )

        col3.metric(
            "✅ Legitimate URLs",
            f"{legitimate:,}",
            delta=f"{legitimate / total * 100:.1f}% of scans",
        )

        st.divider()

        st.subheader("Classification Distribution")

        chart_col, info_col = st.columns(
            [1.5, 1],
            vertical_alignment="center",
        )

        with chart_col:

            donut_fig = create_analytics_donut(
                phishing,
                legitimate,
            )

            if donut_fig is not None:

                st.pyplot(
                    donut_fig,
                    use_container_width=True,
                )

                plt.close(donut_fig)

        with info_col:

            st.markdown("### Scan Breakdown")

            phishing_pct = phishing / total * 100
            legitimate_pct = legitimate / total * 100

            render_html(
                f"""
                <div style="padding: 1rem 0;">
                    <div style="
                        font-size: 1rem;
                        font-weight: 800;
                        color: #FF5263;
                    ">
                        🔴 POTENTIAL PHISHING
                    </div>

                    <div style="
                        font-size: 2.5rem;
                        font-weight: 900;
                        color: #FF5263;
                    ">
                        {phishing:,}
                    </div>

                    <div style="color:#AEBED0; font-size:1rem;">
                        {phishing_pct:.2f}% of total scans
                    </div>
                </div>
                """
            )

            st.progress(phishing / total)

            render_html(
                f"""
                <div style="padding: 1rem 0;">
                    <div style="
                        font-size: 1rem;
                        font-weight: 800;
                        color: #19C99A;
                    ">
                        🟢 LEGITIMATE URLS
                    </div>

                    <div style="
                        font-size: 2.5rem;
                        font-weight: 900;
                        color: #19C99A;
                    ">
                        {legitimate:,}
                    </div>

                    <div style="color:#AEBED0; font-size:1rem;">
                        {legitimate_pct:.2f}% of total scans
                    </div>
                </div>
                """
            )

            st.progress(legitimate / total)

        st.divider()

        st.subheader("📈 URL-wise Phishing Scores")

        score_df = live_df[
            ["url", "phishing_score"]
        ].copy()

        score_df["Scan"] = [
            f"Scan {i + 1}"
            for i in range(len(score_df))
        ]

        score_chart = score_df.set_index(
            "Scan"
        )["phishing_score"]

        st.bar_chart(
            score_chart,
            y_label="Model Phishing Score (%)",
        )

        st.caption(
            "Scores represent model outputs for individually scanned URLs."
        )

        st.divider()

        st.subheader("Complete Scan Analytics")

        display_df = live_df[
            ["time", "url", "status", "score"]
        ].iloc[::-1]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        render_html(
            """
            <div class="dashboard-panel" style="
                text-align:center;
                padding:3.5rem 2rem;
            ">
                <div style="font-size:3rem;">📊</div>
                <h3>No Scan Data Available</h3>
                <p>
                    Start scanning URLs to generate your live
                    classification analytics.
                </p>
            </div>
            """
        )

        st.write("")

        st.button(
            "Open URL Scanner →",
            use_container_width=True,
            on_click=navigate_to,
            args=("URL Scanner",),
        )


# =========================================================
# 15. MODEL EVALUATION
# =========================================================

elif page == "Model Evaluation":

    st.title("🧠 ANN Model Evaluation")

    st.caption(
        "Training and testing performance of the Artificial Neural Network."
    )

    st.info(
        "These are fixed model evaluation results. "
        "Scanning URLs does not retrain the model."
    )

    st.divider()

    st.subheader("Model Performance Metrics")

    if METRICS_PATH.exists():

        try:

            metrics = json.loads(
                METRICS_PATH.read_text(encoding="utf-8")
            )

            if not isinstance(metrics, dict):

                st.warning("Invalid metrics format.")

            else:

                # Only this container is targeted by the
                # smaller Model Evaluation metric CSS rules.
                with st.container(key="evaluation_metrics"):

                    scalar_metrics = [
                        ("Accuracy", metrics.get("accuracy"), "percent"),
                        ("ROC AUC", metrics.get("roc_auc"), "percent"),
                        ("Tuned F1", metrics.get("tuned_f1"), "percent"),
                        ("Threshold", metrics.get("threshold"), "threshold"),
                        (
                            "Tuned Threshold",
                            metrics.get("tuned_threshold"),
                            "threshold",
                        ),
                    ]

                    available = [
                        item for item in scalar_metrics
                        if item[1] is not None
                    ]

                    if available:

                        cols = st.columns(min(len(available), 5))

                        for index, (name, value, kind) in enumerate(available):

                            display = (
                                format_percent(value)
                                if kind == "percent"
                                else format_threshold(value)
                            )

                            cols[index % len(cols)].metric(
                                name,
                                display,
                            )

                    show_class_metrics(
                        metrics,
                        "phishing",
                        "⚠️ Phishing Class",
                    )

                    st.divider()

                    show_class_metrics(
                        metrics,
                        "legitimate",
                        "✅ Legitimate Class",
                    )

        except (OSError, json.JSONDecodeError) as error:

            st.error(f"Unable to read metrics: {error}")

    else:

        st.warning("Evaluation metrics are not available.")

    st.divider()

    graph_files = [
        (
            "📈 Training and Validation Accuracy",
            RESULTS_DIR / "training_accuracy.png",
        ),
        (
            "📉 Training and Validation Loss",
            RESULTS_DIR / "training_loss.png",
        ),
        (
            "🎯 Confusion Matrix",
            RESULTS_DIR / "confusion_matrix.png",
        ),
    ]

    for title, path in graph_files:

        st.subheader(title)

        if path.exists():

            st.image(
                str(path),
                use_container_width=True,
            )

        else:

            st.warning(f"Graph not available: {path.name}")

        st.divider()


# =========================================================
# 16. FOOTER
# =========================================================

render_html(
    """
    <div class="footer">

        <div class="footer-brand">
            🛡️ PhishGuard AI
        </div>

        <div class="footer-description">
            Deep Learning Project |
            ANN-based Phishing URL Screening
        </div>

        <div class="footer-version">
            CYBERSECURITY INTELLIGENCE • VERSION 1.0
        </div>

    </div>
    """
)

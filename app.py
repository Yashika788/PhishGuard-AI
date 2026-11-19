
"""PhishGuard AI - Premium Cybersecurity Dashboard."""

from pathlib import Path
from datetime import datetime
import json
import textwrap

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
# 3. HTML AND DISPLAY HELPERS
# =========================================================

def render_html(html):
    """Render a trusted HTML fragment."""
    st.html(textwrap.dedent(html).strip())


def format_percent(value):
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return str(value)


def format_threshold(value):
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def render_card(title, description):
    render_html(
        f"""
        <div class="section-card">
            <h3>{title}</h3>
            <p class="small-muted">{description}</p>
        </div>
        """
    )


def get_scan_counts(history):
    total = len(history)

    phishing = sum(
        bool(item.get("is_phishing", False))
        for item in history
    )

    legitimate = total - phishing

    return total, phishing, legitimate


def show_class_metrics(metrics, class_key, display_name):

    class_metrics = metrics.get(class_key)

    st.markdown(f"#### {display_name}")

    if not isinstance(class_metrics, dict):
        st.info(
            f"{display_name} metrics are not available."
        )
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
        support_display = (
            f"{int(support):,}"
            if support is not None
            else "N/A"
        )
    except (TypeError, ValueError):
        support_display = str(support)

    col4.metric("Support", support_display)


# =========================================================
# 4. NAVIGATION CALLBACK
# =========================================================

def navigate_to(destination):
    """
    Change the navigation radio value through a callback.

    Streamlit executes this callback before the next script run,
    so the widget state can safely be updated here.
    """
    st.session_state.navigation = destination


# =========================================================
# 5. LOAD CUSTOM CSS
# =========================================================

if CSS_PATH.exists():

    css_content = CSS_PATH.read_text(
        encoding="utf-8"
    )

    st.html(f"<style>{css_content}</style>")

else:
    st.warning(
        "Custom CSS file not found: assets/style.css"
    )


# =========================================================
# 6. SESSION STATE
# =========================================================

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


def clear_previous_result():
    st.session_state.prediction_result = None


# =========================================================
# 7. LOAD ANN MODEL AND SCALER
# =========================================================

@st.cache_resource
def load_artifacts():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained ANN model not found: {MODEL_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}"
        )

    loaded_model = tf.keras.models.load_model(
        MODEL_PATH
    )

    loaded_scaler = joblib.load(
        SCALER_PATH
    )

    if META_PATH.exists():

        loaded_metadata = json.loads(
            META_PATH.read_text(encoding="utf-8")
        )

    else:
        loaded_metadata = {"threshold": 0.5}

    return (
        loaded_model,
        loaded_scaler,
        loaded_metadata
    )


try:

    model, scaler, metadata = load_artifacts()

except Exception as error:

    st.error(
        f"Unable to load ANN model or scaler: {error}"
    )

    st.stop()


# =========================================================
# 8. SIDEBAR NAVIGATION
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
# 9. HERO HEADER
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
# 10. OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.subheader("Security Dashboard")

    st.caption(
        "A quick overview of your URL scanning activity."
    )

    history = st.session_state.scan_history

    total_scans, phishing_count, legitimate_count = (
        get_scan_counts(history)
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🌐 Total URLs Scanned",
        total_scans
    )

    col2.metric(
        "⚠️ Potential Phishing",
        phishing_count
    )

    col3.metric(
        "✅ Legitimate Predictions",
        legitimate_count
    )

    st.divider()

    st.subheader("Quick Access")

    col1, col2 = st.columns(2)

    with col1:

        render_card(
            "🔍 URL Scanner",
            "Analyze website URLs using the trained ANN "
            "model and inspect extracted URL features."
        )

        st.button(
            "Open URL Scanner",
            use_container_width=True,
            on_click=navigate_to,
            args=("URL Scanner",),
        )

    with col2:

        render_card(
            "🧠 Model Evaluation",
            "Explore accuracy, loss, evaluation metrics "
            "and the confusion matrix."
        )

        st.button(
            "View Model Evaluation",
            use_container_width=True,
            on_click=navigate_to,
            args=("Model Evaluation",),
        )

    st.divider()

    st.subheader("Recent Scans")

    if history:

        recent_df = pd.DataFrame(
            history[-5:][::-1]
        )

        st.dataframe(
            recent_df[
                ["time", "url", "status", "score"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No URLs scanned yet. "
            "Open URL Scanner to begin."
        )


# =========================================================
# 11. URL SCANNER PAGE
# =========================================================

elif page == "URL Scanner":

    st.subheader("🔍 Intelligent URL Scanner")

    st.caption(
        "Analyze suspicious URLs using the trained "
        "Artificial Neural Network."
    )

    render_html(
        """
        <div class="section-card">

            <div class="card-eyebrow">
                SCAN ENGINE
            </div>

            <h3>Enter URL for Security Analysis</h3>

            <p class="small-muted">
                The ANN model extracts lexical and structural
                URL features to generate a phishing classification.
            </p>

        </div>
        """
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

            with st.spinner(
                "Analyzing URL with ANN model..."
            ):

                cleaned_url = url.strip()

                features = extract_feature_matrix(
                    [cleaned_url]
                )

                if not isinstance(features, pd.DataFrame):
                    features = pd.DataFrame(features)

                if features.empty:
                    raise ValueError(
                        "URL feature extraction returned no features."
                    )

                expected_features = getattr(
                    scaler,
                    "n_features_in_",
                    None
                )

                if (
                    expected_features is not None
                    and features.shape[1] != expected_features
                ):

                    raise ValueError(
                        f"Feature mismatch: scaler expects "
                        f"{expected_features} features, but extraction "
                        f"returned {features.shape[1]}."
                    )

                scaled_features = scaler.transform(
                    features
                )

                prediction = model.predict(
                    scaled_features,
                    verbose=0
                )

                probability = float(
                    prediction.reshape(-1)[0]
                )

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

            st.error(
                f"URL analysis failed: {error}"
            )

    result = st.session_state.prediction_result

    if result is not None:

        st.divider()

        st.subheader("📊 Analysis Results")

        st.markdown(
            f"**Analyzed URL:** `{result['url']}`"
        )

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

                        <p class="small-muted">
                            The model detected phishing-related
                            URL patterns.
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

                        <p class="small-muted">
                            No phishing pattern was detected
                            by this model.
                        </p>

                    </div>
                    """
                )

        with col2:

            st.metric(
                "Model Phishing Score",
                f"{result['probability']:.2%}"
            )

            st.progress(
                min(
                    max(result["probability"], 0.0),
                    1.0
                )
            )

        st.caption(
            f"Decision threshold: {result['threshold']:.2f}. "
            "This is a model output, not a verified probability of harm."
        )

        st.divider()

        st.subheader(
            "📈 Individual URL Feature Analysis"
        )

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
                y_label="Extracted Feature Value"
            )

            with st.expander(
                "View All Extracted Features"
            ):

                st.dataframe(
                    feature_df,
                    use_container_width=True,
                    hide_index=True,
                )

        else:

            st.warning(
                "No URL features available for visualization."
            )


# =========================================================
# 12. SCAN HISTORY PAGE
# =========================================================

elif page == "Scan History":

    st.subheader("🕘 URL Scan History")

    history = st.session_state.scan_history

    if history:

        st.metric(
            "Total Scans in This Session",
            len(history)
        )

        history_df = pd.DataFrame(
            history[::-1]
        )

        st.dataframe(
            history_df[
                ["time", "url", "status", "score"]
            ],
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
            "No scan history available. "
            "Analyze a URL to create your first record."
        )


# =========================================================
# 13. LIVE SCAN ANALYTICS PAGE
# =========================================================

elif page == "Live Scan Analytics":

    st.subheader("📊 Live Scan Analytics")

    st.caption(
        "Monitor URL predictions generated during the current session."
    )

    history = st.session_state.scan_history

    if history:

        live_df = pd.DataFrame(history)

        total_scans, phishing_count, legitimate_count = (
            get_scan_counts(history)
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("🌐 Total Scans", total_scans)
        col2.metric("⚠️ Potential Phishing", phishing_count)
        col3.metric("✅ Legitimate URLs", legitimate_count)

        st.divider()

        st.subheader("Classification Distribution")

        labels = ["Phishing", "Legitimate"]

        values = [
            phishing_count,
            legitimate_count
        ]

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%" if total_scans else None,
            startangle=90,
            wedgeprops={
                "width": 0.42,
                "edgecolor": "#0b1423",
            },
        )

        ax.set_title(
            "Phishing vs Legitimate URLs"
        )

        ax.axis("equal")

        fig.patch.set_alpha(0)

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.divider()

        st.subheader("URL-wise Phishing Score")

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
            y_label="Model Phishing Score (%)"
        )

        st.caption(
            "Scores represent model outputs for individually scanned URLs."
        )

        st.divider()

        st.subheader("Complete Scan Analytics")

        st.dataframe(
            live_df[
                ["time", "url", "status", "score"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No scan data available yet. "
            "Open URL Scanner and analyze URLs."
        )


# =========================================================
# 14. MODEL EVALUATION PAGE
# =========================================================

elif page == "Model Evaluation":

    st.subheader("🧠 ANN Model Evaluation")

    st.caption(
        "Training and testing performance of the "
        "Artificial Neural Network."
    )

    st.info(
        "These are fixed model evaluation results. "
        "Scanning individual URLs does not retrain the model "
        "or change these graphs."
    )

    st.divider()

    st.subheader("Model Performance Metrics")

    if METRICS_PATH.exists():

        try:

            metrics = json.loads(
                METRICS_PATH.read_text(
                    encoding="utf-8"
                )
            )

            if not isinstance(metrics, dict) or not metrics:

                st.warning(
                    "No evaluation metrics found in metrics.json."
                )

            else:

                scalar_metrics = [
                    (
                        "Accuracy",
                        metrics.get("accuracy"),
                        "percent"
                    ),
                    (
                        "ROC AUC",
                        metrics.get("roc_auc"),
                        "percent"
                    ),
                    (
                        "Tuned F1",
                        metrics.get("tuned_f1"),
                        "percent"
                    ),
                    (
                        "Threshold",
                        metrics.get("threshold"),
                        "threshold"
                    ),
                    (
                        "Tuned Threshold",
                        metrics.get("tuned_threshold"),
                        "threshold"
                    ),
                ]

                available_metrics = [
                    item
                    for item in scalar_metrics
                    if item[1] is not None
                ]

                if available_metrics:

                    metric_columns = st.columns(
                        min(len(available_metrics), 5)
                    )

                    for index, (
                        name,
                        value,
                        value_type
                    ) in enumerate(available_metrics):

                        display_value = (
                            format_percent(value)
                            if value_type == "percent"
                            else format_threshold(value)
                        )

                        metric_columns[
                            index % len(metric_columns)
                        ].metric(
                            name,
                            display_value
                        )

                else:

                    st.warning(
                        "No scalar evaluation metrics were found."
                    )

                st.divider()

                st.subheader("Class-wise Performance")

                show_class_metrics(
                    metrics,
                    "phishing",
                    "⚠️ Phishing Class"
                )

                st.divider()

                show_class_metrics(
                    metrics,
                    "legitimate",
                    "✅ Legitimate Class"
                )

                if metrics.get("tuned_f1") is not None:

                    st.caption(
                        "Tuned F1 is the F1 score associated "
                        "with the selected decision threshold."
                    )

        except (OSError, json.JSONDecodeError) as error:

            st.error(
                f"Unable to read evaluation metrics: {error}"
            )

        except Exception as error:

            st.error(
                f"Unexpected error while displaying metrics: {error}"
            )

    else:

        st.warning(
            f"Evaluation metrics are not available: {METRICS_PATH}"
        )

    st.divider()

    st.subheader("📈 Model Accuracy Graph")

    accuracy_path = (
        RESULTS_DIR / "training_accuracy.png"
    )

    if accuracy_path.exists():

        st.image(
            str(accuracy_path),
            caption="Training and Validation Accuracy",
            use_container_width=True,
        )

    else:

        st.warning(
            "Model accuracy graph is not available."
        )

    st.divider()

    st.subheader("📉 Model Loss Graph")

    loss_path = (
        RESULTS_DIR / "training_loss.png"
    )

    if loss_path.exists():

        st.image(
            str(loss_path),
            caption="Training and Validation Loss",
            use_container_width=True,
        )

    else:

        st.warning(
            "Model loss graph is not available."
        )

    st.divider()

    st.subheader("🎯 Confusion Matrix")

    confusion_path = (
        RESULTS_DIR / "confusion_matrix.png"
    )

    if confusion_path.exists():

        st.image(
            str(confusion_path),
            caption="ANN Model Test Dataset Confusion Matrix",
            use_container_width=True,
        )

    else:

        st.warning(
            "Confusion matrix is not available."
        )


# =========================================================
# 15. FOOTER
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

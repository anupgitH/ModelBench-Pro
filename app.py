
from pathlib import Path
import io

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from modelbench.data.loader import load_csv
from modelbench.data.validator import validate_dataset
from modelbench.data.cleaner import clean_dataset
from modelbench.models.registry import get_model_names
from modelbench.models.trainer import train_model
from modelbench.evaluation.metrics import calculate_metrics
from modelbench.evaluation.cross_validation import run_cross_validation
from modelbench.evaluation.visualizations import (
    plot_confusion_matrix,
    plot_model_comparison,
)
from modelbench.experiments.tracker import (
    save_experiment,
    load_experiments,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ModelBench Pro",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "training_result" not in st.session_state:
    st.session_state.training_result = None

if "cv_result" not in st.session_state:
    st.session_state.cv_result = None

if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None

if "dataset_signature" not in st.session_state:
    st.session_state.dataset_signature = None


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🧪 ModelBench Pro")
st.markdown(
    "### Machine Learning Experiment Tracking & Evaluation Platform"
)
st.caption(
    "Upload data • Train classifiers • Evaluate performance • "
    "Compare experiments"
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv"],
        help="Upload a CSV file containing features and a target column.",
    )

    st.divider()
    st.subheader("Training Settings")

    available_models = get_model_names()

    selected_model = st.selectbox(
        "Select model",
        available_models,
    )

    test_percentage = st.slider(
        "Test data (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
    )

    random_state = st.number_input(
        "Random state",
        min_value=0,
        max_value=10000,
        value=42,
        step=1,
    )

    st.divider()
    st.caption("ModelBench Pro | ML Experiment Workspace")


# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------

if uploaded_file is None:
    st.session_state.dataset_signature = None
    st.session_state.training_result = None
    st.session_state.cv_result = None
    st.session_state.comparison_results = None

    st.info("Upload a CSV file from the sidebar to begin.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📂 Data Analysis")
        st.write("Preview columns, inspect missing values and clean duplicates.")

    with col2:
        st.subheader("🤖 Model Training")
        st.write("Train classification models using reusable ML pipelines.")

    with col3:
        st.subheader("📊 Evaluation")
        st.write("Review performance metrics and compare model experiments.")

    st.stop()


# --------------------------------------------------
# LOAD AND VALIDATE DATA
# --------------------------------------------------

try:
    uploaded_bytes = uploaded_file.getvalue()
    df = load_csv(uploaded_bytes)

except Exception as exc:
    st.error(f"Unable to load the CSV file: {exc}")
    st.stop()


validation_errors = validate_dataset(df)

if validation_errors:
    st.error("The uploaded dataset needs attention.")

    for error in validation_errors:
        st.warning(error)

    st.stop()


# Clear previous results when a different dataset is uploaded.
dataset_signature = (
    uploaded_file.name,
    len(uploaded_bytes),
    hash(uploaded_bytes),
)

if st.session_state.dataset_signature != dataset_signature:
    st.session_state.training_result = None
    st.session_state.cv_result = None
    st.session_state.comparison_results = None
    st.session_state.dataset_signature = dataset_signature


# --------------------------------------------------
# DATA OVERVIEW
# --------------------------------------------------

st.header("📂 Dataset Overview")

row1, row2, row3, row4 = st.columns(4)

row1.metric("Total Rows", f"{len(df):,}")
row2.metric("Total Columns", f"{len(df.columns):,}")
row3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
row4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

with st.expander("Preview dataset", expanded=True):
    st.dataframe(df.head(100), use_container_width=True)

with st.expander("Column information"):
    info_df = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values,
        "Unique Values": df.nunique(dropna=True).values,
    })

    st.dataframe(info_df, use_container_width=True)


# --------------------------------------------------
# CLEANING
# --------------------------------------------------

st.header("🧹 Data Preparation")

remove_duplicates = st.checkbox(
    "Remove duplicate rows and duplicate column names",
    value=True,
)

if remove_duplicates:
    working_df = clean_dataset(df)
else:
    working_df = df.copy()

st.write(
    f"Dataset available for training: **{len(working_df):,} rows** "
    f"and **{len(working_df.columns):,} columns**."
)

if len(working_df) != len(df):
    st.info(
        f"Cleaning removed {len(df) - len(working_df)} duplicate rows."
    )

if len(working_df) < 5 or len(working_df.columns) < 2:
    st.error("At least 5 rows and 2 columns are required.")
    st.stop()


# --------------------------------------------------
# TARGET SELECTION
# --------------------------------------------------

st.header("🎯 Target Configuration")

target_column = st.selectbox(
    "Choose the target column to predict",
    options=list(working_df.columns),
    help="The selected column is the label the model will learn to predict.",
)

target_data = working_df[target_column].dropna()

if target_data.nunique() < 2:
    st.error("The target column must contain at least two distinct classes.")
    st.stop()

if target_data.nunique() > 50:
    st.warning(
        "This target contains more than 50 unique values. "
        "This application is designed for classification; "
        "verify that the selected target is categorical."
    )

st.caption(
    f"Target: `{target_column}` · "
    f"Classes: {target_data.nunique()} · "
    f"Rows with a missing target: "
    f"{int(working_df[target_column].isna().sum())}"
)

st.divider()


# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------

train_tab, compare_tab, cv_tab, history_tab = st.tabs([
    "🚀 Train & Evaluate",
    "📊 Compare Models",
    "🔁 Cross-Validation",
    "📚 Experiment History",
])


# --------------------------------------------------
# TRAIN AND EVALUATE
# --------------------------------------------------

with train_tab:
    st.subheader("Train your selected model")

    st.write(
        f"Selected algorithm: **{selected_model}**"
    )

    train_clicked = st.button(
        "🚀 Train Model",
        type="primary",
        key="train_model_button",
    )

    if train_clicked:
        try:
            with st.spinner("Training model and evaluating predictions..."):
                result = train_model(
                    working_df,
                    target_column=target_column,
                    model_name=selected_model,
                    test_size=test_percentage / 100,
                    random_state=int(random_state),
                )

                metrics = calculate_metrics(
                    result["y_test"],
                    result["predictions"],
                )

                st.session_state.training_result = {
                    "model_name": selected_model,
                    "target_column": target_column,
                    "test_percentage": test_percentage,
                    "pipeline": result["pipeline"],
                    "metrics": metrics,
                    "y_test": result["y_test"],
                    "predictions": result["predictions"],
                }

            st.success("Model trained successfully!")

        except Exception as exc:
            st.error(f"Training failed: {exc}")

    result = st.session_state.training_result

    if result is not None:
        metrics = result["metrics"]

        st.divider()
        st.subheader("Performance Metrics")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        m2.metric("Precision", f"{metrics['precision']:.2%}")
        m3.metric("Recall", f"{metrics['recall']:.2%}")
        m4.metric("F1-score", f"{metrics['f1_score']:.2%}")

        st.subheader("Confusion Matrix")

        fig = plot_confusion_matrix(
            result["y_test"],
            result["predictions"],
        )

        try:
            st.pyplot(fig)
        finally:
            plt.close(fig)

        with st.expander("Detailed classification report"):
            st.json(metrics["classification_report"])

        st.subheader("Save and Export")

        if st.button("💾 Save Experiment", key="save_experiment_button"):
            try:
                record = save_experiment(
                    model_name=result["model_name"],
                    metrics=metrics,
                    notes=(
                        f"Target: {result['target_column']}; "
                        f"test size: {result['test_percentage']}%"
                    ),
                )

                st.success(
                    f"Experiment saved. ID: {record['experiment_id']}"
                )

            except Exception as exc:
                st.error(f"Could not save experiment: {exc}")

        buffer = io.BytesIO()
        joblib.dump(result["pipeline"], buffer)

        st.download_button(
            label="⬇️ Download Trained Model",
            data=buffer.getvalue(),
            file_name="modelbench_model.joblib",
            mime="application/octet-stream",
        )

        st.caption(
            "Only load model files from sources you trust. "
            "Joblib files can execute code when loaded."
        )


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

with compare_tab:
    st.subheader("Compare classification algorithms")

    st.write(
        "Train each selected model using the same target and split settings."
    )

    models_to_compare = st.multiselect(
        "Models to compare",
        options=available_models,
        default=available_models[:min(3, len(available_models))],
    )

    if st.button("📊 Run Model Comparison"):
        if not models_to_compare:
            st.warning("Select at least one model.")
        else:
            comparison = {}

            try:
                with st.spinner("Training and comparing models..."):
                    for model_name in models_to_compare:
                        result = train_model(
                            working_df,
                            target_column=target_column,
                            model_name=model_name,
                            test_size=test_percentage / 100,
                            random_state=int(random_state),
                        )

                        comparison_metrics = calculate_metrics(
                            result["y_test"],
                            result["predictions"],
                        )

                        comparison[model_name] = comparison_metrics

                st.session_state.comparison_results = comparison
                st.success("Model comparison completed.")

            except Exception as exc:
                st.error(f"Comparison failed: {exc}")

    comparison = st.session_state.comparison_results

    if comparison:
        comparison_table = pd.DataFrame(comparison).T

        display_columns = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ]

        st.dataframe(
            comparison_table[display_columns].style.format("{:.2%}"),
            use_container_width=True,
        )

        fig = plot_model_comparison(comparison)

        try:
            st.pyplot(fig)
        finally:
            plt.close(fig)

        best_model = max(
            comparison,
            key=lambda name: comparison[name]["f1_score"],
        )

        st.info(
            f"Highest weighted F1-score in this comparison: **{best_model}**. "
            "Use cross-validation and domain knowledge before choosing a final model."
        )


# --------------------------------------------------
# CROSS-VALIDATION
# --------------------------------------------------

with cv_tab:
    st.subheader("Cross-validation analysis")

    folds = st.slider(
        "Number of folds",
        min_value=2,
        max_value=10,
        value=5,
        key="cv_folds",
    )

    if st.button("🔁 Run Cross-Validation"):
        try:
            with st.spinner("Running cross-validation..."):
                cv_result = run_cross_validation(
                    working_df,
                    target_column=target_column,
                    model_name=selected_model,
                    folds=folds,
                    random_state=int(random_state),
                )

            st.session_state.cv_result = cv_result
            st.success("Cross-validation completed.")

        except Exception as exc:
            st.error(f"Cross-validation failed: {exc}")

    cv_result = st.session_state.cv_result

    if cv_result is not None:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Mean Accuracy", f"{cv_result['mean_accuracy']:.2%}")
        c2.metric("Mean Precision", f"{cv_result['mean_precision']:.2%}")
        c3.metric("Mean Recall", f"{cv_result['mean_recall']:.2%}")
        c4.metric("Mean F1-score", f"{cv_result['mean_f1_score']:.2%}")

        fold_df = pd.DataFrame({
            "Fold": range(1, cv_result["folds"] + 1),
            "Accuracy": cv_result["accuracy_scores"],
            "Precision": cv_result["precision_scores"],
            "Recall": cv_result["recall_scores"],
            "F1-score": cv_result["f1_scores"],
        })

        st.subheader("Fold-by-fold scores")
        st.dataframe(
            fold_df.set_index("Fold").style.format("{:.2%}"),
            use_container_width=True,
        )

        st.line_chart(
            fold_df.set_index("Fold")[["Accuracy", "F1-score"]]
        )


# --------------------------------------------------
# EXPERIMENT HISTORY
# --------------------------------------------------

with history_tab:
    st.subheader("Experiment history")

    try:
        history = load_experiments()

        if not history:
            st.info(
                "No saved experiments yet. Train a model and click "
                "'Save Experiment' to create your first record."
            )
        else:
            history_df = pd.json_normalize(history)

            st.dataframe(
                history_df,
                use_container_width=True,
            )

            valid_comparison = {}

            for experiment in history:
                experiment_metrics = experiment.get("metrics", {})

                required = [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                ]

                if all(key in experiment_metrics for key in required):
                    label = (
                        f"{experiment.get('model_name', 'Model')} "
                        f"({experiment.get('experiment_id', '')[:8]})"
                    )
                    valid_comparison[label] = experiment_metrics

            if valid_comparison:
                st.subheader("Historical performance comparison")

                fig = plot_model_comparison(valid_comparison)

                try:
                    st.pyplot(fig)
                finally:
                    plt.close(fig)

            csv_data = history_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Export Experiment History as CSV",
                data=csv_data,
                file_name="modelbench_experiment_history.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(f"Unable to load experiment history: {exc}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "ModelBench Pro | Built with Python, Streamlit, Pandas and Scikit-learn"
)

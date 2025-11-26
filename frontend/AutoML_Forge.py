"""
AutoML Pipeline Builder - Main Streamlit Application
"""
import streamlit as st
import requests
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="AutoML Forge - Tabular ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load translations
i18n_dir = Path(__file__).parent / "i18n"
with open(i18n_dir / "en.json", "r", encoding="utf-8") as f:
    EN = json.load(f)
with open(i18n_dir / "de.json", "r", encoding="utf-8") as f:
    DE = json.load(f)

# Initialize session state
if "language" not in st.session_state:
    st.session_state.language = "en"

if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None

# Language selector in sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    language = st.selectbox(
        "Language / Sprache",
        options=["en", "de"],
        format_func=lambda x: "🇬🇧 English" if x == "en" else "🇩🇪 Deutsch"
    )
    st.session_state.language = language

# Get translations
t = EN if st.session_state.language == "en" else DE

# Main content
st.title(t["app"]["title"])
st.markdown(t["app"]["subtitle"])

# API configuration
API_BASE_URL = "http://localhost:8000/api"

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    t["tabs"]["upload"],
    t["tabs"]["profile"],
    t["tabs"]["clean"],
    t["tabs"]["train"]
])

# Tab 1: Upload
with tab1:
    st.header(t["upload"]["header"])
    st.markdown(t["upload"]["description"])

    uploaded_file = st.file_uploader(
        t["upload"]["choose_file"],
        type=["csv", "xlsx", "xls", "json", "parquet"]
    )

    if uploaded_file is not None:
        if st.button(t["upload"]["upload_button"]):
            with st.spinner(t["upload"]["uploading"]):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(f"{API_BASE_URL}/upload", files=files)

                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.uploaded_file_id = data["file_id"]

                        st.success(t["upload"]["success"])

                        # Display file info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(t["upload"]["rows"], data["rows"])
                        with col2:
                            st.metric(t["upload"]["columns"], data["columns"])
                        with col3:
                            st.metric(
                                t["upload"]["size"],
                                f"{data['file_size'] / 1024:.2f} KB"
                            )

                        st.info(t["upload"]["next_step"])

                    else:
                        st.error(f"{t['upload']['error']}: {response.text}")

                except Exception as e:
                    st.error(f"{t['upload']['error']}: {str(e)}")

# Tab 2: Profile
with tab2:
    st.header(t["profile"]["header"])

    if st.session_state.uploaded_file_id is None:
        st.warning(t["profile"]["no_file"])
    else:
        if st.button(t["profile"]["generate_button"]):
            with st.spinner(t["profile"]["generating"]):
                try:
                    response = requests.get(
                        f"{API_BASE_URL}/profile/{st.session_state.uploaded_file_id}"
                    )

                    if response.status_code == 200:
                        data = response.json()
                        profile = data["profile"]

                        # Display basic info
                        st.subheader(t["profile"]["basic_info"])
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(t["profile"]["total_rows"], profile["shape"]["rows"])
                        with col2:
                            st.metric(t["profile"]["total_columns"], profile["shape"]["columns"])
                        with col3:
                            st.metric(
                                t["profile"]["memory"],
                                f"{profile['memory_usage']['total_mb']:.2f} MB"
                            )

                        # Missing values
                        st.subheader(t["profile"]["missing_values"])
                        missing_data = []
                        for col, pct in profile["missing"]["percentages"].items():
                            if pct > 0:
                                missing_data.append({
                                    "Column": col,
                                    "Missing (%)": pct,
                                    "Count": profile["missing"]["counts"][col]
                                })

                        if missing_data:
                            st.dataframe(missing_data, width='stretch')
                        else:
                            st.success(t["profile"]["no_missing"])

                        # Data types
                        st.subheader(t["profile"]["data_types"])
                        dtypes_data = [
                            {"Column": col, "Type": dtype}
                            for col, dtype in profile["dtypes"].items()
                        ]
                        st.dataframe(dtypes_data, width='stretch')

                    else:
                        st.error(f"{t['profile']['error']}: {response.text}")

                except Exception as e:
                    st.error(f"{t['profile']['error']}: {str(e)}")

# Tab 3: Clean Data
with tab3:
    st.header(t["clean"]["header"])

    if st.session_state.uploaded_file_id is None:
        st.warning(t["clean"]["no_file"])
    else:
        # Initialize session state
        if 'cleaning_suggestions' not in st.session_state:
            st.session_state.cleaning_suggestions = None
        if 'selected_suggestions' not in st.session_state:
            st.session_state.selected_suggestions = []

        # Analyze button
        if st.button(t["clean"]["analyze_button"]):
            with st.spinner(t["clean"]["analyzing"]):
                try:
                    response = requests.get(
                        f"{API_BASE_URL}/clean/suggestions/{st.session_state.uploaded_file_id}"
                    )

                    if response.status_code == 200:
                        data = response.json()
                        suggestions = data["suggestions"]

                        # Store suggestions in session state
                        st.session_state.cleaning_suggestions = suggestions
                        st.session_state.selected_suggestions = []

                        if len(suggestions) == 0:
                            st.success(t["clean"]["no_issues"])
                    else:
                        st.error(f"{t['clean']['error']}: {response.text}")

                except Exception as e:
                    st.error(f"{t['clean']['error']}: {str(e)}")

        # Display suggestions if they exist (outside button handler so they persist)
        if st.session_state.cleaning_suggestions is not None and len(st.session_state.cleaning_suggestions) > 0:
            suggestions = st.session_state.cleaning_suggestions
            st.info(t["clean"]["issues_found"].replace("{count}", str(len(suggestions))))

            severity_color = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢"
            }

            # Step 1: Show checkboxes to select fixes
            st.markdown("### 📋 Select fixes to apply:")
            for idx, suggestion in enumerate(suggestions):
                col_check, col_info = st.columns([3, 1])
                with col_check:
                    checked = st.checkbox(
                        f"{severity_color.get(suggestion['severity'], '⚪')} **{suggestion['issue']}** (Column: `{suggestion['column']}`)",
                        key=f"apply_{idx}",
                        value=suggestion['issue'] in st.session_state.selected_suggestions
                    )
                    if checked and suggestion['issue'] not in st.session_state.selected_suggestions:
                        st.session_state.selected_suggestions.append(suggestion['issue'])
                    elif not checked and suggestion['issue'] in st.session_state.selected_suggestions:
                        st.session_state.selected_suggestions.remove(suggestion['issue'])

                with col_info:
                    st.caption(f"{suggestion['severity'].upper()} severity")

            # Step 2: Show detailed information in expandable sections
            st.markdown("---")
            st.markdown("### 📖 Detailed explanations:")
            for idx, suggestion in enumerate(suggestions):
                with st.expander(
                    f"{severity_color.get(suggestion['severity'], '⚪')} {suggestion['issue']} - {suggestion['column']}",
                    expanded=False
                ):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"**{t['clean']['suggestion']}:**")
                        st.info(suggestion['suggestion'])

                        st.markdown(f"**{t['clean']['reason']}:**")
                        st.markdown(suggestion['reason'])

                        # Alternatives
                        if suggestion.get('alternatives'):
                            st.markdown(f"**{t['clean']['alternatives']}:**")
                            for alt in suggestion['alternatives']:
                                st.markdown(f"- **{alt['method']}**: {alt['description']}")
                                if 'impact' in alt:
                                    st.caption(f"  ↳ Impact: {alt['impact']}")

                    with col2:
                        st.markdown(f"**{t['clean']['severity']}:** {suggestion['severity'].upper()}")
                        st.markdown(f"**Column:** `{suggestion['column']}`")

                        # Details
                        if suggestion.get('details'):
                            st.markdown(f"**{t['clean']['details']}:**")
                            for key, value in suggestion['details'].items():
                                if isinstance(value, (int, float)):
                                    st.metric(key.replace('_', ' ').title(), value)

            # Apply cleaning button
            if len(st.session_state.get('selected_suggestions', [])) > 0:
                st.markdown("---")
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    if st.button(t["clean"]["apply_button"], type="primary", key="apply_fixes_btn"):
                        with st.spinner(t["clean"]["applying"]):
                            try:
                                clean_response = requests.post(
                                    f"{API_BASE_URL}/clean/execute",
                                    json={
                                        "file_id": st.session_state.uploaded_file_id,
                                        "suggestions_to_apply": st.session_state.selected_suggestions
                                    }
                                )

                                if clean_response.status_code == 200:
                                    clean_data = clean_response.json()
                                    st.success(t["clean"]["success"])
                                    st.metric("Rows after cleaning", clean_data["cleaned_rows"])
                                    st.session_state.uploaded_file_id = clean_data["cleaned_file_id"]
                                    # Don't clear - let user see what was applied
                                    st.info("✨ Fixes applied! Click 'Analyze Data Issues' again to check for remaining issues.")
                                else:
                                    st.error(f"{t['clean']['error']}: {clean_response.text}")
                            except Exception as e:
                                st.error(f"{t['clean']['error']}: {str(e)}")

                with col2:
                    # Pre-fetch script and store in session state
                    if st.button("📥 Prepare Script", key="prepare_script_btn"):
                        try:
                            script_response = requests.post(
                                f"{API_BASE_URL}/clean/export-script",
                                json={
                                    "file_id": st.session_state.uploaded_file_id,
                                    "suggestions_to_apply": st.session_state.selected_suggestions
                                }
                            )

                            if script_response.status_code == 200:
                                st.session_state.cleaning_script = script_response.content
                                st.success("Script ready!")
                            else:
                                st.error("Error preparing script")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

                with col3:
                    # Download button (only shows if script is prepared)
                    if 'cleaning_script' in st.session_state and st.session_state.cleaning_script:
                        st.download_button(
                            label="💾 Save",
                            data=st.session_state.cleaning_script,
                            file_name="clean_data.py",
                            mime="text/x-python",
                            key="download_script_btn"
                        )

# Tab 4: Train Models
with tab4:
    st.header(t["train"]["header"])

    if st.session_state.uploaded_file_id is None:
        st.warning(t["train"]["no_file"])
    else:
        # Get columns for target selection
        if "train_columns" not in st.session_state:
            try:
                response = requests.get(f"{API_BASE_URL}/train/columns/{st.session_state.uploaded_file_id}")
                if response.status_code == 200:
                    st.session_state.train_columns = response.json()["columns"]
            except:
                st.session_state.train_columns = []

        if st.session_state.train_columns:
            # Target column selection
            st.subheader(t["train"]["select_target"])
            st.caption(t["train"]["target_help"])

            target_column = st.selectbox(
                "Target Column",
                options=[col["name"] for col in st.session_state.train_columns],
                label_visibility="collapsed"
            )

            # Show target column info
            if target_column:
                target_info = next((col for col in st.session_state.train_columns if col["name"] == target_column), None)
                if target_info:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Data Type", target_info["dtype"])
                    with col2:
                        st.metric("Unique Values", target_info["unique_values"])
                    with col3:
                        st.metric("Missing", target_info["missing_count"])

                    if target_info["sample_values"]:
                        st.caption(f"Sample values: {', '.join(map(str, target_info['sample_values']))}")

            # Training parameters
            test_size = st.slider(t["train"]["test_size"], 10, 40, 20) / 100

            # Start training button
            if st.button(t["train"]["start_training"], type="primary"):
                with st.spinner(t["train"]["training"]):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/train/start",
                            json={
                                "file_id": st.session_state.uploaded_file_id,
                                "target_column": target_column,
                                "test_size": test_size
                            }
                        )

                        if response.status_code == 200:
                            data = response.json()
                            summary = data["summary"]

                            st.success(f"✅ {t['train']['models_trained']}: {summary['n_models_trained']}")

                            # Problem type
                            st.info(f"🎯 {t['train']['problem_detected']}: **{summary['problem_type'].title()}**")

                            # Best model
                            st.subheader(t["train"]["best_model"])
                            best_model = summary["best_model"]
                            best_metrics = summary["best_model_score"]

                            if best_model and best_metrics:
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown(f"### {best_model}")
                                    if summary["problem_type"] == "classification":
                                        st.metric(t["train"]["accuracy"], f"{best_metrics['accuracy']*100:.2f}%")
                                    else:
                                        st.metric(t["train"]["r2_score"], f"{best_metrics['r2_score']:.4f}")

                                with col2:
                                    # Get CV score for best model
                                    best_result = summary["all_results"][best_model]
                                    cv_mean = best_result["cv_score_mean"]
                                    cv_std = best_result["cv_score_std"]
                                    st.metric(
                                        t["train"]["cv_score"],
                                        f"{cv_mean:.4f}",
                                        delta=f"±{cv_std:.4f}"
                                    )

                            # Model comparison
                            st.subheader(t["train"]["model_comparison"])

                            comparison_data = []
                            for model_name, result in summary["all_results"].items():
                                if "metrics" in result:
                                    metrics = result["metrics"]
                                    if summary["problem_type"] == "classification":
                                        comparison_data.append({
                                            "Model": model_name,
                                            "Accuracy": f"{metrics['accuracy']*100:.2f}%",
                                            "Precision": f"{metrics['precision']*100:.2f}%",
                                            "Recall": f"{metrics['recall']*100:.2f}%",
                                            "F1-Score": f"{metrics['f1_score']*100:.2f}%",
                                            "CV Score": f"{result['cv_score_mean']:.4f}"
                                        })
                                    else:
                                        comparison_data.append({
                                            "Model": model_name,
                                            "R² Score": f"{metrics['r2_score']:.4f}",
                                            "RMSE": f"{metrics['rmse']:.4f}",
                                            "MAE": f"{metrics['mae']:.4f}",
                                            "CV Score": f"{result['cv_score_mean']:.4f}"
                                        })

                            st.dataframe(comparison_data, width='stretch')

                            # Feature importance
                            if best_result.get("feature_importance"):
                                st.subheader(t["train"]["feature_importance"])

                                features = best_result["feature_importance"]
                                importance_data = {
                                    "Feature": [f["feature"] for f in features],
                                    "Importance": [f["importance"] for f in features]
                                }

                                import plotly.express as px
                                fig = px.bar(
                                    x=importance_data["Importance"],
                                    y=importance_data["Feature"],
                                    orientation='h',
                                    labels={"x": "Importance", "y": "Feature"}
                                )
                                fig.update_layout(
                                    yaxis={'categoryorder':'total ascending'},
                                    height=400
                                )
                                st.plotly_chart(fig, width='stretch')

                            # MLflow tracking info
                            st.markdown("---")
                            st.subheader("🔬 Experiment Tracking")
                            st.info("""
                            **All experiments are tracked with MLflow!**

                            View detailed experiment history, compare models, and manage artifacts:

                            1. Run: `python run_mlflow.py` in your terminal
                            2. Open: http://localhost:5000

                            MLflow tracks:
                            - 📊 All model parameters and metrics
                            - 📈 Cross-validation scores
                            - 🔍 Feature importance artifacts
                            - 💾 Trained model artifacts for deployment
                            """)

                        else:
                            st.error(f"{t['train']['error']}: {response.text}")

                    except Exception as e:
                        st.error(f"{t['train']['error']}: {str(e)}")

# Sidebar info
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown(t["sidebar"]["about"])

    st.markdown("### 🚀 Features")
    st.markdown(t["sidebar"]["features"])

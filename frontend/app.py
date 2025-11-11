"""
AutoML Pipeline Builder - Main Streamlit Application
"""
import streamlit as st
import requests
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="AutoML Pipeline Builder",
    page_icon="🤖",
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
                            st.dataframe(missing_data, use_container_width=True)
                        else:
                            st.success(t["profile"]["no_missing"])

                        # Data types
                        st.subheader(t["profile"]["data_types"])
                        dtypes_data = [
                            {"Column": col, "Type": dtype}
                            for col, dtype in profile["dtypes"].items()
                        ]
                        st.dataframe(dtypes_data, use_container_width=True)

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
        if st.button(t["clean"]["analyze_button"]):
            with st.spinner(t["clean"]["analyzing"]):
                try:
                    response = requests.get(
                        f"{API_BASE_URL}/clean/suggestions/{st.session_state.uploaded_file_id}"
                    )

                    if response.status_code == 200:
                        data = response.json()
                        suggestions = data["suggestions"]

                        if len(suggestions) == 0:
                            st.success(t["clean"]["no_issues"])
                        else:
                            st.info(t["clean"]["issues_found"].replace("{count}", str(len(suggestions))))

                            # Store suggestions in session state
                            st.session_state.cleaning_suggestions = suggestions
                            st.session_state.selected_suggestions = []

                            # Display suggestions as cards
                            for idx, suggestion in enumerate(suggestions):
                                severity_color = {
                                    "high": "🔴",
                                    "medium": "🟡",
                                    "low": "🟢"
                                }

                                with st.expander(
                                    f"{severity_color.get(suggestion['severity'], '⚪')} {suggestion['issue']}",
                                    expanded=(suggestion['severity'] == "high")
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

                                    # Checkbox to select this suggestion
                                    if st.checkbox(f"Apply this fix", key=f"apply_{idx}"):
                                        if suggestion['issue'] not in st.session_state.selected_suggestions:
                                            st.session_state.selected_suggestions.append(suggestion['issue'])
                                    else:
                                        if suggestion['issue'] in st.session_state.selected_suggestions:
                                            st.session_state.selected_suggestions.remove(suggestion['issue'])

                            # Apply cleaning button
                            if len(st.session_state.get('selected_suggestions', [])) > 0:
                                st.markdown("---")
                                col1, col2 = st.columns([3, 1])

                                with col1:
                                    if st.button(t["clean"]["apply_button"], type="primary"):
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
                                                else:
                                                    st.error(f"{t['clean']['error']}: {clean_response.text}")
                                            except Exception as e:
                                                st.error(f"{t['clean']['error']}: {str(e)}")

                                with col2:
                                    # Export script button
                                    if st.button(t["clean"]["export_script"]):
                                        try:
                                            script_response = requests.post(
                                                f"{API_BASE_URL}/clean/export-script",
                                                json={
                                                    "file_id": st.session_state.uploaded_file_id,
                                                    "suggestions_to_apply": st.session_state.selected_suggestions
                                                }
                                            )

                                            if script_response.status_code == 200:
                                                st.download_button(
                                                    label="💾 Save Script",
                                                    data=script_response.content,
                                                    file_name="clean_data.py",
                                                    mime="text/x-python"
                                                )
                                        except Exception as e:
                                            st.error(f"Error: {str(e)}")

                    else:
                        st.error(f"{t['clean']['error']}: {response.text}")

                except Exception as e:
                    st.error(f"{t['clean']['error']}: {str(e)}")

# Tab 4: Train (placeholder)
with tab4:
    st.header(t["train"]["header"])
    st.info(t["train"]["coming_soon"])

# Sidebar info
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown(t["sidebar"]["about"])

    st.markdown("### 🚀 Features")
    st.markdown(t["sidebar"]["features"])

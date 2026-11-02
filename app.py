"""Student lookup dashboard with unsupervised K-means segmentation."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
DEFAULT_FEATURES = ["G1", "G2", "studytime", "absences", "failures"]
DEMO_PATH = Path(__file__).with_name("sample_students.csv")

st.set_page_config(page_title="Student Segmentation", page_icon="🎓", layout="wide")
st.title("🎓 Student Academic Profile & Segmentation")
st.write(
    "Search for a student to view her academic profile and the K-means segment assigned "
    "from the selected academic features. Segments are exploratory patterns, not diagnoses or predictions."
)
st.info(
    "Responsible use: treat segments as a starting point for supportive conversations. "
    "Do not rank students or make high-stakes decisions from this model. Keep identifiable student data private."
)


@st.cache_data(show_spinner="Loading the UCI Mathematics dataset...")
def load_uci_data() -> pd.DataFrame:
    """Download the public UCI Mathematics dataset (it contains no student names)."""
    with urlopen(DATA_URL, timeout=30) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        with archive.open("student-mat.csv") as csv_file:
            return pd.read_csv(csv_file, sep=";")


@st.cache_data(show_spinner=False)
def load_demo_data() -> pd.DataFrame:
    """Load fictional sample students included for demonstrating name lookup."""
    return pd.read_csv(DEMO_PATH)


@st.cache_data(show_spinner=False)
def read_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read a comma- or semicolon-separated student CSV."""
    return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")


with st.sidebar:
    st.header("Choose student data")
    source = st.radio("Dataset", ["Fictional demo students", "Upload my CSV", "UCI Mathematics (no names)"])

try:
    if source == "Fictional demo students":
        data = load_demo_data()
        st.caption("Demo records and names are fictional. Use Upload my CSV for your own class data.")
    elif source == "UCI Mathematics (no names)":
        data = load_uci_data()
        data.insert(0, "Student name", [f"UCI record {i + 1:03d}" for i in range(len(data))])
        st.caption("UCI provides no names; temporary record labels are used only to demonstrate lookup.")
    else:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV with one student per row and a name column, such as `student_name`.")
            st.stop()
        data = read_uploaded_csv(uploaded.getvalue())
        st.caption("The uploaded data is processed in this app session. Do not upload identifiable data to a public deployment.")
except Exception as error:
    st.error(f"Could not load the selected data. Check the CSV format or internet connection. Details: {error}")
    st.stop()

name_columns = [column for column in data.columns if column.strip().lower() in {"student_name", "name", "student name"}]
if name_columns:
    name_column = name_columns[0]
else:
    st.error("No student name column found. Add a column named `student_name` or `name` to your CSV.")
    st.stop()

numeric_columns = data.select_dtypes(include="number").columns.tolist()
if len(numeric_columns) < 2:
    st.error("Your dataset needs at least two numeric academic or study-related columns.")
    st.stop()

with st.sidebar:
    st.header("Segmentation settings")
    defaults = [feature for feature in DEFAULT_FEATURES if feature in numeric_columns]
    chosen_features = st.multiselect(
        "Features used to form segments",
        options=numeric_columns,
        default=defaults if len(defaults) >= 2 else numeric_columns[: min(5, len(numeric_columns))],
        help="Names and IDs are excluded automatically. Select academic or study-related numeric features.",
    )
    cluster_count = st.slider("Number of segments", min_value=2, max_value=8, value=3)

if len(chosen_features) < 2:
    st.warning("Select at least two numeric features in the sidebar.")
    st.stop()

model_data = data[chosen_features].copy()
missing_rows = model_data.isna().any(axis=1)
if missing_rows.any():
    st.warning(f"{int(missing_rows.sum())} rows with missing selected values will not receive a segment.")

valid = ~missing_rows
model_data = model_data.loc[valid]
if len(model_data) <= cluster_count:
    st.error("There are too few complete student records for this number of segments.")
    st.stop()

scaler = StandardScaler()
scaled = scaler.fit_transform(model_data)
model = KMeans(n_clusters=cluster_count, random_state=42, n_init=20)
labels = model.fit_predict(scaled)

results = data.loc[valid].copy()
results["ML segment"] = labels
score = silhouette_score(scaled, labels)

st.subheader("Find a student")
search = st.text_input("Type a student name", placeholder="Start typing a name")
name_options = results[name_column].fillna("(Name missing)").astype(str)
if search.strip():
    matches = name_options[name_options.str.contains(search.strip(), case=False, regex=False)].drop_duplicates().tolist()
else:
    matches = name_options.drop_duplicates().tolist()

if not matches:
    st.warning("No matching student. Check the spelling or try fewer letters.")
    st.stop()
selected_name = st.selectbox("Select the student", matches)
matched_index = name_options[name_options == selected_name].index[0]
student = results.loc[matched_index]
student_segment = int(student["ML segment"])

st.subheader(f"Academic profile: {selected_name}")
metric_columns = st.columns(min(len(chosen_features), 5))
for column, feature in zip(metric_columns, chosen_features[:5]):
    column.metric(feature.replace("_", " ").title(), f"{student[feature]:g}")
st.metric("Assigned ML segment", f"Segment {student_segment + 1}")
st.caption("Segment numbers are arbitrary IDs. Compare the profile averages below before interpreting the result.")

profile_source = results.copy()
profile = profile_source.groupby("ML segment")[chosen_features].mean().round(2)
counts = profile_source["ML segment"].value_counts().sort_index().rename("Students")
profile.insert(0, "Students", counts)

left, right = st.columns([1, 2])
with left:
    st.markdown("**How many records are in each segment?**")
    st.dataframe(counts.rename(index=lambda value: f"Segment {value + 1}").to_frame(), use_container_width=True)
with right:
    st.markdown("**Average profile by segment**")
    display_profile = profile.copy()
    display_profile.index = [f"Segment {value + 1}" for value in display_profile.index]
    st.dataframe(display_profile, use_container_width=True)

st.subheader("Explore all student profiles")
show_table = st.checkbox("Show student names and assigned segments", value=False)
if show_table:
    st.dataframe(results[[name_column] + chosen_features + ["ML segment"]], use_container_width=True)

st.subheader("Segment visualization")
if len(chosen_features) == 2:
    coordinates = scaled
    x_axis, y_axis = chosen_features
else:
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(scaled)
    x_axis, y_axis = "Principal component 1", "Principal component 2"
    st.caption(f"The chart's two dimensions represent {pca.explained_variance_ratio_.sum():.1%} of variation in the selected features.")

fig, ax = plt.subplots(figsize=(8, 5))
points = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap="viridis", alpha=0.75)
selected_position = model_data.index.get_loc(matched_index)
ax.scatter(coordinates[selected_position, 0], coordinates[selected_position, 1], s=180, facecolors="none", edgecolors="red", linewidths=2, label="Selected student")
ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)
ax.set_title("Students grouped by selected features")
ax.legend()
fig.colorbar(points, ax=ax, label="Segment ID")
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.metric("Silhouette score", f"{score:.3f}")
st.caption("A higher silhouette score can suggest better-separated groups, but it does not mean the groups are educationally meaningful by themselves.")

st.download_button(
    "Download segmented records as CSV",
    data=results[[name_column] + chosen_features + ["ML segment"]].to_csv(index=False).encode("utf-8"),
    file_name="student_segments.csv",
    mime="text/csv",
)

st.divider()
st.caption("K-means groups records by similarity; it does not explain causes or predict a student's future. Validate findings with educators and the students' real context.")

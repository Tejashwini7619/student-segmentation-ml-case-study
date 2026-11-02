# Student Academic Profile & Segmentation

A student segmentation project with two ways to explore it: a browser-based demo for GitHub Pages and a Python Streamlit app for running locally in VS Code.

## Live demo

[Open the Student Segmentation Dashboard](https://tejashwini7619.github.io/student-segmentation-ml-case-study/)

The live page uses the fictional sample students in `sample_students.csv`. To publish it, enable GitHub Pages for this repository: open **Settings → Pages**, select **Deploy from a branch**, choose branch **main** and folder **/(root)**, then save. GitHub Pages serves the `index.html` browser demo; it does not run the Python Streamlit app.

## Run the browser demo locally

Open `index.html` in VS Code and use the **Live Server** extension, or enable GitHub Pages as described above. The browser demo supports name search, feature selection, K-means segments, profiles, and CSV upload.

## Run in VS Code

1. Install Python 3.10 or later and the Python extension in VS Code.
2. Open this project folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Run these commands:

   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

The Python app opens in a browser. The fictional demo data is included locally. The UCI sample requires internet access.

## Use name lookup with your own data

Choose **Upload my CSV** and upload a CSV with one student per row. Include a name field named `student_name`, `student name`, or `name`, and at least two numeric academic/study columns such as `math_grade`, `study_hours`, or `absences`.

Example structure:

```csv
student_name,math_grade,study_hours,absences
Aarav,82,6,2
Mira,74,4,5
```

The app uses selected numeric columns for clustering and excludes the name field. Type a name in the search box to see that student's feature values, assigned segment, and the average feature profile for every segment. When names repeat, the app currently opens the first matching record; use unique names or add a unique student code if needed.

**Privacy:** The included demo names and records are fictional. Do not put real student names or grades in a public GitHub repository. Use private local data for demos, or anonymize it before publishing. Do not upload identifiable student data to a public/shared app deployment.

## Data sources

- **Fictional demo students:** small made-up dataset included as `sample_students.csv` for safe demonstration of name lookup.
- **UCI Mathematics sample:** the UCI Student Performance dataset has no student names; the app displays temporary record labels to demonstrate profile lookup.
- **Upload my CSV:** use your own authorized data with a student-name column.

The UCI dataset contains historical data from two Portuguese secondary schools. See the [UCI dataset page](https://archive.ics.uci.edu/dataset/320/student%2Bperformance).

## Method

Both versions standardize selected numeric features and apply K-means. They show segment sizes and average feature values; the dashboards also provide a silhouette score and visualization. Cluster IDs are arbitrary; they are not rankings, diagnoses, or predictions. Results depend on data quality, feature selection, and the chosen number of clusters.

## References

- Cortez, P. & Silva, A. (2008). *Using data mining to predict secondary school student performance*. Proceedings of the 5th Future Business Technology Conference (FUBUTEC).
- UCI Machine Learning Repository. [Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student%2Bperformance).
- scikit-learn. [Clustering documentation](https://scikit-learn.org/stable/modules/clustering.html).

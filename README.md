# Student Academic Profile & Segmentation

A Streamlit app for searching a student by name, viewing her selected academic profile, and seeing the exploratory K-means segment formed from academic/study features.

## Run in VS Code

1. Install Python 3.10 or later and the Python extension in VS Code.
2. Open this project folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Run these commands:

   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

The app opens in a browser. The fictional demo data is included locally. The UCI sample requires internet access.

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

The app standardizes the chosen numeric features and fits K-means. It shows segment sizes, average feature values, a silhouette score, and a two-dimensional visualization. Cluster IDs are arbitrary; they are not rankings, diagnoses, or predictions. Results depend on data quality, feature selection, and the chosen number of clusters.

## References

- Cortez, P. & Silva, A. (2008). *Using data mining to predict secondary school student performance*. Proceedings of the 5th Future Business Technology Conference (FUBUTEC).
- UCI Machine Learning Repository. [Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student%2Bperformance).
- scikit-learn. [Clustering documentation](https://scikit-learn.org/stable/modules/clustering.html).

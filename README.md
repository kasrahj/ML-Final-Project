# Voice Feature Analysis and Classification

An academic machine-learning project built around speech recordings. The work turns raw audio into compact signal descriptors, then studies both supervised prediction and unsupervised structure in the resulting feature space. The notebooks include gender classification, a closed-set student identity-recognition experiment, and clustering analyses.

## Technical pipeline

### 1. Audio preparation and feature engineering

[`preprocess.ipynb`](preprocess.ipynb) loads the recording metadata, resamples audio to 44.1 kHz, applies denoising and a band-pass filter, and divides recordings into 100 ms windows with a 50% hop. It selects candidate windows using an energy threshold, then aggregates features from the selected signal. The feature vector includes:

- Zero-crossing rate and RMS energy
- 23 mean log-mel bands
- 13 mean MFCC coefficients
- Mean spectral centroid and bandwidth
- Seven spectral-contrast bands

The outputs are stored in CSV tables for downstream experiments. `correct_format.ipynb` contains an additional preprocessing/formatting path.

### 2. Supervised classification

[`classification.ipynb`](classification.ipynb) prepares a balanced gender-classification subset by sampling 190 examples per class, standardizes the log-mel features, and applies PCA retaining 95% of their variance. It compares an MLP, RBF-kernel SVM, logistic regression, and 5-nearest neighbors, using classification reports, confusion matrices, and ROC/AUC plots to examine results.

The same notebook also explores closed-set identity recognition: a selected group of students is encoded as classes, log-mel features are reduced with PCA, and MLP, linear SVM, random forest, and KNN models are evaluated. This is an experiment over known identities in the dataset, not open-set speaker verification.

### 3. Unsupervised analysis

[`clustering.ipynb`](clustering.ipynb) and [`clustering_shahab.ipynb`](clustering_shahab.ipynb) compare cluster structure in the voice features. The main clustering notebook standardizes features, uses PCA for visualization, and examines K-Means, agglomerative clustering, and DBSCAN. It sweeps candidate cluster counts with silhouette scores and compares cluster assignments with labels using metrics such as adjusted Rand score and homogeneity. The labels are useful for analysis, while clustering itself does not use them to assign clusters.

## Data and artifacts

The repository includes `metadata.csv`, `extracted_features.csv`, `final_dataset.csv`, notebooks, reports, and a sample audio file. The full source corpus is not included. The preprocessing notebook expects audio under `../dataset/HW1_M` and creates intermediate directories for resampled, cleaned, and windowed audio; update those paths for your dataset. The feature tables allow the later classification and clustering work to be inspected without regenerating every audio feature.

The final report is available as [`final_report.pdf`](final_report.pdf), with an editable copy at [`final_report.docx`](final_report.docx).

## Run the notebooks

Use Python with Jupyter and the scientific/audio packages used by the notebooks:

```bash
python -m pip install jupyter numpy pandas matplotlib seaborn scikit-learn librosa soundfile noisereduce scipy tqdm
jupyter lab
```

Open the notebooks in workflow order and run them from this project directory unless you have updated their relative data paths. Results can vary with data selection, preprocessing, train/test split, random seed, and library versions. These notebooks are course experiments rather than a production speech-recognition or authentication system.

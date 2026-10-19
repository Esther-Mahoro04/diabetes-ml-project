# Diabetes Status Classification: Traditional ML vs Deep Learning

Classifying people as **no diabetes, prediabetes or diabetes** from CDC survey answers, and comparing Scikit-Learn models with Keras neural networks.

## Links
- **Report:** [add if hosted]
- **Google Drive** (raw dataset zip, saved models, figures, W&B logs): [DRIVE LINK]
- **Video presentation:** [VIDEO LINK]

## Project summary
- **Question:** can survey answers about health and lifestyle separate the three diabetes statuses?
- **Data:** CDC Diabetes Health Indicators dataset, built from the 2015 BRFSS survey (253,680 rows, 21 features, 3 classes). Source: Kaggle / UCI.
- **Models:** Logistic Regression, Random Forest, Histogram Gradient Boosting (Scikit-Learn); Keras Sequential networks with a `tf.data` pipeline.
- **Experiments:** 13 main experiments covering features, regularisation, class weights, undersampling, SMOTENC, threshold tuning, a cost-based decision rule, and a two-class version.

## Key findings
- Every model ranked people at almost the same level (macro ROC-AUC 0.750 to 0.774).
- Prediabetes was very hard to separate (PR-AUC about 0.03, one-vs-rest AUC 0.687).
- A cost-based decision rule cut the average cost by about 59% against the baseline, with the same model. The costs are assumptions, not medical values.
- On the test set: Logistic Regression + cost rule had the lowest cost (0.548), and the Keras network had the best macro-F1 (about 0.43).

## Repository contents
| File | Description |
|---|---|
| `diabetes_ml_vs_dl.ipynb` | Full notebook: cleaning, split, experiments, evaluation |
| `requirements.txt` | Python dependencies |
| `figures/` | Learning curves, confusion matrices, ROC and precision-recall curves |

The **raw dataset is not in this repository.** It is in the Google Drive folder as `diabetes_raw_dataset.zip`.

## How to run
1. Open the notebook in Google Colab.
2. Download `diabetes_raw_dataset.zip` from the Drive folder above and put it in your own Drive, in a folder named `diabetes_ml_project`.
3. Install the dependencies: `pip install -r requirements.txt`
4. Run the cells in order. The first cell mounts Drive and reads the CSV from the zip.

Notes:
- A fixed random seed (42) is used throughout.
- The test set is used only once, for the three final models.
- Keras results can differ slightly between runs.
- Weights & Biases runs in offline mode. Logs from the final run (E1, E6, E11, E12) are in the Drive folder under `wandb/`.

## Saved models (in the Drive folder)
- `logreg_best.joblib`: Logistic Regression (traditional ML)
- `keras_best.keras`: Keras network, E12 (deep learning)
- `preprocessing.joblib`: scaler, BMI cap and scaled column list needed to use the models

## Limitations
Self-reported survey data from one year and one country; a small prediabetes class (about 700 cases per evaluation set); assumed costs; a single split and seed.

## Author
[YOUR NAME], [MODULE / COURSE], [DATE]

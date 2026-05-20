# AppliedAI
Applied AI, ML &amp; Deep Learning. School projects.

### Group project 1 - Airbnb Stockholm – Price Prediction
```
airbnb_prediction
A regression analysis project predicting nightly Airbnb prices in Stockholm using OLS regression.
This project explores what drives Airbnb pricing in Stockholm by cleaning, analyzing, and modeling listing data from Inside Airbnb.

Dataset: ~4,955 Stockholm listings
Goal: Predict price per night and identify key pricing factors
Methods: Data cleaning, EDA, OLS regression

Parsing and cleaning of raw price strings and categorical variables
Exploratory analysis including pairplots and median price by neighbourhood
One-hot encoding of room type and location variables
OLS regression model with feature importance analysis

pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
```

### Group project 2 - Data reduction - Digits dataset
```
digits_data_reduction

PCA and UMAP applied to scikit-learn's Digits dataset (1797 samples, 64 features, 10 classes), built from mathematical foundations and exposed through an interactive Streamlit app.
Goal
Explore how high-dimensional image data behaves under linear (PCA) and non-linear (UMAP) dimensionality reduction, and understand when each method is appropriate.
Pipeline

Load and inspect the Digits dataset
Standardize features (StandardScaler / MinMaxScaler / none — selectable)
Compute the covariance matrix manually
Extract eigenvalues and eigenvectors with numpy.linalg.eigh
Verify against scikit-learn's PCA
Scree plot with Kaiser criterion (eigenvalue > 1)
2D and 3D PCA scatter plots
UMAP on the top 20 PCA components

Key findings

Three eigenvalues are zero — three corner pixels are constant across all samples, producing NaN gaps in the covariance diagonal.
Poor 2D/3D separation does not imply poor classification. A Random Forest trained on 20 PCA components reaches ~96% accuracy. The classifier operates in the full feature space, not in the projection.
UMAP works much better than PCA for visualization of this dataset, but it functions primarily as an exploration tool rather than a preprocessing step for training.
Scaler choice matters. StandardScaler removes natural variance advantages between features; MinMaxScaler preserves relative pixel distributions.

To run the streamlit app, please run
streamlit run app.py

Stack
Python, NumPy, scikit-learn, UMAP, Plotly, Matplotlib, Seaborn, Streamlit.
```

### Group project 3 — Food Delivery: Unsupervised Learning
```
food_delivery_unsupervised_learning

Customer and restaurant segmentation on a food delivery order history dataset (21,321 orders, 6 restaurants) using unsupervised methods.

Goal

Discover structure in order behavior without labels — segment customers, profile restaurants, and detect anomalies that may indicate operational issues or fraud.

Dataset

Source: [Food Delivery Order History](https://www.kaggle.com/datasets/sujalsuthar/food-delivery-order-history-data) on Kaggle.

Contains restaurant metadata, order timestamps, delivery time and distance, pricing breakdown (subtotal, packaging, discounts, total), customer ratings, cancellations, rider wait time, and preparation duration.

Pipeline

- Load and clean the order data
- Engineer per-customer features (RFM-style: frequency, monetary value, average rating, average items, discount usage)
- Engineer per-restaurant features (average delivery time, average rating, average order value, cancellation rate)
- Scale features
- Apply clustering and anomaly detection
- Interpret and label the resulting segments

Methods

- **KMeans** — customer segmentation based on order behavior
- **DBSCAN** — restaurant performance clustering
- **Isolation Forest** — anomaly detection (unusually long deliveries, suspicious discount patterns, abnormal cancellation rates)
- **Time-based clustering** — peak ordering patterns extracted from timestamps

Findings

Customer profiles emerged along axes of frequency, average spend, and discount usage — including loyal high-value customers, discount hunters, and at-risk users with high cancellation rates. Restaurants separated cleanly into top performers (fast delivery + high ratings), high-volume mid-tier, and problem restaurants (long preparation time + low ratings).

Stack
Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Plotly.
```

### Group project 4 — Telco Customer Churn: Supervised Learning
```
telco_supervised_learning

Predicting customer churn on the Telco Customer Churn dataset using logistic regression and KNN, applied with the exact methodology from the course material.

Goal

Answer the question: *which customers will leave us, and what can we do about it?*

Build interpretable classification models that not only predict churn but also identify the drivers behind it.

Background

The project's first iteration attempted to predict tourist flows from weather data, but produced no signal (low R², non-significant coefficients, structured residuals). The lesson — *models cannot conjure signal that isn't in the data* — led to the choice of Telco, a dataset known to contain a clear churn signal so we could focus on modeling technique rather than data discovery.

Pipeline

- Load and inspect the Telco dataset (clean, no nulls, correct typing)
- Train/test split with `random_state=42`
- Standardize features (`fit_transform` on train, `transform` on test)
- Build logistic regression in `statsmodels` (`sm.add_constant`, `Logit`) for interpretable p-values
- Evaluate with confusion matrix, classification report, ROC-AUC, AUPRC
- Optimize the decision threshold using Youden's J
- Iterate by removing non-significant variables
- Check linearity in log-odds via residual plots
- Build a KNN classifier on the same scaled data, with `k` chosen via cross-validation
- Compare KNN against logistic regression

Why AUPRC matters here

Telco churn is moderately imbalanced (~26% churn). AUPRC is reported alongside ROC-AUC because it is more informative than accuracy when the positive class is the minority.

Stack
Python, pandas, scikit-learn, statsmodels, Matplotlib, Seaborn.
```

### Group project 5 — Steam: Predicting Game Reception with Tree-Based Models
```
steam_devision_trees

Can Steam metadata alone predict whether a game will be positively received by users? We tackle this with decision trees, Random Forest, and XGBoost on a dataset of 90,000+ Steam titles.

Goal

Predict positive user reception from metadata that is **known at launch** — genre, tags, categories, price, achievements, platform support, etc. Review-derived features are deliberately excluded as input, since a game has no reviews before release.

Target: binary classification (successful / not successful).

Pipeline

- Load and inspect the Steam dataset (90,000+ games)
- Filter to games with at least 50 reviews (statistical reliability)
- Feature engineering on tags, genres, categories, platforms, and pricing
- Train/test split, cross-validation
- Grid Search for hyperparameter tuning
- Train three tree-based models: Decision Tree → Random Forest → XGBoost
- Compare with accuracy, precision/recall, and ROC-AUC
- Inspect feature importance via MDI and permutation importance (with variance, not just means)

Why tree-based models

Random Forest and XGBoost are ensembles built on decision trees — a natural progression of the same concept where many trees together generalize better than a single tree. They also handle the dataset's heavy categorical/dummy-encoded structure more gracefully than linear methods.

Findings

- **Random Forest and XGBoost perform similarly**, with XGBoost edging ahead by ~0.01 accuracy.
- **`In-App Purchases` ranks high in MDI importance** — interesting and worth scrutiny, since it's a platform feature rather than a direct quality signal.
- **The 50-review threshold drops ~69% of the dataset**, which is a deliberate trade between coverage and label reliability.

Stack
Python, pandas, scikit-learn, XGBoost, Matplotlib, Seaborn.
```

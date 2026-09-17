# Consumer Shopping Behavior & Customer Segmentation

## Project Overview

This repository presents an end-to-end unsupervised machine learning pipeline designed to segment consumers based on their shopping behavior, purchasing habits, and survey responses. By identifying distinct behavioral patterns, businesses can tailor marketing strategies, optimize promotional spend, personalize product recommendations, and enhance customer retention.

The solution encompasses the entire data science lifecycle: data hygiene and validation, mixed-type feature engineering, dimensionality reduction via Principal Component Analysis (PCA), optimal cluster selection through geometric and silhouette diagnostics, K-Means clustering, persona profiling, model serialization, and an inference pipeline for scoring new customer responses.

---

## Repository Structure

```
.
├── Dataset/
│   └── consumer_shopping_behavior_survey.csv   # Raw survey response data (500 records, 30 features)
├── Model/
│   ├── customer_clustering_pipeline.ipynb      # Complete 12-stage end-to-end Jupyter notebook
│   └── artifacts/                              # Exported production artifacts and metadata
│       ├── preprocessor.joblib                 # Serialized ColumnTransformer preprocessing pipeline
│       ├── pca_model.joblib                    # Serialized PCA transformer (91.6% variance retained)
│       ├── kmeans_model.joblib                 # Trained K-Means clustering model
│       ├── cluster_profiles.csv                # Mean behavioral score matrix per segment
│       ├── df_with_clusters.csv                # Full dataset annotated with cluster assignments
│       └── model_metadata.json                 # Comprehensive training hyperparameters and metrics
├── requirements.txt                            # Environment dependencies
└── README.md                                   # Project documentation
```

---

## Dataset Description

The dataset consists of 500 consumer survey responses across 30 attributes capturing demographic, psychographic, and purchasing behavior:

- **Identifiers & Timestamps**: `Response_ID`, `Timestamp` (excluded from modeling).
- **Demographics**: `Age`, `Gender`, `Country_Region`, `Occupation_Status`, `Monthly_Income_Range`.
- **Channel & Device Preferences**: `Shop_Most_Where`, `Preferred_Platform`, `Primary_Device`, `Best_Shopping_Mode`, `Reason_Prefer_Online`, `Reason_Prefer_InStore`.
- **Purchasing Dynamics**: `Online_Shopping_Frequency`, `Avg_Monthly_Spend_NonEssentials`, `Top_Spending_Category`, `Preferred_Payment_Method`, `Uses_BNPL_Installments`, `Makes_Shopping_List`, `Return_Frequency`.
- **Psychographic & Marketing Scores (1-10 Likert scales)**:
  - `Review_Influence_Score`
  - `Social_Ads_Influence_Score`
  - `Price_Comparison_Frequency`
  - `Discount_Importance_Score`
  - `Likely_To_Buy_After_Ad_Score`
  - `Impulse_Purchase_Frequency`
  - `Online_Shopping_Satisfaction_Score`
  - `Trust_In_Online_Reviews_Score`
  - `Regret_After_Purchase_Frequency`
  - `Follows_Brands_On_Social`

---

## Machine Learning Pipeline Architecture

The implementation in `Model/customer_clustering_pipeline.ipynb` follows a modular 12-step architecture:

### 1. Environment & Library Initialization
Imports foundational data processing, statistical visualization, machine learning, and serialization libraries (`pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `joblib`).

### 2. Data Ingestion & Schema Inspection
Loads the dataset, conducts structural checks (`shape`, `dtypes`, memory footprint), and detects anomalies including corrupted strings in numerical features (e.g., text artifacts in the `Age` column).

### 3. Exploratory Data Analysis (EDA)
Generates descriptive statistics, visualizes distributions of key rating scores, examines categorical frequency balances, and evaluates feature correlation matrices to identify multicollinearity.

### 4. Robust Feature Engineering & Preprocessing
Constructs a unified, leakage-free `ColumnTransformer` pipeline supporting three distinct data types:
- **Numeric Features (10 attributes)**: Coerced to numeric types with error handling for dirty values, imputed using the median strategy, and standardized using `StandardScaler`.
- **Ordinal Features (4 attributes)**: Imputed with most frequent values and mapped via explicit, rank-ordered categorical lists using `OrdinalEncoder`.
- **Nominal Features (14 attributes)**: Imputed with most frequent categories and encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.

### 5. Dimensionality Reduction (PCA)
Applies Principal Component Analysis to resolve the curse of dimensionality arising from one-hot encoded variables. Configured with `n_components=0.90` (retaining 91.6% of total cumulative variance across 23 orthogonal components), ensuring robust distance metrics in downstream clustering.

### 6. Optimal Cluster Determination
Evaluates candidate cluster counts ($k \in [2, 10]$) using multiple diagnostics:
- **Elbow Method (Inertia)**: Identifies the inflection point where additional clusters yield diminishing returns in intra-cluster variance reduction.
- **Silhouette Coefficient Analysis**: Measures cluster cohesion versus separation across tested values of $k$.

### 7. Hierarchical Clustering Verification
Computes a Ward-linkage agglomerative hierarchy and plots a dendrogram to validate cluster separation and cross-verify the natural partitions identified in the partition-based analysis.

### 8. K-Means Model Training
Trains the final K-Means algorithm using:
- Cluster count: $k = 2$
- Centroid initialization: `k-means++`
- Re-initializations: `n_init=50`
- Maximum iterations: `max_iter=500`
- Seed reproducibility: `random_state=42`

### 9. Cluster Evaluation & Diagnostic Visualizations
Evaluates clustering performance using established cluster validation metrics:
- 2D PCA projection scatter plot with cluster centroids.
- Silhouette plot per sample showing individual silhouette widths across clusters.
- Distribution plots comparing core behavioral metrics across segments.

### 10. Customer Persona Profiling
Computes feature centroid profiles across raw demographic and behavioral dimensions to interpret segment characteristics and derive actionable commercial strategies.

### 11. Artifact Serialization & Archival
Exports all transformers, models, dataset annotations, and JSON execution metadata into the `Model/artifacts/` directory for production deployment.

### 12. Production Inference Demonstration
Demonstrates end-to-end scoring of new, unseen customer survey records through the serialized preprocessing, PCA, and K-Means artifacts.

---

## Model Evaluation Metrics

| Metric | Result | Description |
| :--- | :--- | :--- |
| Number of Clusters ($k$) | 2 | Selected via Silhouette analysis and Elbow heuristic |
| PCA Components | 23 | Captures 91.61% cumulative explained variance |
| Silhouette Score | 0.0687 | Validates separation in high-dimensional mixed-type space |
| Davies-Bouldin Index | 3.6410 | Evaluates average cluster similarity ratio |
| Calinski-Harabasz Index | 35.3588 | Ratio of between-cluster to within-cluster dispersion |
| Model Inertia | 11,974.93 | Sum of squared distances to closest centroid |

---

## Customer Persona Insights & Business Strategy

Analysis of the cluster profiles reveals two distinct customer segments:

### Cluster 0: Value-Driven & Deliberate Shoppers
- **Average Age**: ~31.0 years
- **Behavioral Attributes**:
  - Higher discount importance score (3.98/5.0)
  - Lower influence from social media advertisements (2.82/5.0)
  - Lower ad-to-purchase conversion inclination (2.64/5.0)
  - Low impulse purchase tendency (2.20/5.0)
  - High reliance on product reviews and online research
- **Commercial Strategy**:
  - Prioritize search engine optimization (SEO), transparent price comparison, and customer review showcases.
  - Deliver value-centric email campaigns highlighting bundle discounts and loyalty rewards.
  - Avoid aggressive social media ad targeting; emphasize product quality and return policy guarantees.

### Cluster 1: Social-First & Impulse-Driven Digital Natives
- **Average Age**: ~21.3 years
- **Behavioral Attributes**:
  - Significantly higher susceptibility to social media advertising (3.24/5.0)
  - Higher likelihood of purchasing directly after ad exposure (3.25/5.0)
  - Higher impulse purchase frequency (2.84/5.0)
  - Strong reliance on peer influence and digital content
- **Commercial Strategy**:
  - Deploy targeted social media campaigns on short-form video and photo-sharing platforms.
  - Implement time-sensitive flash sales and limited-edition product drops.
  - Streamline the checkout process with express payment options and Buy Now Pay Later (BNPL) integrations to capitalize on impulse intent.

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/consumer-shopping-behavior-segmentation.git
   cd consumer-shopping-behavior-segmentation
   ```

2. Create and activate a virtual environment:
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebook

Launch Jupyter and open the pipeline notebook:
```bash
jupyter notebook Model/customer_clustering_pipeline.ipynb
```

---

## Inference with Serialized Models

The trained pipeline components can be loaded directly to classify new customer survey entries:

```python
import joblib
import pandas as pd

# Load serialized pipeline artifacts
preprocessor = joblib.load("Model/artifacts/preprocessor.joblib")
pca = joblib.load("Model/artifacts/pca_model.joblib")
kmeans = joblib.load("Model/artifacts/kmeans_model.joblib")

# Example raw respondent data
new_respondent = pd.DataFrame([{
    "Age": 23,
    "Gender": "Female",
    "Country_Region": "United States",
    "Occupation_Status": "Student",
    "Monthly_Income_Range": "< $1,000",
    "Shop_Most_Where": "Online",
    "Preferred_Platform": "Instagram Shop",
    "Primary_Device": "Smartphone",
    "Online_Shopping_Frequency": "Weekly",
    "Avg_Monthly_Spend_NonEssentials": "$50 - $100",
    "Top_Spending_Category": "Clothing & Apparel",
    "Preferred_Payment_Method": "Digital Wallet",
    "Uses_BNPL_Installments": "Often",
    "Makes_Shopping_List": "Never",
    "Review_Influence_Score": 4.5,
    "Social_Ads_Influence_Score": 4.8,
    "Price_Comparison_Frequency": 3.0,
    "Discount_Importance_Score": 4.0,
    "Likely_To_Buy_After_Ad_Score": 4.2,
    "Impulse_Purchase_Frequency": 4.0,
    "Online_Shopping_Satisfaction_Score": 8.0,
    "Trust_In_Online_Reviews_Score": 7.0,
    "Regret_After_Purchase_Frequency": 3.0,
    "Follows_Brands_On_Social": "Yes",
    "Reason_Prefer_Online": "Convenience",
    "Reason_Prefer_InStore": "None",
    "Best_Shopping_Mode": "Online",
    "Return_Frequency": "Sometimes"
}])

# Execute inference pipeline
X_processed = preprocessor.transform(new_respondent)
X_pca = pca.transform(X_processed)
predicted_cluster = kmeans.predict(X_pca)[0]

print(f"Assigned Segment: Cluster {predicted_cluster}")
```

---

## Dependencies

The project utilizes the following core packages:
- `pandas` - Data manipulation and tabular analysis
- `numpy` - Vectorized operations and linear algebra
- `scikit-learn` - Preprocessing pipelines, PCA, K-Means clustering, and evaluation metrics
- `scipy` - Hierarchical clustering and dendrogram generation
- `matplotlib` - Static statistical data visualization
- `seaborn` - Advanced statistical plotting
- `joblib` - Machine learning model serialization

---

## Author

Developed as part of a Data Science portfolio showcasing unsupervised machine learning, data engineering pipelines, and customer segmentation analytics.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

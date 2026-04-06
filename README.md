# E-commerce Customer Segmentation & Behavioral Analysis

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology to perform a complete end-to-end data science analysis on an e-commerce transactions dataset. The goal is to segment customers into meaningful groups based on their purchasing behavior using unsupervised learning (PCA and K-Means) and then interpret those segments using a supervised learning model (Decision Tree).

## 🚀 Project Overview

The analysis is structured into 6 distinct phases, each represented by a dedicated Jupyter Notebook:

1.  **Data Understanding (EDA)**: [new_eda.ipynb](new_eda.ipynb)
    *   Relational data merging (Customers, Products, Transactions).
    *   Statistical profiling and behavioral pattern exploration.
    *   FM (Frequency, Monetary) aggregation validation.
2.  **Data Preparation**: [data_preparation.ipynb](data_preparation.ipynb)
    *   Feature Engineering: Aggregating transactions to the Customer grain.
    *   Cleaning: Removing chronological factors (Signup/Transaction dates).
    *   Encoding: One-Hot Encoding for categorical features (Region, Preferred Category).
    *   Scaling: Standardizing features using `StandardScaler`.
3.  **Modeling - PCA**: [pca.ipynb](pca.ipynb)
    *   Dimensionality reduction to handle multi-collinearity.
    *   Variance analysis (Scree Plot) targeting 90% explained variance (7 components).
4.  **Modeling - K-Means**: [kmeans.ipynb](kmeans.ipynb)
    *   Unsupervised clustering using the Elbow Method and Silhouette Analysis.
    *   Customer segmentation and cluster profiling.
5.  **Modeling - Decision Tree**: [new_decision_tree.ipynb](new_decision_tree.ipynb)
    *   Interpretable classification to explain cluster membership.
    *   Extracting "If-Then" rules for business stakeholders.
6.  **Evaluation & Conclusion**: [evaluation_conclusion.ipynb](evaluation_conclusion.ipynb)
    *   Synthesis of findings and cluster quality metrics.
    *   Strategic business recommendations per segment.

## 📊 Key Results

*   **Optimal PCA Components**: 7 components successfully captured >90% of the dataset's variance.
*   **Customer Segments**: Clear behavioral archetypes identified (e.g., "High-Frequency Tech Enthusiasts", "Low-Spend Occasional Shoppers").
*   **Predictive Strength**: The Decision Tree model achieved high accuracy in re-classifying customers into segments, proving the robustness of the clustering logic.

## 🛠️ Installation & Usage

### Prerequisites
*   Python 3.11+
*   Jupyter Notebook / JupyterLab

### Setup
1.  Clone the repository:
    ```bash
    git clone https://github.com/MohamedBenMaaouia/Ecommerce-transaction-data-science-project-.git
    cd Ecommerce-transaction-data-science-project-
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the notebooks in sequence:
    *   Start with `new_eda.ipynb` and follow the numerical order.

## 📁 Repository Structure

*   `data/`: Contains raw CSV files and generated feature matrices.
*   `*.ipynb`: Sequential project notebooks.
*   `requirements.txt`: Project dependencies (pandas, scikit-learn, matplotlib, seaborn).

## 🎓 Academic Context

This project was built as a "Datamining and Data Analysis" academic project, focusing on **interpretation over quantity**. Depth was prioritized in explaining *why* certain methods were chosen and *what* the resulting clusters mean for a business.

---
**Author**: Mohamed Ben Maaouia
**Methodology**: CRISP-DM
**Tools**: Python, Scikit-Learn, Pandas, Matplotlib, Seaborn

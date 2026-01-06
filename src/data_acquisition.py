import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
import os


def download_heart_disease_data():
    """Download heart disease dataset"""
    try:
        # Using UCI Heart Disease dataset from OpenML
        heart_data = fetch_openml(name="heart-statlog", version=1, as_frame=True)
        df = heart_data.frame

        # Rename 'class' column to 'target' and convert to binary
        if "class" in df.columns:
            df["target"] = (df["class"] == "present").astype(int)
            df = df.drop("class", axis=1)

        os.makedirs("data", exist_ok=True)
        df.to_csv("data/heart_disease.csv", index=False)
        print("Dataset downloaded successfully!")
        return df
    except:
        # Fallback: create synthetic data
        np.random.seed(42)
        n_samples = 1000

        data = {
            "age": np.random.randint(25, 80, n_samples),
            "sex": np.random.randint(0, 2, n_samples),
            "chest_pain": np.random.randint(0, 4, n_samples),
            "resting_bp": np.random.randint(90, 200, n_samples),
            "cholesterol": np.random.randint(100, 400, n_samples),
            "fasting_bs": np.random.randint(0, 2, n_samples),
            "resting_ecg": np.random.randint(0, 3, n_samples),
            "max_hr": np.random.randint(60, 220, n_samples),
            "exercise_angina": np.random.randint(0, 2, n_samples),
            "oldpeak": np.random.uniform(0, 6, n_samples),
            "st_slope": np.random.randint(0, 3, n_samples),
            "target": np.random.randint(0, 2, n_samples),
        }

        df = pd.DataFrame(data)
        df.to_csv("data/heart_disease.csv", index=False)
        print("Synthetic dataset created!")
        return df


def perform_eda(df):
    """Perform comprehensive EDA"""
    os.makedirs("plots", exist_ok=True)

    # Basic info
    print("Dataset Shape:", df.shape)
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Target distribution
    plt.figure(figsize=(8, 6))
    df["target"].value_counts().plot(kind="bar")
    plt.title("Target Distribution")
    plt.xlabel("Heart Disease (0=No, 1=Yes)")
    plt.ylabel("Count")
    plt.savefig("plots/target_distribution.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png")
    plt.close()

    # Age distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["age"], bins=20, alpha=0.7, edgecolor="black")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.savefig("plots/age_distribution.png")
    plt.close()

    print("EDA plots saved to 'plots' directory")


if __name__ == "__main__":
    df = download_heart_disease_data()
    perform_eda(df)
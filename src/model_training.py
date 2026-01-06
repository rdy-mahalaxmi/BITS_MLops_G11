import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
import joblib
import mlflow
import mlflow.sklearn
import os

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def fit_transform(self, df):
        """Fit and transform features"""
        df_processed = df.copy()
        
        # Separate features and target
        X = df_processed.drop('target', axis=1)
        y = df_processed['target']
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        return X, y
    
    def transform(self, df):
        """Transform new data using fitted transformers"""
        df_processed = df.copy()
        X = df_processed.drop('target', axis=1) if 'target' in df_processed.columns else df_processed
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X

class ModelTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42)
        }
        self.best_model = None
        self.feature_engineer = FeatureEngineer()
        
    def train_models(self, X, y):
        """Train multiple models with hyperparameter tuning"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {}
        
        for name, model in self.models.items():
            with mlflow.start_run(run_name=name):
                # Hyperparameter tuning
                if name == 'logistic_regression':
                    param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
                else:  # random_forest
                    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
                
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                
                # Predictions
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Cross-validation
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
                
                # Log to MLflow
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("roc_auc", roc_auc)
                mlflow.log_metric("cv_mean", cv_scores.mean())
                mlflow.log_metric("cv_std", cv_scores.std())
                
                # Log model
                mlflow.sklearn.log_model(best_model, name)
                
                results[name] = {
                    'model': best_model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean()
                }
                
                print(f"{name} - ROC-AUC: {roc_auc:.4f}, CV Mean: {cv_scores.mean():.4f}")
        
        # Select best model
        best_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        self.best_model = results[best_name]['model']
        
        print(f"Best model: {best_name}")
        return results

def main():
    # Load data
    df = pd.read_csv('data/heart_disease.csv')
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    X, y = feature_engineer.fit_transform(df)
    
    # Save feature engineer
    os.makedirs('models', exist_ok=True)
    joblib.dump(feature_engineer, 'models/feature_engineer.pkl')
    
    # Create MLflow experiment if it doesn't exist
    try:
        mlflow.set_experiment("heart_disease_prediction")
    except:
        mlflow.create_experiment("heart_disease_prediction")
        mlflow.set_experiment("heart_disease_prediction")
    
    # Train models
    trainer = ModelTrainer()
    trainer.feature_engineer = feature_engineer
    results = trainer.train_models(X, y)
    
    # Save best model
    joblib.dump(trainer.best_model, 'models/best_model.pkl')
    joblib.dump(trainer, 'models/model_trainer.pkl')
    
    print("Training completed!")

if __name__ == "__main__":
    main()
    
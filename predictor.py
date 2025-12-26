import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import warnings
import logging
from data_loader import Job, JobLoader

warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

# Set up logging
logging.basicConfig(filename='grid_search.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def create_responsive_features(df: pd.DataFrame):
    """
    Create responsive features for the dataset, considering job's expected training time,
    job type, and other relevant factors.
    """
    df['estimated_time'] = (
        0.4 * df['model_size'] / 100 +        # Model complexity
        0.3 * df['dataset_size'] / 1000 +    # Dataset size
        0.2 * df['epochs'] * 10 +            # Number of training epochs
        df['batch_size'] * 0.1               # Batch size
    )
    
    df['job_type'] = pd.cut(
        df['estimated_time'],
        bins=[0, 10, 100, 1000, float('inf')],
        labels=['interactive', 'short', 'medium', 'long']
    )
    
    df['responsiveness_score'] = (
        1.0 / (df['model_size'] + 1) * 0.2 +
        1.0 / (df['dataset_size'] + 1) * 0.2 +
        1.0 / (df['epochs'] + 1) * 0.6
    )
    
    df['io_intensity'] = df['dataset_size'] / (df['batch_size'] + 1)
    
    return df


class RuntimePredictor:
    """Runtime predictor supporting multiple regression models"""

    def __init__(self, model_types: List[str] = ["random_forest", "xgboost"]):
        self.model_types = model_types
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False

        self.feature_columns = [
            "model_size",
            "batch_size",
            "dataset_size",
            "epochs",
            "uses_gpu",
            "estimated_time",
            "responsiveness_score",
            "io_intensity"
        ]

    # Model factory with regularization parameters
    def _build_model(self, model_type: str, random_state: int):
        if model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,  # Limit splitting depth to reduce overfitting
                min_samples_leaf=2,   # Limit minimum samples in leaf nodes
                random_state=random_state
            )

        elif model_type == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=100,          # Number of trees
                max_depth=5,               # Maximum depth of trees to reduce overfitting
                random_state=random_state,
                objective="reg:squarederror", 
                alpha=0.3,                    # L1 regularization (increases regularization strength)
                reg_lambda=0.3,              # L2 regularization (increases regularization strength)
                eta=0.1,                      # Learning rate controls model complexity
                subsample=0.8,                # Control the proportion of samples used
                colsample_bytree=0.8,         # Column sampling for each tree
                verbosity=0                   # Disable log output
            )

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    # Data preparation
    def _jobs_to_xy(self, jobs: List[Job]):
        X, y = [], []
        for job in jobs:
            job_dict = {
                'model_size': job.model_size,
                'batch_size': job.batch_size,
                'dataset_size': job.dataset_size,
                'epochs': job.epochs,
                'uses_gpu': job.uses_gpu,
                'true_runtime': job.true_runtime
            }
            
            job_df = pd.DataFrame([job_dict])
            job_df = create_responsive_features(job_df)
            
            X.append([ 
                job_df['model_size'].values[0],
                job_df['batch_size'].values[0],
                job_df['dataset_size'].values[0],
                job_df['epochs'].values[0],
                job_df['uses_gpu'].values[0],
                job_df['estimated_time'].values[0],
                job_df['responsiveness_score'].values[0],
                job_df['io_intensity'].values[0]
            ])
            y.append(job.true_runtime)
        return np.array(X), np.array(y)

    # Hyperparameter optimization (Grid Search)
    def _optimize_model(self, model, X_train, y_train, model_type: str):
        if model_type == "random_forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20],
                'min_samples_split': [5, 10, 15],   # Limit splitting depth to reduce overfitting
                'min_samples_leaf': [2, 4, 6]        # Increase minimum samples in leaf nodes
            }
        elif model_type == "xgboost":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 6, 8],              # Limit the maximum depth of trees
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'alpha': [0, 0.1, 0.2],   # L1 regularization
                'reg_lambda': [0, 0.1, 0.2]  # L2 regularization
            }
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)

        # Log the best parameters
        logger.info(f"Best parameters for {model_type}: {grid_search.best_params_}")

        return grid_search.best_estimator_

    # Training
    def train_model(self, jobs: List[Job], test_size: float = 0.2, random_state: int = 42):
        X, y = self._jobs_to_xy(jobs)
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

        # Use tqdm to show a progress bar
        for model_type in tqdm(self.model_types, desc="Training models", ncols=100):
            model = self._build_model(model_type, random_state)
            optimized_model = self._optimize_model(model, X_train, y_train, model_type)
            self.models[model_type] = optimized_model

            # Evaluate performance on training set
            self._evaluate(optimized_model, X_train, y_train, model_type, dataset_type="Train")

            # Evaluate performance on test set
            self._evaluate(optimized_model, X_test, y_test, model_type, dataset_type="Test")

            if hasattr(optimized_model, "feature_importances_"):
                self._show_feature_importance(optimized_model, model_type)

        self.is_trained = True

    # Evaluate the model
    def _evaluate(self, model, X, y, model_type, dataset_type="Test"):
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        print(f"\n=== {model_type} {dataset_type} Evaluation ===")
        print(f"MAE : {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²  : {r2:.4f}")

    def _show_feature_importance(self, model, model_type):
        print(f"\n=== {model_type} Feature Importance ===")
        if hasattr(model, "feature_importances_"):
            for name, score in zip(self.feature_columns, model.feature_importances_):
                print(f"{name}: {score:.4f}")

    def predict_runtime(self, job: Job) -> dict:
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        X = np.array([[ 
            job.model_size,
            job.batch_size,
            job.dataset_size,
            job.epochs,
            job.uses_gpu
        ]])

        job_dict = {
            'model_size': job.model_size,
            'batch_size': job.batch_size,
            'dataset_size': job.dataset_size,
            'epochs': job.epochs,
            'uses_gpu': job.uses_gpu,
        }

        job_df = pd.DataFrame([job_dict])
        job_df = create_responsive_features(job_df)

        responsive_features = [
            job_df['estimated_time'].values[0],
            job_df['responsiveness_score'].values[0],
            job_df['io_intensity'].values[0]
        ]

        X = np.hstack((X, np.array(responsive_features).reshape(1, -1)))

        X_scaled = self.scaler.transform(X)

        predictions = {}
        for model_type, model in self.models.items():
            predictions[model_type] = max(0.0, float(model.predict(X_scaled)[0]))

        return predictions

    def predict_batch_runtimes(self, jobs: List[Job]):
        return [self.predict_runtime(job) for job in jobs]

    # Save the model
    def save_model(self, rf_path: str, xgb_path: str, scaler_path: str):
        joblib.dump(self.models["random_forest"], rf_path)
        joblib.dump(self.models["xgboost"], xgb_path)
        joblib.dump(self.scaler, scaler_path)

    # Load the model
    def load_model(self, rf_path: str, xgb_path: str, scaler_path: str):
        self.models["random_forest"] = joblib.load(rf_path)
        self.models["xgboost"] = joblib.load(xgb_path)
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True


if __name__ == "__main__":
    loader = JobLoader()
    train_jobs = loader.load_jobs_from_csv("training_jobs.csv")

    predictor = RuntimePredictor(model_types=["random_forest", "xgboost"])
    predictor.train_model(train_jobs)

    # Save each model
    predictor.save_model("random_forest_model.pkl", "xgboost_model.pkl", "scaler.pkl")

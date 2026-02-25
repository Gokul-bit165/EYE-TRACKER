import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class GazeRegressionModel:
    def __init__(self):
        """
        Initializes the regression model using a Neural Network.
        Maps normalized eye features + head pose to screen X,Y.
        """
        # Small Neural Network (Multi-Layer Perceptron)
        self.model_x = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42))
        self.model_y = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42))
        
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> bool:
        """
        Trains the models.
        X_train: shape (N_samples, N_features)
        y_train: shape (N_samples, 2)
        """
        if len(X_train) < 5: 
            return False
            
        try:
            self.model_x.fit(X_train, y_train[:, 0])
            self.model_y.fit(X_train, y_train[:, 1])
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Failed to train regression models: {e}")
            return False

    def predict(self, features: np.ndarray) -> tuple:
        """
        Predicts screen (x, y) coordinates.
        features: 1D array of shape (N_features,)
        """
        if not self.is_trained:
            return None
        
        # Reshape to 2D array for sklearn (1 sample, N features)
        X = features.reshape(1, -1)
        
        pred_x = self.model_x.predict(X)[0]
        pred_y = self.model_y.predict(X)[0]
        
        return (pred_x, pred_y)

    def save_model(self, filepath: str):
        if self.is_trained:
            with open(filepath, 'wb') as f:
                pickle.dump({'model_x': self.model_x, 'model_y': self.model_y}, f)

    def load_model(self, filepath: str) -> bool:
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model_x = data['model_x']
                self.model_y = data['model_y']
                self.is_trained = True
                return True
        except Exception as e:
            print(f"No existing profiling found or failed to load: {e}")
            self.is_trained = False
            return False

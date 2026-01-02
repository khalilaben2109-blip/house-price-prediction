"""
Tests unitaires pour les modèles
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.linear_regression_model import LinearRegressionModel
from models.random_forest_model import RandomForestModel

class TestModels(unittest.TestCase):
    
    def setUp(self):
        """Prépare les données de test"""
        np.random.seed(42)
        self.X_train = pd.DataFrame(np.random.randn(100, 5), 
                                   columns=['f1', 'f2', 'f3', 'f4', 'f5'])
        self.y_train = pd.Series(np.random.randn(100))
        self.X_test = pd.DataFrame(np.random.randn(20, 5), 
                                  columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    
    def test_linear_regression_training(self):
        """Test l'entraînement du modèle de régression linéaire"""
        model = LinearRegressionModel()
        model.train(self.X_train, self.y_train)
        self.assertTrue(model.is_trained)
    
    def test_linear_regression_prediction(self):
        """Test les prédictions du modèle de régression linéaire"""
        model = LinearRegressionModel()
        model.train(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
    
    def test_random_forest_training(self):
        """Test l'entraînement du modèle Random Forest"""
        model = RandomForestModel()
        model.train(self.X_train, self.y_train)
        self.assertTrue(model.is_trained)
    
    def test_random_forest_prediction(self):
        """Test les prédictions du modèle Random Forest"""
        model = RandomForestModel()
        model.train(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

if __name__ == '__main__':
    unittest.main()
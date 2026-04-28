import pandas as pd
import numpy as np
import pytest
from ml_bike_demand_end_to_end.pipelines.nodes import split_data, compute_metrics

def test_split_data():
    # Arrange
    # Creamos un DataFrame falso con 10 filas
    data = {
        "feature_1": range(10),
        "feature_2": range(10, 20),
        "target": range(100, 110)
    }
    df = pd.DataFrame(data)
    
    # Parámetros simulados
    params = {
        "target_params": {
            "new_target_name": "target"
        },
        "train_fraction": 0.8
    }
    
    # Act
    x_train, x_test, y_train, y_test = split_data(df, params)
    
    # Assert
    # Verificamos que las proporciones sean correctas (80% de 10 = 8 filas para train, 2 para test)
    assert len(x_train) == 8
    assert len(x_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2
    
    # Verificamos que el target no esté en las features
    assert "target" not in x_train.columns
    assert "target" not in x_test.columns
    
    # Verificamos que el orden temporal se respeta (no se mezcla)
    # Train debería tener los primeros 8 elementos (índices 0 al 7)
    assert x_train["feature_1"].iloc[0] == 0
    assert x_train["feature_1"].iloc[-1] == 7
    # Test debería tener los últimos 2 elementos (índices 8 y 9)
    assert x_test["feature_1"].iloc[0] == 8
    assert x_test["feature_1"].iloc[-1] == 9

def test_compute_metrics():
    # Arrange
    # Valores reales y predichos conocidos
    y_true = [100.0, 150.0, 200.0]
    y_pred = [110.0, 150.0, 190.0]
    
    # Act
    metrics = compute_metrics(y_true, y_pred)
    
    # Assert
    # Error absoluto: |100-110|=10, |150-150|=0, |200-190|=10. Media = 20/3 = 6.67
    assert metrics["MAE"] == pytest.approx(6.67, 0.01)
    
    # Error cuadrático: 10^2 = 100, 0^2 = 0, 10^2 = 100. Media = 200/3 = 66.67. Raíz = 8.16
    assert metrics["RMSE"] == pytest.approx(8.16, 0.01)
    
    # Porcentaje de error: 10/100=0.1, 0/150=0, 10/200=0.05. Media = 0.15/3 = 0.05. MAPE = 5%
    assert metrics["MAPE"] == pytest.approx(5.0, 0.01)

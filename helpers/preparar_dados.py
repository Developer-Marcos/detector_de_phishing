import openml
from sklearn.model_selection import train_test_split

def carregar_dataset():
      dataset = openml.datasets.get_dataset(4534)
      X, y, _, _, = dataset.get_data(target=dataset.default_target_attribute)

      return X, y

def dividir_dataset(X, y):
      X_train, X_test, y_train, y_test =  train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y ) 

      return X_train, X_test, y_train, y_test
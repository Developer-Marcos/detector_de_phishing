from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

def mostrar_resultados(y_test, prev_y, nome_modelo):
      print(f"Acurácia ({nome_modelo}):", accuracy_score(y_test, prev_y))
      print(f"Relatório de Classificação ({nome_modelo}):")
      print(classification_report(y_test, prev_y))


def decision_tree(X_train, X_test, y_train, y_test):
      modelo_dt = DecisionTreeClassifier(random_state=42)

      modelo_dt.fit(X=X_train, y=y_train)
      prev_y = modelo_dt.predict(X_test)

      mostrar_resultados(y_test=y_test, prev_y=prev_y, nome_modelo="Decision Tree")

def random_forest(X_train, X_test, y_train, y_test):
      modelo_rf = RandomForestClassifier(random_state=42)

      modelo_rf.fit(X=X_train, y=y_train)
      prev_y = modelo_rf.predict(X_test)

      mostrar_resultados(y_test=y_test, prev_y=prev_y, nome_modelo="Random Forest")

def xg_boost(X_train, X_test, y_train, y_test):
      modelo_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

      modelo_xgb.fit(X=X_train, y=y_train)
      prev_y = modelo_xgb.predict(X_test)

      mostrar_resultados(y_test=y_test, prev_y=prev_y, nome_modelo="XGBoost")

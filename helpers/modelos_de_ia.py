from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
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

def svm(X_train, X_test, y_train, y_test):
      modelo_svm = SVC(kernel='rbf', probability=True, random_state=42)

      modelo_svm.fit(X=X_train, y=y_train)
      prev_y = modelo_svm.predict(X_test)

      mostrar_resultados(y_test=y_test, prev_y=prev_y, nome_modelo="SVM")

def voting_classifier(X_train, X_test, y_train, y_test):
      modelo_dt = DecisionTreeClassifier(random_state=42)
      modelo_rf = RandomForestClassifier(random_state=42)
      modelo_xgb = SVC(kernel='rbf', probability=True, random_state=42)

      voting_clf = VotingClassifier(
            estimators=[('dt', modelo_dt),
                        ('rf', modelo_rf),
                        ('svm', modelo_xgb)
            ],
            voting='soft',
            weights=[3,2,1]    
      )

      voting_clf.fit(X=X_train, y=y_train)
      prev_y = voting_clf.predict(X_test)

      mostrar_resultados(y_test=y_test, prev_y=prev_y, nome_modelo="Voting Classifier com os 3 modelos juntos")


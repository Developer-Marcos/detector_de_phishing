from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def mostrar_resultados(y_test, prev_y, nome_modelo):
      print(f"Acurácia ({nome_modelo}): {accuracy_score(y_test, prev_y) * 100:.2f}%")
      print(f"Relatório de Classificação ({nome_modelo}):")
      print(classification_report(y_test, prev_y))

def grafico_comparativo(X_test, y_test, modelos, nomes):
    accuracies = []
    for modelo in modelos:
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc * 100)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(nomes, accuracies, color='#4A90E2', edgecolor='black', linewidth=1)

   
    max_acc = max(accuracies)
    plt.ylim(0, max_acc + 5)  

    plt.ylabel('Acurácia (%)', fontsize=12)
    plt.title('Comparação de Acurácia dos Modelos', fontsize=14)
    
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha='center', va='bottom', fontsize=11, color='black')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



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
      param_grid_dt = {
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5, 10]
      }

      param_grid_rf = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
      }

      param_grid_svc = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
      }

      grid_dt = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=42),
            param_grid=param_grid_dt,
            cv=3,
            n_jobs=1
      )
      grid_rf = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid_rf,
            cv=3,
            n_jobs=1
      )
      grid_svc = GridSearchCV(
            estimator=SVC(kernel='rbf', probability=True, random_state=42),
            param_grid=param_grid_svc,
            cv=3,
            n_jobs=1
      )

      grid_dt.fit(X=X_train, y=y_train)
      grid_rf.fit(X=X_train, y=y_train)
      grid_svc.fit(X=X_train, y=y_train)

      voting_clf = VotingClassifier(
            estimators=[
                  ('dt', grid_dt.best_estimator_),
                  ('rf', grid_rf.best_estimator_),
                  ('svm', grid_svc.best_estimator_)
            ],
            voting='soft',
            weights=[1, 3, 1]
      )

      voting_clf.fit(X=X_train, y=y_train)
      prev_y = voting_clf.predict(X_test)

      X_total = np.concatenate((X_train, X_test), axis=0)
      y_total = np.concatenate((y_train, y_test), axis=0)

      cv_scores = cross_val_score(voting_clf, X_total, y_total, cv=5, scoring='accuracy', n_jobs=1)
      print(f"\nCross-validation (5 folds) - Acurácia média: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

      mostrar_resultados(y_test=y_test, prev_y=prev_y, nome_modelo="Voting Classifier com GridSearch")

      modelos = [
           grid_dt.best_estimator_,
           grid_rf.best_estimator_,
           grid_svc.best_estimator_,
           voting_clf
      ]

      nomes = [
            'Decision Tree',
            'Random Forest',
            'SVC',
            'Voting Classifier'
      ]

      grafico_comparativo(X_test=X_test, y_test=y_test, modelos=modelos, nomes=nomes)


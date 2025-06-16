from helpers.preparar_dados import carregar_dataset, dividir_dataset
from helpers.modelos_de_ia import decision_tree, random_forest, svm, voting_classifier

dados = carregar_dataset()
X = dados[0]
y = dados[1]

X_train, X_test, y_train, y_test = dividir_dataset(X=X, y=y)

# Modelos Treinados separadamente -> Retire um comentário caso queira que seja executado
#decision_tree(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
#random_forest(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
#svm(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

# Modelos agindo juntos por meio do Voting Classifier para o melhor resultado possivel
voting_classifier(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
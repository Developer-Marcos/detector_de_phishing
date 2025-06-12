from helpers.preparar_dados import carregar_dataset, dividir_dataset

dados = carregar_dataset()
X = dados[0]
y = dados[1]

X_train, X_test, y_train, y_test = dividir_dataset(X=X, y=y)

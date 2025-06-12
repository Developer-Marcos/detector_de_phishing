import openml

dataset = openml.datasets.get_dataset(4534)
X, y, _, _, = dataset.get_data(target=dataset.default_target_attribute)

print(X.shape, y.shape)
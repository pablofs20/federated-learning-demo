import numpy as np
from sklearn.neural_network import MLPClassifier

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([0,1,1,0])

# en servidor
model = MLPClassifier(
                activation='logistic',
                max_iter=100,
                hidden_layer_sizes=(2,),
                solver='lbfgs')

# en clientes
model.fit(inputs, expected_output)

print("n outputs es ", model.n_features_in_)

# mezclar en servidor hasta accuracy aceptable

print('score:', model.score(inputs, expected_output)) # outputs 0.5
print('predictions:', model.predict(inputs)) # outputs [0, 0, 0, 0]
print('expected:', np.array([0, 1, 1, 0]))

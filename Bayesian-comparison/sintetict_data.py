import numpy as np
import baycomp
import matplotlib.pyplot as plt

# Generar datos sintéticos: precisiones de dos clasificadores en varios experimentos
np.random.seed(42)  # Para reproducibilidad
classifier_1_acc = np.random.uniform(0.7, 0.9, 30)  # Precisión del clasificador 1
classifier_2_acc = np.random.uniform(0.6, 0.85, 30)  # Precisión del clasificador 2

# Mostrar las precisiones promedio de ambos clasificadores
print(f"Media de precisión del clasificador 1: {np.mean(classifier_1_acc):.3f}")
print(f"Media de precisión del clasificador 2: {np.mean(classifier_2_acc):.3f}")

# Realizar la comparación bayesiana usando el Sign Test
rope = 0.01  # Region of Practical Equivalence (ROPE)
probabilities = baycomp.SignTest.probs(classifier_1_acc, classifier_2_acc, rope=rope)
print(probabilities)

# Mostrar los resultados
print("\nResultados del Sign Test Bayesiano:")
print(f"P(clasificador 1 mejor): {probabilities[0]:.3f}")
print(f"P(diferencia insignificante dentro del ROPE): {probabilities[1]:.3f}")
print(f"P(clasificador 2 mejor): {probabilities[2]:.3f}")

# Visualización opcional 
fig = baycomp.SignTest.plot(classifier_1_acc, classifier_2_acc, rope=rope)
plt.show()

import numpy as np

y = np.array([46800, 49200, 47900, 51300, 50100])
X_dados = np.array([
    [24, 540, 410],
    [26, 580, 415],
    [25, 560, 405],
    [28, 600, 420],
    [27, 590, 418],
])

# Adicionar a coluna de 1s para o intercepto (beta0)
X = np.hstack((np.ones((X_dados.shape[0], 1)), X_dados))

# Coeficientes
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
XTy = X.T @ y
beta_hat = XTX_inv @ XTy

# Previsões para todos os meses
y_previsto = X @ beta_hat
residuos = y - y_previsto
meses = np.arange(1, len(y) + 1)

# Print da tabela com valores calculados
print("=-=-=-=-=-=-=-=-=-=-= MODELO & RESULTADOS =-=-=-=-=-=-=-=-=-=-=")
print(f"Modelo preditivo: ŷ = {beta_hat[0]:.2f} + {beta_hat[1]:.2f}*x1 + {beta_hat[2]:.2f}*x2 + {beta_hat[3]:.2f}*x3\n")

print("{:<5} {:<10} {:<10} {:<10}".format("Mês", "Real (y)", "Previsto (ŷ)", "Resíduo"))
print("-" * 37)
for i in range(len(y)):
    print("{:<5} {:<10.0f} {:<10.2f} {:<10.2f}".format(
        i + 1, y[i], y_previsto[i], residuos[i]
    ))
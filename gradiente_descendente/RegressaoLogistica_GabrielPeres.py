# Entrada: Conjunto de treinamento (X_treino), rótulos (y_treino), taxa
# de aprendizado η (tx_aprend), numero de iteracões T
# Saída: parâmetros β
#
# 1: Inicializar β com valores pequenos aleatórios
# 2: para t = 1 até T faça
# 3:    Calcular predições: pˆ ← σ(X_treinoβ)
# 4:    Calcular gradiente: g ← X_treino^T(pˆ − y_treino)
# 5:    Atualizar parâmetros: β ← β − ηg
# 6:    se critério de parada for satisfeito então
# 7:        Parar
# 8:    fim se
# 9: fim para

import numpy as np
import matplotlib.pyplot as plt

# Dados de treinamento
X_treino = np.array([2, 4, 6, 8])
X = np.c_[np.ones(len(X_treino)), X_treino]
y_treino = np.array([0, 0, 1, 1])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def grad_desc(X, y, eta=0.5, T=20000, eps=1e-8):
    n, d = X.shape
    beta = np.random.rand(d)  # inicialização aleatória de beta0 e beta1
    
    for t in range(T):
        z = X @ beta
        p = sigmoid(z)
        
        # gradiente médio
        grad = (1/n) * (X.T @ (p - y))
        
        beta_new = beta - eta * grad
        
        # critério de parada
        if np.linalg.norm(beta_new - beta) < eps:
            return beta_new
        
        beta = beta_new
    
    return beta


if __name__ == "__main__":
    beta_est = grad_desc(X, y_treino)
    print(f"> beta0 = {beta_est[0]:.4f}")
    print(f"> beta1 = {beta_est[1]:.4f}")

    # Plotar curva sigmoide ajustada
    x_vals = np.linspace(0, 10, 100)
    X_plot = np.c_[np.ones(len(x_vals)), x_vals]
    y_vals = sigmoid(X_plot @ beta_est)

    plt.scatter(X_treino, y_treino, color='red', label="Dados de treino")
    plt.plot(x_vals, y_vals, color='blue', label="Sigmoide ajustada")
    plt.xlabel("X")
    plt.ylabel("P(Y=1|X)")
    plt.legend()
    plt.savefig("sigmoide.png", dpi=300, bbox_inches="tight")

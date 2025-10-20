import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """
    Implementação do Algoritmo de Aprendizagem do Perceptron.
    """
    def __init__(self):
        # Inicializa pesos (w) e bias (b) como None
        self.w = None
        self.b = None

    def _step_function(self, z):
        """ Função de ativação: Retorna +1 se z > 0, e -1 caso contrário. """
        return np.where(z > 0, 1, -1)

    def fit(self, X, y, eta=1.0, epochs=20, seed=42):
        """
        Ajusta os pesos e o bias do Perceptron (Treinamento).

        Parâmetros:
        X : array-like, forma (n_amostras, n_características) - Dados de entrada.
        y : array-like, forma (n_amostras,) - Rótulos de classe (+1 ou -1).
        eta : float - Taxa de aprendizado (Learning Rate).
        epochs : int - Número de épocas (passagens pelo conjunto de dados).
        seed : int - Semente para inicialização aleatória dos pesos.
        """
        # Configurar a semente para reprodutibilidade
        np.random.seed(seed)

        n_samples, n_features = X.shape

        self.w = np.random.rand(n_features) * 0.01
        self.b = 0.0

        for epoch in range(epochs):
            errors = 0
            for xi, target in zip(X, y):
                # 1. Cálculo da Combinação Linear (z)
                # z = w^T * x + b
                z = np.dot(xi, self.w) + self.b
                
                # 2. Previsão da Saída
                y_pred = self._step_function(z)

                # 3. Cálculo do Erro e da Atualização
                update = eta * (target - y_pred)
                
                if update != 0:
                    # Atualização dos pesos
                    self.w += update * xi
                    
                    # Atualização do bias
                    self.b += update
                    errors += 1
            
            if errors == 0:
                print(f"Convergência alcançada na época {epoch + 1}.")
                break
                        
        return self

    def predict(self, X):
        """
        Calcula a previsão de classe para as amostras X.
        """
        if self.w is None:
            raise ValueError("O modelo não foi treinado. Chame fit() primeiro.")
        
        # z = w^T * X + b (para todo o conjunto X)
        z = np.dot(X, self.w) + self.b
        
        # Aplica a função de ativação
        return self._step_function(z)
    
    def plot_decision_boundary(self, X, y, title):
        """ Plota os pontos e a fronteira de decisão aprendida. """

        # Encontrar a reta (x2 = m*x1 + c)
        # y = (-w[0]/w[1]) * x + (-b/w[1])
        x1_min, x1_max = -0.5, 1.5

        # Se w[1] for muito próximo de zero, a reta é vertical (caso raro com inicialização aleatória)
        if np.abs(self.w[1]) < 1e-6:
            # A fronteira é uma linha vertical: x1 = -b/w[0]
            x_reta = -self.b / self.w[0]

            plt.plot([x_reta, x_reta], [-0.5, 1.5], 'k--', linewidth=2)

        else:
            # Reta normal: x2 = m*x1 + c
            slope = -self.w[0] / self.w[1]  # m
            intercept = -self.b / self.w[1] # c

            # Pontos para desenhar a reta
            x1_plot = np.array([x1_min, x1_max])
            x2_plot = slope * x1_plot + intercept

            plt.plot(x1_plot, x2_plot, 'k--', linewidth=2, label='Fronteira de Decisão')

        # Pontos da classe +1 (target=1)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='blue', s=100, label='Classe +1')
        # Pontos da classe -1 (target=-1)
        plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='x', color='red', s=100, label='Classe -1')

        # Configurações do Plot
        plt.title(title)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.xlim(x1_min, x1_max)
        plt.ylim(x1_min, x1_max)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.savefig(f'{title}.png', bbox_inches='tight', dpi=150)


# ----- CÓDIGO DE TESTE ----- #

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_and = np.array([-1, -1, -1, 1])   # Rótulos de saída (y) para o problema AND
y_or = np.array([-1, 1, 1, 1])      # Rótulos de saída (y) para o problema OR


# --- Treinamento do Perceptron para AND ---
perc_and = Perceptron()
print("--- Treinamento AND ---")
perc_and.fit(X, y_and, eta=0.1, epochs=100)

# Teste de previsão
y_pred_and = perc_and.predict(X)
print(f"Previsões AND: {y_pred_and}")
print(f"Pesos finais (w) AND: {perc_and.w}")
print(f"Bias final (b) AND: {perc_and.b}\n")
# Plot para o Problema AND
perc_and.plot_decision_boundary(X, y_and, "Problema_AND")

# --- Treinamento do Perceptron para OR ---
perc_or = Perceptron()
print("--- Treinamento OR ---")
perc_or.fit(X, y_or, eta=0.1, epochs=100)

# Teste de previsão
y_pred_or = perc_or.predict(X)
print(f"Previsões OR: {y_pred_or}")
print(f"Pesos finais (w) OR: {perc_or.w}")
print(f"Bias final (b) OR: {perc_or.b}\n")
# Plot para o Problema OR
perc_or.plot_decision_boundary(X, y_or, "Problema_OR")


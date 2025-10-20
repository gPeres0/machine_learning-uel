import pandas as pd
import numpy as np


class NaiveBayes:
    def __init__(self):
        self.target_name = None
        self.features = None
        self.prior_probs = {}
        self.conditional_probs = {}


    def fit(self, X_train: pd.DataFrame):
        """
        Treina o modelo Naive Bayes com os dados fornecidos.

        ENTRADA:
            X_train (pd.DataFrame): DataFrame de treinamento com as features e a coluna de destino.
        """
        self.target_name = X_train.columns[-1]
        self.features = X_train.columns[:-1]
        
        # Calcula a probabilidade a priori para cada classe
        total_examples = len(X_train)
        self.prior_probs = (X_train[self.target_name].value_counts() / total_examples).to_dict()

        # Calcula as probabilidades condicionais
        for class_name, prior_prob in self.prior_probs.items():
            self.conditional_probs[class_name] = {}

            class_subset = X_train[X_train[self.target_name] == class_name]

            for feature in self.features:
                alpha = 1  # Operador de Laplace
                num_categories = len(X_train[feature].unique())
                denominador = len(class_subset) + alpha * num_categories
                
                # Usa value_counts() para pegar a frequência de cada categoria em feature
                feature_counts = class_subset[feature].value_counts()
                
                all_values = X_train[feature].unique()                
                probabilities = (feature_counts.reindex(all_values, fill_value=0) + alpha) / denominador
                self.conditional_probs[class_name][feature] = probabilities.to_dict()


    def predict(self, new_example: dict):
        """
        Prevê a classe de um novo exemplo.

        ENTRADA:
            new_example (dict): Um dicionário de características e seus valores.

        SAÍDA:
            str: O nome da classe com a maior probabilidade.
        """
        scores = {}
        
        # Itera sobre cada uma das classes
        for class_name, prior_prob in self.prior_probs.items():
            # Inicializa score com a probabilidade a priori
            score = prior_prob
            
            # Multiplica score por cada probabilidade condicional
            for feature, value in new_example.items():
                if feature in self.conditional_probs[class_name] and value in self.conditional_probs[class_name][feature]:
                    score *= self.conditional_probs[class_name][feature][value]
                else:
                    score *= 1e-9  # Evita score zero
            
            scores[class_name] = score

        return max(scores, key=scores.get)
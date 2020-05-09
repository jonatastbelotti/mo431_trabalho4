import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from otimizador import Otimizador
from svm import SVM


class RandomSearch(Otimizador):

    NOME_OTIMIZADOR = "Random search"

    def __init__(self, num_comb):
        super().__init__()
        self.num_combinacoes = num_comb

    def otimizar(self):
        # Definindo os parâmetros a serem utilizados
        parametros = {
            'C': 2 ** np.random.uniform(low=-5.0, high=15.0+1, size=self.num_combinacoes),
            'gamma': 2 ** np.random.uniform(low=-15.0, high=3.0+1, size=self.num_combinacoes),
            'epsilon': np.random.uniform(low=0.05, high=1.0, size=self.num_combinacoes)
        }

        # Executando otimização dos parâmetros
        self.iniciar_tempo()
        randomSCV = RandomizedSearchCV(SVR(kernel='rbf'), parametros, scoring="neg_mean_absolute_error", random_state=0, n_iter=self.num_combinacoes, n_jobs=-1)
        randomSCV.fit(self.X_treinamento, self.Y_treinamento)
        self.finalizar_tempo()

        # Identify optimal hyperparameter values
        C = randomSCV.best_params_['C']
        gamma = randomSCV.best_params_['gamma']
        epsilon = randomSCV.best_params_['epsilon']

        # Treinando SVM final com os parâmetros encontrados
        self.svm = SVM(gamma, C, epsilon)
        self.svm.treinar(self.X_treinamento, self.Y_treinamento)
        self.svm.testar(self.X_teste, self.Y_teste)


# CÓDIGO PRINCIPAL
if __name__ == "__main__":
    r = RandomSearch(125)
    r.otimizar()
    r.imprimir()

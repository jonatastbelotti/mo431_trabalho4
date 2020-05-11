import numpy as np

from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from sklearn.svm import SVR
from scipy.stats import uniform
from scipy.stats import loguniform
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
            'C': loguniform(2**-5, 2**15),
            'gamma': loguniform(2**-15, 2**3),
            'epsilon': uniform(0.0, 1)
        }

        cv_ = ShuffleSplit(n_splits=1, test_size=0.1, train_size=0.9)

        # Executando otimização dos parâmetros
        self.iniciar_tempo()
        randomSCV = RandomizedSearchCV(SVR(kernel='rbf'), parametros, scoring="neg_mean_absolute_error", cv=cv_, n_iter=self.num_combinacoes, n_jobs=-1)
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

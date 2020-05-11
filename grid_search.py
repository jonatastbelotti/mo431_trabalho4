import numpy as np

from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.svm import SVR
from scipy.stats import uniform
from scipy.stats import loguniform
from otimizador import Otimizador
from svm import SVM


class GridSearch(Otimizador):

    NOME_OTIMIZADOR = "Grid Search"

    def __init__(self, tam_grid):
        super().__init__()
        self.tamanho_grid = tam_grid

    def otimizar(self):
        # Definindo os parâmetros a serem utilizados
        parametros = {
            'C': loguniform.rvs(2**-5, 2**15, size=self.tamanho_grid),
            'gamma': loguniform.rvs(2**-15, 2**3, size=self.tamanho_grid),
            'epsilon': uniform.rvs(0.0, 1, size=self.tamanho_grid)
        }

        cv_ = ShuffleSplit(n_splits=1, test_size=0.1, train_size=0.9)

        # Executando otimização dos parâmetros
        self.iniciar_tempo()
        grid = GridSearchCV(estimator=SVR(kernel='rbf'), param_grid=parametros, scoring="neg_mean_absolute_error", cv=cv_, n_jobs=-1)
        grid.fit(self.X_treinamento, self.Y_treinamento)
        self.finalizar_tempo()

        # extraindo os melhores hiperparametros
        C = grid.best_params_['C']
        gamma = grid.best_params_['gamma']
        epsilon = grid.best_params_['epsilon']

        # Treinando SVM final com os parâmetros encontrados
        self.svm = SVM(gamma, C, epsilon)
        self.svm.treinar(self.X_treinamento, self.Y_treinamento)
        self.svm.testar(self.X_teste, self.Y_teste)


# CÓDIGO PRINCIPAL
if __name__ == "__main__":
    g = GridSearch(5)
    g.otimizar()
    g.imprimir()

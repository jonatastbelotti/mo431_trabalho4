import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
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
            'C': 2 ** np.random.uniform(low=-5.0, high=15.0+1, size=self.tamanho_grid),
            'gamma': 2 ** np.random.uniform(low=-15.0, high=3.0+1, size=self.tamanho_grid),
            'epsilon': np.random.uniform(low=0.05, high=1.0, size=self.tamanho_grid)
        }

        # Executando otimização dos parâmetros
        self.iniciar_tempo()
        grid = GridSearchCV(estimator=SVR(kernel='rbf'), param_grid=parametros, scoring="neg_mean_absolute_error", cv=self.tamanho_grid, n_jobs=-1)
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

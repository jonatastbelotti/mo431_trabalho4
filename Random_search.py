from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR

from scipy.stats import uniform
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
            'C': self.intervalo_C,
            'gamma': self.intervalo_gamma,
            'epsilon': self.intervalo_epsilon
        }

        # Executando otimização dos parâmetros
        self.iniciar_tempo()
        randomSCV = RandomizedSearchCV(SVR(kernel='rbf'), parametros, random_state=0, n_iter=self.num_combinacoes, n_jobs=-1)
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

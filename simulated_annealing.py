from otimizador import Otimizador
from svm import SVM
from simanneal import Annealer

import numpy as np


class SVMProblem(Annealer):

    def __init__(self, state, X_, Y_):
        self.X = X_
        self.Y = Y_
        super(SVMProblem, self).__init__(state)  # important!

    def move(self):
        self.state[0] = 2 ** np.random.uniform(low=-5.0, high=15.0+1)
        self.state[1] = 2 ** np.random.uniform(low=-15.0, high=3.0+1)
        self.state[2] = np.random.uniform(low=0.05, high=1.0)

    def energy(self):
        svm = SVM(self.state[1], self.state[0], self.state[2])
        svm.treinar(self.X, self.Y)

        return svm.mae_treinamento


class OtimizacaoSimulatedAnnealing(Otimizador):

    NOME_OTIMIZADOR = "Otimização Simulated annealing"

    def __init__(self, passos):
        super().__init__()

        self.num_passos = passos

    def otimizar(self):
        # Definindo estado inicial da otimização
        estado_inicial = [
            2 ** np.random.uniform(low=-5.0, high=15.0+1),
            2 ** np.random.uniform(low=-15.0, high=3.0+1),
            np.random.uniform(low=0.05, high=1.0)
        ]

        # Instanciando otimizador
        sp = SVMProblem(estado_inicial, self.X_treinamento, self.Y_treinamento)
        sp.steps = self.num_passos

        # Otimizando
        self.iniciar_tempo()
        resp, mae = sp.anneal()
        self.finalizar_tempo()

        # Extraindo os melhores hiperparametros
        C = resp[0]
        gamma = resp[1]
        epsilon = resp[2]

        # Treinando SVM final com os parâmetros encontrados
        self.svm = SVM(gamma, C, epsilon)
        self.svm.treinar(self.X_treinamento, self.Y_treinamento)
        self.svm.testar(self.X_teste, self.Y_teste)


# CÓDIGO PRINCIPAL
if __name__ == "__main__":
    sa = OtimizacaoSimulatedAnnealing(125)
    sa.otimizar()
    sa.imprimir()

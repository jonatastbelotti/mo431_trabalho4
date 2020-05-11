from otimizador import Otimizador
from svm import SVM
from cma import CMAEvolutionStrategy


class OtimizacaoCMA_ES(Otimizador):

    NOME_OTIMIZADOR = "Otimização CMA-ES"

    def __init__(self, chamadas):
        super().__init__()

        self.num_chamadas = chamadas

    def func_objetivo(self, arg):
        C_ = 2 ** (-5 + arg[0] * 20)
        gamma_ = 2 ** (-15 + arg[1] * 18)
        epsilon_ = abs(arg[2])

        svm = SVM(gamma_, C_, epsilon_)
        svm.treinar(self.X_treinamento, self.Y_treinamento)

        return svm.mae_treinamento


    def otimizar(self):
        lw = [0.0, 0.0, 0.0]
        up = [1.0, 1.0, 1.0]
        x0 = 3 * [0.1]
        sigma = 0.25

        # Otimizando
        c = CMAEvolutionStrategy(x0, sigma, {'bounds': [lw, up]})
        self.iniciar_tempo()
        c.optimize(self.func_objetivo, iterations=self.num_chamadas)
        self.finalizar_tempo()

        # Extraindo os melhores hiperparametros
        C_ = 2 ** (-5 + c.best.x[0] * 20)
        gamma_ = 2 ** (-15 + c.best.x[1] * 18)
        epsilon_ = abs(c.best.x[2])

        # Treinando SVM final com os parâmetros encontrados
        self.svm = SVM(gamma_, C_, epsilon_)
        self.svm.treinar(self.X_treinamento, self.Y_treinamento)
        self.svm.testar(self.X_teste, self.Y_teste)


# CÓDIGO PRINCIPAL
if __name__ == "__main__":
    cme = OtimizacaoCMA_ES(125)
    cme.otimizar()
    cme.imprimir()

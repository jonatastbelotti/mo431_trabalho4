from otimizador import Otimizador
from svm import SVM
from pyswarm import pso


class OtimizacaoPSO(Otimizador):

    NOME_OTIMIZADOR = "Otimização PSO"

    def __init__(self, num_part, num_ite):
        super().__init__()

        self.num_particulas = num_part
        self.num_iteracoes = num_ite

    def otimizar(self):
        # Defininfo limites dos hiperparametros
        lb = [-5, -15, 0.05]
        ub = [15, 3, 1.0]

        # Executando otimização dos parâmetros
        self.iniciar_tempo()
        xopt, fopt = pso(self.func_obj_pso, lb, ub, swarmsize=self.num_particulas, maxiter=self.num_iteracoes)
        self.finalizar_tempo()

        # Extraindo os melhores hiperparametros
        C = 2.0 ** xopt[0]
        gamma = 2.0 ** xopt[1]
        epsilon = xopt[2]

        # Treinando SVM final com os parâmetros encontrados
        self.svm = SVM(gamma, C, epsilon)
        self.svm.treinar(self.X_treinamento, self.Y_treinamento)
        self.svm.testar(self.X_teste, self.Y_teste)

    def func_obj_pso(self, params):
        C_ = 2.0 ** params[0]
        gamma_ = 2.0 ** params[1]
        epsilon_ = params[2]

        svm = SVM(gamma_, C_, epsilon_)
        svm.treinar(self.X_treinamento, self.Y_treinamento)

        return svm.mae_treinamento



# CÓDIGO PRINCIPAL
if __name__ == "__main__":
    p = OtimizacaoPSO(11, 11)
    p.otimizar()
    p.imprimir()

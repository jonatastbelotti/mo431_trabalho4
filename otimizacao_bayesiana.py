from hyperopt import hp, fmin, tpe

from otimizador import Otimizador
from svm import SVM


class OtimizacaoBayesiana(Otimizador):

    NOME_OTIMIZADOR = "Otimização bayesiana"

    def __init__(self, num_cham):
        super().__init__()

        self.num_chamadas = num_cham

    def otimizar(self):
        # Definindo os parâmetros a serem utilizados
        parametros = {
            'C': hp.uniform('C', -5, 15),
            'gamma': hp.uniform('gamma', -15, 3),
            'epsilon': hp.uniform('epsilon', 0.05, 1.0)
        }

        # Executando otimização dos parâmetros
        self.iniciar_tempo()
        resp = fmin(self.func_obj_bayesiana, parametros, algo=tpe.suggest, max_evals=self.num_chamadas)
        self.finalizar_tempo()

        # Extraindo os melhores hiperparametros
        C = 2.0 ** resp['C']
        gamma = 2.0 ** resp['gamma']
        epsilon = resp['epsilon']

        # Treinando SVM final com os parâmetros encontrados
        self.svm = SVM(gamma, C, epsilon)
        self.svm.treinar(self.X_treinamento, self.Y_treinamento)
        self.svm.testar(self.X_teste, self.Y_teste)

    def func_obj_bayesiana(self, params):
        C_ = 2.0 ** params['C']
        gamma_ = 2.0 ** params['gamma']
        epsilon_ = params['epsilon']

        svm = SVM(gamma_, C_, epsilon_)
        svm.treinar(self.X_treinamento, self.Y_treinamento)

        return svm.mae_treinamento


# CÓDIGO PRINCIPAL
if __name__ == "__main__":
    b = OtimizacaoBayesiana(125)
    b.otimizar()
    b.imprimir()

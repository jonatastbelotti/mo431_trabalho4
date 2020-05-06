import numpy as np
import os
import time


class Otimizador():

    NOME_OTIMIZADOR = ""

    def __init__(self):
        # Carregando arquivos com os dados de treinamento e teste
        pasta = os.path.dirname(os.path.realpath(__file__))
        self.X_treinamento = np.load(pasta + "/dados/Xtreino5.npy")
        self.Y_treinamento = np.load(pasta + "/dados/ytreino5.npy")
        self.X_teste = np.load(pasta + "/dados/Xteste5.npy")
        self.Y_teste = np.load(pasta + "/dados/yteste5.npy")

        # Criando os intervalos das configurações
        self.intervalo_C = 2.0 ** np.arange(start=-5, stop=15+1)
        self.intervalo_gamma = 2.0 ** np.arange(start=-15, stop=3+1)
        self.intervalo_epsilon = np.arange(start=0.05, stop=1.01, step=0.01)

        self.svm = None

        self.tempo_inicial = None
        self.tempo_final = None
        self.tempo_total = None
    
    def iniciar_tempo(self):
        self.tempo_inicial = time.time()

    def finalizar_tempo(self):
        self.tempo_final = time.time()
        self.tempo_total = self.tempo_final - self.tempo_inicial

    def imprimir(self):
        print("Otimizador " + self.NOME_OTIMIZADOR)
        print("Tempo execução: %.6f segundos" % self.tempo_total)
        print("================================================")
        if self.svm:
            self.svm.imprimir()


# CÓDIGO PRINCIPAL
if __name__ == "__main__":
    o = Otimizador()

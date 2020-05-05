from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import numpy as np
import os


class SVM():

    num_amostra_treinamento = None
    num_amostra_teste = None
    mae_treinamento = None
    mae_teste = None


    def __init__(self, gamma_, C_, epsilon_):
        self.svm = SVR(kernel="rbf", gamma=gamma_, C=C_, epsilon=epsilon_)


    def treinar(self, X, Y):
        self.num_amostra_treinamento = X.shape[0]
        self.svm.fit(X, Y)
        resp = self.svm.predict(X)
        self.mae_treinamento = mean_absolute_error(Y, resp)


    def testar(self, X, Y):
        self.num_amostra_teste = X.shape[0]
        resp = self.svm.predict(X)
        self.mae_teste = mean_absolute_error(Y, resp)
    

    def imprimir(self):
        params = self.svm.get_params()

        print("SVM com núcleo RBF")
        print("C = %f" % params['C'])
        print("Gamma = %f" % params['gamma'])
        print("Epsilon = %f" % params['epsilon'])

        print("Conjunto de Treinamento: %d amostras (%.2f%%)" % (self.num_amostra_treinamento, ((100.0 / (self.num_amostra_treinamento + self.num_amostra_teste)) * self.num_amostra_treinamento)))
        print("MAE Treinamento: %f" % self.mae_treinamento)
        print("Conjunto de Teste: %d amostras (%.2f%%)" % (self.num_amostra_teste, ((100.0 / (self.num_amostra_treinamento + self.num_amostra_teste)) * self.num_amostra_teste)))
        print("MAE Treinamento: %f" % self.mae_teste)



# CÓDIGO PRINCIPAL
if __name__ == "__main__":
    # Carregando arquivos com os dados de treinamento e teste
    pasta = os.path.dirname(os.path.realpath(__file__))
    X_treinamento = np.load(pasta + "/dados/Xtreino5.npy")
    Y_treinamento = np.load(pasta + "/dados/ytreino5.npy")
    X_teste = np.load(pasta + "/dados/Xteste5.npy")
    Y_teste = np.load(pasta + "/dados/yteste5.npy")

    # Instânciando SVM
    svm = SVM(2**-5, 2**-15, 0.05)

    # Treinando SVM
    svm.treinar(X_treinamento, Y_treinamento)

    # Testando SVM
    svm.testar(X_teste, Y_teste)

    # Imprimindo resultado
    svm.imprimir()

import numpy as np
from random import sample

class LRC:
    def __init__(self, lr=1e-3, iters=200, batch_size=100):
        # self.initialized = False
        self.lr = lr
        self.iters = iters
        self.batch_size = batch_size

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, x, w, b):
        return self.sigmoid(np.dot(w.T, x) + b)

    def backward(self, x, pred, y):
        N = x.shape[1]
        dw = (1 / N) * np.dot(x, (pred - y).T)
        db = (1 / N) * np.sum(pred - y)
        return dw, db

    def loss(self, pred, y):
        N = y.shape[0]
        loss = (-1 / N) * np.sum(y * np.log(pred) + (1-y) * np.log(1-pred))
        return loss

    def fit(self, x, y):
        x = x.T
        dim, N = x.shape
        y = np.asarray(y)

        self.w = np.random.randn(dim)
        self.b = 0

        for i in range(self.iters):
            for b in sample(range(N), self.batch_size):
                pred = self.forward(x[:, b:b+1], self.w, self.b)
                dw, db = self.backward(x[:, b:b+1], pred, y[b])
                # abs_grad = np.abs(dw)
                # if abs_grad[abs_grad > 1e-6].size == 0:
                #     break
                assert(dw.shape == self.w.shape)
                self.w -= self.lr * dw
                self.b -= self.lr * db
        # self.lr *= 0.9999
        self.coef_ = self.w

    def predict(self, x):
        pred = self.forward(x.T, self.w, self.b)
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        return pred.tolist()


        

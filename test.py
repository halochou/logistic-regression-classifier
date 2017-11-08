from scipy.io import loadmat
from sklearn.svm import LinearSVC, SVC
from lrc import LRC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

mats = [
'data/classify_d3_k2_saved1.mat',
'data/classify_d3_k2_saved2.mat',
'data/classify_d3_k2_saved3.mat',
'data/classify_d4_k3_saved1.mat',
'data/classify_d4_k3_saved2.mat',
'data/classify_d5_k3_saved1.mat',
'data/classify_d5_k3_saved2.mat',
'data/classify_d99_k50_saved1.mat',
'data/classify_d99_k50_saved2.mat',
'data/classify_d99_k60_saved1.mat',
'data/classify_d99_k60_saved2.mat']

clfs = {
    'svm': SVC(kernel='linear'),
    'svm-rbf': SVC(),
    'lrc': LRC(lr=1e-2, iters=400, batch_size=200),
}

clf_names = ['svm', 'svm-rbf', 'lrc']
for clf_name in clf_names:
    print(clf_name, ':')
    for mat in mats:
        raw_data = loadmat(mat)
        c1 = raw_data['class_1'].T
        c2 = raw_data['class_2'].T
        l1 = c1.shape[0]
        l2 = c2.shape[0]
        s1 = int(l1 * 0.8)
        s2 = int(l2 * 0.8)
        train_data = np.concatenate((c1[:s1, :], c2[:s2, :]), axis=0)
        train_label = [0.] * s1 + [1.] * s2
        test_data = np.concatenate((c1[s1:, :], c2[s2:, :]), axis=0)
        test_label = [0] * (l1-s1) + [1] * (l2-s2)

        clf = clfs[clf_name]
        clf.fit(train_data, train_label)
        test_pred = clf.predict(test_data)
        error = len([1 for (p, l) in zip(test_pred, test_label) if p != l])
        print(mat, 1 - error / len(test_label))

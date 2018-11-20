from python_speech_features import mfcc
from collections import defaultdict
import numpy as np
import sklearn.mixture as skm



sample_data = [('A01M1_0_', [1,2,3,4], 48),
        ('A01M1_1_', [1, 2, 3, 4,5,6], 48),
        ('BC1M1_0_', [1, 2, 3, 4], 48),
        ('BC1M1_1_', [1, 2, 3, 4, 5, 6], 48)]


def get_params(data):
    ret = defaultdict(lambda: [])
    for el in data:
        filename = el[0].split('_')[0]
        number = el[0].split('_')[1]
        assert(len(filename) == 5 and len(number) == 1)
        mfcc_ = mfcc(el[1], samplerate=el[2], appendEnergy=True) # add paremeters
        ret[filename].append((mfcc_, number))
    return ret


def get_test_data(n_test_examples):
    test_mfcc = np.concatenate((np.random.randn(n_test_examples, 1),
                                np.random.randn(n_test_examples, 1) + 50,
                                np.random.randn(n_test_examples, 1) + 100),1)
    test_gmm = skm.GaussianMixture(n_components=3, covariance_type='diag', init_params='random', max_iter=20, n_init=20, tol=0.001)
    test_gmm = test_gmm.fit(test_mfcc)
    return test_mfcc, test_gmm

    # ones_ = [x for x in params_dictionary.values()]
    # labels_ = set(y[1] for x in params_dictionary.values() for y in x)
    # return {k: np.concatenate((np.array([x[0] for x in chain(*ones_) if x[1] == k])),0) for k in labels_}
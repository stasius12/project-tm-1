from python_speech_features import mfcc, delta
from collections import defaultdict, Counter
import numpy as np
import sklearn.mixture as skm
from matplotlib import pyplot as plt
from itertools import product, count
import csv


def get_params(data):
    ret = defaultdict(lambda: [])
    for el in data:
        filename = el[0].split('_')[0]
        number = el[0].split('_')[1]
        assert(len(filename) == 5 and len(number) == 1)
        mfcc_ = mfcc(el[1], samplerate=el[2], appendEnergy=True)  # add paremeters
        mfcc_ = delta(mfcc_, 5)
        ret[filename].append((mfcc_, number))
    return ret


def get_test_data(n_test_examples):
    test_mfcc = np.concatenate((np.random.randn(n_test_examples, 1),
                                np.random.randn(n_test_examples, 1) + 50,
                                np.random.randn(n_test_examples, 1) + 100), 1)
    test_gmm = skm.GaussianMixture(n_components=3, covariance_type='diag', init_params='random', max_iter=20, n_init=20, tol=0.001)
    test_gmm = test_gmm.fit(test_mfcc)
    return test_mfcc, test_gmm


def classificate_mfcc_to_GMM_model(mfcc_matrix, gmm_models):
    log_likelihoods = [] 
    for number, model in sorted(gmm_models.items(), key=lambda x: int(x[0])):
        log_likelihoods.append(np.exp(model.score(mfcc_matrix)))
    return log_likelihoods.index(max(log_likelihoods))


def divide_test(params_dict, n_of_tests_ex=2):
    # change this function to return correct values as it is now
    key_list = list(params_dict.keys())
    print(key_list)
    if n_of_tests_ex == 2:
        from sklearn.model_selection import KFold
        x = [i for i in range(len(key_list))]
        kf = KFold(11)
        rt = [(list(map(lambda z: key_list[z], i)), list(map(lambda z: key_list[z], j))) for i, j in kf.split(x)]

    else:
        from sklearn.model_selection import train_test_split
        rt = []
        counter = Counter()
        for i in count(1):
            if len(rt) >= len(key_list):
                break
            split_ = train_test_split(key_list, test_size=n_of_tests_ex)
            if not any([1 for i in rt if len(set(split_[1]) - set(i[1])) < n_of_tests_ex - 1]):
                counter += Counter(split_[1])
                diff = max(counter.values()) - min(counter.values())
                if not any(filter(lambda z: z > n_of_tests_ex, counter.values())) and (diff < 2 or (diff < 3 and min(counter.values()) == 0)):
                    rt.append(split_)
                else:
                    counter -= Counter(split_[1])
            if len(rt) == len(key_list) - 1:  # if only one set of values is required, then fill it in appropriately
                pom = [z[0] for z in counter.items() if z[1] == n_of_tests_ex - 1]
                rt.append([list(set(key_list) - set(pom)), pom])
                counter += Counter(pom)

        print(counter)

    train_keys = [a[0] for a in rt]
    test_keys = [a[1] for a in rt]
    return train_keys, test_keys


def plot_conf_matrix(matrix, percentage_vals=True):
    fig, axs = plt.subplots()
    plt.imshow(matrix, cmap='coolwarm')
    plt.colorbar()
    sum_ = np.sum(matrix)/10
    # for i, j in filter(lambda x: x[0] == x[1], product(range(10), range(10))):
    for i, j in product(range(10), range(10)):
        if not matrix[j, i] == 0:
            result = int((matrix[j, i] / sum_) * 100) if percentage_vals else int(matrix[j, i])
            fontsize_ = 12 if i ==j else 9
            if not result == 0:
                result = "%s" % result + "%" if percentage_vals else result
                plt.text(i, j, result, horizontalalignment="center", fontname='serif', fontsize=fontsize_, fontweight=700)
    ticks = list(range(10))
    plt.xticks(ticks)
    plt.yticks(ticks)
    axs.tick_params(labeltop=True, labelbottom=False, top=True, bottom=False, labelsize=12)
    plt.title("Recognition ratio: %s" % calc_recogn_ratio(matrix, as_str=True), y=1.05, fontsize=15, fontweight=700)
    plt.show()


def calc_recogn_ratio(confusion_matrix, as_str=False):
    value = np.sum(confusion_matrix.diagonal()) / np.sum(confusion_matrix)
    return value if not as_str else "%s" % int(value*100) + "%"


def write_to_csv(data, file, delimiter=' '):
    if type(data) == list or type(data) == dict:
        if type(data) == dict:
            data = data.items()
        with open(file, 'a') as csvf:
            writer = csv.writer(csvf, delimiter=delimiter)
            for el in data:
                writer.writerow(el)

    # ones_ = [x for x in params_dictionary.values()]
    # labels_ = set(y[1] for x in params_dictionary.values() for y in x)
    # return {k: np.concatenate((np.array([x[0] for x in chain(*ones_) if x[1] == k])),0) for k in labels_}

from python_speech_features import mfcc, delta
from collections import defaultdict, Counter
import numpy as np
import sklearn.mixture as skm
from matplotlib import pyplot as plt
from itertools import product, count, combinations
import csv


def get_mfcc(data, samplerate, appendEnergy=True, delta_=True, deltadelta_=True):
    mfcc_ = mfcc(data, samplerate=samplerate, appendEnergy=appendEnergy)
    if delta_:
        mfcc_delta = delta(mfcc_, 5)
        mfcc_ = np.concatenate((mfcc_, mfcc_delta), 1)
        if deltadelta_:
            mfcc_delta_delta = delta(mfcc_delta, 5)
            mfcc_ = np.concatenate((mfcc_, mfcc_delta_delta), 1)
    return mfcc_


def get_params(data):
    ret = defaultdict(lambda: [])
    for el in data:
        filename = el[0].split('_')[0]
        number = el[0].split('_')[1]
        assert(len(filename) == 5 and len(number) == 1)
        mfcc_ = get_mfcc(el[1], samplerate=el[2])
        ret[filename].append((mfcc_, number))
    return ret


def get_test_data(n_test_examples):
    test_mfcc = np.concatenate((np.random.randn(n_test_examples, 1),
                                np.random.randn(n_test_examples, 1) + 50,
                                np.random.randn(n_test_examples, 1) + 100), 1)
    test_gmm = skm.GaussianMixture(n_components=3, covariance_type='diag', init_params='random', max_iter=20, n_init=20, tol=0.001)
    test_gmm = test_gmm.fit(test_mfcc)
    return test_mfcc, test_gmm


def classificate_mfcc_to_GMM_model(mfcc_matrix, gmm_models, ll=False):
    log_likelihoods = [] 
    for number, model in sorted(gmm_models.items(), key=lambda x: int(x[0])):
        log_likelihoods.append(np.exp(model.score(mfcc_matrix)))
    ll_val = max(log_likelihoods)
    idx = log_likelihoods.index(ll_val)
    return idx if not ll else idx, ll_val


def divide_test(params_dict, n_of_tests_ex=2):
    # change this function to return correct values as it is now
    key_list = list(params_dict.keys())
    print(key_list)
    if n_of_tests_ex == 2:
        from sklearn.model_selection import KFold
        x = [i for i in range(len(key_list))]
        kf = KFold(11)
        rt = [(list(map(lambda z: key_list[z], i)), list(map(lambda z: key_list[z], j))) for i, j in kf.split(x)]
        test_keys = [x[1] for x in rt]
        train_keys = [x[0] for x in rt]

    else:
        # from sklearn.model_selection import train_test_split
        rt = []
        counter = Counter()
        for test_split in combinations(key_list, n_of_tests_ex):
            if len(rt) >= len(key_list):
                break
            # split_ = train_test_split(key_list, test_size=n_of_tests_ex)
            if not any([1 for curr_test_split in rt if len(set(test_split) - set(curr_test_split)) < n_of_tests_ex - 1]):
                counter += Counter(test_split)
                diff = max(counter.values()) - min(counter.values())
                if not any(filter(lambda z: z > n_of_tests_ex, counter.values())) and diff < 2:
                    rt.append(test_split)
                    print(counter)
                else:
                    counter -= Counter(test_split)
            if len(rt) == len(key_list) - 1:  # if only one set of values is required, then fill it in appropriately
                pom = [z[0] for z in counter.items() if z[1] == n_of_tests_ex - 1]
                rt.append([list(set(key_list) - set(pom)), pom])
                counter += Counter(pom)

        print(counter)
        test_keys = rt
        train_keys = [list(set(key_list) - set(x)) for x in test_keys]

    print(test_keys)
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


def write_to_csv(data, file, delimiter=' ', option='w'):
    if type(data) == list or type(data) == dict:
        if type(data) == dict:
            data = data.items()
        with open(file, option) as csvf:
            writer = csv.writer(csvf, delimiter=delimiter)
            if len(np.array(data).shape) == 1:
                writer.writerow(data)
            else:
                for el in data:
                    writer.writerow(el)

    # ones_ = [x for x in params_dictionary.values()]
    # labels_ = set(y[1] for x in params_dictionary.values() for y in x)
    # return {k: np.concatenate((np.array([x[0] for x in chain(*ones_) if x[1] == k])),0) for k in labels_}

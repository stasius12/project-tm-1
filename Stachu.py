from python_speech_features import mfcc, delta
from collections import defaultdict, Counter
import numpy as np
import sklearn.mixture as skm
from matplotlib import pyplot as plt
from itertools import product, count, combinations
from Kuba import validate, get_gmm_models, get_labels_dictionary
from Mati import read_files
import csv
from itertools import zip_longest

def score_evaluate_waves():
    # GMM MODELS
    gmm_models_ = get_gmm_models(get_labels_dictionary(get_params(read_files())), 8, 40)

    # EVALUATION SET
    file_list_ = sorted(read_files('eval'), key=lambda x: x[0])
    mfcc_matrices_for_evaluation_set = {k[0]: get_mfcc(k[1], k[2]) for k in file_list_}
    ret = [(k, ) + classificate_mfcc_to_GMM_model(v, gmm_models_) for k, v in mfcc_matrices_for_evaluation_set.items()]
    write_to_csv(ret, 'results.csv', delimiter=',', option='w')

def cross_validation(params_dict, n_components=8, n_iters=50, cov_type ='diag', n_of_tests_ex=2):
    train_keys, test_keys = divide_test(params_dict, n_of_tests_ex=n_of_tests_ex)
    sum_matrix = np.zeros((10, 10))
    for i in range(len(test_keys)):
        train_set_params = dict((k, params_dict[k]) for k in train_keys[i])
        test_set_params = dict((k, params_dict[k]) for k in test_keys[i])
        #print(train_set_params.keys())
        #print(test_set_params.keys())
        conf_matrix = validate(train_set_params, test_set_params, n_components, n_iters, cov_type)
        #print(conf_matrix)
        #print("Recognition Ratio for test %s: " % i, calc_recogn_ratio(conf_matrix))
        sum_matrix = np.add(sum_matrix, conf_matrix)
    return sum_matrix


def test_optimal_number_of_components():
    ratios = []
    diagonals = []
    for i in range(1,5, 200):
        print("\n ===== %s components =====" % i)
        test_matrix = cross_validation(i, 100)
        print(test_matrix)
        rr = "%.2f" % calc_recogn_ratio(test_matrix)
        print("Recognition Ratio: ", rr)
        ratios.append(rr)
        diagonals.append(list(np.diag(test_matrix)))
    write_to_csv(list(zip_longest(range(1, 21), ratios)), 'test_n_components.csv')
    write_to_csv(diagonals, 'test_n_components.csv')

def get_params(data, winlen = 0.025, numcep = 13, nfilt = 26, nfft = 512, appendEnergy=True, delta_=True, deltadelta_=True):
    ret = defaultdict(lambda: [])
    for el in data:
        filename = el[0].split('_')[0]
        number = el[0].split('_')[1]
        assert(len(filename) == 5 and len(number) == 1)
        dat= el[1]
        sample_rate = el[2]
        mfcc_ = get_mfcc(dat, sample_rate, winlen, numcep, nfilt, nfft, appendEnergy, delta_, deltadelta_ )
        ret[filename].append((mfcc_, number))
    return ret

def get_mfcc(data, samplerate, winlen, numcep, nfilt, nfft, appendEnergy, delta_, deltadelta_):
    mfcc_ = mfcc(data, winlen=winlen, numcep=numcep, nfilt=nfilt, nfft=nfft, samplerate=samplerate, appendEnergy=appendEnergy)
    if delta_:
        mfcc_delta = delta(mfcc_, 5)
        mfcc_ = np.concatenate((mfcc_, mfcc_delta), 1)
        if deltadelta_:
            mfcc_delta_delta = delta(mfcc_delta, 5)
            mfcc_ = np.concatenate((mfcc_, mfcc_delta_delta), 1)
    return mfcc_


def get_test_data(n_test_examples):
    test_mfcc = np.concatenate((np.random.randn(n_test_examples, 1),
                                np.random.randn(n_test_examples, 1) + 50,
                                np.random.randn(n_test_examples, 1) + 100), 1)
    test_gmm = skm.GaussianMixture(n_components=3, covariance_type='diag', init_params='random', max_iter=20, n_init=20, tol=0.001)
    test_gmm = test_gmm.fit(test_mfcc)
    return test_mfcc, test_gmm


def classificate_mfcc_to_GMM_model(mfcc_matrix, gmm_models, ll=True):
    log_likelihoods = [] 
    for number, model in sorted(gmm_models.items(), key=lambda x: int(x[0])):
        log_likelihoods.append(np.exp(model.score(mfcc_matrix)))
    ll_val = max(log_likelihoods)
    idx = log_likelihoods.index(ll_val)
    return idx if not ll else idx, ll_val


def divide_test(params_dict, n_of_tests_ex=2):
    # change this function to return correct values as it is now
    key_list = list(params_dict.keys())
    #print(key_list)
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

# if ELSE:
#     data = []
#     with open('test_n_components.csv', 'r') as csvf:
#         reader = csv.reader(csvf, delimiter=' ', quotechar='|')
#         for row in reader:
#             data.append(row)
#     data = np.array(data[21:])
#     print(data[:, 5])
#     plt.plot(range(1, 21), [float(i) for i in data[:, 9]])
#     plt.xticks(range(1, 21))
#     plt.grid()
#     plt.show()
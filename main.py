# MAIN FILE containing only calls to functions
from Stachu import get_params, get_test_data, divide_test, plot_conf_matrix, calc_recogn_ratio, write_to_csv, get_mfcc, classificate_mfcc_to_GMM_model
from Mati import read_files, divide_data_set_to_cross_validate
from Kuba import get_labels_dictionary, get_gmm_models, show_model_and_data, validate
from useful_funcs import evaluate, load_results, load_keys
import numpy as np
from itertools import zip_longest
import csv
from matplotlib import pyplot as plt


def cross_validation(n_components, n_iters, n_of_tests_ex=2):
    train_keys, test_keys = divide_test(params_dict, n_of_tests_ex=n_of_tests_ex)
    sum_matrix = np.zeros((10, 10))
    for i in range(len(test_keys)):
        train_set_params = dict((k, params_dict[k]) for k in train_keys[i])
        test_set_params = dict((k, params_dict[k]) for k in test_keys[i])
        print(train_set_params.keys())
        print(test_set_params.keys())
        conf_matrix = validate(train_set_params, test_set_params, n_components, n_iters)
        print(conf_matrix)
        print("Recognition Ratio for test %s: " % i, calc_recogn_ratio(conf_matrix))
        sum_matrix = np.add(sum_matrix, conf_matrix)
    return sum_matrix


def test_optimal_number_of_components():
    ratios = []
    diagonals = []
    for i in range(1, 21):
        print("\n ===== %s components =====" % i)
        test_matrix = cross_validation(i, 100)
        print(test_matrix)
        rr = "%.2f" % calc_recogn_ratio(test_matrix)
        print("Recognition Ratio: ", rr)
        ratios.append(rr)
        diagonals.append(list(np.diag(test_matrix)))
    write_to_csv(list(zip_longest(range(1, 21), ratios)), 'test_n_components.csv')
    write_to_csv(diagonals, 'test_n_components.csv')


def score_evaluate_waves():
    # GMM MODELS
    gmm_models_ = get_gmm_models(get_labels_dictionary(get_params(read_files())), 8, 40)

    # EVALUATION SET
    file_list_ = sorted(read_files('eval'), key=lambda x: x[0])
    mfcc_matrices_for_evaluation_set = {k[0]: get_mfcc(k[1], k[2]) for k in file_list_}
    ret = [(k, ) + classificate_mfcc_to_GMM_model(v, gmm_models_) for k, v in mfcc_matrices_for_evaluation_set.items()]
    write_to_csv(ret, 'results.csv', delimiter=',', option='w')

score_evaluate_waves()
evaluate()

TEST = False  # set to True if you want to test it
CROSS_VALIDATE = False
PLOTTING = False
ELSE = False


if CROSS_VALIDATE or TEST:
    """
    -> [(nazwa_pliku, wave, fs),....]
    """
    file_list = read_files()

    """
    -> {'speaker_id': [(mfcc, label), (...) ],
    ...
    }
    """
    params_dict = get_params(file_list)


    """
    -> {'label': macierz_mfcc,
    ...
    }
    """
    labels_dict = get_labels_dictionary(params_dict)
    gmm_models = get_gmm_models(labels_dict, 20, 200)


if CROSS_VALIDATE:
    test_matrix = cross_validation(8, 40, n_of_tests_ex=2)
    print(test_matrix)
    print("Recognition Ratio: ", calc_recogn_ratio(test_matrix))

    np.save('conf_matrix', test_matrix)


if PLOTTING:
    conf_matrix = np.load('conf_matrix.npy')
    plot_conf_matrix(conf_matrix, percentage_vals=True)
    plot_conf_matrix(conf_matrix, percentage_vals=False)


if ELSE:
    data = []
    with open('test_n_components.csv', 'r') as csvf:
        reader = csv.reader(csvf, delimiter=' ', quotechar='|')
        for row in reader:
            data.append(row)
    data = np.array(data[21:])
    print(data[:, 5])
    plt.plot(range(1, 21), [float(i) for i in data[:, 9]])
    plt.xticks(range(1, 21))
    plt.grid()
    plt.show()




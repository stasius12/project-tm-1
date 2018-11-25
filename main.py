# MAIN FILE containing only calls to functions
from Stachu import get_params, get_test_data, divide_test, plot_conf_matrix, calc_recogn_ratio, write_to_csv
from Mati import read_files, divide_data_set_to_cross_validate
from Kuba import get_labels_dictionary, get_gmm_models, show_model_and_data, validate
import numpy as np
from itertools import zip_longest
import csv
from matplotlib import pyplot as plt


def cross_validation(n_components, n_iters):
    train_keys, test_keys = divide_test(params_dict)
    sum_matrix = np.zeros((10, 10))
    for i in range(len(test_keys)):
        train_set_params = dict((k, params_dict[k]) for k in train_keys[i])
        test_set_params = dict((k, params_dict[k]) for k in test_keys[i])
        conf_matrix = validate(train_set_params, test_set_params, n_components, n_iters)
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


TEST = False  # set to True if you want to test it
CROSS_VALIDATE = False
PLOTTING = False
TEST_COMP = False

if CROSS_VALIDATE or TEST or TEST_COMP:
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


if TEST:
    mfcc_, gmm_ = get_test_data(1000)
    labels_dict['test'] = mfcc_
    gmm_models['test'] = gmm_
    print(gmm_.means_)
    which_test = 2  # number from 0 to 2
    show_model_and_data(labels_dict, gmm_models, 'test', which_test, [which_test*50-2, which_test*50+2])

elif CROSS_VALIDATE:
    test_matrix = cross_validation(8, 40)
    print(test_matrix)
    print("Overall Recognition Ratio: ", calc_recogn_ratio(test_matrix))
    np.save('conf_matrix', test_matrix)


if PLOTTING:
    conf_matrix = np.load('conf_matrix.npy')
    plot_conf_matrix(conf_matrix, percentage_vals=False)


if TEST_COMP:
    test_optimal_number_of_components()


else:
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










# CROSS VALIDATION (liczba_testow)
"""
1. Funkcja zwracajaca liste indeksow testowych i treningowych, jako lista tupli par list
{
2. Funkcja przyjmujaca jedna pare trening/test do crosswalidacji
    a)f labels_dict dla modelu treningowego -> gmm_models
    b)testowy_zbior = wyciagnac z get_params testowych mowcow
    c)conf_matrix = inicjalizacja pustej confusion matrix
    d)Iteracja po mowcy z testowy_zbior
        d1) iteracja po cyfrach(labelach)
            - inicjalizacja pustej tablicy prawdopodobienstwa classif_prop = []
            d2) iteracja po modelach
                - obliczenie prawdopodbienstwa classyfikacji prop
                - classif_prop.append(prop)
            e) max_idx = znajdz index maksymalnej wartości z clasif_prop
            f) conf_matrix[label_idx, max_idx] += 1
}     
"""

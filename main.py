# MAIN FILE containing only calls to functions
from Stachu import get_params, get_test_data, classificate_mfcc_to_GMM_model
from Mati import read_files
from Kuba import get_labels_dictionary, get_gmm_models, show_model_and_data, validate, calc_recogn_ratio

TEST = False # set to True if you want to test it

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

# funkcja ktora przyjmuje params i zwraca tylko TRENINGOWE labels
"""
-> {'label': macierz_mfcc,
    ...
    }
"""
labels_dict = get_labels_dictionary(params_dict)
gmm_models = get_gmm_models(labels_dict, 20, 200)


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


if TEST:
    mfcc_, gmm_ = get_test_data(1000)
    labels_dict['test'] = mfcc_
    gmm_models['test'] = gmm_
    print(gmm_.means_)
    which_test = 2 # number from 0 to 2
    show_model_and_data(labels_dict, gmm_models, 'test', which_test, [which_test*50-2, which_test*50+2])
else:
   # show_model_and_data(labels_dict, gmm_models, '9', 0, [-100, 100])
   #  for i in [str(i) for i in range(10)]:
   #      print('Number %s classificate as: %s' % (i, classificate_mfcc_to_GMM_model(labels_dict[i], gmm_models)))
    key_list = list(params_dict.keys())
    train_keys = key_list[0:17]
    test_keys = key_list[17:23]
    train_set_params = dict((k, params_dict[k]) for k in train_keys if k in params_dict)
    test_set_params = dict((k, params_dict[k]) for k in test_keys if k in params_dict)
    conf_matrix = validate(train_set_params, test_set_params, 8, 200)
    print(conf_matrix)
    print("Recgnition ratio: ", calc_recogn_ratio(conf_matrix))
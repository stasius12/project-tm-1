# MAIN FILE containing only calls to functions
from Stachu import get_params, get_test_data, classificate_mfcc_to_GMM_model
from Mati import read_files
from Kuba import get_labels_dictionary, get_gmm_models,show_model_and_data

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
    c)Iteracja po mowcy z testowy_zbior
        1. Dla kazdej liczby:
            a) wektor bledu -> zwraca numer cyfry
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
    for i in [str(i) for i in range(10)]:
        print('Number %s classificate as: %s' % (i, classificate_mfcc_to_GMM_model(labels_dict[i], gmm_models)))

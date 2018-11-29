from scipy.io import wavfile
import os
import os.path
import itertools


def read_files(path_='waves'):
    matrix_of_names = os.listdir(path_)
    matrix_of_param = []
    for el in matrix_of_names:
        freq, data = wavfile.read(path_ + '/%s' % el)
        matrix_of_param.append((el, data, freq))
    return matrix_of_param


def divide_data_set_to_cross_validate(params_dict,number_of_pairs):
    # change this function to return correct values as it is now

    key_list = list(params_dict.keys())
    print("Key list :", key_list)
    train_keys = list()
    test_keys = list()
    j = 0
    for i in list(itertools.permutations(params_dict.keys(),number_of_pairs)):

        if j > number_of_pairs:
            break
        else:
            train_keys.append(i)
            j = j + 1

    for i in len(key_list):

        test_keys.append(list(set(key_list) - set(train_keys[i])))


    print("Train keys", train_keys)
    print("Test keys", test_keys)


    train_set_params = dict((k,params_dict[k]) for k in train_keys if k in params_dict)
    test_set_params = dict((k, params_dict[k]) for k in test_keys if k in params_dict)
    return train_set_params, test_set_params

from scipy.io import wavfile
import os
import os.path


def read_files():
    matrix_of_names = os.listdir('waves')
    matrix_of_param = []
    for el in matrix_of_names:
        freq, data = wavfile.read('waves/%s' % el)
        matrix_of_param.append((el, data, freq))
    return matrix_of_param


def divide_data_set_to_cross_validate(params_dict):
    # change this function to return correct values as it is now
    key_list = list(params_dict.keys())
    train_keys = key_list[0:17]
    test_keys = key_list[17:23]
    train_set_params = dict((k, params_dict[k]) for k in train_keys if k in params_dict)
    test_set_params = dict((k, params_dict[k]) for k in test_keys if k in params_dict)
    return train_set_params, test_set_params

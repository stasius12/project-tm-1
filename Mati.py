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

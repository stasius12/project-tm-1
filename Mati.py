import wave
import os
import os.path


def read_files():
    matrix_of_names = os.listdir('waves')
    matrix_of_param = []
    for el in matrix_of_names:
        data = wave.open('waves/%s' % el)
        freq = data.getframerate()
        matrix_of_param.append((el, data, freq))
    return matrix_of_param

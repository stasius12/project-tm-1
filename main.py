# MAIN FILE containing only calls to functions
from Stachu import get_params, plot_conf_matrix, calc_recogn_ratio, cross_validation, score_evaluate_waves
from Mati import read_files
from useful_funcs import evaluate

#MFCC params
WINLEN = 0.025
NUMCEP = 13
NFILT = 26
NFFT = 512
APPEND_ENERGY = True
DELTA = True
DELTA_DELTA = True

#GMM parms
N_COMPONENTS = 8
N_ITERS = 100
COV_TYPE ='diag'

file_list = read_files()
params_dict = get_params(file_list, WINLEN, NUMCEP, NFILT, NFFT, APPEND_ENERGY, DELTA, DELTA_DELTA)

OPTION = 0 # 0 - cross validation, 1 - evaluation, 2 - tests
if OPTION == 0:
    conf_matrix = cross_validation(params_dict, N_COMPONENTS, N_ITERS, COV_TYPE,  n_of_tests_ex=2)
    print(conf_matrix)
    print("Recognition Ratio: ", calc_recogn_ratio(conf_matrix))
    plot_conf_matrix(conf_matrix, percentage_vals=True)
    plot_conf_matrix(conf_matrix, percentage_vals=False)
elif OPTION == 1:
    score_evaluate_waves()
    evaluate()

elif OPTION == 2:
    from tests import  test_winlen, test_numcep, test_nfilt, test_nfft, test_ncomponents, test_niters
    test_winlen(file_list, 'window_length_tests.csv')
    test_numcep(file_list, 'num_cepstrum_tests.csv')
    test_nfilt(file_list, 'nfilt_tests.csv')
    test_ncomponents(file_list, 'ncomponents_tests2.csv')
    test_nfft(file_list, 'nfft_tests.csv')
    test_niters(file_list, 'niters_tests.csv')
    #file_list = read_files() -> [(nazwa_pliku, wave, fs),....]
    #params_dict = get_params(file_list)-> {'speaker_id': [(mfcc, label), (...) ],
    #get_mfcc -> {'label': macierz_mfcc,









from Stachu import get_params, cross_validation, calc_recogn_ratio, write_to_csv
import os.path

def test_winlen(file_list, csv_file_name):
    if not os.path.isfile(csv_file_name):
        print('-------WIN LENGTH TESTING -------')
        winLens = [0.01, 0.025, 0.05, 0.07, 0.1, 0.2, 0.25, 0.3, 0.035, 0.4, 0.45, 0.5, 0.55, 0.6]
        output = {}
        for winLen in winLens:
            params_dict = get_params(file_list, winlen=winLen, nfft=2048)
            confusion_matrix = cross_validation(params_dict)
            recogn_ratio = calc_recogn_ratio(confusion_matrix)
            output[winLen] = recogn_ratio
        write_to_csv(output, csv_file_name)

def test_numcep(file_list, csv_file_name):
    if not os.path.isfile(csv_file_name):
        print('-------NUMBER OF CEPSTRUMS TESTING -------')
        cep_nums = range(4,14)
        output = {}
        for cep_num in cep_nums:
            print(cep_num, ' cepstrums testing ...')
            params_dict = get_params(file_list, numcep=cep_num,)
            confusion_matrix = cross_validation(params_dict)
            recogn_ratio = calc_recogn_ratio(confusion_matrix)
            output[cep_num] = recogn_ratio
        write_to_csv(output, csv_file_name)
        print('-------NUMBER OF CEPSTRUMS TESTING FINISHED----')

def test_nfilt(file_list, csv_file_name):
    if not os.path.isfile(csv_file_name):
        print('-------NUMBER OF FILTERS TESTING -------')
        filter_nums = range(5, 26)
        output = {}
        for filter_num in filter_nums:
            print(filter_num, ' nfilt testing ...')
            params_dict = get_params(file_list, nfilt=filter_num)
            confusion_matrix = cross_validation(params_dict)
            recogn_ratio = calc_recogn_ratio(confusion_matrix)
            output[filter_num] = recogn_ratio
        write_to_csv(output, csv_file_name)
        print('-------NUMBER OF FILTERS TESTING FINISHED----')

def test_nfft(file_list, csv_file_name):
    if not os.path.isfile(csv_file_name):
        print('-------NUMBER OF NFFT TESTING-------')
        nfft_nums = [64, 128, 256, 512, 1024, 2048, 4096]
        output = {}
        for nfft_num in nfft_nums:
            print(nfft_num, ' nfft testing ...')
            params_dict = get_params(file_list, nfft=nfft_num)
            confusion_matrix = cross_validation(params_dict)
            recogn_ratio = calc_recogn_ratio(confusion_matrix)
            output[nfft_num] = recogn_ratio
        write_to_csv(output, csv_file_name)
        print('-------NFFT TESTING FINISHED----')

def test_ncomponents(file_list, csv_file_name):
    if not os.path.isfile(csv_file_name):
        print('-------NUMBER OF COMPONENTS TESTING-------')
        ncomponents_nums = range(1, 20)
        output = {}
        for ncomponents_num in ncomponents_nums:
            print(ncomponents_num, ' components testing ...')
            params_dict = get_params(file_list)
            confusion_matrix = cross_validation(params_dict, n_components=ncomponents_num)
            recogn_ratio = calc_recogn_ratio(confusion_matrix)
            output[ncomponents_num] = recogn_ratio
        write_to_csv(output, csv_file_name)
        print('-------NUMBER OF COMPONENTS TESTING FINISHED----')

def test_niters(file_list, csv_file_name):
    if not os.path.isfile(csv_file_name):
        print('-------NUMBER OF ITERATIONS TESTING-------')
        n_iter_nums = range(1, 20)
        output = {}
        for n_iter_num in n_iter_nums:
            print(n_iter_num, ' n_iters testing ...')
            params_dict = get_params(file_list)
            confusion_matrix = cross_validation(params_dict, n_iters=n_iter_num)
            recogn_ratio = calc_recogn_ratio(confusion_matrix)
            output[n_iter_num] = recogn_ratio
        write_to_csv(output, csv_file_name)
        print('-------NUMBER OF ITERATIONS TESTING FINISHED----')
# MAIN FILE containing only calls to functions
from Stachu import get_params, get_test_data
from Mati import read_files
from Kuba import get_labels_dictionary, get_gmm_models,show_model_and_data

TEST = False # set to True if you want to test it

file_list= read_files()
params_dict = get_params(file_list)
labels_dict = get_labels_dictionary(params_dict)
gmm_models = get_gmm_models(labels_dict, 20, 200)
if TEST:
    mfcc_, gmm_ = get_test_data(1000)
    labels_dict['test'] = mfcc_
    gmm_models['test'] = gmm_
    print(gmm_.means_)
    which_test = 2 # number from 0 to 2
    show_model_and_data(labels_dict, gmm_models, 'test', which_test, [which_test*50-2, which_test*50+2])
else:
    show_model_and_data(labels_dict, gmm_models, '9', 1, [0, 30])

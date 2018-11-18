# MAIN FILE containing only calls to functions
from Stachu import get_params
from Mati import read_files
from Kuba import get_labels_dictionary
from Kuba import get_gmm_models
from Kuba import show_model_and_data

file_list= read_files()
params_dict = get_params(file_list)
labels_dict = get_labels_dictionary(params_dict)
gmm_models = get_gmm_models(labels_dict, 20, 200)
show_model_and_data(labels_dict, gmm_models, '8', 1, [-100, 100])
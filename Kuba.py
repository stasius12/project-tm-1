import matplotlib.pyplot as plt
import numpy as np
import sklearn.mixture as skm


def get_gmm_models(labels_dictionary, n, n_iter):
    gmm_models = {}
    for curr_label in labels_dictionary:
        gmm_obj = skm.GaussianMixture(n_components=n, covariance_type='diag', init_params='random', max_iter=n_iter, n_init=20, tol=0.001)
        gmm_obj.fit(labels_dictionary[curr_label])
        gmm_models[curr_label] = gmm_obj
    return gmm_models


def get_labels_dictionary(params_dictionary):
    output_dictionary = {}
    for speaker_id in params_dictionary:
        tuple_list = params_dictionary[speaker_id]
        for tup in tuple_list:
            mfcc = tup[0]
            label = tup[1]
            output_dictionary = concat_mfcc(output_dictionary, label, mfcc)
    return output_dictionary


def concat_mfcc(output_dictionary, label, mfcc):
    if label in output_dictionary:
        mfcc_old = output_dictionary[label]
        mfcc_new = np.concatenate((mfcc_old, mfcc))
        output_dictionary[label] = mfcc_new
        return output_dictionary
    else:
        output_dictionary[label] = mfcc
        return output_dictionary


def show_model_and_data(mfcc_atributes, gmm_models, label, attribute_num, xlim):
    mfcc_matrix = mfcc_atributes[label]
    gmm_obj = gmm_models[label]
    means = gmm_obj.means_[:, attribute_num]
    covs = gmm_obj.covariances_[:, attribute_num]
    weights = gmm_obj.weights_

    fig = plt.figure()
    subplot = fig.add_subplot(111)
    title = 'Liczba: %s, Cecha: C%s' % (label, attribute_num)
    subplot.set_title(title)
    subplot.set_xlabel('wartosc atrybutu [-]')
    subplot.set_ylabel('prawdopodobienstwo wystapienia [-]')

    bins = np.linspace(xlim[0], xlim[1], 100)
    x = np.linspace(xlim[0], xlim[1], 1000)
    n, bins, patches = subplot.hist(mfcc_matrix[:, attribute_num], bins, density=True)
    max_count = max(n)

    dist_sum = np.array([])
    for component in range(0, len(means)):
        mean = means[component]
        cov = covs[component]
        weight = weights[component]
        dist = gaussian(x, mean, cov) * max_count * weight
        if dist_sum.size == 0:
            dist_sum = dist
        else:
            dist_sum = dist + dist_sum

        subplot.plot(x, dist)
    subplot.plot(x, dist_sum)
    plt.show()

def gaussian(x, mu, sig):
    factor = 1
    return factor * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

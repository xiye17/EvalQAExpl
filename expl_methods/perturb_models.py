import numpy as np
import scipy as sp
import sklearn
from sklearn.linear_model import Ridge, LinearRegression
from scipy.special import binom
from itertools import combinations

## modified from original lime implementation
def lime_kernel(d, kernel_width=25):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

def lime_feature_selection(data, labels, weights, num_features, method):
    clf = Ridge(alpha=0.01, fit_intercept=True)
    clf.fit(data, labels, sample_weight=weights)

    coef = clf.coef_
    if sp.sparse.issparse(data):
        coef = sp.sparse.csr_matrix(clf.coef_)
        weighted_data = coef.multiply(data[0])
        # Note: most efficient to slice the data before reversing
        sdata = len(weighted_data.data)
        argsort_data = np.abs(weighted_data.data).argsort()
        # Edge case where data is more sparse than requested number of feature importances
        # In that case, we just pad with zero-valued features
        if sdata < num_features:
            nnz_indexes = argsort_data[::-1]
            indices = weighted_data.indices[nnz_indexes]
            num_to_pad = num_features - sdata
            indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
            indices_set = set(indices)
            pad_counter = 0
            for i in range(data.shape[1]):
                if i not in indices_set:
                    indices[pad_counter + sdata] = i
                    pad_counter += 1
                    if pad_counter >= num_to_pad:
                        break
        else:
            nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
            indices = weighted_data.indices[nnz_indexes]
        return indices
    else:
        weighted_data = coef * data[0]
        feature_weights = sorted(
            zip(range(data.shape[1]), weighted_data),
            key=lambda x: np.abs(x[1]),
            reverse=True)
        return np.array([x[0] for x in feature_weights[:num_features]])

def lime_explain_instance_with_data(neighborhood_data, neighborhood_labels, distances, feature_selection='auto'):
    weights = lime_kernel(distances)
    labels_column = neighborhood_labels
    # used_features = self.feature_selection(neighborhood_data, labels_column, weights, num_features, feature_selection)
    model_regressor = Ridge(alpha=1, fit_intercept=True)
    easy_model = model_regressor
    easy_model.fit(neighborhood_data, labels_column, sample_weight=weights)
    return easy_model.coef_    

def lime_feat_labels_distances(doc_size, classifier_fn, num_samples=5000, distance_metric='cosine'):

    def distance_fn(x):
        return sklearn.metrics.pairwise.pairwise_distances(
            x, x[0], metric=distance_metric).ravel() * 100

    sample = np.random.randint(1, doc_size + 1, num_samples - 1)
    data = np.ones((num_samples, doc_size))
    data[0] = np.ones(doc_size, dtype=np.int64)
    features_range = range(doc_size)
    for i, size in enumerate(sample, start=1):
        inactive = np.random.choice(features_range, size, replace=False)
        data[i, inactive] = 0
    
    labels = np.array([classifier_fn(d) for d in data])
    distances = distance_fn(sp.sparse.csr_matrix(data))
    return data, labels, distances

def run_lime_attribution(args, doc_size, classifier_fn):
    data, labels, distances = lime_feat_labels_distances(doc_size, classifier_fn)
    return lime_explain_instance_with_data(data, labels, distances)


## modified from shap lime implementation

# hyper-paremeter inherited from oringinal shap implementation
def shap_feat_label_weights(doc_size, classifier_fn, verbose=False):
    # not working for small seq for now, needs more complex way
    num_sample = 2 * doc_size + 2 ** 11
    if num_sample > (2 ** doc_size - 2):
        num_sample = (2 ** doc_size - 2)

    if verbose:
        print('Doc size', doc_size, 'Num sample', num_sample)

    num_included = 0
    subset_size = 1
    data = []
    kernel_weights = []
    
    features_idx = np.arange(doc_size)
    while True:
        num_left = num_sample - num_included
        num_sample_of_size = int(binom(doc_size, subset_size))
        if not (2 * subset_size == doc_size):
            num_sample_of_size *= 2
        # not able to contain all of this size
        if num_left < num_sample_of_size:    
            if verbose:
                print('for size', subset_size, 'needed', num_sample_of_size, 'remaining', num_left)
            break
        if verbose:
            print('including everything of', subset_size)
        weight_of_size = (doc_size - 1) / (subset_size * (doc_size - subset_size)) / binom(doc_size, subset_size)
        # add all combination of this size
        for inds in combinations(features_idx, subset_size):
            pos_mask = np.ones(doc_size, dtype=np.int64)
            pos_mask[np.array(inds, dtype=np.int64)] = 0
            data.append(pos_mask)
            kernel_weights.append(weight_of_size)
            if not (2 * subset_size == doc_size):
                neg_mask = 1 - pos_mask
                data.append(neg_mask)
                kernel_weights.append(weight_of_size)
        subset_size += 1
        num_included += num_sample_of_size

    num_fixed_data = num_included
    unincluded_size = subset_size
    num_left = num_sample - num_included
    if num_left > 0:
        remaining_possible_size = list(range(unincluded_size, doc_size + 1 - unincluded_size))
        remaining_possible_weight = np.array([(doc_size - 1) / (s * (doc_size - s)) for s in remaining_possible_size])
        remaining_sample_weight = remaining_possible_weight.sum() / num_left
        if verbose:
            print('already included', num_fixed_data)
            print('remaining size to sample from', remaining_possible_size)
            print('with the following weight', remaining_possible_weight)
        
        # normalize sampling rate to 1
        remaining_possible_weight = remaining_possible_weight / remaining_possible_weight.sum()
        size_sample = np.random.choice(remaining_possible_size, int(num_left / 2), p=remaining_possible_weight)
        
        for size in size_sample:
            selected = np.random.choice(features_idx, size, replace=False)
            pos_mask = np.ones(doc_size, dtype=np.int64)
            pos_mask[np.array(selected, dtype=np.int64)] = 0
            neg_mask = 1 - pos_mask
            data.append(pos_mask)
            data.append(neg_mask)
            kernel_weights.append(remaining_sample_weight)
            kernel_weights.append(remaining_sample_weight)
        
        if verbose:
            print('sum of fixed weights', sum(kernel_weights[:num_fixed_data]))
            print('theoratical sum of fixed weights',
                    2 * sum([(doc_size - 1) / (s * (doc_size - s)) for s in range(1, unincluded_size)]))
            print('sum of randomly sampled weights', sum(kernel_weights[num_fixed_data:]))
            print('theratical sum of randomly sampled weights',
                    sum([(doc_size - 1) / (s * (doc_size - s)) for s in remaining_possible_size]))

    data = np.stack(data)
    labels = np.array([classifier_fn(d) for d in data])
    kernel_weights = np.array(kernel_weights)
    return data, labels, kernel_weights


def shap_explain_instance_with_data(data, labels, weights):
    model_regressor = LinearRegression(fit_intercept=False)    
    model_regressor.fit(data, labels, sample_weight=weights)
    return model_regressor.coef_    

def run_shap_attribution(args, doc_size, classifier_fn):
    data, labels, weights = shap_feat_label_weights(doc_size, classifier_fn)
    # print(data.shape, labels.shape, weights.shape)
    return shap_explain_instance_with_data(data, labels, weights)

if __name__=='__main__':
    dummy_fn = lambda x: np.sum(x)
    # run_shap_attribution(None, 4, dummy_fn)
    # run_shap_attribution(None, 10, dummy_fn)
    # run_shap_attribution(None, 20, dummy_fn)
    val = run_shap_attribution(None, 100, dummy_fn)
    print(val)
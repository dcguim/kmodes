import pandas as pd
import numpy as np
import unittest
from kmodes.util.dissim import matching_dissim, ng_dissim
from kmodes.kprototypes import KPrototypes
from kmodes.util.init_methods import init_cao
from kmodes.util import encode_features

data_dtypes = {
    0: 'float64',
    1: 'float64',
    2: str,
    3: str
    }


def generate_test_data():
    """Generate mixed data with 3 features each with a probability 1-p, of being
    missing. This function is for documentation-only purposes on how the data
    was generated - the data found in the 2num2cat_missing.csv file should be
    used as is within test methods repeatability of results"""
    seed = 123
    for p in range(5, 11):
        generate_test_dataset(p/10, seed)

def generate_test_dataset(p, seed=None):
    # generate numerical data
    mean1 = [2, 8]
    cov1 = [[0.5, 2], [0, 1]]
    x1, y1 = np.random.default_rng(seed).multivariate_normal(mean1, cov1, 300).T
    mean2 = [-1, 3]
    cov2 = [[0.2, 0], [0.5, 2]]
    x2, y2 = np.random.default_rng(seed).multivariate_normal(mean2, cov2, 300).T
    mean3 = [2, 5]
    cov3 = [[2, 1.6], [2, 1]]
    x3, y3 = np.random.default_rng(seed).multivariate_normal(mean3, cov3, 300).T
    # generate categorical data
    np.random.seed(seed)
    cat1 = np.random.binomial(2, 0.95, 300)
    cat1 = pd.Series(cat1).replace({2: 'red', 1: 'purple', 0: 'blue'})
    cat2 = np.random.binomial(2, 0.9, 300)
    cat2 = pd.Series(cat2).replace({2: 'purple', 1: 'blue', 0: 'red'})
    cat3 = np.random.binomial(2, 0.85, 300)
    cat3 = pd.Series(cat3).replace({2: 'blue', 1: 'purple', 0: 'red'})
    cat_rnd_cs = pd.concat([cat1, cat2, cat3], axis=0)
    # generating another noisy categorical data
    cat1 = pd.Series(f'value_{v}' for v in np.random.binomial(30, 0.2, 300)).T
    cat2 = pd.Series(f'value_{v}' for v in np.random.binomial(30, 0.2, 300)).T
    cat3 = pd.Series(f'value_{v}' for v in np.random.binomial(30, 0.2, 300)).T
    cat_noisy = pd.concat([cat1, cat2, cat3], axis=0)
    # putting it all together
    xs = pd.concat([pd.Series(x1), pd.Series(x2), pd.Series(x3)], axis=0)
    ys = pd.concat([pd.Series(y1), pd.Series(y2), pd.Series(y3)], axis=0)
    X = pd.concat([xs, ys, cat_rnd_cs, cat_noisy], axis=1)
    def gen_random_rem_obs(p):
        rem_obs = np.random.binomial(1, p, size=X.shape[0])
        return pd.Series(rem_obs).map({0: True, 1: False})
    rem_obs = gen_random_rem_obs(p)
    X.loc[rem_obs, 0] = np.nan
    rem_obs = gen_random_rem_obs(p)
    X.loc[rem_obs, 1] = np.nan
    rem_obs = gen_random_rem_obs(p)
    # even if the data data might have differnt ways of represent missing data,
    # it is the developers responsibility to convert them to np.nan
    X.loc[rem_obs, 2] = ''
    rem_obs = gen_random_rem_obs(p)
    X.loc[rem_obs, 3] = ''
    X.to_csv(f'kmodes/tests/data/2num2cat_missing_{1-p:.1f}.csv',
             sep=';', index=False, encoding='utf-8')


def read_test_data(p):
    """Read the data from csv file, it supports data with 10, 20, 30, 40 and
    50% of missing probabilities. The arg `p` behaves like in
    `generate_test_data(p)`"""
    Xdf = pd.read_csv(f'kmodes/tests/data/2num2cat_missing_{1-p:.1f}.csv',
                      sep=';', encoding='utf-8')
    Xdf.dropna(axis=0, how='all', inplace=True)
    return Xdf.values


class TestKProtoTypesMissing(unittest.TestCase):

    def test_encode_feature(self):
        Xmiss50prec = read_test_data(0.5)
        categorical = [2, 3]
        Xcat = np.asanyarray(Xmiss50prec[:, categorical])
        Xcat_encoded, _ = encode_features(Xcat)
        expected = pd.read_csv('kmodes/tests/data/expected_test_encoded_features.csv',
                               sep=';', encoding='utf-8',
                               dtype={0: 'int32', 1: 'int32'}).values
        for v in range(len(categorical)):
            np.testing.assert_array_equal(Xcat_encoded[:, v], expected[:, v])

    def test_init_cao(self):
        for diss in [ng_dissim, matching_dissim]:
            n_clusters = 3
            for p in range(5, 10):
                X = read_test_data(p/10)
                categorical = [2, 3]
                Xcat = np.asanyarray(X[:, categorical])
                Xcat, _ = encode_features(Xcat)
                centroids = init_cao(Xcat, n_clusters=n_clusters, dissim=diss)
                print(centroids)
                for c in range(n_clusters):
                    for v in range(len(categorical)):
                        self.assertTrue(not np.isnan(centroids[c][v]))
                        self.assertTrue(centroids[c][v] != -1)

    def test_kprototype_not_provide_missing_flag(self):
        kprot = KPrototypes(n_clusters=3, max_iter=10, n_jobs=1)
        for p in range(5, 10):
            # notice there are nans in the input data X
            X = read_test_data(p/10)
            categorical = [2, 3]
            with self.assertRaises(ValueError):
                kprot.fit(X,
                          categorical=categorical,
                          missing_obs=False)

    def test_kprototype_single(self):
        kprot = KPrototypes(n_clusters=3, max_iter=10, n_jobs=1)
        for p in range(5, 10):
            X = read_test_data(p/10)
            categorical = [2, 3]
            kprot_fit = kprot.fit(X, categorical=categorical, missing_obs=True)
            ## kprot_fit.cluster_centroids_


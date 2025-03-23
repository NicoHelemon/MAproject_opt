import numpy as np
from itertools import product
from scipy.sparse.linalg import eigsh
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score

def empirical_B_C(A, Z, K):
	B, C = np.zeros((K, K)), np.zeros((K, K))

	t_idx = np.triu_indices_from(A, k=1)
	Z_row, Z_col = Z[t_idx[0]], Z[t_idx[1]]

	for k, l in product(range(K), repeat=2):
		mask = (Z_row == k) & (Z_col == l) | (Z_row == l) & (Z_col == k)
		block_values = A[t_idx][mask].flatten()

		if block_values.size:
			B[k, l] = block_values.mean()
			C[k, l] = block_values.var(ddof = block_values.size > 1)

	return B, C

def spectral_embedding(A, d=2, mode = 'sqrt-sclaled'):
	vals, vecs = eigsh(A, k=d, which='LM')
	if mode == 'sqrt-scaled':
		return vecs * np.sqrt(np.abs(vals))
	elif mode == 'scaled':
		return vecs * np.abs(vals)
	else:
		return vecs

def fit_GMM(X):
	gmm = GaussianMixture(n_components=2, covariance_type='full').fit(X)
	return gmm.predict(X), gmm.means_, gmm.covariances_

def rand_index(Z_true, Z_pred):
	return adjusted_rand_score(Z_true, Z_pred)

def label_permutation(Z_true, Z_pred):
	matches_no_swap = np.sum(Z_true == Z_pred)
	matches_swap = np.sum(Z_true == (1 - Z_pred))
	if matches_swap > matches_no_swap:
		return 1 - Z_pred
	else:
		return Z_pred

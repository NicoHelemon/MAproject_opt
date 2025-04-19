import numpy as np

from .Transformations import *
from .WSBM import *

from scipy.sparse.linalg import eigsh
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score

from itertools import permutations, product

class TWSBMInstance:
	def __init__(self, model = None, transformation = None, 
			  A = None, Z = None, seed = None, emb_mode = 'sqrt-scaled'):
		self.model = None
		self.model_name = None
		self.transformation = None
		self.transform_name = None
		self.A, self.Z = None, None
		self.X = None
		self.Z_hat, self.M, self.Σ = None, None, None
		self.C_true, self.C_graph, self.C_embedding = None, None, None
		self.RAND = None

		if model is None and A is None:
			raise ValueError("At least one of model, A must be provided")

		self.model = model
		if transformation is None: transformation = IdentityTransform()
		self.transformation = transformation
		if A is None and Z is None:
			A, Z = model(seed = seed)
			self.A, self.Z = transformation(A), Z
		else:
			self.A, self.Z = A, Z

		self.model_name = model.name
		self.transform_name = transformation.name

		self.X = self.__spectral_embedding(self.A, mode = emb_mode)
		self.Z_hat, self.M, self.Σ = self.__fit_GMM(self.X)

		if self.model is not None:
			B, C = self.model.theoretical_B_C(self.transformation)
			self.C_true = self.__chernoff_information_graph(B, C, self.model.Π)

		B_hat, C_hat, Π_hat = self.__empirical_B_C(self.A, self.Z_hat)
		self.C_graph = self.__chernoff_information_graph(B_hat, C_hat, Π_hat)

		self.C_embedding = self.__chernoff_information_embedding(self.M, self.Σ, len(self.Z_hat))

		self.RAND = adjusted_rand_score(self.Z, self.Z_hat)

		"""
		print(f"ρ = {self.model.ρ}, Π = {self.model.Π.ravel()} {self.model_name}, T = {self.transform_name}")

		def MSE(U, U_hat):
			U_hat_anti_T = np.rot90(U_hat, 2).T
			return 1 / 3 * min(np.sum(np.triu((U - U_hat)**2)), np.sum(np.triu((U - U_hat_anti_T)**2)))

		B_hat_Z_true, C_hat_Z_true, _ = self.__empirical_B_C(self.A, self.Z)

		print(f"MSE B = {MSE(B, B_hat_Z_true):.6f}")
		print(f"MSE C = {MSE(C, C_hat_Z_true):.6f}")"""

	"""
	def __empirical_B_C(self, A, Z):
		K = len(np.unique(Z))
		B, C, Π = np.zeros((K, K)), np.zeros((K, K)), np.zeros((K, K))

		t_idx = np.triu_indices_from(A, k=1)
		Z_row, Z_col = Z[t_idx[0]], Z[t_idx[1]]

		for k, l in product(range(K), repeat=2):
			mask = (Z_row == k) & (Z_col == l) | (Z_row == l) & (Z_col == k)
			block_values = A[t_idx][mask].flatten()

			if k == l:
				Π[k, l] = np.count_nonzero(Z == k) / len(Z)

			if block_values.size:
				B[k, l] = block_values.mean()
				C[k, l] = block_values.var(ddof = block_values.size > 1)

		return B, C, Π"""
	
	def __empirical_B_C(self, A, Z):
		K     = len(np.unique(Z))
		nodes = np.bincount(Z, minlength=K)
		edges = nodes[:, None] * nodes[None, :] - np.diag(nodes)
		Π     = np.diag(nodes / len(Z))

		H  = np.eye(K)[Z]
		S1 = H.T @ A @ H
		S2 = H.T @ A**2 @ H

		with np.errstate(divide='ignore', invalid='ignore'):
			B = S1 / edges
			C = (S2 - B * S1) / (edges - 1)
			B = np.nan_to_num(B)
			C = np.nan_to_num(C)
		return B, C, Π

	def __spectral_embedding(self, A, mode, d=2):
		vals, vecs = eigsh(A.astype(np.float32), k=d, which='LM')
		if mode == 'sqrt-scaled':
			return vecs * np.sqrt(np.abs(vals))
		elif mode == 'scaled':
			return vecs * np.abs(vals)
		elif mode == 'raw':
			return vecs
		else:
			raise ValueError(f"Unknown embedding mode: {mode}")

	def __fit_GMM(self, X):
		gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42).fit(X)
		return gmm.predict(X), gmm.means_, gmm.covariances_
	
	def __chernoff_information_graph(self, B, C, Π):	
		K = B.shape[0]
		e = np.eye(K)
		if np.prod(np.diag(C)) == 0:
			return 0

		def objective(t, k, l):
			S_kl_t = (1 - t) * np.diag(C[k]) + t * np.diag(C[l])
			ek_el = (e[k] - e[l])[:, None]
			matrix_res = ek_el.T @ B @ Π @ np.linalg.inv(S_kl_t) @ B @ ek_el
			return 0.5 * t * (1 - t) * matrix_res.item()
		
		def neg_objective(t, k, l):
			return -objective(t, k, l)
		
		c = np.inf
		for k, l in permutations(range(K), 2):
			res = minimize_scalar(neg_objective, bounds=(0, 1), method='bounded', args=(k, l))
			c = min(c, -res.fun)

		return c

	def __chernoff_information_embedding(self, X, Σ, n):
		K = X.shape[0]

		def objective(t, k, l):
			Σ_kl_t = (1 - t) * Σ[k] + t * Σ[l]
			xk_xl = X[k] - X[l]
			matrix_res = xk_xl.T @ np.linalg.inv(Σ_kl_t) @ xk_xl
			return 0.5 * t * (1 - t) * matrix_res.item()
		
		def neg_objective(t, k, l):
			return -objective(t, k, l)
		
		c = np.inf
		for k, l in permutations(range(K), 2):
			res = minimize_scalar(neg_objective, bounds=(0, 1), method='bounded', args=(k, l))
			c = min(c, -res.fun)

		return c / n
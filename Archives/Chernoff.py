import numpy as np
from scipy.optimize import minimize_scalar
from itertools import permutations

def chernoff_information_graph(B, C, Π):	
	K = B.shape[0]
	e = np.eye(K)
	C += np.finfo(float).eps * np.eye(K)

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

def chernoff_information_embedding(X, Σ, n):
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

	return c
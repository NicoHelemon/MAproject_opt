from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm

# Abstract class for weight transformations
class WeightTransform(ABC):
	@abstractmethod
	def __call__(self, A):
		# Apply the transformation to matrix A.
		pass

# Identity transformation (no changes to A)
class IdentityTransform(WeightTransform):
	def __init__(self):
		self.name = "Identity"

	def __call__(self, A):
		return A

# Opposite transformation (1 - A)
class OppositeTransform(WeightTransform):
	def __init__(self):
		self.name = "Opposite"

	def __call__(self, A):
		A = 1 - A
		np.fill_diagonal(A, 0)
		return A

# Logarithmic transformation (-log(A))
class LogTransform(WeightTransform):
	def __init__(self):
		self.name = "Logarithmic"

	def __call__(self, A):
		A = np.clip(A, np.finfo(A.dtype).eps, None)
		A = -np.log(A)
		np.fill_diagonal(A, 0)
		return A

# Threshold transformation (binary thresholding)
class ThresholdTransform(WeightTransform):
	def __init__(self, τ):
		self.name = f"Threshold (τ = {τ})"
		self.τ = τ

	def __call__(self, A):
		A = (A <= self.τ).astype(float)
		np.fill_diagonal(A, 0)
		return A

# Rank transformation (normalized ranking of upper-triangular values)
class RankTransform(WeightTransform):
	def __init__(self):
		self.name = "Rank"

	def __call__(self, A):
		n = A.shape[0]
		N = n * (n - 1) // 2

		# Extract upper triangular values
		iu = np.triu_indices(n, k=1)
		upper_vals = A[iu]

		# Compute normalized ranks
		normalized_ranks = (np.argsort(np.argsort(upper_vals)) + 1) / float(N + 1)

		# Create output matrix
		R = np.zeros_like(A, dtype=float)
		R[iu] = normalized_ranks
		R = R + R.T
		
		return R
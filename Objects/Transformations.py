import numpy as np

from abc import ABC, abstractmethod
from scipy.stats import rankdata

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
		self.id = 'Id'

	def __call__(self, A):
		return A

# Opposite transformation (1 - A)
class OppositeTransform(WeightTransform):
	def __init__(self):
		self.name = "Opposite"
		self.id = 'Opp'

	def __call__(self, A):
		A = 1 - A
		np.fill_diagonal(A, 0)
		return A

# Logarithmic transformation (-log(A))
class LogTransform(WeightTransform):
	def __init__(self):
		self.name = "Logarithmic"
		self.id = 'Log'

	def __call__(self, A):
		A = np.clip(A, np.finfo(A.dtype).eps, None)
		A = -np.log(A)
		np.fill_diagonal(A, 0)
		return A

# Threshold transformation (binary thresholding)
class ThresholdTransform(WeightTransform):
	def __init__(self, τ = 0.1):
		self.name = f"Threshold (τ = {τ})"
		self.id = 'Thr'
		self.τ = τ

	def __call__(self, A):
		A = (A <= self.τ).astype(float)
		np.fill_diagonal(A, 0)
		return A

# Rank transformation (normalized ranking of upper-triangular values)
class RankTransform(WeightTransform):
	def __init__(self):
		self.name = "Rank"
		self.id = 'Rank'

	def __call__(self, A):

		iu = np.triu_indices_from(A, k=1)
		ranks = rankdata(A[iu], method='ordinal')

		R = np.zeros_like(A, dtype=float)
		R[iu] = ranks / (ranks.size + 1)
		R = R + R.T
		np.fill_diagonal(R, 0)
		
		return R
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
		tA = A.copy()
		return tA

# Opposite transformation (1 - A)
class OppositeTransform(WeightTransform):
	def __init__(self):
		self.name = "Opposite"
		self.id = 'Opp'

	def __call__(self, A):
		tA = A.copy()
		tA[tA > 0] = 1 - tA[tA > 0]
		return tA

# Logarithmic transformation (-log(A))
class LogTransform(WeightTransform):
	def __init__(self):
		self.name = "Logarithmic"
		self.id = 'Log'

	def __call__(self, A):
		tA = A.copy()
		tA[tA > 0] = -np.log(tA[tA > 0])
		return tA

# Threshold transformation (binary thresholding)
class ThresholdTransform(WeightTransform):
	def __init__(self, τ = 0.1):
		self.name = f"Threshold (τ = {τ})"
		self.id = 'Thr'
		self.τ = τ

	def __call__(self, A):
		tA = A.copy()
		tA[tA > 0] = (tA[tA > 0] <= self.τ).astype(float)
		return tA
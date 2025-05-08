import numpy as np

from abc import ABC, abstractmethod
from scipy.stats import rankdata
from matplotlib import colors

cmap_blue = colors.LinearSegmentedColormap.from_list("blue", ["#00FFFF", "#0050FF"])
norm_pow_γ = colors.Normalize(vmin=0.5, vmax=2.0)

cmap_pink_red = colors.LinearSegmentedColormap.from_list("pink_red", ["#FFB2C6", "#FF0000"])
norm_qtl_q = colors.Normalize(vmin=0.01, vmax=0.5)


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
		self.color = cmap_blue(norm_pow_γ(1))

	def __call__(self, A):
		tA = A.copy()
		return tA

# Opposite transformation (1 - A)
class OppositeTransform(WeightTransform):
	def __init__(self):
		self.name = "Opposite"
		self.id = 'Opp'
		self.color = 'orange'

	def __call__(self, A):
		tA = np.clip(A, 0, None)
		tA[tA > 0] = 1 - tA[tA > 0]
		return tA

# Logarithmic transformation (-log(A))
class LogTransform(WeightTransform):
	def __init__(self):
		self.name = "Logarithmic"
		self.id = 'Log'
		self.color = 'green'

	def __call__(self, A):
		tA = np.clip(A, 0, None)
		tA[tA > 0] = -np.log(tA[tA > 0])
		return tA

# Threshold transformation (binary thresholding)
class ThresholdTransform(WeightTransform):
	def __init__(self, τ = 0.05):
		self.name = f"Threshold (τ = {τ})"
		self.id = f'Thr-{τ}'
		self.τ = τ
		self.color = 'black'

	def __call__(self, A):
		tA = A.copy()
		tA[tA > 0] = (tA[tA > 0] <= self.τ).astype(float)
		return tA
	
class RankTransform(WeightTransform):
	def __init__(self):
		self.name = "Rank"
		self.id = 'Rank'
		self.color = 'purple'

	def __call__(self, A):
		iu = np.triu_indices_from(A, k=1)
		p_iu = A[iu] > 0
		ranks = rankdata(A[iu][p_iu], method='ordinal')

		tA = np.zeros_like(A, dtype=float)
		tA[iu[0][p_iu], iu[1][p_iu]] = ranks / (ranks.size + 1)
		tA = tA + tA.T
		np.fill_diagonal(tA, 0)
		
		return tA
	
class QuantileTransform(WeightTransform):
	def __init__(self, q = 0.1):
		self.name = f"Quantile (q = {q})"
		self.id = f'Q-{int(q*100):02d}'
		self.q = q
		self.color = cmap_pink_red(norm_qtl_q(q))

	def __call__(self, A):
		tA = A.copy()
		τ = np.quantile(tA[tA > 0], self.q)
		tA[tA > 0] = (tA[tA > 0] <= τ).astype(float)
		return tA
	
class PowerTransform(WeightTransform):
	def __init__(self, γ = 1.41):
		self.name = f"Power (γ = {γ})"
		self.id = f'Pow-{γ}'
		self.γ = γ
		self.color = cmap_blue(norm_pow_γ(γ))

	def __call__(self, A):
		tA = A.copy()
		tA = tA ** self.γ
		return tA
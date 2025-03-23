import numpy as np
import scipy.stats as stats
from Transformations import *
from Helper import *
from Chernoff import *

class betaWSBM:
	def __init__(self, n, ρ, Π, α):
		# n: number of nodes
		# ρ: probability of observing a link
		# Π: probabilities for community membership
		# α: shape parameters for the beta distribution
		self.n = n
		self.ρ = ρ
		self.Π = Π
		self.K = Π.shape[0]

		self.α = α

		self.name = f"Beta-WSBM(α11 = {α[0, 0]}, α12 = {α[0, 1]})"

	def __call__(self, seed=None):
		np.random.seed(seed)
		# Community membership
		Z = np.random.choice(np.arange(self.K), size=self.n, p=np.diag(self.Π))
		
        # Mixture model (1-ρ)δ_1 + ρBeta(α_{Z_i,Z_j}, 1)
		A = np.ones((self.n, self.n))
		mask = np.random.rand(self.n, self.n) < self.ρ
		A[mask] = stats.beta.rvs(self.α[Z[:, None], Z[None, :]][mask], 1)
		
		A = np.triu(A) + np.triu(A, 1).T
		np.fill_diagonal(A, 0)
		
		return A, Z
	
	def theoretical_B_C(self, T = None):
		ρ, α = self.ρ, self.α
		if T is None or isinstance(T, IdentityTransform):
			B = 1 - ρ / (α + 1.0)
			C = 1 - (2 * ρ / (α + 2.0)) - B**2
			return B, C
		elif isinstance(T, OppositeTransform):
			B, C = self.theoretical_B_C(IdentityTransform())
			return 1 - B, C
		elif isinstance(T, LogTransform):
			B = ρ / α
			C = ρ * (2.0 / α**2) - B**2
			return B, C
		elif isinstance(T, ThresholdTransform):
			B = ρ * (T.τ ** α)
			C = B * (1 - B)
			return B, C
		else:
			raise ValueError("Invalid transformation T")
	
class lognormWSBM:
	def __init__(self, n, ρ, Π, Σ, ExpMu = np.array([[1, 1 - 1e-6], [1 - 1e-6, 1]])):
		# n: number of nodes
		# ρ: probability of observing a link
		# Π: probabilities for community membership
		# Σ: shape parameter for the lognormal distribution
		# ExpMu: scale parameter for the lognormal distribution
		self.n = n
		self.ρ = ρ
		self.Π = Π
		self.K = Π.shape[0]

		self.Σ = Σ
		self.ExpMu = ExpMu

		self.name = f"Lognorm-WSBM(σ11 = {Σ[0, 0]}, σ12 = {Σ[0, 1]})"

	def __call__(self, seed=None):
		np.random.seed(seed)
		# Community membership
		Z = np.random.choice(np.arange(self.K), size=self.n, p=np.diag(self.Π))
		
        # Mixture model (1-ρ)δ_1 + ρLognorm(Σ_{Z_i,Z_j}, ExpMu_{Z_i,Z_j})
		A = np.ones((self.n, self.n))
		mask = np.random.rand(self.n, self.n) < self.ρ
		A[mask] = stats.lognorm.rvs(s=self.Σ[Z[:, None], Z[None, :]][mask],
									scale=self.ExpMu[Z[:, None], Z[None, :]][mask])
		
		A = np.triu(A) + np.triu(A, 1).T
		np.fill_diagonal(A, 0)
		
		return A, Z
	
	def theoretical_B_C(self, T = None):
		ρ, Σ, ExpMu = self.ρ, self.Σ, self.ExpMu
		Mu = np.log(ExpMu)
		if T is None or isinstance(T, IdentityTransform):
			B = 1 - ρ + ρ * ExpMu * np.exp(Σ ** 2 / 2)
			C = 1 - ρ + ρ * ExpMu ** 2 * np.exp(2 * Σ ** 2) - B**2
			return B, C
		elif isinstance(T, OppositeTransform):
			B, C = self.theoretical_B_C(IdentityTransform())
			return 1 - B, C
		elif isinstance(T, LogTransform):
			B = -ρ * Mu
			C = ρ * Σ ** 2 + ρ * (1 - ρ) * Mu ** 2
			return B, C
		elif isinstance(T, ThresholdTransform):
			B = ρ * norm.cdf((np.log(T.τ) - Mu) / Σ)
			C = B * (1 - B)
			return B, C
		else:
			raise ValueError("Invalid transformation T")
		

class TWSBInstance:
	def __init__(self, model, transformation, A = None, Z = None, seed = None):
		self.model = model
		self.transformation = transformation
		if A is None or Z is None:
			A, Z = model(seed = seed)
			self.A, self.Z = transformation(A), Z
		else:
			self.A, self.Z = A, Z

		self.model_name = model.name
		self.transform_name = transformation.name

		self.X = spectral_embedding(self.A)
		self.Z_hat, self.M, self.Σ = fit_GMM(self.X)
		self.Z_hat = label_permutation(self.Z, self.Z_hat)

		B, C = self.model.theoretical_B_C(self.transformation)
		B_hat, C_hat = empirical_B_C(self.A, self.Z_hat, 2)

		self.C_true = chernoff_information_graph(B, C, self.model.Π)
		self.C_graph = chernoff_information_graph(B_hat, C_hat, self.model.Π)
		self.C_embedding = chernoff_information_embedding(self.M, self.Σ, self.model.n)
		self.RAND = rand_index(self.Z, self.Z_hat)

		
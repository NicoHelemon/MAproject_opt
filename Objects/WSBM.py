import numpy as np
from scipy.stats import norm, lognorm, beta
from scipy.integrate import quad_vec
from itertools import product, permutations
import seaborn as sns

from .Transformations import *
from Plotting.StringHelper import sub, sup

# Constants I:

n = 1000

###########################
	
def pi_init(π):
	if isinstance(π, float):
		return np.array([[π, 0], [0, 1 - π]])
	elif isinstance(π, np.ndarray) and π.shape == (2, 2):
		if np.trace(π) == 1 and π[0, 1] == π[1, 0] == 0:
			return π
	else:
		raise ValueError("Invalid π parameter")
	
def edges_block_proportions(K, Π, N):
	n_edges = np.zeros((K, K))
	n = np.diag(Π) * N
	for i, j in product(range(K), repeat=2):
		if i == j:
			n_edges[i, j] = n[i] * (n[i] - 1) / 2
		else:
			n_edges[i, j] = n[i] * n[j]

	return n_edges / (n * (n - 1) / 2)

def linspace_exclusive(start, stop, N):
	return np.linspace(start, stop, N + 2)[1:-1]

###########################

class WSBM(ABC):
	@abstractmethod
	def __call__(self, seed=None):
		pass

	@abstractmethod
	def theoretical_B_C(self, T = None):
		pass

class betaWSBM(WSBM):
	param_name = 'α'
	name = 'Beta'

	def alpha_init(α11, α12, α22):
		return np.array([[α11, α12], [α12, α22]])

	α22_fixed = 1.0

	def __init__(self, ρ, Π, α, n = n, p22 = 'fixed'):
		# n: number of nodes
		# ρ: probability of observing a link
		# Π: probabilities for community membership
		# α: shape parameters for the beta distribution

		self.n = n
		self.ρ = ρ
		self.Π = pi_init(Π)
		self.K = self.Π.shape[0]

		α11, α12 = α
		if p22 == 'fixed':
			self.α = betaWSBM.alpha_init(α11, α12, betaWSBM.α22_fixed)
		elif p22 == 'p11':
			self.α = betaWSBM.alpha_init(α11, α12, α11)
		else:
			raise ValueError("Invalid p22 parameter - must be 'fixed' or 'p11'")

		if self.α[0, 0] != self.α[1, 1]:
			self.name = f'Beta-WSBM:\nα{sub("11")} = {self.α[0, 0]}, α{sub("12")} = {self.α[0, 1]}, α{sub("22")} = {self.α[1, 1]}'
		else:
			self.name = f'Beta-WSBM:\nα{sub("11")} = α{sub("22")} = {self.α[0, 0]}, α{sub("12")} = {self.α[0, 1]}'

	def __call__(self, seed=None):
		np.random.seed(seed)
		# Community membership
		Z = np.random.choice(np.arange(self.K), size=self.n, p=np.diag(self.Π))
		
		# Mixture model (1-ρ)δ_0 + ρBeta(α_{Z_i,Z_j}, 1)
		A = np.zeros((self.n, self.n))
		mask = np.random.rand(self.n, self.n) < self.ρ
		A[mask] = beta.rvs(self.α[Z[:, None], Z[None, :]][mask], 1)
		
		A = np.triu(A) + np.triu(A, 1).T
		np.fill_diagonal(A, 0)

		return A, Z
	
	def theoretical_B_C(self, T = None):
		ρ, α = self.ρ, self.α
		if T is None or isinstance(T, IdentityTransform):
			B = ρ * α / (α + 1.0)
			C = ρ * α / (α + 2.0) - B**2
			return B, C
		elif isinstance(T, OppositeTransform):
			B = ρ / (α + 1.0)
			C = 2 * ρ / ((α + 1.0) * (α + 2.0)) - B**2
			return B, C
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
	
class lognormWSBM(WSBM):
	param_name = 'σ'
	name = 'LogN'

	def compute_mu(σ, quantile=0.99):
		return - norm.ppf(quantile) * σ
	
	def sigma_init(σ11, σ12, σ22):
		return np.array([[σ11, σ12], [σ12, σ22]])

	ExpMu = np.exp(compute_mu(np.ones((2,2)))) - 1e-6 * (1 - np.eye(2))
	σ22_fixed = 1

	def __init__(self, ρ, Π, Σ, n = n, p22 = 'fixed'):
		# n: number of nodes
		# ρ: probability of observing a link
		# Π: probabilities for community membership
		# Σ: shape parameter for the lognormal distribution
		# ExpMu: scale parameter for the lognormal distribution
		self.n = n
		self.ρ = ρ
		self.Π = pi_init(Π)
		self.K = self.Π.shape[0]

		σ11, σ12 = Σ
		if p22 == 'fixed':
			self.Σ = lognormWSBM.sigma_init(σ11, σ12, lognormWSBM.σ22_fixed)
		elif p22 == 'p11':
			self.Σ = lognormWSBM.sigma_init(σ11, σ12, σ11)
		else:
			raise ValueError("Invalid p22 parameter - must be 'fixed' or 'p11'")

		if self.Σ[0, 0] != self.Σ[1, 1]:
			self.name = f'Lognorm-WSBM: μ = {np.log(lognormWSBM.ExpMu[0, 1]):.2f}\nσ{sub("11")} = {self.Σ[0, 0]}, σ{sub("12")} = {self.Σ[0, 1]}, σ{sub("22")} = {self.Σ[1, 1]}'
		else:
			self.name = f'Lognorm-WSBM: μ = {np.log(lognormWSBM.ExpMu[0, 1]):.2f}\nσ{sub("11")} = σ{sub("22")} = {self.Σ[0, 0]}, σ{sub("12")} = {self.Σ[0, 1]}'

	def __call__(self, seed=None):
		np.random.seed(seed)
		# Community membership
		Z = np.random.choice(np.arange(self.K), size=self.n, p=np.diag(self.Π))
		
		# Mixture model (1-ρ)δ_0 + ρLognorm(Σ_{Z_i,Z_j}, ExpMu_{Z_i,Z_j})
		A = np.zeros((self.n, self.n))
		mask = np.random.rand(self.n, self.n) < self.ρ
		A[mask] = lognorm.rvs(s=self.Σ[Z[:, None], Z[None, :]][mask],
									scale=lognormWSBM.ExpMu[Z[:, None], Z[None, :]][mask])

		A = np.triu(A) + np.triu(A, 1).T
		np.fill_diagonal(A, 0)

		A[(A <= 0) | (A >= 1)] = 0
		
		return A, Z
	
	def theoretical_B_C(self, T = None):
		ρ, Σ = self.ρ, self.Σ
		Mu = np.log(lognormWSBM.ExpMu)
		E1 = np.exp(Mu + Σ ** 2 / 2)
		E2 = np.exp(2 * Mu + 2 * Σ ** 2)

		if T is None or isinstance(T, IdentityTransform):
			B = ρ * E1
			C = ρ * E2 - B**2
			return B, C
		elif isinstance(T, OppositeTransform):
			B = ρ * (1 - E1)
			C = ρ * (1 - 2*E1 + E2) - B**2
			return B, C
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


# Constants II:

RHOS = [0.25, 0.5]
PIS = [0.1, 0.5]
ALPHAS = [(0.1, 1.0), (0.5, 0.5)]
SIGMAS = [(1, 0.5), (0.1, 0.5)]
MODELS = [betaWSBM, lognormWSBM]
MODELS_AND_PARAMS = list(product([betaWSBM], ALPHAS)) + list(product([lognormWSBM], SIGMAS))
TRANSFORMS = [IdentityTransform(), OppositeTransform(), LogTransform(), ThresholdTransform()]
RHOS_PIS_MODELS = list(product(RHOS, PIS, MODELS))

TRANSFORMS_ID = [t.id for t in TRANSFORMS]
TRANSFORMS_MAP = {t.id : t for t in TRANSFORMS}
METRICS_ID = ['Rand', 'C_true', 'C_graph', 'C_embed']
METRICS_NAME = ["Rand index", "True Chernoff information", "Chernoff graph-estimation", "Chernoff embedding-estimation"]
METRICS_MAP = dict(zip(METRICS_ID, METRICS_NAME))
METRICS_ID_COSMETIC_MAP = {'C_true' : f'C{sup("true")}', 'C_graph' : f'C{sup("graph")}', 'C_embed' : f'C{sup("embed")}'}

BIASES = ['abs', 'rel', 'log']
BIASES_NAME = ['Absolute bias', 'Relative bias', 'Log-ratio bias']
BIASES_MAP = dict(zip(BIASES, BIASES_NAME))

TRANSFORMS_CMAP = dict(zip(TRANSFORMS + ['Argmax'], ['blue', 'orange', 'green', 'red', 'black']))
RHOS_PIS_MODELS_CMAP = dict(zip(RHOS_PIS_MODELS, sns.color_palette("tab10", len(RHOS_PIS_MODELS))))

P22S = ['fixed', 'p11']
EMB_MODES = ['sqrt-scaled', 'scaled', 'raw']

def emb_mode_p22_path_str(emb_mode, p22):
	return f"{emb_mode}_p22_{p22}"

		
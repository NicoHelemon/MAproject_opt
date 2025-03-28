import numpy as np
from scipy.stats import norm, lognorm, beta
from scipy.integrate import quad_vec
from itertools import product, permutations
import seaborn as sns

from .Transformations import *
from Plotting.StringHelper import sub, sup

# Constants I:

n = 1000
ExpMu = np.array([[1, 1 - 1e-6], 
				  [1 - 1e-6, 1]])

###########################

def alpha_init(α, α22 = 1.0):
	if isinstance(α, (tuple, list, np.ndarray)) and len(α) == 2:
		α11, α12 = α
		return np.array([[α11, α12], [α12, α22]])
	elif isinstance(α, np.ndarray) and α.shape == (2, 2):
		return α
	else:
		raise ValueError("Invalid α parameter")
	
def sigma_init(Σ, σ22 = 1.0):
	if isinstance(Σ, (tuple, list, np.ndarray)) and len(Σ) == 2:
		σ11, σ12 = Σ
		return np.array([[σ11, σ12], [σ12, σ22]])
	elif isinstance(Σ, np.ndarray) and Σ.shape == (2, 2):
		return Σ
	else:
		raise ValueError("Invalid Σ parameter")
	
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


class WSBM(ABC):
	@abstractmethod
	def __call__(self, seed=None):
		pass

	@abstractmethod
	def theoretical_B_C(self, T = None):
		pass

class betaWSBM(WSBM):
	param_name = 'α'

	def __init__(self, ρ, Π, α, n = n):
		# n: number of nodes
		# ρ: probability of observing a link
		# Π: probabilities for community membership
		# α: shape parameters for the beta distribution

		self.n = n
		self.ρ = ρ
		self.Π = pi_init(Π)
		self.K = self.Π.shape[0]

		self.α = alpha_init(α)

		self.name = f'Beta-WSBM(α{sub(" 11")} = {self.α[0, 0]}, α{sub(" 12")} = {self.α[0, 1]})'

	def __call__(self, seed=None):
		np.random.seed(seed)
		# Community membership
		Z = np.random.choice(np.arange(self.K), size=self.n, p=np.diag(self.Π))
		
		# Mixture model (1-ρ)δ_1 + ρBeta(α_{Z_i,Z_j}, 1)
		A = np.ones((self.n, self.n))
		mask = np.random.rand(self.n, self.n) < self.ρ
		A[mask] = beta.rvs(self.α[Z[:, None], Z[None, :]][mask], 1)
		
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
		elif isinstance(T, RankTransform):
			n, ρ, Π, α, K = self.n, self.ρ, self.Π, self.α, self.K

			P = edges_block_proportions(K, Π, n)

			B  = np.zeros((K, K))
			C  = np.zeros((K, K))

			blocks = list(permutations(range(K), 2))

			S1 = sum(P[r,s] / (α + α[r,s]) for (r,s) in blocks)
			S2 = sum(P[r1,s1] * P[r2,s2] / (α + α[r1,s1] + α[r2,s2]) 
			 			 for (r1,s1) in blocks for (r2,s2) in blocks)
				
			B = (1 - ρ) + ρ**2 * α * S1
			C = (1 - ρ) + ρ**3 * α * S2 - B**2			
			return B, C
		else:
			raise ValueError("Invalid transformation T")
	
class lognormWSBM(WSBM):
	param_name = 'σ'

	def __init__(self, ρ, Π, Σ, ExpMu = ExpMu, n = n):
		# n: number of nodes
		# ρ: probability of observing a link
		# Π: probabilities for community membership
		# Σ: shape parameter for the lognormal distribution
		# ExpMu: scale parameter for the lognormal distribution
		self.n = n
		self.ρ = ρ
		self.Π = pi_init(Π)
		self.K = self.Π.shape[0]

		self.Σ = sigma_init(Σ)
		self.ExpMu = ExpMu

		self.name = f'Lognorm-WSBM(σ{sub(" 11")} = {self.Σ[0, 0]}, σ{sub(" 12")} = {self.Σ[0, 1]})'

	def __call__(self, seed=None):
		np.random.seed(seed)
		# Community membership
		Z = np.random.choice(np.arange(self.K), size=self.n, p=np.diag(self.Π))
		
		# Mixture model (1-ρ)δ_1 + ρLognorm(Σ_{Z_i,Z_j}, ExpMu_{Z_i,Z_j})
		A = np.ones((self.n, self.n))
		mask = np.random.rand(self.n, self.n) < self.ρ
		A[mask] = lognorm.rvs(s=self.Σ[Z[:, None], Z[None, :]][mask],
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
		elif isinstance(T, RankTransform):
			n, Π, K = self.n, self.Π, self.K

			P = edges_block_proportions(K, Π, n)

			def CDF_Lognorm(x):
				F_ΣExpMU = np.vectorize(lambda σ, expmu : lognorm(s=σ, scale=expmu).cdf)(Σ, ExpMu)
				return np.vectorize(lambda f, x: f(x), otypes=[float])(F_ΣExpMU, x)
			
			def PDF_Lognorm(x):
				F_ΣExpMU = np.vectorize(lambda σ, expmu : lognorm(s=σ, scale=expmu).pdf)(Σ, ExpMu)
				return np.vectorize(lambda f, x: f(x), otypes=[float])(F_ΣExpMU, x)

			def H(x):
				return (1- ρ) * (1.0 if x == 0 else 0.0) + ρ * np.sum(np.triu(P * CDF_Lognorm(x)))
			
			H1 = H(1)
			I1, _ = quad_vec(lambda x: H(x) * PDF_Lognorm(x), 0, np.inf)
			I2, _ = quad_vec(lambda x: H(x) ** 2 * PDF_Lognorm(x), 0, np.inf)

			B = (1- ρ) * H1 + ρ * I1
			C = (1- ρ) * H1**2 + ρ * I2 - B**2

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
TRANSFORMS = [IdentityTransform(), OppositeTransform(), LogTransform(), ThresholdTransform(), RankTransform()]
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

TRANSFORMS_CMAP = dict(zip(TRANSFORMS, sns.color_palette("tab10", len(TRANSFORMS))))
RHOS_PIS_MODELS_CMAP = dict(zip(RHOS_PIS_MODELS, sns.color_palette("tab10", len(RHOS_PIS_MODELS))))

		
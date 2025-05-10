import numpy as np
from functools import partial
from scipy.stats import norm, lognorm, beta
from scipy.integrate import quad_vec
from scipy.optimize import brentq
from itertools import product, combinations_with_replacement
import seaborn as sns

from .Transformations import *
from Plotting.StringHelper import sub, sup
import time

# Constants I:

n = 1000

###########################

def param_init(p11, p12, p22):
	return np.array([[p11, p12], [p12, p22]])
	
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

	return n_edges / (N * (N - 1) / 2)

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
	p22_fixed = 1.0

	def instance_name_str(α):
		if α[0, 0] != α[1, 1]:
			return f'Beta-WSBM:\nα{sub("11")} = {α[0, 0]}, α{sub("12")} = {α[0, 1]}, α{sub("22")} = {α[1, 1]}'
		else:
			return f'Beta-WSBM:\nα{sub("11")} = α{sub("22")} = {α[0, 0]}, α{sub("12")} = {α[0, 1]}'

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
			self.α = param_init(α11, α12, betaWSBM.p22_fixed)
		elif p22 == 'p11':
			self.α = param_init(α11, α12, α11)
		else:
			raise ValueError("Invalid p22 parameter - must be 'fixed' or 'p11'")

		self.name = betaWSBM.instance_name_str(self.α)

	def __call__(self, seed=None):
		np.random.seed(seed)
		# Community membership
		Z = np.random.choice(np.arange(self.K), size=self.n, p=np.diag(self.Π))
		Z_i, Z_j = Z[:, None], Z[None, :]
		
		# Mixture model (1-ρ)δ_0 + ρBeta(α_{Z_i,Z_j}, 1)
		A = np.zeros((self.n, self.n))
		drawn_edges_idx = np.random.rand(self.n, self.n) < self.ρ
		A[drawn_edges_idx] = beta.rvs(self.α[Z_i, Z_j][drawn_edges_idx], 1)
		
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
		elif isinstance(T, RankTransform):
			n, Π, K = self.n, self.Π, self.K

			P = edges_block_proportions(K, Π, n)

			B  = np.zeros((K, K))
			C  = np.zeros((K, K))

			blocks = list(combinations_with_replacement(range(K), 2))

			S1 = sum(P[r,s] / (α + α[r,s]) for (r,s) in blocks)
			S2 = sum(P[r1,s1] * P[r2,s2] / (α + α[r1,s1] + α[r2,s2]) 
			 			 for (r1,s1) in blocks for (r2,s2) in blocks)

			B = ρ * α * S1
			C = ρ * α * S2 - B**2

			return B, C
		elif isinstance(T, QuantileTransform):
			π = np.diag(self.Π)

			def CDF(τ):
				return (π[:, None] * π[None, :] * τ ** α).sum()
			
			τ_q = brentq(lambda τ: CDF(τ) - T.q, 0, 1)

			B = ρ * (τ_q ** α)
			C = B * (1 - B)
			return B, C
		elif isinstance(T, PowerTransform):
			B = ρ * α / (α + T.γ)
			C = ρ * α / (α + 2.0*T.γ) - B**2
			return B, C
		else:
			raise ValueError("Invalid transformation T")
	
class lognormWSBM(WSBM):
	param_name = 'σ'
	name = 'LogN'
	p22_fixed = 1

	def compute_mu(σ, quantile=0.99):
		return - norm.ppf(quantile) * σ

	ExpMu = np.exp(compute_mu(np.ones((2,2)))) - 1e-6 * (1 - np.eye(2))
	μ = np.log(ExpMu[0, 1])

	def instance_name_str(Σ, μ = μ):
		if Σ[0, 0] != Σ[1, 1]:
			return f'Lognorm-WSBM: μ = {μ:.2f}\nσ{sub("11")} = {Σ[0, 0]}, σ{sub("12")} = {Σ[0, 1]}, σ{sub("22")} = {Σ[1, 1]}'
		else:
			return f'Lognorm-WSBM: μ = {μ:.2f}\nσ{sub("11")} = σ{sub("22")} = {Σ[0, 0]}, σ{sub("12")} = {Σ[0, 1]}'

	def __init__(self, ρ, Π, Σ, n = n, p22 = 'fixed', tail_control = None):
		# n: number of nodes
		# ρ: probability of observing a link
		# Π: probabilities for community membership
		# Σ: shape parameter for the lognormal distribution
		# ExpMu: scale parameter for the lognormal distribution
		self.n = n
		self.ρ = ρ
		self.Π = pi_init(Π)
		self.K = self.Π.shape[0]
		self.tail_control = tail_control

		σ11, σ12 = Σ
		if p22 == 'fixed':
			self.Σ = param_init(σ11, σ12, lognormWSBM.p22_fixed)
		elif p22 == 'p11':
			self.Σ = param_init(σ11, σ12, σ11)
		else:
			raise ValueError("Invalid p22 parameter - must be 'fixed' or 'p11'")

		self.name = lognormWSBM.instance_name_str(self.Σ)

	def __call__(self, seed=None):
		np.random.seed(seed)
		# Community membership
		Z = np.random.choice(np.arange(self.K), size=self.n, p=np.diag(self.Π))
		Z_i, Z_j = Z[:, None], Z[None, :]
		
		# Mixture model (1-ρ)δ_0 + ρLognorm(Σ_{Z_i,Z_j}, ExpMu_{Z_i,Z_j})
		A = np.zeros((self.n, self.n))
		drawn_edges_idx = np.random.rand(self.n, self.n) < self.ρ

		s = self.Σ[Z_i, Z_j][drawn_edges_idx]
		scale = lognormWSBM.ExpMu[Z_i, Z_j][drawn_edges_idx]
		
		edges = lognorm.rvs(s=s, scale=scale, size=s.shape)
		if self.tail_control is not None:
			above_t = edges > self.tail_control
			while np.any(above_t):
				edges[above_t] = lognorm.rvs(s[above_t], scale=scale[above_t], size=above_t.sum())
				above_t = edges > self.tail_control

		A[drawn_edges_idx] = edges

		A = np.triu(A) + np.triu(A, 1).T
		np.fill_diagonal(A, 0)
		
		return A, Z
	
	def theoretical_B_C(self, T = None):
		ρ, Σ = self.ρ, self.Σ
		ExpMu = lognormWSBM.ExpMu
		Mu = np.log(ExpMu)
		E1 = np.exp(Mu + Σ ** 2 / 2)
		E2 = np.exp(2 * Mu + 2 * Σ ** 2)

		if self.tail_control is not None:
			α  = (np.log(self.tail_control) - Mu) / Σ
			t0 = norm.cdf(α)
			t1 = norm.cdf(α - Σ) / t0
			t2 = norm.cdf(α - 2*Σ) / t0

			E1 = E1 * t1
			E2 = E2 * t2
		else:
			α = 1.0
			t0 = 1.0

		if T is None or isinstance(T, IdentityTransform):
			B = ρ * E1
			C = ρ * E2 - B**2
			return B, C
		elif isinstance(T, OppositeTransform):
			B = ρ * (1 - E1)
			C = ρ * (1 - 2*E1 + E2) - B**2
			return B, C
		elif isinstance(T, LogTransform):
			λ = norm.pdf(α) / t0 if self.tail_control is not None else 0
			F1 = Mu - Σ * λ
			F2 = Σ ** 2 * (1 - α*λ - λ**2)

			B = -ρ * F1
			C = ρ * F2 + ρ * (1 - ρ) * F1 ** 2
			return B, C
		elif isinstance(T, ThresholdTransform):
			B = ρ * norm.cdf((np.log(T.τ) - Mu) / Σ) / t0
			C = B * (1 - B)
			return B, C
		elif isinstance(T, RankTransform):
			n, Π, K = self.n, self.Π, self.K

			P = np.triu(edges_block_proportions(K, Π, n))

			CDF = partial(lognorm.cdf,  s=Σ, scale=ExpMu)
			PDF = partial(lognorm.pdf,  s=Σ, scale=ExpMu)

			def integrand(x):
				PDF_x = PDF(x) / t0
				h_x = np.sum(P * CDF(x)) / t0

				return np.stack((h_x * PDF_x, h_x**2 * PDF_x))
			
			I, _ = quad_vec(integrand, 0, 1, epsabs = 1e-9, epsrel = 1e-7)

			B = ρ * I[0]
			C = ρ * I[1] - B**2

			return B, C
		elif isinstance(T, QuantileTransform):
			π = np.diag(self.Π)

			def CDF(τ):
				return (π[:, None] * π[None, :] * norm.cdf((np.log(τ) - Mu) / Σ) / t0).sum()
			
			τ_q = brentq(lambda τ: CDF(τ) - T.q, np.finfo(float).tiny, 1)

			B = ρ * norm.cdf((np.log(τ_q) - Mu) / Σ) / t0
			C = B * (1 - B)
			return B, C
		elif isinstance(T, PowerTransform):
			γ = T.γ
			γM = γ * Mu
			γΣ = γ * Σ
			γt1 = norm.cdf(α - γ*Σ) / t0 if self.tail_control is not None else 1.0
			γt2 = norm.cdf(α - 2*γ*Σ) / t0 if self.tail_control is not None else 1.0
			γE1 = np.exp(γM + γΣ ** 2 / 2) * γt1
			γE2 = np.exp(2 * γM + 2 * γΣ ** 2) * γt2

			B = ρ * γE1
			C = ρ * γE2 - B**2
			return B, C
		else:
			raise ValueError(f"Invalid transformation {T}")


# Constants II:

def sigmoid(x, x0, k):
	z = k*(x - x0)
	z = np.clip(z, -700, +700)
	return 1/(1 + np.exp(-z))

def step(x, x0):
	return np.where(x > x0, 1, 0)

def sigmoid_w95(x, x0, w):
	if w == 0:
		return step(x, x0)
	else:
		return sigmoid(x, x0, np.log(19)/w)

RHOS = [0.1, 0.25, 0.5]
PIS = [0.1, 0.25, 0.5]
ALPHAS = [(0.1, 1.0), (0.5, 0.5)]
SIGMAS = [(1, 0.5), (0.1, 0.5)]
MODELS = [betaWSBM, lognormWSBM]
MODELS_AND_PARAMS = list(product([betaWSBM], ALPHAS)) + list(product([lognormWSBM], SIGMAS))

TRANSFORMS_MIN  = [OppositeTransform(), LogTransform(), RankTransform()]
TRANSFORMS_QTL  = [QuantileTransform(q) for q in [0.01, 0.05, 0.1, 0.25, 0.5]]
TRANSFORMS_POW  = [PowerTransform(γ) for γ in [(np.sqrt(2) ** i).round(2) for i in [-2, -1, 0, 1, 2]]]
TRANSFORMS      = TRANSFORMS_POW[2:4] + TRANSFORMS_QTL[2:4] + TRANSFORMS_MIN
TRANSFORMS_EXT  = TRANSFORMS_POW + TRANSFORMS_QTL + TRANSFORMS_MIN
RHOS_PIS_MODELS = list(product(RHOS, PIS, MODELS))

TRANSFORMS_ID = [t.id for t in TRANSFORMS]
TRANSFORMS_MAP = {t.id : t for t in TRANSFORMS}
METRICS_ID = ['Rand', 'GMM_score','C_true', 'gC_true', 'C_graph', 'gC_graph', 'C_embed', 'gC_embed']
METRICS_NAME = ["Rand index",
				"GMM score", 
				"True Chernoff information", 
				"Gated true Chernoff information",
				"Chernoff graph-estimation",
				"Gated Chernoff graph-estimation",
				"Chernoff embedding-estimation",
				"Gated Chernoff embedding-estimation"]
METRICS_MAP = dict(zip(METRICS_ID, METRICS_NAME))

CHERNOFFS_ID = METRICS_ID[2:]
VANILLA_METRICS_ID = ['Rand', 'GMM_score', 'C_true', 'C_graph', 'C_embed']
NON_GATED_CHERNOFFS_ID = ['C_true', 'C_graph', 'C_embed']
GATED_CHERNOFFS_ID = ['gC_true', 'gC_graph', 'gC_embed']
GATING_FUNCTIONS = dict(zip(GATED_CHERNOFFS_ID, 
							[lambda x: sigmoid_w95(x, x0=1, w=0),
							 lambda x: sigmoid_w95(x, x0=1, w=0), 
							 lambda x: sigmoid_w95(x, x0=1, w=0)]))
CHERNOFFS_ID_COSMETIC_MAP = dict(zip(CHERNOFFS_ID, [''.join([C.split('_')[0], sup(C.split('_')[1])]) for C in CHERNOFFS_ID]))
CHERNOFFS_CMAP = dict(zip(CHERNOFFS_ID, ['yellow', 'gold', 'cyan', 'teal', 'magenta', 'mediumvioletred']))


BIASES = ['abs', 'rel', 'log']
BIASES_NAME = ['Absolute bias', 'Relative bias', 'Log-ratio bias']
BIASES_MAP = dict(zip(BIASES, BIASES_NAME))

TRANSFORMS_CMAP = {t : t.color for t in TRANSFORMS_EXT}

CMAP = CHERNOFFS_CMAP | TRANSFORMS_CMAP

P22S = ['fixed', 'p11']
EMB_MODES = ['sqrt-scaled', 'scaled', 'raw']

def emb_mode_p22_path_str(emb_mode, p22):
	return f"{emb_mode}_p22_{p22}"

		
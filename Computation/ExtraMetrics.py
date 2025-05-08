import numpy as np
import warnings
from scipy.stats import spearmanr, ConstantInputWarning
from sklearn.metrics import auc
from scipy.ndimage import convolve

from Objects.WSBM import *

def sigmoid_gmm_score(window_95 = 10, x0 = -5):
	k  = np.log(19)/window_95
	x0 = x0

	def sigmoid(x):
		return 1/(1 + np.exp(-k*(x - x0)))
	
	return sigmoid

def kernel(window = 3, dim = 2, exterior_total_weight = None):
	if exterior_total_weight is None: 
		exterior_total_weight = dim * 0.5
	K = np.ones([window] * dim)
	K = K * exterior_total_weight / (np.sum(K) - 1)
	K[tuple([slice(1, -1)] * dim)] = 1
	return K
	
def local_weighted_average(tensor, K = None):
	if K is None: K = kernel(dim = tensor.ndim - 1)
	assert tensor.ndim == K.ndim + 1, "Tensor and kernel must matching number of dimensions"
	M = convolve(tensor.astype(float).sum(axis=-1), K, mode='constant', cval=0.0)
	M = M / convolve(np.ones(tensor.shape[:-1]), K, mode='constant', cval=0.0)
	return M / tensor.shape[-1]

def best_transform_metrics(m, transforms = TRANSFORMS_EXT.copy()):
	def stack_grids(m, metric):
		return np.stack([m[t][metric] for t in transforms], axis=-1)

	Rand = stack_grids(m, 'Rand')
	m['Rand Max'] = np.max(Rand, axis=-1)

	for t in transforms:
		regret = m['Rand Max'] - m[t]['Rand']

		m[t]['Rand Avg'] = np.mean(m[t]['Rand'])
		m[t]['Regret Avg'] = np.mean(regret)
		m[t]['Regret Area'] = np.count_nonzero(regret) / np.prod(regret.shape)
		m[t]['Regret Avg on Positive Regret'] = np.mean(regret[regret > 0])
		m[t]['Rand Avg on Positive Regret'] = np.mean(m[t]['Rand'][regret > 0])
		m[t]['Rand Avg on Null Regret'] = np.mean(m[t]['Rand'][regret == 0])
		m[t]['Ahead Ratio'] = np.mean(m[t]['Rand'] == m['Rand Max'])

	for C in CHERNOFFS_ID:
		m[f'{C}-Best Transform'] = {}
		mBestC = m[f'{C}-Best Transform']
		mBestC['Arg'] = np.argmax(stack_grids(m, C), axis=-1).astype(int)
		mBestC['Transform Area'] = np.bincount(mBestC['Arg'].ravel(), minlength=len(transforms)) / np.prod(mBestC['Arg'].shape)
		mBestC['Rand'] = np.take_along_axis(Rand, mBestC['Arg'][..., None], axis=-1).squeeze(-1)
		mBestC['Rand Avg'] = np.mean(mBestC['Rand'])

		regret = m['Rand Max'] - mBestC['Rand']
		mBestC['Regret'] = regret
		mBestC['Regret Avg'] = np.mean(regret)
		mBestC['Regret Area'] = np.count_nonzero(regret) / np.prod(regret.shape)
		mBestC['Regret Avg on Positive Regret'] = np.mean(regret[regret > 0])
		mBestC['Rand Avg on Positive Regret'] = np.mean(mBestC['Rand'][regret > 0])
		mBestC['Rand Avg on Null Regret'] = np.mean(mBestC['Rand'][regret == 0])

		mBestC['Ahead Ratio'] = np.mean(mBestC['Rand'] == m['Rand Max'])

		for t in transforms:
			mBestC[t] = {}
			idx_t = mBestC['Arg'] == transforms.index(t)
			mBestC[t]['Rand Avg'] = np.mean(mBestC['Rand'][idx_t])
			regret_t = regret[idx_t]
			mBestC[t]['Regret Avg'] = np.mean(regret_t)
			# Relative to total area vs relative to area of the best transform (... /np.count_nonzero(idx_t))
			mBestC[t]['Regret Area'] = np.count_nonzero(regret_t) / np.prod(regret.shape)
			mBestC[t]['Regret Avg on Positive Regret'] = np.mean(regret_t[regret_t > 0])
			mBestC[t]['Rand Avg on Positive Regret'] = np.mean(mBestC['Rand'][idx_t & (regret > 0)])
			mBestC[t]['Rand Avg on Null Regret'] = np.mean(mBestC['Rand'][idx_t & (regret == 0)])

	return m

def bias(m, eps = np.finfo(float).eps):
	def abs_bias(true, pred):
		return np.abs(true - pred)

	def rel_bias(true, pred):
		return abs_bias(true, pred) / (true + eps)

	def log_bias(true, pred):
		return np.log(pred / (true + eps))

	m['Bias'] = {}
	for C in CHERNOFFS_ID[1:]:
		m['Bias'][C] = {}
		m['Bias'][C]['abs'] = abs_bias(m['C_true'], m[C])
		m['Bias'][C]['rel'] = rel_bias(m['C_true'], m[C])
		m['Bias'][C]['log'] = log_bias(m['C_true'], m[C])

	return m

def partial_correlation(true, pred, num_ticks = 100):
		true_flat = true.ravel()
		pred_flat = pred.ravel()

		idx = np.argsort(pred_flat)[::-1]
		true_ordered = true_flat[idx]
		pred_ordered = pred_flat[idx]

		N = len(true_ordered)
		step = max(1, N // num_ticks)
		ticks = np.arange(step, N+1, step)

		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=ConstantInputWarning)
			corrs = [spearmanr(true_ordered[:n], pred_ordered[:n])[0] for n in ticks]
		corrs = np.nan_to_num(corrs, nan=0)
		ticks = ticks / N
		partial_corrs = ((ticks * 100).astype(int), corrs)
		# Btw auc (btwn 0 and 1, which is the case) == mean 

		return corrs[-1], auc(ticks, corrs), partial_corrs

def correlation(m):
	m['Correlation'] = {}

	m['Correlation']['Rand']  = {metric: partial_correlation(m['Rand'], m[metric]) 
							  for metric in CHERNOFFS_ID}
	m['Correlation']['C_true'] = {metric: partial_correlation(m['C_true'], m[metric]) 
							   for metric in CHERNOFFS_ID[1:]}

	return m

def aggregate_metrics(metrics, transforms = TRANSFORMS_EXT.copy()):
	for (rho, pi, m), t in product(RHOS_PIS_MODELS, transforms):
		metrics[f'rho:{rho}']	= {}
		metrics[f'pi:{pi}']  	= {}
		metrics[m]   			= {}
		metrics[t]   			= {}

	for m, t in product(MODELS, transforms):
		metrics[m][t] = {}
			
	for m_id in METRICS_ID:
		metrics[m_id] = np.concatenate([metrics[rpm][t][m_id].ravel() 
										for rpm, t in product(RHOS_PIS_MODELS, transforms)])
		
		for rpm in RHOS_PIS_MODELS:
			metrics[rpm][m_id] = np.concatenate([metrics[rpm][t][m_id].ravel() 
												 for t in transforms])
			
		for rho in RHOS:
			metrics[f'rho:{rho}'][m_id] = np.concatenate([metrics[(rho, pi, m)][t][m_id].ravel() 
												 for pi, m, t in product(PIS, MODELS, transforms)])
			
		for pi in PIS:
			metrics[f'pi:{pi}'][m_id] = np.concatenate([metrics[(rho, pi, m)][t][m_id].ravel()
												 for rho, m, t in product(RHOS, MODELS, transforms)])
			
		for model in MODELS:
			metrics[model][m_id] = np.concatenate([metrics[(rho, pi, model)][t][m_id].ravel() 
												   for rho, pi, t in product(RHOS, PIS, transforms)])
			
			for t in transforms:
				metrics[model][t][m_id] = np.concatenate([metrics[(rho, pi, model)][t][m_id].ravel() 
														   for rho, pi in product(RHOS, PIS)])
		
		for t in transforms:
			metrics[t][m_id] = np.concatenate([metrics[rpm][t][m_id].ravel() 
											   for rpm in RHOS_PIS_MODELS])
			
	return metrics
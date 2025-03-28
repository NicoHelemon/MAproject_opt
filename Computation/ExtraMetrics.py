import numpy as np
import warnings
from scipy.stats import spearmanr, ConstantInputWarning
from sklearn.metrics import auc

from Objects.WSBM import *

def best_transform_metrics(m):
	def stack_grids(m, metric):
		return np.stack([m[t][metric] for t in TRANSFORMS], axis=-1)

	C_graph = stack_grids(m, 'C_graph')
	C_embed = stack_grids(m, 'C_embed')
	Rand    = stack_grids(m, 'Rand')
	max_Rand = np.max(Rand, axis=-1)

	m['C_graph-Best Transform'] = {}
	BT_arg_C_graph = np.argmax(C_graph, axis=-1)
	m['C_graph-Best Transform']['Arg'] = BT_arg_C_graph
	m['C_graph-Best Transform']['Rand'] = np.take_along_axis(Rand, BT_arg_C_graph[..., None], axis=-1).squeeze(-1)
	m['C_graph-Best Transform']['Regret'] = max_Rand - m['C_graph-Best Transform']['Rand']

	m['C_embed-Best Transform'] = {}
	BT_arg_C_embed = np.argmax(C_embed, axis=-1)
	m['C_embed-Best Transform']['Arg'] = BT_arg_C_embed
	m['C_embed-Best Transform']['Rand'] = np.take_along_axis(Rand, BT_arg_C_embed[..., None], axis=-1).squeeze(-1)
	m['C_embed-Best Transform']['Regret'] = max_Rand - m['C_embed-Best Transform']['Rand']

	return m

def bias(m, eps = np.finfo(float).eps):
	def abs_bias(true, pred):
		return np.abs(true - pred)

	def rel_bias(true, pred):
		return abs_bias(true, pred) / (true + eps)

	def log_bias(true, pred):
		return np.log(pred / (true + eps))

	m['Bias'] = {}
	m['Bias']['C_graph'] = {'abs': abs_bias(m['C_true'], m['C_graph']), 'rel': rel_bias(m['C_true'], m['C_graph']), 'log': log_bias(m['C_true'], m['C_graph'])}
	m['Bias']['C_embed'] = {'abs': abs_bias(m['C_true'], m['C_embed']), 'rel': rel_bias(m['C_true'], m['C_embed']), 'log': log_bias(m['C_true'], m['C_embed'])}

	return m

def correlation(m):
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

		return corrs[-1], auc(ticks, corrs), partial_corrs

	m['Correlation'] = {}
	m['Correlation']['Rand']  = {metric: partial_correlation(m['Rand'], m[metric]) for metric in ['C_true', 'C_graph', 'C_embed']}
	m['Correlation']['C_true'] = {metric: partial_correlation(m['C_true'], m[metric]) for metric in ['C_graph', 'C_embed']}

	return m
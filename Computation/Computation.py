
import numpy as np
import os

from joblib import Parallel, delayed
from tqdm import tqdm
from joblib import dump

from Objects.TWSBMInstance import *


def simulate_in_grid(N, batch, rep, model, model_params, transformations,
					 p22='fixed', emb_mode='sqrt-scaled'):
	
	def simulate_one_grid_point(i, j, p11, p12, rho, pi, model, rep, transformations,
							 emb_mode, p22, base_seed):
		metrics =  {f'{t.id}_{m_id}'   : np.zeros(rep) for t in transformations for m_id in METRICS_ID}
		metrics |= {f'{t.id}_GMM_score': np.zeros(rep) for t in transformations}

		m = model(rho, pi, (p11, p12), p22=p22)

		seed = base_seed
		for k in range(rep):
			A, Z = m(seed=seed)
			for t in transformations:
				G = TWSBMInstance(model=m, transformation=t, A=t(A), Z=Z, emb_mode=emb_mode)
				metrics[f'{t.id}_C_true'][k]  	= G.C_true
				metrics[f'{t.id}_C_graph'][k] 	= G.C_graph
				metrics[f'{t.id}_C_embed'][k] 	= G.C_embedding
				metrics[f'{t.id}_Rand'][k]    	= G.RAND
				metrics[f'{t.id}_GMM_score'][k]	= G.GMM_score

			seed += 1

		return (i, j, metrics)

	rho, pi = model_params
	p11_linspace = linspace_exclusive(0, 1, N)
	p12_linspace = linspace_exclusive(0, 1, N)

	base_seed = batch * N**2 * rep

	tasks = [
		(i, j, p11, p12, rho, pi, model, rep, transformations, emb_mode, p22, base_seed + (i * N + j) * rep)
		for i, p11 in enumerate(p11_linspace)
		for j, p12 in enumerate(p12_linspace)
	]

	n_jobs = max(1, int(os.cpu_count()*3/4))
	print(f"Using {n_jobs} threads (out of {os.cpu_count()})")
	results = Parallel(n_jobs=n_jobs, backend="loky")(
		delayed(simulate_one_grid_point)(*task)
		for task in tqdm(tasks, desc="Simulation", total=len(tasks))
	)

	metrics = {
		f'{t.id}_{m_id}': np.zeros((N, N, rep))
		for t in transformations
		for m_id in METRICS_ID
	} | {
		f'{t.id}_GMM_score': np.zeros((N, N, rep))
		for t in transformations
	}

	for i, j, result in results:
		for key in result:
			metrics[key][i, j, :] = result[key]

	file = f"{model.__name__}_{rho}_{pi}".replace('.', '')
	path = f"Computation/{emb_mode_p22_path_str(emb_mode, p22)}/{file}"
	os.makedirs(path, exist_ok=True)
	np.savez_compressed(f"{path}/{batch}.npz", **metrics)

def simulate_in_line_with_one_varying_param(
	N = 100,
	R = 100,
	model = betaWSBM,
	rho   = 0.25,
	pi    = 0.25,
	p11   = 0.25,
	p12   = 0.5,
	t     = PowerTransform(1),
	varying_param = 'rho',
	varying_param_bounds = (0, 0.5)):

	try:
		if issubclass(varying_param, WeightTransform):
			assert isinstance(t, varying_param), "t must be of the same type as varying_param"
	except TypeError:
		assert isinstance(varying_param, str), "varying_param must be a string or a WeightTransform subclass"
	params = np.linspace(*varying_param_bounds, N)
	emb_mode, p22 = EMB_MODES[0], P22S[0]

	metrics_id = VANILLA_METRICS_ID + ['pi_1']

	def _run_for_vp(vp, rho=rho, pi=pi, p11=p11, p12=p12, t=t):
		if varying_param == 'rho':
			rho = vp
		elif varying_param == 'pi':
			pi = vp
		elif varying_param == 'p11':
			p11 = vp
		elif varying_param == 'p12':
			p12 = vp
		elif issubclass(varying_param, WeightTransform):
			t = varying_param(vp)
		else:
			raise ValueError(f"Unknown varying parameter: {varying_param}")

		m = model(rho, pi, (p11, p12), p22=p22)

		graphs = []
		for j in range(R):
			A, Z = m(42+j)
			G = TWSBMInstance(model = m, transformation = t, A = t(A), Z = Z, emb_mode = emb_mode)
			graphs.append(G.to_dict())
		metrics_for_vp = {m_id : {} for m_id in metrics_id}
		for m_id in metrics_id:
			metrics_for_vp[m_id]['mean'] = np.mean([g[m_id] for g in graphs])
			metrics_for_vp[m_id]['std']  = np.std([g[m_id]  for g in graphs])
		return (graphs[0], metrics_for_vp)

	n_jobs = max(1, int(os.cpu_count()*3/4))
	print(f"Using {n_jobs} threads (out of {os.cpu_count()})")
	all_results = Parallel(n_jobs=n_jobs, backend="loky")(
		delayed(_run_for_vp)(vp) for vp in tqdm(params, desc="Simulation"))
	
	graphs, metrics_for_vps = zip(*all_results)

	metrics = {m_id : {} for m_id in metrics_id}
	for m_id in metrics_id:
		metrics[m_id]['mean'] = np.array([m[m_id]['mean'] for m in metrics_for_vps])
		metrics[m_id]['std']  = np.array([m[m_id]['std']  for m in metrics_for_vps])

	fixed_params = {
		'model' : model,
		'rho'	: rho,
		'pi'	: pi,
		'p11'	: p11,
		'p12'	: p12,
		't'		: t,
	}
	varying_param_str = varying_param if isinstance(varying_param, str) else varying_param.__name__
	dico = {
		'fixed_params': fixed_params,
		'varying_param': varying_param_str,
		'varying_param_bounds': varying_param_bounds,
		'metrics': metrics,
		'graphs': graphs,
	}

	name = f'{model.name}_{t.id}_{varying_param_str}_{varying_param_bounds[0]}_{varying_param_bounds[1]}'
	path = f"Computation/data_for_gifs"
	os.makedirs(path, exist_ok=True)
	dump(dico, f"{path}/{name}.joblib", compress=('lz4', 5))
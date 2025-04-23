
import numpy as np
import time
import os

from joblib import Parallel, delayed
from tqdm import tqdm

from Objects.TWSBMInstance import *


def simulate_in_grid(N, batch, rep, model, model_params, transformations,
					 p22='fixed', emb_mode='sqrt-scaled'):
	
	def simulate_one_grid_point(i, j, p11, p12, rho, pi, model, rep, transformations,
							 emb_mode, p22, base_seed):
		metrics = {f'{t.id}_{m_id}' : np.zeros(rep) for t in transformations for m_id in METRICS_ID}

		m = model(rho, pi, (p11, p12), p22=p22)

		seed = base_seed
		for k in range(rep):
			A, Z = m(seed=seed)
			for t in transformations:
				G = TWSBMInstance(model=m, transformation=t, A=t(A), Z=Z, emb_mode=emb_mode)
				metrics[f'{t.id}_C_true'][k]  = G.C_true
				metrics[f'{t.id}_C_graph'][k] = G.C_graph
				metrics[f'{t.id}_C_embed'][k] = G.C_embedding
				metrics[f'{t.id}_Rand'][k]   = G.RAND

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

	results = Parallel(n_jobs=6, backend="loky")(
		delayed(simulate_one_grid_point)(*task)
		for task in tqdm(tasks, desc="Simulation", total=len(tasks))
	)

	metrics = {
		f'{t.id}_{m_id}': np.zeros((N, N, rep))
		for t in transformations
		for m_id in METRICS_ID
	}

	for i, j, result in results:
		for key in result:
			metrics[key][i, j, :] = result[key]

	file = f"{model.__name__}_{rho}_{pi}".replace('.', '')
	path = f"Computation/{emb_mode_p22_path_str(emb_mode, p22)}/{file}"
	os.makedirs(path, exist_ok=True)
	np.savez_compressed(f"{path}/{batch}.npz", **metrics)
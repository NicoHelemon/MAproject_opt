
import numpy as np
import time
import os

from Objects.TWSBMInstance import *

def r_dot(x):
	return str(x).replace('.', '')

def simulate_in_grid(N, rep, model, model_params, transformations, 
					 p22 = 'fixed', emb_mode = 'sqrt-scaled'):
	rho, pi = model_params

	p11_linspace = linspace_exclusive(0, 1, N)
	p12_linspace = linspace_exclusive(0, 1, N)

	metrics = {f'{t.id}_{m_id}' : np.zeros((N, N, rep)) for t in transformations for m_id in METRICS_ID}

	total_steps = N ** 2 * rep
	steps_done = 0

	start_time = time.time()

	for i, p11 in enumerate(p11_linspace):
		for j, p12 in enumerate(p12_linspace):
			m = model(rho, pi, (p11, p12), p22 = p22)
			for k in range(rep):
				seed = k * N ** 2 + i * N + j
				A, Z = m(seed = seed)
				for t in transformations:
					G = TWSBMInstance(model = m, transformation = t, A = t(A), Z = Z, emb_mode = emb_mode)
					metrics[f'{t.id}_C_true'][i, j, k]  = G.C_true
					metrics[f'{t.id}_C_graph'][i, j, k] = G.C_graph
					metrics[f'{t.id}_C_embed'][i, j, k] = G.C_embedding
					metrics[f'{t.id}_Rand'][i, j, k]   = G.RAND

				steps_done += 1
				if steps_done % N == 0:
					elapsed = time.time() - start_time
					fraction_done = steps_done / total_steps
					estimated_total_time = elapsed / fraction_done
					eta = estimated_total_time - elapsed
					elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
					eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
					print(f"Progress: {steps_done}/{total_steps} "
						f"({fraction_done*100:.1f}%). "
						f"Elapsed: {elapsed_str}. ETA: {eta_str}.")

	os.makedirs(f"Computation/{emb_mode_p22_path_str(emb_mode, p22)}/Grids", exist_ok = True)
	np.savez_compressed(f"Computation/Grids/{model.__name__}_{r_dot(rho)}_{r_dot(pi)}.npz", **metrics)

def simulate_in_line(N, rep, model, model_params, transformations, p11 = None, p12 = None,
					 p22 = 'fixed', emb_mode = 'sqrt-scaled'):
	rho, pi = model_params

	assert p11 is None or p12 is None
	if p11 is None: 
		p11_linspace = linspace_exclusive(0, 1, N)
	else:
		assert p11 > 0 and p11 < 1
		fixed_param = (f'{model.param_name}11', p11)
		p11_linspace = np.array([p11]*N)
	if p12 is None:
		p12_linspace = linspace_exclusive(0, 1, N)
	else:
		assert p12 > 0 and p12 < 1
		fixed_param = (f'{model.param_name}12', p12)
		p12_linspace = np.array([p12]*N)

	p_linspace = zip(p11_linspace, p12_linspace)

	metrics = {f'{t.id}_{m_id}' : np.zeros((N, rep)) for t in transformations for m_id in METRICS_ID}

	total_steps = N * rep
	steps_done = 0

	start_time = time.time()

	for i, (p11, p12) in enumerate(p_linspace):
		m = model(rho, pi, (p11, p12), p22 = p22)
		for j in range(rep):	
			seed = j * N + i
			A, Z = m(seed = seed)
			for t in transformations:
				G = TWSBMInstance(model = m, transformation = t, A = t(A), Z = Z, emb_mode = emb_mode)
				metrics[f'{t.id}_C_true'][i, j]  = G.C_true
				metrics[f'{t.id}_C_graph'][i, j] = G.C_graph
				metrics[f'{t.id}_C_embed'][i, j] = G.C_embedding
				metrics[f'{t.id}_Rand'][i, j]   = G.RAND

			steps_done += 1
			if steps_done % (max(1, N // 100)) == 0:
				elapsed = time.time() - start_time
				fraction_done = steps_done / total_steps
				estimated_total_time = elapsed / fraction_done
				eta = estimated_total_time - elapsed
				elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
				eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
				print(f"Progress: {steps_done}/{total_steps} "
					f"({fraction_done*100:.1f}%). "
					f"Elapsed: {elapsed_str}. ETA: {eta_str}.")
			
	metrics['fixed_param'] = fixed_param
				
	path = f"Computation/{emb_mode_p22_path_str(emb_mode, p22)}/Lines"
	os.makedirs(path, exist_ok = True)
	param_str, param_val = fixed_param
	file = f"{model.__name__}_{rho}_{pi}_{param_str}_{param_val}".replace('.', '')
	np.savez_compressed(f"{path}/{file}.npz", **metrics)
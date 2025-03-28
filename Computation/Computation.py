
import numpy as np
import time
import os

from Objects.TWSBMInstance import *

def r_dot(x):
	return str(x).replace('.', '')

def simulate_in_grid(N, model, model_params, transformations):
	rho, pi = model_params

	p11_linspace = np.linspace(0.01, 1, N)
	p12_linspace = np.linspace(0.01, 1, N)

	metrics = {f'{t.id}_{m_id}' : np.zeros((N, N)) for t in transformations for m_id in METRICS_ID}

	total_steps = N ** 2
	steps_done = 0

	start_time = time.time()

	for i, p11 in enumerate(p11_linspace):
		for j, p12 in enumerate(p12_linspace):
			
			m = model(rho, pi, (p11, p12))
			A, Z = m(42)
			for t in transformations:
				G = TWSBMInstance(model = m, transformation = t, A = t(A), Z = Z)
				metrics[f'{t.id}_C_true'][i, j]  = G.C_true
				metrics[f'{t.id}_C_graph'][i, j] = G.C_graph
				metrics[f'{t.id}_C_embed'][i, j] = G.C_embedding
				metrics[f'{t.id}_Rand'][i, j]   = G.RAND

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
				
	os.makedirs("Computation/Grids", exist_ok = True)
	np.savez_compressed(f"Computation/Grids/{model.__name__}_{r_dot(rho)}_{r_dot(pi)}.npz", **metrics)
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from scipy.stats import pearsonr
import matplotlib.patches as mpatches
import matplotlib.colors as colors

from WSBM import *

def get_metrics_in_grid(N, model, params, transformation, refinement=1, method='cubic'):
	assert isinstance(refinement, int) and refinement >= 1

	n, rho, Pi = params
	if model == betaWSBM:
		def model_on(Alpha):
			return model(n, rho, Pi, Alpha)
		
	elif model == lognormWSBM:
		def model_on(Sigma):
			return model(n, rho, Pi, Sigma)
		
	else:
		raise ValueError("Model not supported")
	
	p11_linspace = np.linspace(0.01, 1, N)
	p12_linspace = np.linspace(0.01, 1, N)

	C_true      = np.zeros((N, N))
	C_graph     = np.zeros((N, N))
	C_embedding = np.zeros((N, N))
	RAND        = np.zeros((N, N))

	total_steps = N ** 2
	steps_done = 0

	start_time = time.time()

	for i, p11 in enumerate(p11_linspace):
		for j, p12 in enumerate(p12_linspace):
			P = np.array([[p11, p12], 
						  [p12, 1.0]])
			
			G = TWSBInstance(model_on(P), transformation, seed = 42)

			C_true[i, j]      = G.C_true
			C_graph[i, j]     = G.C_graph
			C_embedding[i, j] = G.C_embedding
			RAND[i, j]        = G.RAND
		
			steps_done += 1
			if steps_done % 100 == 0:
				elapsed = time.time() - start_time
				fraction_done = steps_done / total_steps
				estimated_total_time = elapsed / fraction_done
				eta = estimated_total_time - elapsed
				elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
				eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
				print(f"Progress: {steps_done}/{total_steps} "
					f"({fraction_done*100:.1f}%). "
					f"Elapsed: {elapsed_str}. ETA: {eta_str}.")
				
	Corr_RAND   = [pearsonr(C.ravel(), RAND.ravel()) for C in [C_true, C_graph, C_embedding]]
	Corr_C_true = [pearsonr(C_true.ravel(), C.ravel()) for C in [C_graph, C_embedding]]
				
	def clip_and_refine(Z, clip_min = True, clip_max = True):
		a_min = Z.min() if clip_min else None
		a_max = Z.max() if clip_max else None
		refined_Z = refine_grid(p11_linspace, p12_linspace, Z, refinement=refinement, method=method)
		return np.clip(refined_Z, a_min = a_min, a_max = a_max)

	if refinement > 1:
		C_true      = clip_and_refine(C_true, clip_max = False)
		C_graph     = clip_and_refine(C_graph, clip_max = False)
		C_embedding = clip_and_refine(C_embedding, clip_max = False)
		RAND        = clip_and_refine(RAND)

	Grids = [C_true, C_graph, C_embedding, RAND]
				
	return Grids, Corr_RAND, Corr_C_true, N * refinement

def refine_grid(x_linspace, y_linspace, Z, refinement=4, method='cubic'):
	Ny, Nx = Z.shape

	X, Y = np.meshgrid(x_linspace, y_linspace)
	X_refined, Y_refined = np.meshgrid(np.linspace(x_linspace.min(), x_linspace.max(), Nx * refinement), 
							   np.linspace(y_linspace.min(), y_linspace.max(), Ny * refinement))

	Z_refined = griddata(np.column_stack((X.ravel(), Y.ravel())), Z.ravel(), (X_refined, Y_refined), method=method)
	
	return Z_refined

def plot_metrics_heatmaps(model, model_params, transformation,
					   Grids, Corr_RAND,
					   N):
	"""
	Displays four heatmaps (2×2) for:
	  • RAND index
	  • True Chernoff matrix
	  • Chernoff graph-estimator
	  • Chernoff embedding-estimator

	The colorbar limits are determined by excluding the extreme values based on the given percentile
	(default is 90), and are shared among all four heatmaps.
	"""

	C_true, C_graph, C_embedding, RAND = Grids
	Corr_RAND = [(None, None)] + Corr_RAND
	
	# Determine parameter name based on the model
	param = "alpha" if model == betaWSBM else "sigma"
	
	# Create list of matrices and their corresponding titles
	matrices = [RAND, C_true, C_graph, C_embedding]
	titles = [
		"RAND index",
		"True Chernoff information",
		"Chernoff graph-estimation",
		"Chernoff embedding-estimation"
	]
	
	# Compute common color limits by excluding the extreme values

	values = np.concatenate([mat.ravel() for mat in matrices[1:]])
	vmin = np.percentile(values, 5)
	vmax = np.percentile(values, 80)
	norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
	#norm = colors.PowerNorm(gamma=0.1, vmin=vmin, vmax=vmax, clip=True)
	
	# Plot setup: create a 2×2 subplot figure
	fig, axes = plt.subplots(2, 2, figsize=(11.5, 10))
	axes = axes.flatten()

	n, rho, Pi = model_params
	global_title = (
		f"model: {model.__name__}, transformation: {transformation.name}\n"
		f"n = {n}, ρ = {rho}, Π = {Pi.tolist()}"
	)
	fig.suptitle(global_title, fontsize=14)
	
	# Plot each heatmap with shared color limits and formatted ticks
	for ax, mat, title, (corr, pval) in zip(axes, matrices, titles, Corr_RAND):
		if title == "RAND index":
			# For RAND index, use the 'Reds' colormap with default scaling
			sns.heatmap(mat, cmap='Reds', ax=ax, cbar=True)
		else:
			# For the Chernoff matrices, use 'Blues' colormap with shared vmin/vmax limits
			sns.heatmap(mat, cmap='Blues', ax=ax, norm=norm, cbar=True)
			pval_ineq = '<' if pval <= 0.05 else '>'
			pval_check = '✓' if pval <= 0.05 else '✗'
			patch = mpatches.Patch(color='none', 
							   label=f"corr(R, C) = {corr:.3f},  pval {pval_ineq} 0.05 {pval_check}")
			ax.legend(handles=[patch], loc='lower right')
		ax.set_title(title)
		ax.set_xticks(np.linspace(0, N, 5))
		ax.set_xticklabels(np.linspace(0, 1, 5).round(2))
		ax.set_yticks(np.linspace(0, N, 5))
		ax.set_yticklabels(np.linspace(0, 1, 5).round(2))
		ax.set_xlabel(f"{param}12")
		ax.set_ylabel(f"{param}11")
		ax.invert_yaxis()
		# Make spines visible with a specific line width
		for spine in ax.spines.values():
			spine.set_visible(True)
			spine.set_linewidth(1)
	
	plt.tight_layout()
	plt.show()


def plot_bias_heatmaps(model, model_params, transformation,
                       Grids, Corr_C_true,
                       N,
                       abs_mode=False):
    """
    Displays six heatmaps (2×3) comparing C_graph and C_embedding against C_true:
    
      • Column 1: Bias (or absolute bias if abs_mode=True)
      • Column 2: Relative Bias (or absolute relative bias if abs_mode=True)
      • Column 3: Log-Ratio Bias

    Colorbar limits are determined by excluding extreme 10% values,
    using the 5th and 80th percentiles of the data.
    For signed metrics (when abs_mode is False) a symmetric logarithmic normalization (SymLogNorm)
    is used to center zero.
    For nonnegative measures (when abs_mode is True) LogNorm is applied.
    For the log-ratio (which is already on a log scale), a linear normalization (Normalize) is used.
    
    The global title includes model parameters and the correlations (with p-values)
    for the graph- and embedding-based estimates.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import seaborn as sns

    # Unpack grids (the fourth element is unused)
    C_true, C_graph, C_embedding, _ = Grids
    eps = np.finfo(C_true.dtype).eps

    # Compute raw metrics
    bias_graph = C_graph - C_true
    bias_emb   = C_embedding - C_true

    rel_graph = bias_graph / (C_true + eps)
    rel_emb   = bias_emb   / (C_true + eps)

    log_graph = np.log(C_graph / (C_true + eps))
    log_emb   = np.log(C_embedding / (C_true + eps))

    # If absolute mode is enabled, work with absolute values for bias/relative bias
    if abs_mode:
        bias_graph = np.abs(bias_graph)
        bias_emb   = np.abs(bias_emb)
        rel_graph  = np.abs(rel_graph)
        rel_emb    = np.abs(rel_emb)

    # --- Compute normalization limits via percentiles ---

    # For bias
    values_bias = np.concatenate([bias_graph.ravel(), bias_emb.ravel()])
    vmin_bias = np.percentile(values_bias, 5)
    vmax_bias = np.percentile(values_bias, 90)

    # For relative bias
    values_rel = np.concatenate([rel_graph.ravel(), rel_emb.ravel()])
    vmin_rel = np.percentile(values_rel, 5)
    vmax_rel = np.percentile(values_rel, 90)

    # For log-ratio bias (log-transformed, so use linear normalization)
    values_log = np.concatenate([log_graph.ravel(), log_emb.ravel()])
    vmin_log = np.percentile(values_log, 5)
    vmax_log = np.percentile(values_log, 90)

    # Configure normalization
    if abs_mode:
        norm_bias = colors.LogNorm(vmin=vmin_bias, vmax=vmax_bias, clip=True)
        norm_rel  = colors.LogNorm(vmin=vmin_rel, vmax=vmax_rel, clip=True)
    else:
        # For signed data, symmetrize the limits around zero
        lim_bias = max(abs(vmin_bias), abs(vmax_bias))
        norm_bias = colors.SymLogNorm(
            linthresh=lim_bias * 0.05 if lim_bias != 0 else 1e-3,
            vmin=-lim_bias, vmax=lim_bias, base=10, clip=True)
        
        lim_rel = max(abs(vmin_rel), abs(vmax_rel))
        norm_rel = colors.SymLogNorm(
            linthresh=lim_rel * 0.05 if lim_rel != 0 else 1e-3,
            vmin=-lim_rel, vmax=lim_rel, base=10, clip=True)
    
    # For log-ratio, use linear normalization
    bound = max(abs(vmin_log), abs(vmax_log))
    norm_log = colors.TwoSlopeNorm(vmin=-bound, vcenter = 0, vmax=bound)

    # --- Plot setup ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    n, rho, Pi = model_params

    # Extract correlations and p-values for graph and embedding.
    # Each element in Corr_C_true is assumed to be a tuple (corr, pval)
    corr_graph, pval_graph = Corr_C_true[0]
    corr_emb, pval_emb     = Corr_C_true[1]
    
    # Prepare p-value inequality and check symbols for graph estimator
    pval_ineq_graph = '<' if pval_graph <= 0.05 else '>'
    pval_check_graph = '✓' if pval_graph <= 0.05 else '✗'
    # Prepare p-value inequality and check symbols for embedding estimator
    pval_ineq_emb = '<' if pval_emb <= 0.05 else '>'
    pval_check_emb = '✓' if pval_emb <= 0.05 else '✗'
    
    global_title = (
        f"model: {model.__name__}, transformation: {transformation.name}\n"
        f"n = {n}, ρ = {rho}, Π = {Pi.tolist()}\n"
        f"Corr (C_true, C_graph) = {corr_graph:.3f} (pval {pval_ineq_graph} 0.05 {pval_check_graph}),\n"
        f"Corr (C_true, C_embedding) = {corr_emb:.3f} (pval {pval_ineq_emb} 0.05 {pval_check_emb})"
    )
    fig.suptitle(global_title, fontsize=16)

    titles = ["Bias", "Relative Bias", "Log-Ratio Bias"]
    # For bias and relative bias: use sequential colormap if abs_mode, diverging otherwise.
    cmaps = [("Blues" if abs_mode else "RdBu"),
             ("Blues" if abs_mode else "RdBu"),
             "RdBu"]

    data_graph = [bias_graph, rel_graph, log_graph]
    data_emb   = [bias_emb,   rel_emb,   log_emb]
    norms      = [norm_bias,  norm_rel,  norm_log]

    param = "alpha" if model == betaWSBM else "sigma"

    for i, title in enumerate(titles):
        sns.heatmap(data_graph[i], ax=axes[0, i], cmap=cmaps[i], norm=norms[i], cbar=True)
        axes[0, i].set_title(f"Chernoff graph-estimator\n{title}")
        sns.heatmap(data_emb[i], ax=axes[1, i], cmap=cmaps[i], norm=norms[i], cbar=True)
        axes[1, i].set_title(f"Chernoff embedding-estimator\n{title}")

        for ax in (axes[0, i], axes[1, i]):
            ax.set_xticks(np.linspace(0, N, 5))
            ax.set_xticklabels(np.linspace(0, 1, 5).round(2))
            ax.set_yticks(np.linspace(0, N, 5))
            ax.set_yticklabels(np.linspace(0, 1, 5).round(2))
            ax.set_xlabel(f"{param}12")
            ax.set_ylabel(f"{param}11")
            ax.invert_yaxis()

    for ax in axes.flatten():
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)

    plt.tight_layout()
    plt.show()
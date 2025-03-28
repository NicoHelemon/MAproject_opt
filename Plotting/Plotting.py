
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from scipy.stats import spearmanr
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

from Objects.WSBM import *
from .StringHelper import *

RHOS_PIS_MODELS = list(product(RHOS[:1], PIS[:1], MODELS))
n = 1000
K = 2

def plot_embedding(rho, pi, metrics, n = n):
	def label_permutation(Z_true, Z_pred):
		matches_no_swap = np.sum(Z_true == Z_pred)
		matches_swap = np.sum(Z_true == (1 - Z_pred))
		if matches_swap > matches_no_swap:
			return 1 - Z_pred
		else:
			return Z_pred
	
	def switch(mode='Truth'):
		n_rows, n_cols = len(metrics), len(list(metrics.values())[0].values())
		fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 + 2.5*n_cols, 2 + 3*n_rows))
		global_title = model_str(n, rho, pi)
		fig.suptitle(global_title, fontsize=20)
		for i, (model, model_metrics) in enumerate(metrics.items()):
			axes[i, 0].set_ylabel(model.name + "\n")
			for j, G in enumerate(model_metrics.values()):
				Z, X, Z_hat, M, Σ, C_true, C_graph, C_embedding, RAND = G.Z, G.X, G.Z_hat, G.M, G.Σ, G.C_true, G.C_graph, G.C_embedding, G.RAND
				Z_hat = label_permutation(Z, Z_hat)
				ax = axes[i][j]
				plt.sca(ax)
				if mode == 'Truth':
					ax.scatter(X[:, 0], X[:, 1], c=Z, cmap='bwr', marker='.', alpha=0.2)
				else:
					ax.scatter(X[:, 0], X[:, 1], c=Z_hat, cmap='bwr', marker='.', alpha=0.2)
				ax.set_xticks([])
				ax.set_yticks([])
				for mean, cov in zip(M, Σ):
					eigenvalues, eigenvectors = np.linalg.eigh(cov)
					angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
					width, height = 2 * np.sqrt(6 * eigenvalues)
					ellip = plt.matplotlib.patches.Ellipse(
						mean, width, height, angle=angle, edgecolor='k', facecolor='none', linestyle='solid'
					)
					ax.add_patch(ellip)
				transform_name = G.transform_name + "\n" if i == 0 else ""
				title = (
					transform_name
					+ f"RI: {RAND:.2f} "
					+ f"CT: {C_true:.5f}\n"
					+ f"CG: {C_graph:.5f} "
					+ f"CE: {C_embedding:.5f}"
				)
				ax.set_title(title)
		plt.tight_layout()
		plt.show()

	widgets.interact(switch, mode=['Truth', 'Prediction'])

def plot_scatter_Rand_vs_Chernoff(metrics, n_points_ratio_displayed = 1.0, n = n, K = K):
	skip = int(np.ceil(1/n_points_ratio_displayed))
	λ, b = 1, -7
	def sigmoid(x, λ = λ, b = b): return 1/(1+np.exp(-λ*(x-b)))

	n_points =  len(metrics['Rand'][::skip])
	n_points_ratio_displayed_str = ""
	if n_points_ratio_displayed < 1.0:
		n_points_ratio_displayed_str = f" ({n_points_ratio_displayed * 100:.0f}% of total)"

	fig, axes = plt.subplots(2, 3, figsize=(18, 14))
	global_title = (f"Rand vs Chernoff information scatter plots\n"
					f"For WSBM graphs of size n = {n}, with K = {2} communities\n"
					f"Number of points displayed (per plot): {n_points}{n_points_ratio_displayed_str}\n")
	fig.suptitle(global_title, fontsize=20)

	for i, (ax, m_id) in enumerate(zip(axes[0], METRICS_ID[1:])):
		for t in TRANSFORMS:
			x, y = metrics[t][m_id], metrics[t]['Rand']
			x, y = x[::skip], y[::skip]
			x, y = x[x > 0], y[x > 0]

			x = sigmoid(np.log(x))
			ax.scatter(x, y, s=0.5, alpha=0.5, color=TRANSFORMS_CMAP[t],
			  label=f'{t.name}\nS-corr = {spearmanr(x, y)[0]:.2f}')

		ax.set_title(f'{METRICS_MAP[m_id]}\nSpearman correlation = {spearmanr(metrics[m_id], metrics["Rand"])[0]:.2f}\n')
		#ax.set_xlabel(f'Sigmoid{{λ={λ}}}(ln({m_id}) - {abs(b)})\n', fontsize=12)
		if i == 0:
			ax.set_ylabel(METRICS_MAP['Rand'], fontsize=12)
		ax.set_xlim(0, 1)
		ax.grid(True, linewidth=0.5)
		leg = ax.legend(scatterpoints=1, markerscale=8)
		for hand in leg.legend_handles:
			hand.set_alpha(1)
		for spine in ax.spines.values():
			spine.set_visible(True)
			spine.set_linewidth(1)

	for i, (ax, m_id) in enumerate(zip(axes[1], METRICS_ID[1:])):
		for rho, pi, model in RHOS_PIS_MODELS:
			x, y = metrics[(rho, pi, model)][m_id], metrics[(rho, pi, model)]['Rand']
			x, y = x[::skip], y[::skip]
			x, y = x[x > 0], y[x > 0]

			x = sigmoid(np.log(x))
			ax.scatter(x, y, s=0.5, alpha=0.5, color = RHOS_PIS_MODELS_CMAP[(rho, pi, model)],
			  label=f'ρ={rho} π={pi} {model.__name__[:-4].capitalize()}\nS-corr = {spearmanr(x, y)[0]:.2f}')

		if i == 0:
			ax.set_ylabel(METRICS_MAP['Rand'], fontsize=12)
		ax.set_xlim(0, 1)
		ax.set_xlabel(f'\n Sigmoid(ln({METRICS_ID_COSMETIC_MAP[m_id]}))\n', fontsize=12)
		ax.grid(True, linewidth=0.5)
		ax.legend(scatterpoints=1, markerscale=8)
		for hand in leg.legend_handles:
			hand.set_alpha(1)
		for spine in ax.spines.values():
			spine.set_visible(True)
			spine.set_linewidth(1)

	#plt.tight_layout(rect=[0, 0, 1, 0.93])
	plt.tight_layout()
	plt.show()

def plot_metrics_heatmap(rho, pi, model, transformation, metrics, shared = False, log = False, corr_info = True, n = n):
	if shared:
		values = np.concatenate([metrics[m_id].ravel() for m_id in METRICS_ID[1:]])
		vmin, vmax = np.min(values[values > 0]), np.max(values)
	else:
		vmin = vmax = None
	
	fig, axes = plt.subplots(2, 2, figsize=(11.5, 10))
	fig.set_dpi(300)
	axes = axes.flatten()

	suptitle = f"Metrics heatmaps\n" + model_str(n, rho, pi, model, transformation)
	fig.suptitle(suptitle, fontsize=14)
	
	for i, (ax, m_id) in enumerate(zip(axes, METRICS_ID)):
		metric_grid = metrics[m_id]
		N = metric_grid.shape[0]
		if m_id == 'Rand':
			sns.heatmap(metric_grid, cmap='Reds', ax=ax, cbar=True)
		else:
			if log:
				norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
			else:
				norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
			sns.heatmap(metric_grid, cmap='Blues', ax=ax, cbar=True, norm=norm)
			if corr_info:
				corr, auc_corr, partial_corrs = metrics['Correlation']['Rand'][m_id]
				ns, corrs = partial_corrs
				min_corrs, max_corrs = np.min(corrs), np.max(corrs)
				abs_max = min_corrs if abs(min_corrs) > abs(max_corrs) else max_corrs
				ax_ins = ax.inset_axes([0.65, 0.675, 0.25, 0.25])
				ax_ins.plot(ns, corrs)
				ax_ins.set_xlabel(f"Top % C-largest\npoints considered", fontsize=6)
				ax_ins.set_ylabel(f'S-corr(R, C top%)', fontsize=6)
				ax_ins.set_ylim(-1.05, 1.05)
				ax_ins.set_xticks([0, 50, 100])
				ax_ins.set_yticks([-1, 0, 1])
				ax_ins.tick_params(axis='both', labelsize=4)
				ax_ins.axhline(corr, color='blue', linestyle='-', linewidth=1, label=f'S-corr    = {corr:.2f}')
				ax_ins.axhline(abs_max, color='red', linestyle='-', linewidth=1, label=f"max S-c = {abs_max:.2f}")
				ax_ins.axhline(0, color='black', linestyle='-',  linewidth=0.5)

				ax_ins.fill_between(ns, corrs, 0, facecolor='yellow', alpha=0.3)
				ax_ins.plot([], [], marker='s', linestyle='None', markerfacecolor='yellow', 
					label=f"AUC       = {auc_corr:.2f}", alpha=0.5)
				leg = ax_ins.legend(fontsize=8, bbox_to_anchor=(0.635, -1, 0.5, 0.5))

				for j, handle in enumerate(leg.legend_handles):
					handle.set_markersize(5)
					handle.set_markeredgewidth(0.5)
					handle.set_markeredgecolor('black')
					handle.set_markerfacecolor(handle.get_color())
					if j == 2: 
						handle.set_marker('s')
						handle.set_markerfacecolor('yellow')
						handle.set_alpha(0.5)

				ax_ins.set_title(f"Spearman‑corr(R, {METRICS_ID_COSMETIC_MAP[m_id]})", fontsize=8)

		ax.set_title(METRICS_MAP[m_id])
		ax.set_xticks(np.linspace(0, N, 5))
		ax.set_xticklabels(np.linspace(0, 1, 5).round(2))
		ax.set_yticks(np.linspace(0, N, 5))
		ax.set_yticklabels(np.linspace(0, 1, 5).round(2))
		if i == 2 or i == 3:
			ax.set_xlabel(f'{model.param_name}{sub(" 12")}', fontsize = 12)
		if i == 0 or i == 2:
			ax.set_ylabel(f'{model.param_name}{sub(" 11")}', fontsize = 12)
		ax.invert_yaxis()
		for spine in ax.spines.values():
			spine.set_visible(True)
			spine.set_linewidth(1)
	
	plt.tight_layout()
	plt.show()

def plot_bias_heatmap(rho, pi, model, transformation, metrics, log = True, n = n):
	fig, axes = plt.subplots(4, 2, figsize=(12, 20))

	suptitle = (f"Bias of Chernoff information estimators vs true Chernoff information\n"
				f"{model_str(n, rho, pi, model, transformation)}")
	fig.suptitle(suptitle, fontsize=14)

	m = metrics['Correlation']['C_true']
	min_corr = min(np.min(m['C_graph'][2][1]), np.min(m['C_embed'][2][1]))

	for ax, m_id in zip(axes[0], METRICS_ID[2:]):
		corr, auc_corr, partial_corrs = metrics['Correlation']['C_true'][m_id]
		ns, corrs = partial_corrs
		min_corrs, max_corrs = np.min(corrs), np.max(corrs)
		abs_max = min_corrs if abs(min_corrs) > abs(max_corrs) else max_corrs
		ax.plot(ns, corrs)
		ax.set_xlabel(f"Top % {METRICS_ID_COSMETIC_MAP[m_id]}-largest points considered")
		ax.set_ylabel(f'S-correlation({METRICS_ID_COSMETIC_MAP["C_true"]}, {METRICS_ID_COSMETIC_MAP[m_id]} Top%)')
		ax.set_ylim(-1.05, 1.05)
		ax.set_xticks([0, 25, 50, 75, 100])
		ax.set_yticks([-1, -0.5, 0, 0.5, 1])
		ax.axhline(corr, color='blue', linestyle='-', linewidth=1, label=f'S-corr    = {corr:.2f}')
		ax.axhline(abs_max, color='red', linestyle='-', linewidth=1, label=f"max S-c = {abs_max:.2f}")
		ax.axhline(-1, color='black', linestyle='-',  linewidth=0.5)
		ax.axhline(0, color='black', linestyle='-',  linewidth=0.5)
		ax.axhline(1, color='black', linestyle='-',  linewidth=0.5)
		ax.fill_between(ns, corrs, 0, facecolor='yellow', alpha=0.3)
		ax.plot([], [], marker='s', linestyle='None', markerfacecolor='yellow', label=f"AUC       = {auc_corr:.2f}")
		leg = ax.legend(loc = 'upper right')

		for j, handle in enumerate(leg.legend_handles):
			handle.set_markersize(5)
			handle.set_markerfacecolor(handle.get_color())
			if j == 2: 
				handle.set_marker('s')
				handle.set_markerfacecolor('yellow')
				handle.set_markersize(8)
				handle.set_alpha(0.5)
			handle.set_markeredgewidth(0.5)
			handle.set_markeredgecolor('black')

		ax.set_title(f'Spearman‑corr({METRICS_ID_COSMETIC_MAP["C_true"]}, {METRICS_ID_COSMETIC_MAP[m_id]})')
	
	for axbias, bias in zip(axes[1:], BIASES):
		for ax, m_id in zip(axbias, METRICS_ID[2:]):
			bias_grid = metrics['Bias'][m_id][bias]
			N = bias_grid.shape[0]
			
			values = np.concatenate((metrics['Bias']['C_graph'][bias].ravel(), 
							metrics['Bias']['C_embed'][bias].ravel()))
			vmin, vmax = np.percentile(values, 5), np.percentile(values, 95)
			bound = max(abs(vmin), abs(vmax))
			if bias == 'log':
				norm = colors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
				cmap = 'RdBu'
			else:
				if log == True:
					norm = colors.SymLogNorm(linthresh=bound * 0.05 if bound != 0 else 1e-3,
											vmin=-bound, vmax=bound, base=10, clip=True)
				else:
					norm = colors.Normalize(vmin=vmin, vmax=vmax)
				cmap = 'Blues'
				
			sns.heatmap(bias_grid, ax=ax, cmap=cmap, norm=norm)
			ax.set_title(f'{BIASES_MAP[bias]} {METRICS_ID_COSMETIC_MAP[m_id]} vs {METRICS_ID_COSMETIC_MAP["C_true"]}')
			ax.set_xticks(np.linspace(0, N, 5))
			ax.set_xticklabels(np.linspace(0, 1, 5).round(2))
			ax.set_yticks(np.linspace(0, N, 5))
			ax.set_yticklabels(np.linspace(0, 1, 5).round(2))
			if bias == 'log':
				ax.set_xlabel(f'{model.param_name}{sub(" 12")}', fontsize = 12)
			if m_id == 'C_graph':
				ax.set_ylabel(f'{model.param_name}{sub(" 11")}', fontsize = 12)
			ax.invert_yaxis()
			for spine in ax.spines.values():
				spine.set_visible(True)
				spine.set_linewidth(1)


	plt.tight_layout()
	plt.show()

def plot_best_transform_heatmaps(rho, pi, model, metrics, n=n):
	rows = ['C_graph-Best Transform', 'C_embed-Best Transform']
	cols = ['Arg', 'Rand', 'Regret']

	fig, axes = plt.subplots(2, 3, figsize=(16, 10), gridspec_kw={'width_ratios': [0.8, 1, 1]})
	fig.set_dpi(300)
	fig.suptitle(
		f"Best‑Transform Metrics on Model: {model.__name__}\n" + model_str(n, rho, pi),
		fontsize=14
	)

	for i, row in enumerate(rows):
		for j, col in enumerate(cols):
			ax = axes[i, j]
			grid = metrics[row][col]
			N = grid.shape[0]

			if col == 'Arg':
				cmap = ListedColormap(list(TRANSFORMS_CMAP.values())[:-1])
				norm = BoundaryNorm(np.arange(-0.5, cmap.N + 0.5, 1), cmap.N)
				ax.pcolormesh(grid, cmap = cmap, norm = norm, shading='auto')
				area = np.bincount(grid.ravel(), minlength = len(TRANSFORMS)) / N ** 2
				area_map = dict(zip(TRANSFORMS, area))
				sorted_TRANSFORMS = sorted(TRANSFORMS, key=lambda t: area_map[t], reverse=True)
				handles = [Patch(facecolor=TRANSFORMS_CMAP[t], label=f'{t.id}: {area_map[t]:.2f}') 
			   			   for t in sorted_TRANSFORMS]
				ax.legend(title = 'Transforms: Area',
					handles=handles, loc="upper left", handlelength=1, handleheight=1)
			elif col == 'Rand':
				norm = colors.Normalize(vmin=0, vmax=1, clip=True)
				sns.heatmap(grid, ax=ax, norm=norm, cmap = 'Reds')

				mean_rand_transforms_map = {t: np.mean(metrics[t]['Rand'].ravel()) for t in TRANSFORMS}
				mean_rand_transforms_map['Argmax'] = np.mean(grid.ravel())
				case = lambda t : t.id if t != 'Argmax' else 'Best'
				sorted_TRANSFORMS = sorted(TRANSFORMS + ['Argmax'], key=lambda t: mean_rand_transforms_map[t], reverse=True)
				handles = [Patch(facecolor=TRANSFORMS_CMAP[t], label=f'{case(t)}: {mean_rand_transforms_map[t]:.2f}') 
			   			   for t in sorted_TRANSFORMS]
				ax.legend(title = 'Transforms: Avg(Rand)}', handles=handles, loc="upper left", handlelength=1, handleheight=1)

			else:
				norm = colors.Normalize(vmin=0, vmax=1, clip=True)
				sns.heatmap(grid, ax=ax, norm=norm, cmap = 'Purples')
				title = (f'Avg(Regret) = {np.mean(grid.ravel()):.2f}\n'
						 f'Area(Regret > 0) = {np.count_nonzero(grid > 0) / N ** 2:.2f}')
				ax.legend(title = title, loc="upper left")

			title = f'{METRICS_ID_COSMETIC_MAP[row[:7]]}-Best Transform{": " + col if col != "Arg" else ""}'
			ax.set_title(title)

			ticks = np.linspace(0, N, 5)
			labels = np.linspace(0, 1, 5).round(2)
			ax.set_xticks(ticks); ax.set_xticklabels(labels)
			ax.set_yticks(ticks); ax.set_yticklabels(labels)
			if col != 'Arg':
				ax.invert_yaxis()

			if i == 1:
				ax.set_xlabel(f"{model.param_name}{sub(' 12')}", fontsize=12)
			if j == 0:
				ax.set_ylabel(f"{model.param_name}{sub(' 11')}", fontsize=12)

			for spine in ax.spines.values():
				spine.set_visible(True)
				spine.set_linewidth(1)

	plt.tight_layout()
	plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from scipy.stats import spearmanr
from matplotlib import colors
from ipywidgets import interact
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from matplotlib import cm
from pathlib import Path
import matplotlib as mpl

from Objects.WSBM import *
from .StringHelper import *
from Computation.ExtraMetrics import partial_correlation

n = 1000
K = 2

class Plotter:
	def __init__(self, folder_path):
		self.folder_path = f"Plots/{folder_path}"
		Path(self.folder_path).mkdir(parents=True, exist_ok=True)


	def save_file(self, out_path, file_name, dpi = 200, eps = False, close = True):
		Path(f'{self.folder_path}/{out_path}').mkdir(parents = True, exist_ok = True)

		file_name = file_name.replace('.', '')
		plt.savefig(f'{self.folder_path}/{out_path}/{file_name}.png', dpi=dpi)
		if eps: plt.savefig(out_path + f'/{file_name}.eps'.replace(' ', '_'), dpi=dpi)
		if close: plt.close()

	def plot_embedding(self, rho, pi, metrics, n = n, subfolder = ''):
		def label_permutation(Z_true, Z_pred):
			matches_no_swap = np.sum(Z_true == Z_pred)
			matches_swap = np.sum(Z_true == (1 - Z_pred))
			if matches_swap > matches_no_swap: return 1 - Z_pred
			else: return Z_pred
		
		def switch(mode):
			n_rows, n_cols = len(metrics), len(list(metrics.values())[0].values())
			fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 + 2.5*n_cols, 2 + 3*n_rows))
			mode_str = "True community labels" if mode == 'Truth' else "Predicted community labels"
			global_title = mode_str + '\n' + model_str(n, rho, pi)
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
						ax.scatter(X[:, 0], X[:, 1], c=Z_hat, cmap='PuOr', marker='.', alpha=0.2)
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
			self.save_file(f'Embeddings/{subfolder}', f'{rho}_{pi}', dpi=400)

		switch('Truth')
		#switch('Prediction')

	def plot_scatter_Rand_vs_Chernoff(self, metrics, n_points_ratio_displayed=1.0,
									C_transform = 'Sigmoid-Ln', n=n, K=K):

		if C_transform == 'Sigmoid-Ln':
			λ, b = 1, -7
			def transform(x, λ=λ, b=b):
				return 1 / (1 + np.exp(-λ * (np.log(x) - b)))
		elif C_transform == 'Ln':
			def transform(x):
				return np.log(x)
		else:
			raise ValueError("Invalid C_transform. Choose 'Sigmoid-Ln' or 'Ln'.")

		skip = int(np.ceil(1 / n_points_ratio_displayed))
		n_points = len(metrics['Rand'][::skip])
		n_points_ratio_displayed_str = ""
		#if n_points_ratio_displayed < 1.0:
		#	n_points_ratio_displayed_str = f" ({n_points_ratio_displayed * 100:.0f}% du total)"

		fig, axes = plt.subplots(4, 3, figsize=(18, 24))
		global_title = (f"Rand vs Chernoff information scatter plots\n"
						f"For WSBM graphs of size n = {n}, with K = {K} communities\n\n\n")
		fig.suptitle(global_title, fontsize=20)
		sub_title = f"Number of points displayed (per plot): {n_points}{n_points_ratio_displayed_str}\n"
		fig.text(0.5, 0.935, sub_title, ha='center', fontsize=14)
		
		def scatter_polishing(ax, i):
			if i == 0: ax.set_ylabel(METRICS_MAP['Rand'], fontsize=12)
			if C_transform == 'Sigmoid-Ln':
				vmin = 0.0
				vmax = 1.0
			else:
				vals = np.concatenate([metrics[m_id] for m_id in CHERNOFFS_ID])
				vals = vals[vals > 0]
				vmin = np.log(np.min(vals))
				vmax = np.log(np.max(vals))
			ax.set_xlim(vmin, vmax)
			ax.grid(True, linewidth=0.5)
			leg = ax.legend(scatterpoints=1, markerscale=8, loc = 'upper left')
			for hand in leg.legend_handles:
				hand.set_alpha(1)
			for spine in ax.spines.values():
				spine.set_visible(True)
				spine.set_linewidth(1)

		def get_xysc(key, m_id):
			x, y = metrics[key][m_id][::skip], metrics[key]['Rand'][::skip]
			x, y = x[x>0], y[x>0]
			corr, auc_corr, (_, corrs) = partial_correlation(x, y)
			min_corrs, max_corrs = np.min(corrs), np.max(corrs)
			abs_max = min_corrs if abs(min_corrs) > abs(max_corrs) else max_corrs
			x = transform(x)
			return x, y, corr, auc_corr, abs_max

		for i, (ax, m_id) in enumerate(zip(axes[0], CHERNOFFS_ID)):
			cmap = cm.viridis
			nc = len(RHOS)
			for j, rho in enumerate(RHOS):
				x, y, corr, auc_corr, max_corr = get_xysc(f'rho:{rho}', m_id)
				label_str = (f'ρ = {rho}: '
							f'S-corr = {corr:.2f}')#\n'
							#f'auc-c = {auc_corr:.2f}, '
							#f'max-c = {max_corr:.2f}')
				ax.scatter(x, y, s=0.5, alpha=0.5,
						color = cmap(j / nc),
						label = label_str)
				
			scatter_polishing(ax, i)

			x, y = metrics[m_id], metrics['Rand']
			x, y = x[x>0], y[x>0]
			corr, auc_corr, (_, corrs) = partial_correlation(x, y)
			min_corrs, max_corrs = np.min(corrs), np.max(corrs)
			abs_max = min_corrs if abs(min_corrs) > abs(max_corrs) else max_corrs
			title = (f'{METRICS_MAP[m_id]}\nSpearman correlation = {corr:.2f}')#\n'
			#f'AUC(Partial S-corr) = {auc_corr:.2f}, Max(Partial S-corr) = {abs_max:.2f}')
			ax.set_title(title, fontsize=12)

		for i, (ax, m_id) in enumerate(zip(axes[1], CHERNOFFS_ID)):
			cmap = cm.plasma
			nc = len(PIS)
			for j, pi in enumerate(PIS):
				x, y, corr, auc_corr, max_corr = get_xysc(f'pi:{pi}', m_id)
				label_str = (f'π = {pi}: '
							f'S-corr = {corr:.2f}')#\n'
							#f'auc-c = {auc_corr:.2f}, '
							#f'max-c = {max_corr:.2f}')
				ax.scatter(x, y, s=0.5, alpha=0.5,
						color = cmap(j / nc),
						label = label_str)
				
			scatter_polishing(ax, i)

		for i, (ax, m_id) in enumerate(zip(axes[2], CHERNOFFS_ID)):
			cmap = cm.inferno
			nc = len(MODELS)
			for j, model in enumerate(MODELS):
				x, y, corr, auc_corr, max_corr = get_xysc(model, m_id)
				label_str = (f'{model.name}: '
							f'S-corr = {corr:.2f}')#\n'
							#f'auc-c = {auc_corr:.2f}, '
							#f'max-c = {max_corr:.2f}')
				ax.scatter(x, y, s=0.5, alpha=0.5,
						color = cmap(j / nc),
						label = label_str)
				
			scatter_polishing(ax, i)

		def transform_str(m_id):
			m_id_str = CHERNOFFS_ID_COSMETIC_MAP[m_id]
			if C_transform == 'Sigmoid-Ln':
				return f'\nSigmoid(ln({m_id_str}))'
			else:
				return f'\nln({m_id_str})'

		for i, (ax, m_id) in enumerate(zip(axes[3], CHERNOFFS_ID)):
			for t in TRANSFORMS:
				x, y, corr, auc_corr, max_corr = get_xysc(t, m_id)
				label_str = (f'{t.name}: '
							f'S-corr = {corr:.2f}')#\n'
							#f'auc-c = {auc_corr:.2f}, '
							#f'max-c = {max_corr:.2f}')
				ax.scatter(x, y, s=0.5, alpha=0.5,
						color = TRANSFORMS_CMAP[t],
						label = label_str)
				
			scatter_polishing(ax, i)
			ax.set_xlabel(transform_str(m_id), fontsize=12)

		plt.tight_layout()
		self.save_file('', f'Rand_vs_Chernoff_{C_transform}', dpi=400)

	def plot_metrics_heatmap(self, rho, pi, model, transformation, metrics, shared = False, log = False, corr_info = True, n = n):
		if shared:
			values = np.concatenate([metrics[m_id] for m_id in CHERNOFFS_ID])
			vmin, vmax = np.min(values[values > 0]), np.max(values)
			# use quantiles to set vmin and vmax
		else:
			vmin = vmax = None
		
		fig, axes = plt.subplots(2, 2, figsize=(11.5, 10))
		axes = axes.flatten()

		suptitle = f"Metrics heatmaps\n" + model_str(n, rho, pi, model, transformation)
		fig.suptitle(suptitle, fontsize=14)
		
		for i, (ax, m_id) in enumerate(zip(axes, METRICS_ID)):
			metric_grid = metrics[m_id]
			N = metric_grid.shape[0]
			if m_id == 'Rand':
				sns.heatmap(metric_grid, cmap='Reds', ax=ax, vmin = -0.05, vmax = 1, cbar=True)
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

					ax_ins.set_title(f"Spearman‑corr(R, {CHERNOFFS_ID_COSMETIC_MAP[m_id]})", fontsize=8)

			ax.set_title(METRICS_MAP[m_id])
			ax.set_xticks(np.linspace(0, N, 5))
			ax.set_xticklabels(np.linspace(0, 1, 5).round(2))
			ax.set_yticks(np.linspace(0, N, 5))
			ax.set_yticklabels(np.linspace(0, 1, 5).round(2))
			if i == 2 or i == 3:
				ax.set_xlabel(f'{model.param_name}{sub("12")}', fontsize = 12)
			if i == 0 or i == 2:
				ax.set_ylabel(f'{model.param_name}{sub("11")}', fontsize = 12)
			ax.invert_yaxis()
			for spine in ax.spines.values():
				spine.set_visible(True)
				spine.set_linewidth(1)
		
		plt.tight_layout()
		m_str = f'{model.name}_{rho}_{pi}'.replace('.', '')
		self.save_file(f'Heatmaps_by_MT/Metrics/By_Model/{m_str}', f'{transformation.id}', dpi=400, close=False)
		self.save_file(f'Heatmaps_by_MT/Metrics/By_Transform/{transformation.id}', m_str, dpi=400)

	def plot_bias_heatmap(self, rho, pi, model, transformation, metrics, log = True, n = n):
		fig, axes = plt.subplots(4, 2, figsize=(12, 20))

		suptitle = (f"Bias of Chernoff information estimators vs true Chernoff information\n"
					f"{model_str(n, rho, pi, model, transformation)}")
		fig.suptitle(suptitle, fontsize=14)

		for ax, m_id in zip(axes[0], METRICS_ID[2:]):
			corr, auc_corr, partial_corrs = metrics['Correlation']['C_true'][m_id]
			ns, corrs = partial_corrs
			min_corrs, max_corrs = np.min(corrs), np.max(corrs)
			abs_max = min_corrs if abs(min_corrs) > abs(max_corrs) else max_corrs
			ax.plot(ns, corrs)
			ax.set_xlabel(f"Top % {CHERNOFFS_ID_COSMETIC_MAP[m_id]}-largest points considered")
			ax.set_ylabel(f'S-correlation({CHERNOFFS_ID_COSMETIC_MAP["C_true"]}, {CHERNOFFS_ID_COSMETIC_MAP[m_id]} Top%)')
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

			ax.set_title(f'Spearman‑corr({CHERNOFFS_ID_COSMETIC_MAP["C_true"]}, {CHERNOFFS_ID_COSMETIC_MAP[m_id]})')
		
		for axbias, bias in zip(axes[1:], BIASES):
			for ax, m_id in zip(axbias, METRICS_ID[2:]):
				bias_grid = metrics['Bias'][m_id][bias]
				N = bias_grid.shape[0]
				
				values = np.concatenate((metrics['Bias']['C_graph'][bias], 
								metrics['Bias']['C_embed'][bias]))
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
				ax.set_title(f'{BIASES_MAP[bias]} {CHERNOFFS_ID_COSMETIC_MAP[m_id]} vs {CHERNOFFS_ID_COSMETIC_MAP["C_true"]}')
				ax.set_xticks(np.linspace(0, N, 5))
				ax.set_xticklabels(np.linspace(0, 1, 5).round(2))
				ax.set_yticks(np.linspace(0, N, 5))
				ax.set_yticklabels(np.linspace(0, 1, 5).round(2))
				if bias == 'log':
					ax.set_xlabel(f'{model.param_name}{sub("12")}', fontsize = 12)
				if m_id == 'C_graph':
					ax.set_ylabel(f'{model.param_name}{sub("11")}', fontsize = 12)
				ax.invert_yaxis()
				for spine in ax.spines.values():
					spine.set_visible(True)
					spine.set_linewidth(1)

		plt.tight_layout()
		m_str = f'{model.name}_{rho}_{pi}'.replace('.', '')
		self.save_file(f'Heatmaps_by_MT/Bias/By_Model/{m_str}', f'{transformation.id}', dpi=400, close=False)
		self.save_file(f'Heatmaps_by_MT/Bias/By_Transform/{transformation.id}', m_str, dpi=400)

	def plot_best_transform_heatmaps(self, rho, pi, model, metrics, n=n):
		chernoffs = CHERNOFFS_ID.copy()
		cols = ['Arg', 'Rand', 'Regret']

		fig, axes = plt.subplots(3, 3, figsize=(16, 16), gridspec_kw={'width_ratios': [0.8, 1, 1]})
		fig.suptitle(
			f"Best‑Transform Metrics on Model: {model.name}\n" + model_str(n, rho, pi),
			fontsize=14
		)

		for i, C in enumerate(chernoffs):
			for j, col in enumerate(cols):
				ax = axes[i, j]
				grid = metrics[f'{C}-Best Transform'][col]
				N = grid.shape[0]

				if col == 'Arg':
					cmap = ListedColormap(list(TRANSFORMS_CMAP.values()))
					norm = BoundaryNorm(np.arange(-0.5, cmap.N + 0.5, 1), cmap.N)
					ax.pcolormesh(grid, cmap = cmap, norm = norm, shading='auto')
					area_map = dict(zip(TRANSFORMS, metrics[f'{C}-Best Transform']['Transform Area']))
					sorted_TRANSFORMS = sorted(TRANSFORMS, key=lambda t: area_map[t], reverse=True)
					handles = [Patch(facecolor=TRANSFORMS_CMAP[t], label=f'{t.id}: {area_map[t]:.2f}') 
							for t in sorted_TRANSFORMS]
					ax.legend(title = 'Transforms: Area',
						handles=handles, loc="upper left", handlelength=1, handleheight=1)
				elif col == 'Rand':
					norm = colors.Normalize(vmin=0, vmax=1, clip=True)
					sns.heatmap(grid, ax=ax, norm=norm, cmap = 'Reds')

					mean_rand_transforms_map = {t: np.mean(metrics[t]['Rand']) for t in TRANSFORMS}
					mean_rand_transforms_map[C] = metrics[f'{C}-Best Transform']['Rand Avg']
					case = lambda t : t.id if t in TRANSFORMS else t
					sorted_TRANSFORMS = sorted(TRANSFORMS + [C], key=lambda t: mean_rand_transforms_map[t], reverse=True)
					handles = [Patch(facecolor=CMAP[t], label=f'{case(t)}: {mean_rand_transforms_map[t]:.2f}') 
							for t in sorted_TRANSFORMS]
					ax.legend(title = 'Transforms: Avg(Rand)}', handles=handles, loc="upper left", handlelength=1, handleheight=1)

				else:
					norm = colors.Normalize(vmin=0, vmax=1, clip=True)
					sns.heatmap(grid, ax=ax, norm=norm, cmap = 'Purples')
					avg_regret = f"Avg(Reg) = {metrics[f'{C}-Best Transform']['Regret Avg']:.2f}"
					area_positive_r = f"Area(Reg>0) = {metrics[f'{C}-Best Transform']['Regret Area']:.2f}"
					avg_positive_r = f"Avg(Reg[Reg>0]) = {metrics[f'{C}-Best Transform']['Regret Avg on Positive Regret']:.2f}"

					handle_avg = mpatches.Patch(facecolor='none', edgecolor='none', label=avg_regret)
					handle_area = mpatches.Patch(facecolor='none', edgecolor='none', label=area_positive_r)
					handle_avg_positive_r = mpatches.Patch(facecolor='none', edgecolor='none', label=avg_positive_r)

					ax.legend(handles=[handle_avg, handle_area, handle_avg_positive_r], loc="upper left",
							handlelength=0, handleheight=0)

				title = f'{CHERNOFFS_ID_COSMETIC_MAP[C]}-Best Transform{": " + col if col != "Arg" else ""}'
				ax.set_title(title)

				ticks = np.linspace(0, N, 5)
				labels = np.linspace(0, 1, 5).round(2)
				ax.set_xticks(ticks); ax.set_xticklabels(labels)
				ax.set_yticks(ticks); ax.set_yticklabels(labels)
				if col != 'Arg':
					ax.invert_yaxis()

				if i == 1:
					ax.set_xlabel(f"{model.param_name}{sub('12')}", fontsize=12)
				if j == 0:
					ax.set_ylabel(f"{model.param_name}{sub('11')}", fontsize=12)

				for spine in ax.spines.values():
					spine.set_visible(True)
					spine.set_linewidth(1)

		plt.tight_layout()
		self.save_file(f'Best_Transform/{model.name}', f'{model.name}_{rho}_{pi}', dpi=400)

	def modulable_bar_plots(self, ax, L, fontsize=10):
		x_offset = 0
		x_centers = []

		for tuple in L:
			space_0 = tuple[0][-1]
			x_offset_start = x_offset + space_0
			for area, height, height_2, text, text_2, color, space in tuple:
				x_offset += space
				ax.bar(x_offset, height, width=area, color=color, align='edge')
				ax.bar(x_offset, height_2, width=area, bottom=height, color='grey', align='edge')
				ax.text(x_offset + area/2, height/2, str(text), ha='center', va='center', fontsize=fontsize)
				ax.text(x_offset + area/2, height + height_2/2, str(text_2), ha='center', va='center', fontsize=fontsize)
				x_offset += area

			x_centers.append(x_offset_start + (x_offset - x_offset_start) / 2)
			
		ax.set_xticks(x_centers)

		return x_offset + space_0
	
	def plot_transforms_rand(self, model, metrics_g, mode = 'No regret'):
		_, ax = plt.subplots(figsize=(8, 6))

		chernoffs = CHERNOFFS_ID.copy()
		transforms = TRANSFORMS.copy()

		transforms_with_argmax_C = transforms + [f"{chernoff}-Best Transform" for chernoff in chernoffs]
		transforms_with_argmax_C.sort(key=lambda t: metrics_g[t]['Rand Avg'], reverse=True)

		m_transforms = {t : metrics_g[t] for t in transforms_with_argmax_C}

		def color_helper(t):
			if isinstance(t, str):
				return CHERNOFFS_CMAP[t.split('-')[0]]
			else:
				return TRANSFORMS_CMAP[t]
			
		if mode == 'No regret':
			L = [((1,
				m['Rand Avg'], 
				#m['Regret Avg'],
				0,
				f"{m['Rand Avg']:.2f}",
				#f"{m['Regret Avg']:.2f}",
				"",
				color_helper(t), 
				0.1),) for t, m in m_transforms.items()]
		elif mode == 'With regret':
			L1 = [(1 - m['Regret Area'],
				m['Rand Avg on Null Regret'],
				0,
				#f"{m['Rand Avg on Null Regret']:.2f}",
				"",
				"",
				color_helper(t),
				0.1) for t, m in m_transforms.items()]
			L2 = [(m['Regret Area'],
				m['Rand Avg on Positive Regret'],
				m['Regret Avg on Positive Regret'],
				#f"{m['Rand Avg on Positive Regret']:.2f}",
				#f"{m['Regret Avg on Positive Regret']:.2f}",
				"", "",
				color_helper(t),
				0.025) for t, m in m_transforms.items()]

			L = zip(L1, L2)

			L1_perc = [(1 - m['Regret Area'], -0.05, 0, f"  {((1 - m['Regret Area']) * 100):.0f}%", 
						"", "white", 0.1) for t, m in m_transforms.items()]
			L2_perc = [(m['Regret Area'], -0.05, 0, f"  {(m['Regret Area'] * 100):.0f}%", 
						"", "white", 0.025) for t, m in m_transforms.items()]

			L_perc = zip(L1_perc, L2_perc)
			self.modulable_bar_plots(ax, L_perc, fontsize=8)
		else:
			raise ValueError("Invalid mode. Choose 'No regret' or 'With regret'.")

		x_offset = self.modulable_bar_plots(ax, L)

		def label_helper(t):
			if isinstance(t, str):
				return CHERNOFFS_ID_COSMETIC_MAP[t.split('-')[0]]
			else:
				return t.id

		ax.hlines(0, 0, x_offset, color='black', lw=0.5)
		ax.set_xticklabels([label_helper(t) for t in transforms_with_argmax_C])
		ax.set_ylabel('Avg Rand')
		ax.set_xlim(-0.25, x_offset+0.25)
		
		if mode == 'No regret':
			ax.set_title(f'Transformations Average Rand (over {len(RHOS_PIS_MODELS)} graph models)\n')
			Avg_Rand_Max = np.mean(metrics_g['Rand Max'])
			ax.hlines(Avg_Rand_Max, 0, 
					x_offset, color='black', lw=1, label=f'Avg(Rand Max): {Avg_Rand_Max:.2f}', linestyle='--')
			ax.legend()
			ax.set_ylim(-0.05, 1)
			ax.set_xlabel('\nTransformations')

		if mode == 'With regret':
			ax.set_title(f'Transformations Average Rand & Regret (on {model.name} Model)\nConditioned on Null Regret vs Positive Regret\n')
			ax.set_xlabel('\n\nTransformations')
			ax.text(
				x_offset/2,
				-0.09,
				"(Null Regret vs Positive Regret Ratio)", 
				transform=ax.get_xaxis_transform(),
				ha="center", va="top",
				fontsize=8)
			ax.set_ylim(-0.075, 1)
			ax.set_ylim(-0.05, 1.05)
			ax.legend(handles=[mpatches.Patch(color='grey', label='Rand Regret')], loc='upper left', fontsize=8)

		mode = 'No_Regret' if mode == 'No regret' else 'With_Regret'
		plt.tight_layout()
		self.save_file(f'Best_Transform/{model.name}', f'Transforms_Rands_{mode}', dpi=400)

	def plot_transforms_rand_for_best_transform(self, model, metrics_g, chernoff):
		_, ax = plt.subplots(figsize=(8, 6))

		m_chernoff = metrics_g[f"{chernoff}-Best Transform"]

		def t_area(t):
			return m_chernoff['Transform Area'][TRANSFORMS.index(t)]

		transforms = TRANSFORMS.copy()
		transforms.sort(key=lambda t: t_area(t), reverse=True)
		transforms = [t for t in transforms if t_area(t) > 0.01]

		m_transforms = {t : m_chernoff[t] for t in transforms}
			
		L = [((t_area(t), 
			m['Rand Avg'], 
			m['Regret Avg'],
			f"{m['Rand Avg']:.2f}" if t_area(t) > 0.06 and m['Rand Avg'] > 0.045 else "",
			f"{m['Regret Avg']:.2f}" if t_area(t) > 0.06 and m['Regret Avg'] > 0.045 else "",
			TRANSFORMS_CMAP[t], 
			0.01),) for t, m in m_transforms.items()]

		x_offset = self.modulable_bar_plots(ax, L)

		L_perc = [((t_area(t), -0.05, 0, f"  {((t_area(t)) * 100):.0f}%", "", "white", 0.01),) for t in transforms]
		
		self.modulable_bar_plots(ax, L_perc, fontsize=8)

		ax.hlines(0, 0, x_offset, color='black', lw=0.5)
		ax.set_xticklabels([t.id for t in transforms], rotation=45)
		ax.text(
				x_offset/2,
				-0.1,
				"(Transformation Selection Proportions)", 
				transform=ax.get_xaxis_transform(),
				ha="center", va="top",
				fontsize=8)
		ax.set_xlabel('\nTransformations')
		ax.set_ylabel('Avg Rand')
		ax.set_ylim(-0.075, 1.05)
		ax.set_xlim(-0.05, x_offset+0.05)
		ax.set_title(f'{CHERNOFFS_ID_COSMETIC_MAP[chernoff]}-Best Transform underlying Transformation Selection\nWith average (on {model.name} Model) Rand, Regret & Selection by Transformation\n')

		ax.legend(handles=[mpatches.Patch(color='grey', label='Rand Regret')], loc='upper right', fontsize=8)

		plt.tight_layout()
		self.save_file(f'Best_Transform/{model.name}', f'{chernoff}_Best_Transform_Transforms_Rands', dpi=400)

	def plot_best_transform_lines(self, rho, pi, model, metrics_l, p11, p12, p22 = 'fixed', n = n):
		def darken_color(color, factor=0.5, alpha=0.5):
			rgb = np.array(mpl.colors.to_rgb(color))
			darkened = rgb * factor
			return (darkened[0], darkened[1], darkened[2], alpha)

		assert isinstance(p11, int) and p12 == 'varying' or p11 == 'varying' and isinstance(p12, int)

		if p11 == 'varying':
			metrics = metrics_l['p12'][p12][(rho, pi, model)]
		else:
			metrics = metrics_l['p11'][p11][(rho, pi, model)]

		chernoffs = CHERNOFFS_ID.copy()
		N = len(metrics[TRANSFORMS[0]]['Rand'])
		x = linspace_exclusive(0, 1, N)

		if p11 == 'varying':
			p11 = model.param_name + sub('11')
			varying_param = p11
			p12 = round(x[p12], 2)		
		else:
			p11 = round(x[p11], 2)
			p12 = model.param_name + sub('12')
			varying_param = p12
		p22 = model.p22_fixed if p22 == 'fixed' else p11

		fig, axes = plt.subplots(2, 1, figsize=(10, 12))
		suptitle_str = f"Best Transform Metrics on Model: {model.instance_name_str(param_init(p11, p12, p22))}\n{model_str(n, rho, pi)}"
		fig.suptitle(suptitle_str, fontsize=14)

		ax1, ax2 = axes
		for t in TRANSFORMS:
			y = metrics[t]['Rand']
			y_std = metrics[t]['std']['Rand']
			ax1.plot(x, y,
					label=f"{t.name}:\n Avg(Rand) = {np.mean(y):.2f}, Lead Ratio = {metrics[t]['Ahead Ratio']:.2f}",
					color=TRANSFORMS_CMAP[t],
					linewidth=2)
			ax1.fill_between(x, y - y_std, y + y_std, color=TRANSFORMS_CMAP[t], alpha=0.1)
			ax2.plot(x, y,
					label=f"",
					color=darken_color(TRANSFORMS_CMAP[t]),
					linewidth=1)
		
		for i, C in enumerate(chernoffs):
			mBestC = metrics[f'{C}-Best Transform']
			y_best = mBestC['Rand']

			shift = 0.02 * (i - (len(chernoffs) - 1) / 2)
			
			ax2.plot(x, y_best + shift,
					label=(f"{C}-Best Transform:\n"
							f"Avg(Rand) = {mBestC['Rand Avg']:.2f}\nLead Ratio = {mBestC['Ahead Ratio']:.2f}\n"
							f"Area(Regret) = {mBestC['Regret Avg']:.2f}"),
					color=CHERNOFFS_CMAP[C],
					linewidth=5,
					alpha = 0.75)
			
			
		ax1.set_title(f"Transforms RAND", fontsize=12)
		ax2.set_title(f"Best Transforms RAND", fontsize=12)
		ax2.set_xlabel(sub(varying_param), fontsize=14)
		for ax in [ax1, ax2]:
			ax.set_ylim(-0.1, 1.05)
			ax.set_xlim(0, 1)
			ax.set_ylabel("Rand index", fontsize=12)
			ax.set_xticks(np.linspace(0, 1, 5))
			ax.legend(handlelength=2, handleheight=2, fontsize=9, loc='upper left')
		
		plt.tight_layout()
		plt.gcf().set_dpi(400)
		plt.show()

	def plot_chernoffs_lines(self, rho, pi, model, metrics_l, p11, p12, p22 = 'fixed', n = n):
		assert isinstance(p11, int) and p12 == 'varying' or p11 == 'varying' and isinstance(p12, int)

		if p11 == 'varying':
			metrics = metrics_l['p12'][p12][(rho, pi, model)]
		else:
			metrics = metrics_l['p11'][p11][(rho, pi, model)]

		chernoffs = CHERNOFFS_ID.copy()
		N = len(metrics[TRANSFORMS[0]]['Rand'])
		x = linspace_exclusive(0, 1, N)

		if p11 == 'varying':
			p11 = model.param_name + sub('11')
			varying_param = p11
			p12 = round(x[p12], 2)		
		else:
			p11 = round(x[p11], 2)
			p12 = model.param_name + sub('12')
			varying_param = p12
		p22 = model.p22_fixed if p22 == 'fixed' else p11

		fig, axes = plt.subplots(3, 1, figsize=(10, 15))
		suptitle_str = f"Best Transform Metrics on Model: {model.instance_name_str(param_init(p11, p12, p22))}\n{model_str(n, rho, pi)}"
		fig.suptitle(suptitle_str, fontsize=14)
		
		chernoffs = CHERNOFFS_ID.copy()
		
		for ax, C in zip(axes, chernoffs):
			for t in TRANSFORMS:
				y =  metrics[t][C]
				y_std = metrics[t]['std'][C]
				ax.plot(x, y, label = t.name, color = TRANSFORMS_CMAP[t], linewidth=2)
				ax.fill_between(x, y - y_std, y + y_std, color=TRANSFORMS_CMAP[t], alpha=0.1)
				ax.set_xlim(0, 1)
				ax.set_xticks(np.linspace(0, 1, 5))
				ax.set_ylabel(CHERNOFFS_ID_COSMETIC_MAP[C], fontsize=12)
				ax.legend(handlelength=2, handleheight=2, fontsize=9, loc='upper left')
			
			
		axes[0].set_title(f"Chernoffs", fontsize=12)
		axes[2].set_xlabel(sub(varying_param), fontsize=14)
			
		plt.tight_layout()
		plt.gcf().set_dpi(400)
		plt.show()

	def plot_line_sliding(self, plotting_function, rho, pi, model, metrics_l, p22 = 'fixed', n = n, slider = 'p11'):		
		description = model.param_name + sub(slider[1:])
		N = len(metrics_l['p11'][0][RHOS_PIS_MODELS[0]][TRANSFORMS[0]]['Rand'])
		x = linspace_exclusive(0, 1, N)
		step = x[1] - x[0]  # assume uniform grid

		def plot_best_transform_at(p_real):
			# find the nearest integer index
			j = int(np.round((p_real - x[0]) / step))
			return plotting_function(
				rho, pi, model, metrics_l,
				p11=j if slider_name=='p11' else 'varying',
				p12=j if slider_name=='p12' else 'varying',
				p22=p22, n=n
			)

		# now build a slider whose values *are* the x’s
		slider_name = 'p11'
		slider = widgets.FloatSlider(
			value=x[50],
			min=x[0],
			max=x[-1],
			step=step,
			description=description,
			continuous_update=False
		)

		interact(plot_best_transform_at, p_real=slider)
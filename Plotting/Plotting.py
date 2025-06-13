
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from scipy.stats import spearmanr
from scipy.linalg import orthogonal_procrustes
from matplotlib import colors
from ipywidgets import interact
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm, PowerNorm
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patches as mpatches
from matplotlib import cm
from pathlib import Path
import matplotlib as mpl
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from matplotlib.artist import Artist
from brokenaxes import brokenaxes
from matplotlib.lines import Line2D
import math
import matplotlib.gridspec as gridspec
from joblib import load, dump
import imageio
from scipy.stats import gaussian_kde
import os
from scipy.ndimage import convolve1d
from moviepy import ImageSequenceClip
from Objects.WSBM import *
from .StringHelper import *
from Computation.ExtraMetrics import partial_correlation

n = 1000
K = 2


# 2) Handler qui dessine le dégradé
class HandlerColormap(HandlerBase):
	def __init__(self, cmap, norm, orientation='horizontal', **kw):
		super().__init__(**kw)
		self.cmap = cmap
		self.norm = norm
		self.orientation = orientation

	def create_artists(self, legend, orig_handle,
					   xdescent, ydescent, width, height,
					   fontsize, trans):
		grad = np.linspace(0, 1, 256).reshape(1, -1)
		if self.orientation == 'vertical':
			grad = grad.T
		img = legend.axes.imshow(
			grad,
			cmap=self.cmap,
			norm=self.norm,
			extent=[xdescent, xdescent+width, ydescent, ydescent+height],
			origin='lower',
			transform=trans,
			aspect='auto'
		)
		return [img]

class Plotter:
	def __init__(self, folder_path = "", eps = False):
		eps_str = "eps" if eps else "png"
		folder_path_str = f"/{folder_path}" if folder_path != "" else ""
		self.folder_path = f"Plots{folder_path_str}/{eps_str}"
		self.eps = eps
		Path(self.folder_path).mkdir(parents=True, exist_ok=True)


	def save_file(self, out_path, file_name, dpi = 200, close = True):
		Path(f'{self.folder_path}/{out_path}').mkdir(parents = True, exist_ok = True)

		file_name = file_name.replace('.', '')
		if not self.eps:
			plt.savefig(f'{self.folder_path}/{out_path}/{file_name}.png', dpi=dpi)
		else:
			plt.savefig(f'{self.folder_path}/{out_path}/{file_name}.eps', dpi=dpi)
		if close: plt.close()

	def plot_embedding(self, rho, pi, metrics, n = n, subfolder = '', 
					mode = 'Truth', ellipse = True, q_outliers = 0, show_stats = True):
		def label_permutation(Z_true, Z_pred):
			matches_no_swap = np.sum(Z_true == Z_pred)
			matches_swap = np.sum(Z_true == (1 - Z_pred))
			if matches_swap > matches_no_swap: return 1 - Z_pred
			else: return Z_pred
		
		def switch(mode):
			n_rows, n_cols = len(metrics), len(list(metrics.values())[0].values())
			fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 + 3*n_cols, 2 + 3*n_rows), squeeze=False)
			mode_str = "True community labels" if mode == 'Truth' else "Predicted community labels"
			underyling_model = list(metrics.keys())[0]
			global_title = mode_str + f' on {underyling_model.model_name_with_law_params()}\n' + model_str(n, rho, pi)
			if self.eps: global_title = empty_string_except(global_title)
			fig.suptitle(global_title, fontsize=20)
			for i, (model, model_metrics) in enumerate(metrics.items()):
				axes[i, 0].set_ylabel(model.param_matrix_str + '\n', fontsize=8)

				C_trues     = [G.C_true     for G in model_metrics.values()]
				C_graphs    = [G.C_graph    for G in model_metrics.values()]
				C_embeddings= [G.C_embedding for G in model_metrics.values()]
				j_ct, j_cg, j_ce = map(int, (np.argmax(C_trues), np.argmax(C_graphs), np.argmax(C_embeddings)))

				for j, G in enumerate(model_metrics.values()):
					Z, X, Z_hat, M, Σ, C_true, C_graph, C_embedding, RAND = G.Z, G.X, G.Z_hat, G.M, G.Σ, G.C_true, G.C_graph, G.C_embedding, G.RAND
					gated_GMM_score = sigmoid_w95(G.GMM_score, -5, 10)
					Z_hat = label_permutation(Z, Z_hat)
					ax = axes[i][j]
					plt.sca(ax)

					distances = np.linalg.norm(X - X.mean(axis=0), axis=1)
					idx_outliers = np.argsort(distances)[-int(len(X) * q_outliers):] if q_outliers > 0 else []
					mask = np.ones(len(X), dtype=bool)
					mask[idx_outliers] = False
					X, Z, Z_hat = X[mask], Z[mask], Z_hat[mask]

					t = np.clip(RAND, 0, 1)**2
					u = np.clip(gated_GMM_score, 0, 1)
					base = np.array([1 - t, 1, 1 - t])
					grey = np.array([0.5, 0.5, 0.5])
					facecolor = u * base + (1 - u) * grey

					ax.scatter(X[:, 0],
								X[:, 1],
								c=Z if mode=='Truth' else Z_hat,
								cmap='bwr' if mode=='Truth' else ListedColormap(["deepskyblue","hotpink"]),
								marker='.',
								alpha=0.2 + (1 - u) * 0.4 if show_stats else 0.2)
					
					ax.set_xticks([])
					ax.set_yticks([])

					if ellipse:
						for mean, cov in zip(M, Σ):
							eigenvalues, eigenvectors = np.linalg.eigh(cov)
							angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
							width, height = 2 * np.sqrt(6 * eigenvalues)
							ellip = plt.matplotlib.patches.Ellipse(
								mean, width, height, angle=angle, edgecolor='k', facecolor='none', linestyle='solid'
							)
							ax.add_patch(ellip)
					Π_hat = list(np.sort(np.diag(G.Π_hat)).round(2))
					stats = [f"ARI:  {RAND:.2f}",
							 f"ΠẐ:   {Π_hat}",
							 f"GS:   {G.GMM_score:.2f}",
							 f"CT:    {C_true:.5f}",
							 f"CG:   {C_graph:.5f}",
							 f"CE:   {C_embedding:.5f}"]
					#if q_outliers > 0:
					#	stats.append(f'\n{q_outliers*100:.1f}% X outside')
					#for stat in stats:
					#	ax.plot([], [], ' ', label=stat, linestyle = None, marker = "")
					if show_stats:
						ax.set_facecolor(facecolor)
						invisible_handles = [Line2D([], [], color='none') for _ in stats]
						leg = ax.legend(
							handles=invisible_handles,
							labels=stats,
							loc="upper left",
							fontsize=8,
							handlelength=0,
							handletextpad=0,
						)
						for h in invisible_handles:
							h.set_visible(False)
						for txt in leg.get_texts():
							s = txt.get_text()
							if (j == j_ct and s.startswith("CT:")) or (j == j_cg and s.startswith("CG:")) or (j == j_ce and s.startswith("CE:")):
								txt.set_fontweight("bold")

					if i == 0:
						ax.set_title(G.transform_name + "\n", fontsize=12)
			plt.tight_layout()
			rho_str, pi_str = float_to_str(rho), float_to_str(pi)
			mode_str = "" if mode == 'Truth' else f"_pred"
			ellipse_str = "" if ellipse else "_no_ellipse"
			show_stats_str = "" if show_stats else "_no_stats"
			self.save_file(f'Embeddings/{subfolder}', f'{pi_str}_{rho_str}{mode_str}{ellipse_str}{show_stats_str}', dpi=400)

		switch(mode)

	def plot_correlation(self, model, metrics_g, m_id1, m_id2, color = 'Reds'):
		col_labels = list(product(RHOS, PIS))
		row_labels = TRANSFORMS_EXT
		data = np.zeros((len(row_labels), len(col_labels)))

		for i, t in enumerate(row_labels):
			for j, (rho, pi) in enumerate(col_labels):
				data[i, j] = metrics_g[model][(rho, pi, t)]['Correlation'][m_id1][m_id2]

		df   = pd.DataFrame(data, index=row_labels, columns=col_labels)
		df_t = df.T

		index = df_t.index
		rhos  = [ρ for ρ,_ in index]
		pis   = [π for _,π in index]

		unique_rhos    = sorted(set(rhos), key=rhos.index)
		group_indices  = {ρ: [i for i,(ρ_,_) in enumerate(index) if ρ_==ρ]
						for ρ in unique_rhos}
		group_centers  = {ρ: np.mean(idxs) for ρ,idxs in group_indices.items()}
		group_bounds   = {ρ: (min(idxs)-0.5, max(idxs)+0.5) 
						for ρ,idxs in group_indices.items()}

		n0, m0 = df_t.shape
		df_t['pi_total'] = [
			metrics_g[model][f'pi:{π}']['Correlation'][m_id1][m_id2] if 3 <= i < 6
			else np.nan for i, (ρ, π) in enumerate(df_t.index)
		]
		df_t['rho_total'] = [
			metrics_g[model][f'rho:{ρ}']['Correlation'][m_id1][m_id2] if i % 3 == 1
			else -metrics_g[model][f'rho:{ρ}']['Correlation'][m_id1][m_id2] for i, (ρ, π) in enumerate(df_t.index)
		]

		df_t.loc['transform_total'] = (
			[metrics_g[model][t]['Correlation'][m_id1][m_id2]
			for t in df_t.columns[:-2]]
			+ [np.nan, np.nan]
		)

		fig, ax = plt.subplots(
			figsize=(
				max(6, len(df_t.columns)*0.5),
				max(6, len(df_t.index)*0.25)
			)
		)
		cmap = plt.get_cmap(color)
		im   = ax.imshow(np.abs(df_t.values), cmap=cmap, vmin=0, vmax=1, aspect='equal')

		ax.hlines(n0 - 0.5, -0.5, m0 + 2 - 0.5, color='black', linewidth=1)
		ax.vlines(m0 - 0.5, -0.5, n0 + 1 - 0.5, color='black', linewidth=1)
		ax.vlines(m0 - 0.5 + 1, -0.5, n0 + 1 - 1.5, color='black', linewidth=1)

		ax.set_xticks(np.arange(len(df_t.columns) - 2))
		ax.set_xticklabels([c.id for c in df_t.columns[:-2]], rotation=45, fontsize=8)
		ax.set_xlabel('Transformations t', fontsize=10)
		ax.xaxis.tick_top()
		ax.xaxis.set_label_position('top')

		ax.set_yticks(np.arange(len(df_t.index) - 1))
		ax.set_yticklabels([
			f"{idx[1]:.2f}"
			for idx in df_t.index[:-1]
		], fontsize=6)
		ax.yaxis.set_tick_params(length=4)
		ax.set_ylabel('ρ × π\n\n\n', fontsize=10, rotation=90)

		for ρ in unique_rhos:
			y0, y1 = group_bounds[ρ]
			ax.hlines([y0, y1], -0.5, len(df_t.columns)-0.5,
					colors='black', linewidth=1)
			ax.text(-1.5, group_centers[ρ], f"{ρ:.2f}",
					va='center', ha='right', fontsize=8)
			
		n0, m0 = df_t.shape

		for i in range(n0):
			for j in range(m0):
				ax.text(
					j, i,
					f"{df_t.iat[i, j]:.2f}" if j < m0 - 2 or df_t.iat[i, j] > 0 else "",
					ha='center', va='center',
					fontsize=8, color='black',
					fontweight='bold' if (i == n0 - 1) or (j == m0 - 1) or (j == m0 - 2) else 'normal'
				)

		plt.title(f"{model.name}-WSBM: Spearman's correlation Heatmap\n" + 
			f"{METRICS_ID_COSMETIC_MAP[m_id1]}  vs  {METRICS_ID_COSMETIC_MAP[m_id2]}" +
			f" (Overall s-corr: {metrics_g[model]['Correlation'][m_id1][m_id2]:.2f})", fontsize=12)
		plt.tight_layout()
		m_id1 = m_id1[1:] if m_id1.startswith('g') else m_id1
		m_id2 = m_id2[1:] + '_g' if m_id2.startswith('g') else m_id2
		self.save_file(f'Correlation/{model.name}_{m_id1}', m_id2, dpi=400)

	def plot_metrics_heatmap(self, rho, pi, model, transformation, metrics, shared = False, log = False, corr_info = True, n = n):
		if shared:
			values = np.concatenate([metrics[m_id] for m_id in CHERNOFFS_ID])
			vmin, vmax = np.min(values[values > 0]), np.max(values)
			# use quantiles to set vmin and vmax
		else:
			vmin = vmax = None
		
		fig, axes = plt.subplots(len(METRICS_ID) // 2, 2, figsize=(10, 5 * len(METRICS_ID) // 2 - 1))
		axes = axes.flatten()

		suptitle = f"Metrics heatmaps\n" + model_str(n, rho, pi, model, transformation)
		fig.suptitle(suptitle, fontsize=14)
		
		for i, (ax, m_id) in enumerate(zip(axes, METRICS_ID)):
			metric_grid = metrics[m_id]
			N = metric_grid.shape[0]
			if m_id == 'Rand':
				sns.heatmap(metric_grid, cmap='Reds', ax=ax, vmin = -0.05, vmax = 1, cbar=True)
			elif m_id == 'GMM_score':
				sns.heatmap(metric_grid, cmap='Greens', ax=ax, vmin = 1, vmax = 6, cbar=True)
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
			if i == len(METRICS_ID) or i == len(METRICS_ID) - 1:
				ax.set_xlabel(f'{model.param_name}{sub("12")}', fontsize = 12)
			if i % 2 == 0:
				ax.set_ylabel(f'{model.param_name}{sub("11")}', fontsize = 12)
			ax.invert_yaxis()
			for spine in ax.spines.values():
				spine.set_visible(True)
				spine.set_linewidth(1)
		
		plt.tight_layout()
		rho_str, pi_str = float_to_str(rho), float_to_str(pi)
		m_str = f'{model.name}_{rho_str}_{pi_str}'.replace('.', '')
		self.save_file(f'Heatmaps_by_MT/Metrics/By_Model/{m_str}', f'{transformation.id}', dpi=400, close=False)
		self.save_file(f'Heatmaps_by_MT/Metrics/By_Transform/{transformation.id}', m_str, dpi=400)

	def plot_bias_heatmap(self, rho, pi, model, transformation, metrics, log = True, n = n):
		fig, axes = plt.subplots(4, len(CHERNOFFS_ID[2:]), figsize=(6*len(CHERNOFFS_ID[2:])+0.5, 20))

		suptitle = (f"Bias of (a)Chernoff information estimators vs true (a)Chernoff information\n"
					f"{model_str(n, rho, pi, model, transformation)}")
		fig.suptitle(suptitle, fontsize=14)

		for ax, m_id in zip(axes[0], CHERNOFFS_ID[2:]):
			true_C = 'C_true' if m_id[0] == 'C' else 'gC_true'
			corr, auc_corr, partial_corrs = metrics['Correlation'][true_C][m_id]
			ns, corrs = partial_corrs
			min_corrs, max_corrs = np.min(corrs), np.max(corrs)
			abs_max = min_corrs if abs(min_corrs) > abs(max_corrs) else max_corrs
			ax.plot(ns, corrs)
			ax.set_xlabel(f"Top % {CHERNOFFS_ID_COSMETIC_MAP[m_id]}-largest points considered")
			ax.set_ylabel(f'S-correlation({CHERNOFFS_ID_COSMETIC_MAP[true_C]}, {CHERNOFFS_ID_COSMETIC_MAP[m_id]} Top%)')
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

			ax.set_title(f'Spearman‑corr({CHERNOFFS_ID_COSMETIC_MAP[true_C]}, {CHERNOFFS_ID_COSMETIC_MAP[m_id]})')
		
		for axbias, bias in zip(axes[1:], BIASES):
			for ax, m_id in zip(axbias, CHERNOFFS_ID[2:]):
				gated_prefix = 'g' if m_id[0] == 'g' else ''
				bias_grid = metrics['Bias'][gated_prefix + 'C_true'][m_id][bias]
				N = bias_grid.shape[0]
				
				values = np.concatenate((metrics['Bias'][gated_prefix + 'C_true'][gated_prefix + 'C_graph'][bias], 
							 metrics['Bias'][gated_prefix + 'C_true'][gated_prefix + 'C_embed'][bias]))
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
				ax.set_title(f'{BIASES_MAP[bias]} {CHERNOFFS_ID_COSMETIC_MAP[m_id]} vs {CHERNOFFS_ID_COSMETIC_MAP[gated_prefix + "C_true"]}')
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
		rho_str, pi_str = float_to_str(rho), float_to_str(pi)
		m_str = f'{model.name}_{rho_str}_{pi_str}'.replace('.', '')
		self.save_file(f'Heatmaps_by_MT/Bias/By_Model/{m_str}', f'{transformation.id}', dpi=400, close=False)
		self.save_file(f'Heatmaps_by_MT/Bias/By_Transform/{transformation.id}', m_str, dpi=400)

	def plot_best_transform_heatmaps(self, rho, pi, model, metrics, n=n, transforms = None, gated = False):
		if transforms is None: transforms = TRANSFORMS_EXT.copy()
		chernoffs = GATED_CHERNOFFS_ID.copy() if gated else NON_GATED_CHERNOFFS_ID.copy()
		cols = ['Arg', 'Rand', 'Regret']

		fig, axes = plt.subplots(len(chernoffs), 3, figsize=(15, 5*len(chernoffs)), gridspec_kw={'width_ratios': [1, 1, 1]})
		global_title = f"Best‑Transform Metrics on Model: {model.name}\n" + model_str(n, rho, pi)
		if self.eps: global_title = empty_string_except(global_title)
		fig.suptitle(global_title, fontsize=14)

		for i, C in enumerate(chernoffs):
			for j, col in enumerate(cols):
				ax = axes[i, j]
				grid = metrics[f'{C}-Best Transform'][col]
				N = grid.shape[0]

				if col == 'Arg':
					cmap = ListedColormap(list(TRANSFORMS_CMAP.values()))
					norm = BoundaryNorm(np.arange(-0.5, cmap.N + 0.5, 1), cmap.N)
					ax.pcolormesh(grid, cmap = cmap, norm = norm, shading='auto')
					area_map = dict(zip(transforms, metrics[f'{C}-Best Transform']['Transform Area']))
					sorted_TRANSFORMS = sorted(transforms, key=lambda t: area_map[t], reverse=True)
					sorted_TRANSFORMS = filter(lambda t: area_map[t] > 0.015, sorted_TRANSFORMS)
					handles = [Patch(facecolor=TRANSFORMS_CMAP[t], label=f'{t.id}: {area_map[t]:.2f}') 
							for t in sorted_TRANSFORMS]
					ax.legend(title = 'Selection Proportion:',
						handles=handles, loc="upper left", handlelength=1, handleheight=1)
					ax.set_title(f'Underlying Transformation Selection')
				elif col == 'Rand':
					norm = colors.Normalize(vmin=0, vmax=1, clip=True)
					sns.heatmap(grid, ax=ax, norm=norm, cmap = 'Reds', cbar=False)

					#mean_rand_transforms_map = {t: metrics[t]['Rand Avg'] for t in transforms}
					#mean_rand_transforms_map[C] = metrics[f'{C}-Best Transform']['Rand Avg']
					#case = lambda t : t.id if t in transforms else t
					#sorted_TRANSFORMS = sorted(transforms + [C], key=lambda t: mean_rand_transforms_map[t], reverse=True)
					#handles = [Patch(facecolor=CMAP[t], label=f"{case(t).replace('_', ' ')}: {mean_rand_transforms_map[t]:.2f}") 
					#		for t in sorted_TRANSFORMS]
					#handles = [Patch(facecolor=CMAP[C], label=f"{C.replace('_', ' ')} : {metrics[f'{C}-Best Transform']['Rand Avg']:.2f}     ")]
					#ax.legend(title = 'Transforms: Avg(Rand)', handles=handles, loc="upper left", handlelength=1, handleheight=1)
					avg_rand_index = f"Avg(ARI) = {metrics[f'{C}-Best Transform']['Rand Avg']:.2f}"
					ax.legend(handles=[mpatches.Patch(facecolor='none', edgecolor='none', label=avg_rand_index)], loc="upper left", handlelength=0, handleheight=0)
					ax.set_title(f'Adjusted Rand Index')

					if i == 0:
						cax = inset_axes(ax,
										width="3%",      # width: 3% of parent_bbox width
										height="30%",    # height: 30% of parent_bbox height
										loc='upper right',
										borderpad=1)

						# Create colorbar manually
						sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
						sm.set_array([])  # Only needed for older matplotlib versions
						fig.colorbar(sm, cax=cax, ticks=[0.0, 0.5, 1.0], ticklocation='left')

				else:
					norm = colors.Normalize(vmin=0, vmax=1, clip=True)
					sns.heatmap(grid, ax=ax, norm=norm, cmap = 'Purples', cbar=False)
					avg_regret =		f"Avg(Reg)              = {metrics[f'{C}-Best Transform']['Regret Avg']:.2f}"
					area_positive_r = 	f"Area(Reg>0)        = {metrics[f'{C}-Best Transform']['Regret Area']:.2f}"
					avg_positive_r = 	f"Avg(Reg[Reg>0]) = {metrics[f'{C}-Best Transform']['Regret Avg on Positive Regret']:.2f}"

					handle_avg = mpatches.Patch(facecolor='none', edgecolor='none', label=avg_regret)
					handle_area = mpatches.Patch(facecolor='none', edgecolor='none', label=area_positive_r)
					handle_avg_positive_r = mpatches.Patch(facecolor='none', edgecolor='none', label=avg_positive_r)

					ax.legend(handles=[handle_avg, handle_area, handle_avg_positive_r], loc="upper left",
							handlelength=0, handleheight=0)
					ax.set_title(f'Regret')
					if i == 0:
						cax = inset_axes(ax,
										width="3%",      # width: 3% of parent_bbox width
										height="30%",    # height: 30% of parent_bbox height
										loc='upper right',
										borderpad=1)

						# Create colorbar manually
						sm = plt.cm.ScalarMappable(cmap='Purples', norm=norm)
						sm.set_array([])  # Only needed for older matplotlib versions
						fig.colorbar(sm, cax=cax, ticks=[0.0, 0.5, 1.0], ticklocation='left')

				#title = f'{CHERNOFFS_ID_COSMETIC_MAP[C]}-Best Transform{": " + col if col != "Arg" else ""}'
				#ax.set_title(title)

				ticks = np.linspace(0, N, 5)
				labels = np.linspace(0, 1, 5).round(2)
				ax.set_xticks(ticks); ax.set_xticklabels(labels)
				ax.set_yticks(ticks); ax.set_yticklabels(labels)
				if col != 'Arg':
					ax.invert_yaxis()

				if i == len(chernoffs) - 1:
					ax.set_xlabel(f"{model.param_name}{sub('12')}", fontsize=12)
				if j == 0:
					ax.set_ylabel(f"{CHERNOFFS_ID_COSMETIC_MAP[C]}-Optimal Transformation\n\n{model.param_name}{sub('11')}", fontsize=12)

				for spine in ax.spines.values():
					spine.set_visible(True)
					spine.set_linewidth(1)

		plt.tight_layout()
		rho_str, pi_str = float_to_str(rho), float_to_str(pi)
		gated_str = '_gated' if gated else ''
		self.save_file(f'Best_Transform/{model.name}', f'{model.name}_{pi_str}_{rho_str}{gated_str}', dpi=400)


	def plot_best_transform_scatter_rand_vs_regretratio(self, rho, pi, model, metrics, transforms=None):
		if transforms is None:
			transforms = TRANSFORMS_EXT.copy()

		fig, ax = plt.subplots(figsize=(8, 8))

		# --- Plot base transforms ---
		base_handles = []
		for t in transforms:
			x = 1 - metrics[t]['Regret Area']
			y = metrics[t]['Rand Avg']
			h = ax.scatter(
				x, y,
				marker='x', s=50,
				color=CMAP[t],
				label=getattr(t, 'id', str(t)).replace('_', ' ') + f":\n{y:.3f}, {x:.2f}"
			)
			base_handles.append((h, y))

		# --- Plot non-gated Chernoff ---
		chernoffs_handles = []
		for t in NON_GATED_CHERNOFFS_ID:
			key = f"{t}-Best Transform"
			x = 1 - metrics[key]['Regret Area']
			y = metrics[key]['Rand Avg']
			h = ax.scatter(
				x, y,
				marker='s', s=80,
				color=CMAP[t], edgecolor=CMAP[t],
				linewidth=0,
				label=t.replace('_', ' ') + f":\n{y:.3f}, {x:.2f}"
			)
			chernoffs_handles.append((h, y))

		# --- Plot gated Chernoff ---
		for t in GATED_CHERNOFFS_ID:
			key = f"{t}-Best Transform"
			x = 1 - metrics[key]['Regret Area']
			y = metrics[key]['Rand Avg']
			h = ax.scatter(
				x, y,
				marker='s', s=80,
				color=CMAP[t[1:]], edgecolor='k',
				linewidth=2,
				label=t.replace('_', ' ') + f":\n{y:.3f}, {x:.2f}"
			)
			chernoffs_handles.append((h, y))

		# Styling axes
		ax.set_xlabel('Null Regret Ratio', fontsize=12)
		ax.set_ylabel('Adjusted Rand Index', fontsize=12)
		if rho is not None and pi is not None:
			global_title = f"Transforms Rand Avg vs Null Regret Ratio on Model:\n{model.name} " + model_str(n, rho, pi)[:-1]
		else:
			global_title = f"Transforms Rand Avg vs Null Regret Ratio on {model.name} Model\n"
		if self.eps: global_title = empty_string_except(global_title)
		ax.set_title(global_title, fontsize=14)
		ax.set_xlim(-0.05, 1.0)
		ax.set_ylim(-0.05, 1.0)
		ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

		y = np.mean(metrics['Rand Max'])
		ax.axhline(y=y, color='k', linestyle='--')
		ax.text(
			0.5, y+0.05,
			f"Max ARI: {y:.3f}",
			transform=ax.get_xaxis_transform(),
			va='center', ha='center'
		)

		base_handles = sorted(base_handles, key=lambda hy: hy[1], reverse=True)
		sorted_handles = [h for h, _ in base_handles]
		sorted_labels  = [h.get_label() for h in sorted_handles]

		# --- First legend: base transforms (bottom right) ---
		leg1 = ax.legend(
			handles=sorted_handles,
			labels=sorted_labels,
			title="Transforms:",
			loc="lower right",
			frameon=True,
			bbox_to_anchor=(0.8, 0, 0.2, 1),
			mode='expand',
		)
		ax.add_artist(leg1)

		chernoffs_handles = sorted(chernoffs_handles, key=lambda hy: hy[1], reverse=True)
		sorted_handles = [h for h, _ in chernoffs_handles]
		sorted_labels  = [h.get_label() for h in sorted_handles]

		# --- Second legend: Chernoffs (upper right) ---
		leg2 = ax.legend(
			handles=sorted_handles,
			labels=sorted_labels,
			title="C-opt transform:\nARI, NRR",
			loc="lower right",
			frameon=True,
			bbox_to_anchor=(0.609, 0, 0.2, 1),
			mode='expand',
		)

		plt.tight_layout()
		if rho is not None and pi is not None:
			rho_str, pi_str = float_to_str(rho), float_to_str(pi)
			self.save_file(f'Best_Transform/{model.name}', f'{model.name}_{pi_str}_{rho_str}_NRR', dpi=400)
		else:
			self.save_file(f'Best_Transform/{model.name}_Aggregated', f'{model.name}__Rand_vs_NoRegretRatio', dpi=400)


	def modulable_bar_plots(self, ax, L, fontsize=10):
		x_offset = 0
		x_centers = []

		for tuple in L:
			space_0 = tuple[0][-1] if len(tuple[0]) == 7 else tuple[0][-2]
			x_offset_start = x_offset + space_0
			for tup in tuple:
				if len(tup) == 7:
					area, height, height_2, text, text_2, color, space = tup
					linewidth = 0
				elif len(tup) == 8:
					area, height, height_2, text, text_2, color, space, linewidth = tup
				x_offset += space
				ax.bar(x_offset, height, width=area, color=color, align='edge', linewidth=linewidth, edgecolor = 'black')
				ax.bar(x_offset, height_2, width=area, bottom=height, color='grey', align='edge', linewidth=linewidth, edgecolor = 'black')
				ax.text(x_offset + area/2, height/2, str(text), ha='center', va='center', fontsize=fontsize)
				ax.text(x_offset + area/2, height + height_2/2, str(text_2), ha='center', va='center', fontsize=fontsize)
				x_offset += area

			x_centers.append(x_offset_start + (x_offset - x_offset_start) / 2)
			
		ax.set_xticks(x_centers)

		return x_offset + space_0
	
	def plot_transforms_rand(self, model, metrics_g, mode = 'No regret', 
						  transforms = None, chernoffs = None, name = 'Transforms'):
		_, ax = plt.subplots(figsize=(8, 6))

		if transforms is None: transforms = TRANSFORMS.copy()
		if chernoffs is None: chernoffs = CHERNOFFS_ID.copy()

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
				f"{m['Rand Avg']:.4f}" if m['Rand Avg'] > 0.1 else "",
				#f"{m['Regret Avg']:.2f}",
				"",
				color_helper(t), 
				0.1, 
				1.5 if isinstance(t, str) and t.startswith('gC') else 0),) for t, m in m_transforms.items()]
		elif mode == 'With regret':
			L1 = [(1 - m['Regret Area'],
				m['Rand Avg on Null Regret'],
				0,
				#f"{m['Rand Avg on Null Regret']:.2f}",
				"",
				"",
				color_helper(t),
				0.1,
				1 if isinstance(t, str) and t.startswith('gC') else 0) for t, m in m_transforms.items()]
			L2 = [(m['Regret Area'],
				m['Rand Avg on Positive Regret'],
				m['Regret Avg on Positive Regret'],
				#f"{m['Rand Avg on Positive Regret']:.2f}",
				#f"{m['Regret Avg on Positive Regret']:.2f}",
				"", "",
				color_helper(t),
				0.025,
				1 if isinstance(t, str) and t.startswith('gC') else 0) for t, m in m_transforms.items()]

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
			ax.set_title(f'Transformations Average Rand & Regret on {model.name} Model\n')
			Avg_Rand_Max = np.mean(metrics_g['Rand Max'])
			ax.hlines(Avg_Rand_Max, 0, 
					x_offset, color='black', lw=1, label=f'Avg(Rand Max): {Avg_Rand_Max:.4f}', linestyle='--')
			ax.legend()
			ax.set_ylim(-0.05, 1)
			ax.set_xlabel('\nTransformations')

		if mode == 'With regret':
			ax.set_title(f'Transformations Average Rand & Regret on {model.name} Model\nConditioned on Null Regret vs Positive Regret\n')
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
		self.save_file(f'Best_Transform/{model.name}_Aggregated', f'{model.name}_{name}_Rands_{mode}', dpi=400)

	def plot_transforms_rand_for_best_transform(self, model, metrics_g, chernoff, 
											 transforms = None):
		_, ax = plt.subplots(figsize=(8, 6))

		m_chernoff = metrics_g[f"{chernoff}-Best Transform"]

		if transforms is None:
			transforms = TRANSFORMS_EXT.copy()
		transforms_fixed = transforms.copy()

		def t_area(t):
			return m_chernoff['Transform Area'][transforms_fixed.index(t)]
		
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
		ax.set_xticklabels([(t.id if t_area(t) > 0.03 else '') for t in transforms], rotation=45)
		ax.text(
				x_offset/2,
				-0.125,
				"(Transformation Selection Proportions)", 
				transform=ax.get_xaxis_transform(),
				ha="center", va="top",
				fontsize=8)
		ax.set_xlabel('\nTransformations')
		ax.set_ylabel('Avg Rand')
		ax.set_ylim(-0.075, 1.05)
		ax.set_xlim(-0.05, x_offset+0.05)
		ax.set_title(f'{CHERNOFFS_ID_COSMETIC_MAP[chernoff]}-Best Transform underlying Transformation Selection\nWith average, on {model.name} Model, Rand, Regret & Selection by Transformation\n')

		ax.legend(handles=[mpatches.Patch(color='grey', label='Rand Regret')], loc='upper right', fontsize=8)

		plt.tight_layout()
		chernoff = chernoff[1:] + '_g' if chernoff.startswith('g') else chernoff
		self.save_file(f'Best_Transform/{model.name}_Aggregated', f'{model.name}_Best_Transform_Transforms_Rands_{chernoff}', dpi=400)

	def plot_rand_by_sigmoid_params_model_wise(self, C, best_rand_avg):
		models = MODELS.copy()
		model_colormaps = dict(zip(MODELS, ['Blues', 'Oranges']))

		x0_list, w_list = zip(*best_rand_avg[models[0]][C].keys())
		x0_list, w_list = set(x0 for x0 in x0_list if isinstance(x0, float)), set(w for w in w_list if isinstance(w, float))
		shifts, windows = np.array(sorted(x0_list)), np.array(sorted(w_list))

		ylim_size = 0.025
		ylim0 = best_rand_avg[models[1]][C]["no gating"] + np.array([-1, 1]) * ylim_size
		ylim1 = best_rand_avg[models[0]][C]["no gating"] + np.array([-1, 1]) * ylim_size

		fig = plt.figure(figsize=(6, 6))
		bax = brokenaxes(
			ylims=(ylim0, ylim1),
			hspace=.05,
			despine=True,
			fig=fig)

		for ax in bax.axs:
			ax.set_facecolor('#505050')  # anthracite

		# 2) Plot curves and annotate
		max_info = {}
		for model in models:
			# Plot sigmoid curves per window
			cmap = plt.get_cmap(model_colormaps[model])
			colors = cmap(np.linspace(0.3, 1.0, len(windows)))
			for idx, w in enumerate(windows):
				y_vals = [best_rand_avg[model][C][(x0, w)] for x0 in shifts]
				bax.plot(shifts, y_vals, color=colors[idx], alpha=0.5)

			# Horizontal line for no gating with annotation
			y0 = best_rand_avg[model][C]["no gating"]
			bax.hlines(y0, xmin=min(shifts), xmax=max(shifts), linestyles='-', linewidth=1, colors='black')
			mid_x = (min(shifts) + max(shifts)) / 2
			bax.annotate(
				f"Non-Gated:\nR = {y0:.4f}",
				xy=(mid_x, y0), xytext=(-20, -15), textcoords='offset points', va='center', fontsize='small'
			)

			"""
			ymax = best_rand_avg[model]["max"]
			print(ymax)
			bax.hlines(ymax, xmin=min(shifts), xmax=max(shifts), linestyles='--', linewidth=1, colors='black')
			bax.annotate(
				f"Max:\nR = {ymax:.4f}",
				xy=(mid_x, ymax), xytext=(-20, -15), textcoords='offset points', va='center', fontsize=6
			)"""


			# Find max Rand Avg across (x0, w) for sigmoid
			best_val = -np.inf
			best_params = None
			
			for x0 in shifts:
				for w in windows:
					val = best_rand_avg[model][C][(x0, w)]
					if val > best_val:
						best_val = val
						best_params = (x0, w)
			max_info[model] = (best_params, best_val)

		# Annotate max points and compare across models
		for model in models:
			cmap = plt.get_cmap(model_colormaps[model])
			colors = cmap(windows / windows[-1])
			(x0_m, w_m), best_val = max_info[model]
			color_model = colors[np.where(windows == w_m)[0][0]]

			# Marker for model max
			bax.scatter([x0_m], [best_val], marker='o', s=25, edgecolors='k', facecolors=color_model, zorder=5)
			bax.annotate(
				f"s₀={x0_m}, w={w_m:.2f}\nSigm(s₀,w)-Gated:\nR = {best_val:.4f}",
				xy=(x0_m, best_val), textcoords='offset points', xytext=(-30, 7.5), fontsize='small'
			)

			# Mark and label other models at the same (x0, w)
			for other in models:
				if other is not model:
					val_other = best_rand_avg[other][C][(x0_m, w_m)]
					bax.scatter([x0_m], [val_other], marker='o', s=25, edgecolors='k', facecolors=color_model, zorder=5)
					bax.annotate(
						f"R = {val_other:.4f}",
						xy=(x0_m, val_other), textcoords='offset points', fontsize='small', xytext=(-20, -12.5),
					)

		for ax, model in zip(bax.axs, models):
			cmap = plt.get_cmap(model_colormaps[model])
			norm = mpl.colors.Normalize(vmin=0, vmax=1)

			ax.legend(
				handles = [mpl.cm.ScalarMappable(norm=norm, cmap=cmap)],
				labels = [f"w ∈ [{windows[0]:.0f}, {windows[-1]:.0f}]"],
				handler_map={mpl.cm.ScalarMappable: HandlerColormap(cmap, norm)},
				loc='upper left',
				fontsize='small',
				title_fontsize='small',
				#facecolor='black',
				framealpha=1
			)
			

		bax.legend(
			handles = [Line2D([0], [0], color=plt.get_cmap(model_colormaps[model])(windows/windows[-1])[-5],
						lw=2) for model in models],
			labels = [model.name for model in models],
			loc='lower left',
			fontsize='small',
			title_fontsize='small',
			#facecolor='black',
			framealpha=1)


		bax.set_xlabel('s₀')
		bax.set_xlim(-15.5, 5.5)
		bax.set_xticks([-15, -10, -5, 0, 5])
		bax.set_ylabel('Rand Avg\n')
		global_title = f"Rand Avg for Sigmoid(s₀,w)(GMM-score)-Gated {CHERNOFFS_ID_COSMETIC_MAP[C[1:]]}\n"
		if self.eps: global_title = empty_string_except(global_title)
		bax.set_title(global_title)

		self.save_file(f'Sigmoid_Params_Optimization', f'{C}_model_wise', dpi=400)

	def plot_rand_by_sigmoid_params(self, C, best_rand_avg, color = 'Greens'):
		models = MODELS.copy()

		x0_list, w_list = zip(*best_rand_avg[MODELS[0]][C].keys())
		x0_list, w_list = set(x0 for x0 in x0_list if isinstance(x0, float)), set(w for w in w_list if isinstance(w, float))
		shifts, windows = np.array(sorted(x0_list)), np.array(sorted(w_list))

		ylim_size = 0.05
		y0_vals = [best_rand_avg[m][C]["no gating"] for m in models]
		avg_baseline = np.mean(y0_vals)
		ymax_vals = [best_rand_avg[m]["max"] for m in models]
		avg_max = np.mean(ymax_vals)
		ylim = avg_baseline + np.array([-1, 1]) * ylim_size

		fig = plt.figure(figsize=(6, 4))
		ax = fig.add_subplot(1, 1, 1)
		ax.set_facecolor('#505050')  # anthracite
		ax.set_ylim(*ylim)

		cmap = plt.get_cmap(color)
		colors = cmap(windows / windows[-1])

		max_val = -np.inf
		max_params = None

		for idx, w in enumerate(windows):
			y_vals = [
				(best_rand_avg[models[0]][C][(x0, w)] +
				best_rand_avg[models[1]][C][(x0, w)]) / 2
				for x0 in shifts]
			ax.plot(shifts, y_vals, color=colors[idx], alpha=0.5)

			for x0, y in zip(shifts, y_vals):
				if y > max_val:
					max_val = y
					max_params = (x0, w)

		# ligne horizontale baseline
		ax.hlines(avg_baseline,
				xmin=min(shifts), xmax=max(shifts),
				linestyles='-', linewidth=1, colors='black')
		mid_x = (min(shifts) + max(shifts)) / 2
		ax.annotate(
			f"Non-Gated Avg:\nR = {avg_baseline:.4f}",
			xy=(mid_x, avg_baseline),
			xytext=(-20, -15),
			textcoords='offset points',
			va='center',
			fontsize='small'
		)

		ax.hlines(avg_max,
				xmin=min(shifts), xmax=max(shifts),
				linestyles='--', linewidth=1, colors='black')
		mid_x = (min(shifts) + max(shifts)) / 2
		ax.annotate(
			f"Max Avg:\nR = {avg_max:.4f}",
			xy=(mid_x, avg_max),
			xytext=(-20, -15),
			textcoords='offset points',
			va='center',
			fontsize='small'
		)

		# annotation du point global maximal
		x0_m, w_m = max_params
		idx_max = np.where(windows == w_m)[0][0]
		color_max = colors[idx_max]
		ax.scatter([x0_m], [max_val],
				marker='o', s=25,
				edgecolors='k', facecolors=color_max,
				zorder=5)
		shift_x = 30 if x0_m < -14 else 0
		ax.annotate(
			f"s₀={x0_m}, w={w_m:.2f}\nSigm(s₀,w)-Gated Avg:\nR = {max_val:.4f}",
			xy=(x0_m, max_val),
			textcoords='offset points',
			xytext=(-30 + shift_x, 7.5),
			fontsize='small'
		)

		norm = mpl.colors.Normalize(vmin=0, vmax=1)

		ax.legend(
			handles = [mpl.cm.ScalarMappable(norm=norm, cmap=cmap)],
			labels = [f"w ∈ [{windows[0]:.0f}, {windows[-1]:.0f}]"],
			handler_map={mpl.cm.ScalarMappable: HandlerColormap(cmap, norm)},
			loc='lower left',
			fontsize='small',
			title_fontsize='small',
			#facecolor='black',
			framealpha=1
		)

		ax.set_xticks([-15, -10, -5, 0, 5])
		ax.set_xlim(-15.5, 5.5)
		ax.set_xlabel('s₀')
		ax.set_ylabel('Rand Avg\n')
		global_title = f"Rand Avg for Sigmoid(s₀,w)(GMM-score)-Gated {CHERNOFFS_ID_COSMETIC_MAP[C[1:]]}\n"
		if self.eps: global_title = empty_string_except(global_title)
		ax.set_title(global_title)

		plt.tight_layout()
		self.save_file(f'Sigmoid_Params_Optimization', f'{C}_grouped', dpi=400)


#################################################################


	def plot_metrics_for_varying_param(self, model = betaWSBM,
									t = PowerTransform(1),
									varying_param = 'rho',
									varying_param_bounds = (0, 0.5),
									h = np.array([1, 6, 15, 20, 15, 6, 1]) / 64):
		
		try:
			if issubclass(varying_param, WeightTransform):
				assert isinstance(t, varying_param), "t must be of the same type as varying_param"
		except TypeError:
			assert isinstance(varying_param, str), "varying_param must be a string or a WeightTransform subclass"

		if varying_param == 'rho':
			varying_param_label_str = 'ρ'
		elif varying_param == 'pi':
			varying_param_label_str = 'π₁'
		elif varying_param == 'p11':
			varying_param_label_str = f'{model.param_name}{sub("11")}'
		elif varying_param == 'p12':
			varying_param_label_str = f'{model.param_name}{sub("12")}'
		elif issubclass(varying_param, WeightTransform):
			varying_param_label_str = f'({t.__class__.__name__} {t.param_name})'
		else:
			raise ValueError(f"Unknown varying_param: {varying_param}")
			
		varying_param_str = varying_param if isinstance(varying_param, str) else varying_param.__name__
		name = f'{model.name}_{t.id}_{varying_param_str}_{varying_param_bounds[0]}_{varying_param_bounds[1]}'
		path = f"Computation/data_for_gifs"
		dico = load(f"{path}/{name}.joblib")
		
		metrics = dico['metrics']
		N = len(metrics['Rand']['mean'])
		params = np.linspace(*varying_param_bounds, N)
		idxs = np.arange(0, N, N // 10)

		rho = dico['fixed_params']['rho']
		pi  = dico['fixed_params']['pi']
		p11 = dico['fixed_params']['p11']
		p12 = dico['fixed_params']['p12']

		n_top = 4  # <-- set this to however many cells you want in the top row
		fig   = plt.figure(figsize=(12, 8))
		outer = gridspec.GridSpec(3, 1, height_ratios=[1, 10, 10], figure=fig)
		global_title = f"{varying_param_label_str}-induced changes in\n Clustering Metrics and Chernoff Informations"
		if self.eps: global_title = empty_string_except(global_title)
		fig.suptitle(global_title, fontsize=18)

		# top row of arbitrary length
		top_gs   = outer[0].subgridspec(1, n_top)
		axes_top = [fig.add_subplot(top_gs[0, i]) for i in range(n_top)]

		# re–define your 2×2 bottom grid
		bottom_gs = outer[1:].subgridspec(2, 2)
		axes = np.array([[fig.add_subplot(bottom_gs[i, j]) for j in range(2)]
						for i in range(2)])

		ax = axes_top[0]
		ax.axis('off')
		ax.text(0.5, 0.5, f"Model: {model.model_name_with_law_params()}\nn = {1000}, K = {2}", 
			ha='center', va='center', fontsize=12)

		ax = axes_top[1]
		ax.axis('off')
		varying = None
		if varying_param == 'p11': varying = (0, 0)
		if varying_param == 'p12': varying = (0, 1)
		ax.text(
			0.5, 0.5, 
			model.param_matrix_str(np.array([[p11, p12], [p12, model.p22_fixed]]), varying),
			ha='center', va='center',
			fontsize=10,
		)

		ax = axes_top[2]
		ax.axis('off')
		rho_str = 'ρ' if varying_param == 'rho' else f'{rho:.2f}'
		pi_str  = 'π₁' if varying_param == 'pi' else f'{pi:.2f}'
		ax.text(0.5, 0.5, f"ρ = {rho_str}\nπ₁ = {pi_str}", ha='center', va='center', fontsize=12)

		ax = axes_top[3]
		ax.axis('off')
		tr_str = t.name if isinstance(varying_param, str) else t.name.split("=", 1)[0] + f"= {t.param_name})"
		ax.text(0.5, 0.5, f"Transformation:\n{tr_str}", ha='center', va='center', fontsize=12)

		# 1) Rand Index
		ax = axes[0, 0]
		mean = convolve1d(metrics['Rand']['mean'], h, mode='reflect')
		std  = metrics['Rand']['std']
		ax.plot(params, mean, color = 'red')
		ax.errorbar(params[idxs], mean[idxs], yerr=std[idxs], fmt='none', capsize=0, color = 'red')
		ax.set_ylim(-0.1, 1.05)
		ax.set_yticks(np.linspace(0, 1, 6))
		ax.set_title('Adjusted Rand Index')
		ax.set_ylabel('ARI')

		# 2)
		ax = axes[0, 1]
		for key in ['C_true', 'C_graph', 'C_embed']:
			mean = convolve1d(metrics[key]['mean'], h, mode='reflect')
			std  = metrics[key]['std']
			ax.plot(params, mean, label=key.replace('_', ' '), color = CHERNOFFS_CMAP[key])
			if key != 'C_true':
				shift = -1 if key == 'C_graph' else 1
				ax.errorbar(params[idxs + shift], mean[idxs + shift], 
				yerr=std[idxs + shift], fmt='none', capsize=0, color = CHERNOFFS_CMAP[key])
		ax.set_title('Chernoff Informations')
		trim = lambda a, x: a[int(len(a)*x) : len(a) - int(len(a)*x)]
		max_c = np.max(np.concatenate([trim(metrics[key]['mean'], 0.01) for key in ['C_true', 'C_graph', 'C_embed']]))

		def round_to_25_two_sig(x):
			if x == 0:
				return 0.0
			sign  = -1 if x < 0 else 1
			y     = abs(x)
			exp   = math.floor(math.log10(y))
			shift = 1 - exp
			s     = y * 10**shift
			s_r   = (math.ceil if sign > 0 else math.floor)(s / 25) * 25
			return sign * s_r / 10**shift
		
		max_c = round_to_25_two_sig(max_c)
		ax.set_ylim(-0.05*max_c, 1.05*max_c)
		ax.set_yticks(np.linspace(0, max_c, 6))
		ax.set_ylabel('C')
		ax.legend()

		# 3) GMM Score
		ax = axes[1, 0]
		mean = convolve1d(metrics['GMM_score']['mean'], h, mode='reflect')
		std  = metrics['GMM_score']['std']
		ax.plot(params, mean, color = 'green')
		ax.errorbar(params[idxs], mean[idxs], yerr=std[idxs], fmt='none', capsize=0, color = 'green')
		ax.set_title('GMM Score')
		ax.set_ylim(-0.25, 5.25)
		ax.set_yticks(np.linspace(0, 5, 6))
		ax.set_ylabel('Score')

		# 4) π₁ Estimate
		ax = axes[1, 1]
		mean = convolve1d(metrics['pi_1']['mean'], h, mode='reflect')
		std  = metrics['pi_1']['std']
		ax.plot(params, mean, color = 'black')
		ax.errorbar(params[idxs], mean[idxs], yerr=std[idxs], fmt='none', capsize=0, color = 'black')
		ax.set_title('Estimated π₁')
		ax.set_ylim(-0.025, 0.525)
		ax.set_yticks(np.linspace(0, 0.5, 6))
		ax.set_ylabel('Estimated π₁')
		if varying_param == 'pi':
			ax.plot(params, params, color = 'black', linestyle='--', label='True π₁')
		else:
			ax.plot(params, [pi] * N, color = 'black', linestyle='--', label='True π₁')
		ax.legend()

		for ax in axes.flat:
			labelpad_shift = 0
			if not isinstance(varying_param, str) and issubclass(varying_param, WeightTransform):
				varying_param_label_str = t.param_name
				if varying_param == PowerTransform:
					ax.set_xticks(np.linspace(*varying_param_bounds, 7))
					labelpad_shift = 1.5
			ax.set_xlabel(varying_param_label_str, labelpad=-2.5 + labelpad_shift)
			

		plt.tight_layout()
		self.save_file(f'Metrics_Varying_Params', f'{name}', dpi=400)

	def plot_embedding_for_varying_param(self, n, dico,
								   model = betaWSBM,
								   t = PowerTransform(1),
								   varying_param = 'rho',
								   varying_param_bounds = (0, 0.5),
								   shift_factor_x = 0, shift_factor_y = 0,
								   scaling_factor_x = 1, scaling_factor_y = 1,
								   h = np.array([1, 6, 15, 20, 15, 6, 1]) / 64,
								   KDE = True):
	
		try:
			if issubclass(varying_param, WeightTransform):
				assert isinstance(t, varying_param), "t must be of the same type as varying_param"
		except TypeError:
			assert isinstance(varying_param, str), "varying_param must be a string or a WeightTransform subclass"

		if varying_param == 'rho':
			varying_param_label_str = 'ρ'
		elif varying_param == 'pi':
			varying_param_label_str = 'π₁'
		elif varying_param == 'p11':
			varying_param_label_str = f'{model.param_name}{sub("11")}'
		elif varying_param == 'p12':
			varying_param_label_str = f'{model.param_name}{sub("12")}'
		elif issubclass(varying_param, WeightTransform):
			varying_param_label_str = f'({t.__class__.__name__} {t.param_name})'
		else:
			raise ValueError(f"Unknown varying_param: {varying_param}")
		
		metrics = dico['metrics']
		embedding_metrics = dico['displayed_graphs_metrics']
		N = len(metrics['Rand']['mean'])
		params = np.linspace(*varying_param_bounds, N)

		rho = dico['fixed_params']['rho']
		pi  = dico['fixed_params']['pi']
		p11 = dico['fixed_params']['p11']
		p12 = dico['fixed_params']['p12']

		fig = plt.figure(figsize=np.array([13, 8])*1.25)
		fig.suptitle(f"{varying_param_label_str}-induced changes in the Embeddings, Clustering Metrics and Chernoff Informations", fontsize=18)

		outer = gridspec.GridSpec(6, 2,
			width_ratios=[5, 2.25],
			height_ratios=[1, 1, 5, 5, 5, 5],
			figure=fig,
			wspace=0.05
		)

		# 1) Big plot on the left spanning all rows
		ax_big = fig.add_subplot(outer[:, 0])

		# 2) Top-right: 2×2 grid
		top_right_gs = outer[0:2, 1].subgridspec(2, 2, wspace=0.2)
		axes_top = [fig.add_subplot(top_right_gs[i, j]) for i in range(2) for j in range(2)]

		# 3) Bottom-right: 4 stacked rows
		bottom_right_gs = outer[2:, 1].subgridspec(4, 1)
		axes_bottom = [fig.add_subplot(bottom_right_gs[i, 0])for i in range(4)]

		ax = axes_top[0]
		ax.axis('off')
		ax.text(0.5, 0.5, f"Model: {model.model_name_with_law_params()}\nn = {1000}, K = {2}", 
			ha='center', va='center', fontsize=12)

		ax = axes_top[2]
		ax.axis('off')
		varying = None
		if varying_param == 'p11': varying = (0, 0)
		if varying_param == 'p12': varying = (0, 1)
		ax.text(
			0.5, 0.5, 
			model.param_matrix_str(np.array([[p11, p12], [p12, model.p22_fixed]]), varying),
			ha='center', va='center',
			fontsize=11,
		)

		ax = axes_top[3]
		ax.axis('off')
		rho_str = 'ρ' if varying_param == 'rho' else f'{rho:.2f}'
		pi_str  = 'π₁' if varying_param == 'pi' else f'{pi:.2f}'
		ax.text(0.5, 0.5, f"ρ = {rho_str}\nπ₁ = {pi_str}", ha='center', va='center', fontsize=12)

		ax = axes_top[1]
		ax.axis('off')
		tr_str = t.name if isinstance(varying_param, str) else t.name.split("=", 1)[0] + f"= {t.param_name})"
		ax.text(0.5, 0.5, f"Transformation:\n{tr_str}", ha='center', va='center', fontsize=12)

		ax = ax_big
		G = dico['graphs'][n]
		X, Z, M, Σ = G['X'], G['Z'], G['M'], G['Σ']

		if n > 0:
			X_prev = dico['graphs'][n-1]['X']
			#X_centered, X_prev_centered = X - X.mean(axis=0), X_prev - X_prev.mean(axis=0)
			#R, _ = orthogonal_procrustes(X_centered, X_prev_centered)
			#X = X_centered @ R.T + X_prev.mean(axis=0)

			d0 = np.mean( np.linalg.norm(X - X_prev, axis=1))
			X_flipped = X * np.array([-1, 1])
			d1 = np.mean( np.linalg.norm(X_flipped - X_prev, axis=1))
			if d1 < d0:
				X = X_flipped
				M = M * np.array([-1, 1])
				Σ = np.diag([-1, 1]) @ Σ @ np.diag([-1, 1])
				dico['graphs'][n]['X'] = X

		ax.scatter(X[:, 0], X[:, 1], c=Z, cmap='bwr', marker='.', alpha=0.5)
		ax.set_xticks([])
		ax.set_yticks([])
		ref_window  = np.array(dico['embedding_ref_window'])
		shift		= ref_window * np.array([shift_factor_x, shift_factor_y])
		scaling 	= np.stack([-ref_window, ref_window]) / 2 * np.array([scaling_factor_x, scaling_factor_y])
		limits  	= X.mean(axis=0) + scaling + shift
		x_limits, y_limits = limits[:, 0], limits[:, 1]
		xmin, xmax, ymin, ymax = x_limits[0], x_limits[1], y_limits[0], y_limits[1]

		# --- 2) build a regular grid over that window ---
		if KDE:
			xi = np.linspace(xmin, xmax, 200)
			yi = np.linspace(ymin, ymax, 200)
			Xg, Yg = np.meshgrid(xi, yi)
			grid = np.vstack([Xg.ravel(), Yg.ravel()])

			my_Reds  = LinearSegmentedColormap.from_list('WhiteToRed',  [(0.0, '#ffffff'), (1.0, '#ff0000')])
			my_Blues = LinearSegmentedColormap.from_list('WhiteToBlue', [(0.0, '#ffffff'), (1.0, '#0000ff')])

			for label, cmap in zip(np.unique(Z), [my_Blues, my_Reds]):
				Xc = X[Z == label]
				if Xc.shape[0] < 3 or np.linalg.matrix_rank(np.cov(Xc.T)) < 2:
					continue  # need at least 2 points for kde
				kde = gaussian_kde(Xc.T, bw_method=1)
				Zc = kde(grid).reshape(Xg.shape)

				# optional: normalize each map so they’re on comparable scales
				Zc /= Zc.max()

				# --- 4) draw it under your points with some transparency ---
				ax.imshow(
					Zc,
					origin='lower',
					extent=[xmin, xmax, ymin, ymax],
					cmap=cmap,
					alpha=0.6,
					aspect='auto'
				)


		ax.set_xlim(x_limits)
		ax.set_ylim(y_limits)

		def plot_embedding_metrics(ax, m_id, label, color):
			ax.plot(params[:n+1], embedding_metrics[m_id][:n+1], color = color, linewidth=1, label=label)
			ax.plot(params[n], embedding_metrics[m_id][n], marker='o', color=color, markersize=4)

		for mean, cov in zip(M, Σ):
			eigenvalues, eigenvectors = np.linalg.eigh(cov)
			angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
			width, height = 2 * np.sqrt(6 * eigenvalues)
			ellip = plt.matplotlib.patches.Ellipse(
				mean, width, height, angle=angle, edgecolor='k', facecolor='none', linestyle='solid'
			)
			ax.add_patch(ellip)

		# 1) Rand Index
		ax = axes_bottom[0]
		rand_mean = convolve1d(metrics['Rand']['mean'], h, mode='reflect')
		ax.plot(params, rand_mean, color = 'red', linewidth=0.5, alpha=0.5)
		plot_embedding_metrics(ax, 'Rand', 'ARI', 'red')
		ax.set_ylim(-0.1, 1.05)
		ax.set_yticks(np.linspace(0, 1, 6))

		ax = axes_bottom[1]
		for key in ['C_true', 'C_graph', 'C_embed']:
			C_mean = convolve1d(metrics[key]['mean'], h, mode='reflect')
			ax.plot(params, C_mean, color = CHERNOFFS_CMAP[key], linewidth=0.5, alpha=0.5)
			plot_embedding_metrics(ax, key, key.replace('_', ' '), CHERNOFFS_CMAP[key])
		trim = lambda a, x: a[int(len(a)*x) : len(a) - int(len(a)*x)]
		max_c = np.max(np.concatenate([trim(metrics[key]['mean'], 0.01) for key in ['C_true', 'C_graph', 'C_embed']]))

		def round_to_25_two_sig(x):
			if x == 0:
				return 0.0
			sign  = -1 if x < 0 else 1
			y     = abs(x)
			exp   = math.floor(math.log10(y))
			shift = 1 - exp
			s     = y * 10**shift
			s_r   = (math.ceil if sign > 0 else math.floor)(s / 25) * 25
			return sign * s_r / 10**shift
		
		max_c = round_to_25_two_sig(max_c)
		ax.set_ylim(-0.05*max_c, 1.05*max_c)
		ax.set_yticks(np.linspace(0, max_c, 6))

		# 3) GMM Score
		ax = axes_bottom[2]
		GMM_score_mean = convolve1d(metrics['GMM_score']['mean'], h, mode='reflect')
		ax.plot(params, GMM_score_mean, color = 'green', linewidth=0.5, alpha=0.5)
		plot_embedding_metrics(ax, 'GMM_score', 'GMM Score', 'green')
		ax.set_ylim(-0.25, 5.25)
		ax.set_yticks(np.linspace(0, 5, 6))

		# 4) π₁ Estimate
		ax = axes_bottom[3]
		pi_1_mean = convolve1d(metrics['pi_1']['mean'], h, mode='reflect')
		ax.plot(params, pi_1_mean, color = 'black', linewidth=0.5, alpha=0.5)
		plot_embedding_metrics(ax, 'pi_1', 'Estimated π₁', 'black')
		ax.set_ylim(-0.025, 0.525)
		ax.set_yticks(np.linspace(0, 0.5, 6))
		if varying_param == 'pi':
			ax.plot(params, params, color = 'black', linestyle='--', label='True π₁', linewidth=0.5)
		else:
			ax.plot(params, [pi] * N, color = 'black', linestyle='--', label='True π₁', linewidth=0.5)

		for ax in axes_bottom[-1:]:
			if not isinstance(varying_param, str) and issubclass(varying_param, WeightTransform):
				varying_param_label_str = t.param_name
				if varying_param == PowerTransform:
					ax.set_xticks(np.linspace(*varying_param_bounds, 7))
			ax.set_xlabel(varying_param_label_str)
			ax.yaxis.set_ticks_position('right')
			ax.legend(loc='lower right', fontsize=8)
		for ax in axes_bottom[:-1]:
			ax.set_xticks([])
			ax.yaxis.set_ticks_position('right')
			ax.legend(loc='lower right', fontsize=8)

		mask_out = (X[:, 0] < xmin) | (X[:, 0] > xmax) | (X[:, 1] < ymin) | (X[:, 1] > ymax)
		percentage_out = mask_out.sum() / X.shape[0] * 100

		Π_hat = list(np.sort(np.diag(G['Π_hat'])).round(2))
		Π_hat = [f"{pi:.2f}" for pi in Π_hat]
		stats = [f"\nARI:   {G['Rand']:.2f}",
					f"ΠẐ:   {Π_hat}",
					f"GS:   {G['GMM_score']:.2f}",
					f"CT:    {G['C_true']:.5f}",
					f"CG:   {G['C_graph']:.5f}",
					f"CE:   {G['C_embed']:.5f}",
					f'\n{percentage_out:.1f}% X outside']
		
		for stat in stats: ax_big.plot([], [], linestyle='', marker='', color='none', label=stat, alpha=0)
		ax_big.legend(
					loc='upper left',
					title=f'{varying_param_label_str} = {params[n]:.2f}',
					title_fontsize=18,
					handlelength=0,
					handletextpad=0)

		delta = 0.01   # fraction of figure width to shift right
		for ax in axes_top:
			pos = ax.get_position()
			ax.set_position([
				pos.x0 + delta,
				pos.y0,
				pos.width,
				pos.height
			])

		return fig
	
	def embedding_gif(self, model = betaWSBM,
				  t = PowerTransform(1),
				  varying_param = 'rho',
				  varying_param_bounds = (0, 0.5),
				  shift_factor_x = 0, shift_factor_y = 0,
				  scaling_factor_x = 1, scaling_factor_y = 1,
				  fps = 32,
				  KDE = True,
				  generate_frames = True):
		
		varying_param_str = varying_param if isinstance(varying_param, str) else varying_param.__name__
		name = f'{model.name}_{t.id}_{varying_param_str}_{varying_param_bounds[0]}_{varying_param_bounds[1]}'
		path = f"Computation/data_for_gifs"
		dico = load(f"{path}/{name}.joblib")

		displayed_graphs_metrics = {}
		for m_id in VANILLA_METRICS_ID + ['pi_1']:
			displayed_graphs_metrics[m_id] = [G[m_id] for G in dico['graphs']]

		dico['displayed_graphs_metrics'] = displayed_graphs_metrics

		def select_reference_limits(graphs, top_percent=0.1, expand_factor = 1):
			sorted_graphs = sorted(graphs, key=lambda G: G['Rand'], reverse=True)
			n_top = max(1, int(len(sorted_graphs) * top_percent))
			candidates = sorted_graphs[:n_top]
			centers = []
			limits = []
			for G in candidates:
				X = G['X']
				centers.append(X.mean(axis=0))
				fig, ax = plt.subplots()
				ax.scatter(X[:, 0], X[:, 1], c=G.get('Z', None), cmap='bwr', marker='.', alpha=0.5)
				xlim = ax.get_xlim()
				ylim = ax.get_ylim()
				plt.close(fig)
				limits.append((xlim, ylim))

			x_min, x_max = np.mean([l[0][0] for l in limits]), np.mean([l[0][1] for l in limits])
			y_min, y_max = np.mean([l[1][0] for l in limits]), np.mean([l[1][1] for l in limits])
			dx, dy = (x_max - x_min) * expand_factor, (y_max - y_min) * expand_factor
			return (dx, dy)
		
		dico['embedding_ref_window'] = select_reference_limits(dico['graphs'], top_percent=0.1, expand_factor=1.25)

		filenames = []

		N = len(dico['graphs'])
		name = name.replace('.', '')
		if generate_frames:
			for i in range(0, N):
				fig = self.plot_embedding_for_varying_param(i, dico,
											model = model,
											t = t,
											varying_param = varying_param,
											varying_param_bounds = varying_param_bounds,
											shift_factor_x = shift_factor_x, shift_factor_y = shift_factor_y,
											scaling_factor_x = scaling_factor_x, scaling_factor_y = scaling_factor_y,
											KDE = KDE)
				
				path = f'{self.folder_path}/Embedding_Gifs/Frames/{name}'
				fname = f"frame_{i:03d}.png"
				Path(path).mkdir(parents=True, exist_ok=True)
				fig.savefig(f'{path}/{fname}', dpi=200)
				plt.close(fig)
				filenames.append(f'{path}/{fname}')

		else:
			filenames = [f'{self.folder_path}/Embedding_Gifs/Frames/frame_{i:03d}.png' for i in range(N)]

		#name = name.replace('.', '')
		#clip = ImageSequenceClip(filenames, fps=fps)
		# plugin='ImageMagick' often gives better colors, or try program='ffmpeg'
		#clip.write_gif(f'{self.folder_path}/Embedding_Gifs/{name}.gif', fps=fps, program='ffmpeg')

		#name = name.replace('.', '')
		#with imageio.get_writer(f'{self.folder_path}/Embedding_Gifs/{name}.gif', mode='I', fps=fps) as writer:
		#	for fname in filenames:
		#		image = imageio.imread(fname)
		#		writer.append_data(image)

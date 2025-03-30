from .Plotting import *
from scipy.ndimage import gaussian_filter, convolve

n = 1000
K = 2

gaussian_sigma = 0.8
blur_itersions = 1

kernel_size = 3
kernel = np.ones((kernel_size, kernel_size), dtype=float)
kernel /= kernel.sum()

def art_plot_scatter_Rand_vs_Chernoff(metrics, n_points_ratio_displayed=1.0,
                                  C_transform='Sigmoid-Ln', n=n, K=K):
    # Define the transformation function based on C_transform
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
    
    # Create a 3x3 grid of subplots (instead of 5x3)
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # Remove any global title or text-based decoration.
    # (On purpose, the artistic version omits titles, labels, legends, grids, etc.)
    
    # Function to apply the "artistic" style:
    #   - Remove ticks, labels, legends, titles, grid lines
    #   - Set background color to an anthracite grey
    #   - Hide spines
    def artistic_polishing(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.grid(False)
        ax.set_facecolor('#4B4B4B')  # Anthracite grey background
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Helper function to get the x and y values for scatter plotting.
    # It filters out non-positive x values, computes Spearman correlation (unused here),
    # and applies the chosen transformation on x.
    def get_xysc(key, m_id):
        x, y = metrics[key][m_id][::skip], metrics[key]['Rand'][::skip]
        valid = x > 0
        x = x[valid]
        y = y[valid]
        s_corr = spearmanr(x, y)[0]
        x = transform(x)
        return x, y, s_corr

    # --- Row 1: Grouping by ρ ---
    # (Initial block previously at row index 1 using cmap.viridis)
    for i, (ax, m_id) in enumerate(zip(axes[0], METRICS_ID[1:])):
        cmap = cm.viridis
        nc = len(RHOS)
        for j, rho in enumerate(RHOS):
            x, y, _ = get_xysc(f'rho:{rho}', m_id)
            ax.scatter(x, y, s=0.5, alpha=0.5, color=cmap(j / nc))
        artistic_polishing(ax)
    
    # --- Row 2: Grouping by π ---
    # (Initial block previously at row index 2 using cmap.plasma)
    for i, (ax, m_id) in enumerate(zip(axes[1], METRICS_ID[1:])):
        cmap = cm.plasma
        nc = len(PIS)
        for j, pi in enumerate(PIS):
            x, y, _ = get_xysc(f'pi:{pi}', m_id)
            ax.scatter(x, y, s=0.5, alpha=0.5, color=cmap(j / nc))
        artistic_polishing(ax)
    
    # --- Row 3: Grouping by Model ---
    # (Initial block previously at row index 3 using cmap.inferno)
    for i, (ax, m_id) in enumerate(zip(axes[2], METRICS_ID[1:])):
        cmap = cm.inferno
        nc = len(MODELS)
        for j, model in enumerate(MODELS):
            x, y, _ = get_xysc(model, m_id)
            ax.scatter(x, y, s=0.5, alpha=0.5, color=cmap(j / nc))
        artistic_polishing(ax)
    
    plt.tight_layout()
    save_file('ArtisticPlots', f'Rand_vs_Chernoff_{C_transform}', dpi=500)

def art_plot_metrics_heatmap(rho, pi, model, transformation, metrics, shared=False, log=False, corr_info=True, n=n):
    # Determine vmin and vmax if using shared scaling across metrics
    if shared:
        values = np.concatenate([metrics[m_id] for m_id in METRICS_ID[1:]])
        vmin, vmax = np.min(values[values > 0]), np.max(values)
    else:
        vmin = vmax = None

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10))
    axes = axes.flatten()

    # Remove any overall title or text; we work with purely visual, "artistic" plots.
    for i, (ax, m_id) in enumerate(zip(axes, METRICS_ID)):
        # Apply a gaussian blur to the metric grid before plotting
        metric_grid = metrics[m_id]
        #blurred = gaussian_filter(metric_grid, sigma=gaussian_sigma)
        blurred = metric_grid.copy()
        #for _ in range(blur_itersions):
        #    blurred = convolve(blurred, kernel, mode='reflect')

        # Select colormap and normalization based on the metric id
        if m_id == 'Rand':
            cmap = 'Reds'
            # For Rand, no normalization is applied
            sns.heatmap(blurred, cmap=cmap, ax=ax, xticklabels=False, yticklabels=False, cbar=False)
        else:
            cmap = 'Blues'
            if log:
                norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
            else:
                norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
            sns.heatmap(blurred, cmap=cmap, ax=ax, norm=norm, xticklabels=False, yticklabels=False, cbar=False)
        
        # Remove all textual elements, ticks, labels and titles
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    plt.tight_layout()
    m_str = f'{model.name}_{rho}_{pi}'.replace('.', '')
    save_file(f'ArtisticPlots/Metrics_Heatmap/', f'{transformation.id}_{m_str}', dpi=500)

def art_plot_bias_heatmap(rho, pi, model, transformation, metrics, log=True, n=n):
    # Create a 3x2 grid of subplots (one row per bias, two columns per metric)
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    
    # Iterate over biases (assumed to be of length 3) and over the two metrics (METRICS_ID[2:])
    for row_idx, bias in enumerate(BIASES):
        for col_idx, m_id in enumerate(METRICS_ID[2:]):
            ax = axes[row_idx, col_idx]
            
            # Obtain the bias grid and apply a Gaussian blur to it
            bias_grid = metrics['Bias'][m_id][bias]
            #blurred_grid = bias_grid.copy()
            #blurred_grid = gaussian_filter(bias_grid, sigma=gaussian_sigma)
            blurred_grid = bias_grid.copy()
            #for _ in range(blur_itersions):
            #    blurred_grid = convolve(blurred_grid, kernel, mode='reflect')

            N = blurred_grid.shape[0]
            
            # Determine normalization parameters based on percentiles over both C_graph and C_embed
            values = np.concatenate((metrics['Bias']['C_graph'][bias], 
                                     metrics['Bias']['C_embed'][bias]))
            vmin, vmax = np.percentile(values, 5), np.percentile(values, 95)
            bound = max(abs(vmin), abs(vmax))
            
            # Set normalization and colormap according to the bias type and log flag
            if bias == 'log':
                norm = colors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
                cmap = 'RdBu'
            else:
                if log:
                    norm = colors.SymLogNorm(linthresh=bound * 0.05 if bound != 0 else 1e-3,
                                             vmin=-bound, vmax=bound, base=10, clip=True)
                else:
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)
                cmap = 'Blues'
            
            # Plot the heatmap without colorbar and without any tick or text decorations
            sns.heatmap(blurred_grid, ax=ax, cmap=cmap, norm=norm, cbar=False,
                        xticklabels=False, yticklabels=False)
            
            # Remove all text elements and ticks
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Invert the y-axis if needed for consistency
            ax.invert_yaxis()
    
    plt.tight_layout()
    m_str = f'{model.name}_{rho}_{pi}'.replace('.', '')
    save_file(f'ArtisticPlots/Bias_Heatmap/By_Model', f'{transformation.id}_{m_str}', dpi=500)

from scipy.ndimage import gaussian_filter

def art_plot_best_transform_heatmaps(rho, pi, model, metrics, n=n):
    rows = ['C_graph-Best Transform', 'C_embed-Best Transform']
    cols = ['Rand', 'Regret']

    # Create a 2x3 grid of subplots without extra gridspec_kw
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Remove any global title; we're aiming for an artistic, minimalistic output.
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            ax = axes[i, j]
            grid = metrics[row][col]
            # Apply a Gaussian blur to the grid
            #blurred = gaussian_filter(grid, sigma=gaussian_sigma)
            blurred = grid.copy()
            #for _ in range(blur_itersions):
            #    blurred = convolve(blurred, kernel, mode='reflect')
            #N = blurred.shape[0]

            if col == 'Rand':
                norm = colors.Normalize(vmin=0, vmax=1, clip=True)
                sns.heatmap(blurred, ax=ax, norm=norm, cmap='Reds', cbar=False,
                            xticklabels=False, yticklabels=False)
            else:  # Regret
                norm = colors.Normalize(vmin=0, vmax=1, clip=True)
                sns.heatmap(blurred, ax=ax, norm=norm, cmap='Purples', cbar=False,
                            xticklabels=False, yticklabels=False)

            # Remove all text, ticks, labels, and legends
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)

    plt.tight_layout()
    save_file('ArtisticPlots/Best_Transform', f'Grid_{model.name}_{rho}_{pi}', dpi=500)
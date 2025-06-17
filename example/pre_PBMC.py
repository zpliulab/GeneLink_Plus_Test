import pandas as pd
from scipy.io import mmread
import os
import scanpy as sc
import magic
import numpy as np
import matplotlib.pyplot as plt
import random
import tarfile

def pre_PBMCs():
    # Set global font to Times New Roman for consistent figure styling
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16

    # Set random seed to ensure reproducibility of any stochastic steps
    np.random.seed(12)
    random.seed(12)

    # ----------------------------
    # 1. Data extraction & loading
    # ----------------------------
    tar_file_path = 'pbmc8k_filtered_gene_bc_matrices.tar.gz'
    extracted_dir = 'pbmc8k_filtered_gene_bc_matrices'
    if not os.path.exists(extracted_dir):
        with tarfile.open(tar_file_path, 'r:gz') as tar:
            tar.extractall()
        print(f"Extracted {tar_file_path}.")

    basic_dir = os.path.join(extracted_dir, 'filtered_gene_bc_matrices', 'GRCh38')
    genes_dir = os.path.join(basic_dir, 'genes.tsv')
    barcodes_dir = os.path.join(basic_dir, 'barcodes.tsv')
    matrix_dir = os.path.join(basic_dir, 'matrix.mtx')

    # Load gene names and ensure they are strings
    genes = pd.read_csv(genes_dir, header=None, sep='\t', usecols=[1]).squeeze("columns")
    genes = genes.astype('str')

    # Load cell barcodes
    barcodes = pd.read_csv(barcodes_dir, header=None, sep='\t').squeeze("columns")

    # Read sparse expression matrix and convert to dense
    matrix = mmread(matrix_dir)
    matrix_dense = matrix.todense()

    # Create DataFrame: rows = genes, columns = cells
    gene_expression = pd.DataFrame(matrix_dense, index=genes, columns=barcodes)
    gene_expression.index.name = None

    # Save raw count matrix for inspection
    output_path = os.path.join(basic_dir, 'raw_counts.csv')
    gene_expression.to_csv(output_path)
    print(f"Raw counts saved to {output_path}")

    # ---------------------------------------
    # 2. Build AnnData & basic preprocessing
    # ---------------------------------------
    adata = sc.AnnData(gene_expression.T)
    adata.obs.index.name = None
    adata.var.index.name = None
    adata.var_names_make_unique()

    # Filter genes: keep genes expressed in ≥5% of cells
    min_cells = int(adata.n_obs * 0.05)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    # Filter cells: keep cells expressing ≥200 genes
    sc.pp.filter_cells(adata, min_genes=200)

    # ----------------
    # 3. Normalization
    # ----------------
    # Normalize total counts per cell to 10,000
    # This ensures comparability of expression across cells
    sc.pp.normalize_total(adata, target_sum=1e4)

    # ------------------------------
    # 4. MAGIC imputation & smoothing
    # ------------------------------
    # Apply MAGIC to impute dropout and denoise:
    #  - KNN graph: K = 5 nearest neighbors (default)
    #  - Diffusion steps: t = 3 (default)
    magic_operator = magic.MAGIC()  
    adata = magic_operator.fit_transform(adata)  

    # Log-transform counts to stabilize variance
    sc.pp.log1p(adata)

    # Save imputed expression for downstream use
    gene_expression_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    gene_expression_df = gene_expression_df.round(3).T
    output_path = os.path.join(basic_dir, 'imputed_counts.csv')
    gene_expression_df.to_csv(output_path)
    print(f"Imputed counts saved to {output_path}")

    # -------------------------------
    # 5. Dimensionality reduction & PCA
    # -------------------------------
    # Compute top 40 PCs using 'arpack' for speed
    sc.tl.pca(adata, svd_solver='arpack')

    # ----------------------------------
    # 6. Nearest-neighbor graph & t-SNE
    # ----------------------------------
    # Build neighborhood graph with 10 neighbors in PC space
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.tsne(adata)

    # ---------------------
    # 7. Clustering (Leiden)
    # ---------------------
    # Perform Leiden clustering at resolution 0.1
    sc.tl.leiden(adata, resolution=0.1)

    # --------------------------------
    # 8. Marker gene ranking & plotting
    # --------------------------------
    # Identify top 30 marker genes per cluster with t-test
    sc.tl.rank_genes_groups(adata, 'leiden', method='t-test', n_genes=30)
    sc.pl.rank_genes_groups(adata, n_genes=30, sharey=False)

    # Visualize t-SNE colored by selected marker genes
    genes_to_plot = ['CD79A', 'MS4A1', 'RUNX3', 'IL32', 'CCR7', 'MAL', 'CD14', 'CD36']
    for gene in genes_to_plot:
        sc.pl.tsne(adata, color=gene, show=True)

    # Visualize t-SNE colored by Leiden cluster assignments
    sc.pl.tsne(adata, color=['leiden'], show=True)

    # ----------------------------------
    # 9. Map clusters to known cell types
    # ----------------------------------
    cell_type_mapping = {'0': 'CD8', '1': 'CD4', '2': 'CD14 Monocytes', '3': 'B'}
    adata.obs['cell_type'] = adata.obs['leiden'].map(cell_type_mapping)

    # Export expression per cell type
    for cell_type in adata.obs['cell_type'].unique():
        sub_data = adata[adata.obs['cell_type'] == cell_type, :]
        sub_df = pd.DataFrame(sub_data.X, index=sub_data.obs_names, columns=sub_data.var_names)
        sub_df = sub_df.round(3).T
        fname = f'{cell_type}_imputed.csv'
        sub_df.to_csv(os.path.join(basic_dir, fname))
        print(f"Data for {cell_type} saved to {fname}")

    # --------------------------------
    # 10. Final marker analysis & plot
    # --------------------------------
    sc.tl.rank_genes_groups(adata, 'cell_type', method='t-test_overestim_var', n_genes=30)
    sc.pl.rank_genes_groups(adata, n_genes=30, xlabel='Adjusted t score', sharey=False)
    sc.pl.tsne(adata, color='cell_type', title='tSNE by Cell Type')

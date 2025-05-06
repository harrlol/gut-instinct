import scipy.sparse as sp
import torch
import numpy as np


def get_expression(sdata, dataset_patch):
    # get expression data
    gene_names = sdata['anucleus'].var['gene_symbols'].values
    gene_exp_matrix = sdata['anucleus'].X

    # need to pair up cell id with the cell's index in expression matrix
    cell_id_to_idx = {cell_id: i for i, cell_id in enumerate(sdata['anucleus'].obs['cell_id'].values)}
    valid_cell_ids = dataset_patch.valid_cell_ids
    valid_cell_idx = [cell_id_to_idx[cell_id] for cell_id in valid_cell_ids if cell_id in cell_id_to_idx] # taking the intercept just in case

    # subset expression matrix by filtered cell
    dataset_expression = gene_exp_matrix[valid_cell_idx]

    if sp.issparse(dataset_expression):
        dataset_expression = dataset_expression.toarray()
    return dataset_expression

def uni_helper(patch, uni_model):
    with torch.inference_mode():
        feature_emb = uni_model(patch)
        return feature_emb

def extract_features(loader, model):
    features = []
    cell_ids = []

    with torch.no_grad():
        for batch in loader:
            patches = batch['patch'].to('cuda')
            output = uni_helper(patches, model)

            # reshape
            output = output.squeeze()

            features.append(output.cpu().numpy())
            cell_ids.extend(batch['cell_id'])

    features = np.vstack(features)
    feature_cell_ids = np.array(cell_ids)
    return features, feature_cell_ids

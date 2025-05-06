import h5py
import numpy as np

def calculate_avg_gene_expression(sdata, patch, patch_coords, cell_coords, gene_expression_df):
    """
    Calculates the average gene expression for cells within a patch.
    """
    x_start, y_start = patch_coords
    x_end, y_end = x_start + patch.shape[0], y_start + patch.shape[1]

    cell_ids_in_patch = [
        cell_id
        for cell_id, (x, y) in cell_coords.items()
        if x_start <= x < x_end and y_start <= y < y_end
    ]

    if cell_ids_in_patch:
        avg_expression = gene_expression_df.loc[cell_ids_in_patch].mean(axis=0).values
    else:
        avg_expression = np.zeros(gene_expression_df.shape[1])

    return avg_expression


def generate_and_save_patches(image, sdata, cell_coords, gene_expression_df,
                              window_size=224, stride=16, output_file="patches_and_expressions.h5",
                              cell_threshold=5, center_size=16):
    """
    Generates patches, calculates average gene expressions, and saves to HDF5.
    Filters patches with less than cell_threshold cells.
    """
    with h5py.File(output_file, 'w') as hf:
        patches_group = hf.create_group('patches')
        expressions_group = hf.create_group('expressions')

        for i in range(0, image.shape[0] - window_size + 1, stride):
            for j in range(0, image.shape[1] - window_size + 1, stride):
                # calculate 16x16 coordinates
                center_x_start = i + (window_size - center_size) // 2
                center_y_start = j + (window_size - center_size) // 2
                center_x_end = center_x_start + center_size
                center_y_end = center_y_start + center_size

                # check if any cells
                cells_in_center = any(
                    center_x_start <= x < center_x_end and center_y_start <= y < center_y_end
                    for x, y in cell_coords.values()
                )

                # skip if not
                if not cells_in_center:
                    continue

                # generate and save patch
                patch = image[i:i + window_size, j:j + window_size]
                patch_coords = (i, j)

                avg_expression = calculate_avg_gene_expression(
                    sdata, patch, patch_coords, cell_coords, gene_expression_df
                )

                patch_key = f'{i}_{j}'
                patches_group.create_dataset(patch_key, data=patch)
                expressions_group.create_dataset(patch_key, data=avg_expression)


def load_patches_and_expressions(h5_file="patches_and_expressions.h5"):
    """
    Loads patches and average gene expressions from the HDF5 file.
    """

    with h5py.File(h5_file, 'r') as hf:
        patches = [hf['patches'][key][()] for key in hf['patches'].keys()]
        expressions = [hf['expressions'][key][()] for key in hf['expressions'].keys()]

    return patches, expressions
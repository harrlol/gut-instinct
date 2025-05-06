#!/bin/bash
set -e

ENV_NAME=".venv"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # root dir of repo
echo "$REPO_DIR"
cd "$REPO_DIR"
python3 -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

pip install ipykernel
python -m ipykernel install --user --name gutinstinct --display-name "Python (gutinstinct)"

echo "✅ Virtual environment setup complete!"
echo "👉 Run 'source .venv/bin/activate' and 'jupyter notebook' to begin."



# uncomment the below if you don't have access to the data

# ## get data files
# mkdir -p zipped_files
# gdown 1463tF6_IcVcCQCZudCFU0fz1_GbHA8ak --output zipped_files/DC1.zarr.zip
# gdown 18KynQo9F-YKREG5kDTI9InqP7NG2L0cx --output zipped_files/DC5.zarr.zip
# gdown 1f7akUHKC4UFm_X9KWRZVsUkYOlbHUA5U --output zipped_files/UC1_I.zarr.zip
# gdown 1UGhC68uyaVQBFXsrvkHveU4rFiwjgTlm --output zipped_files/UC1_NI.zarr.zip
# gdown 1OLIh0rbCBwDifYDgT4fR-mhltHCxDr34 --output zipped_files/UC6_I.zarr.zip
# gdown 1RlJDxHXTkZffhfSS-ZXzt8vRWaoRumUp --output zipped_files/UC6_NI.zarr.zip
# gdown 1YK0DqV1EffewNmXDb7lITNwK-BE2bWv4 --output zipped_files/UC7_I.zarr.zip
# gdown 1e7qAk9A_z6uO-jhnXrw3KDIrML3tqG02 --output zipped_files/UC9_I.zarr.zip

# ## unzip files
# mkdir -p raw
# for file in zipped_files/*.zip; do
#     base=$(basename "$file" .zip)
#     target_dir="raw/$base"
#     mkdir -p "$target_dir"
#     echo "Unzipping $file to $target_dir..."
#     unzip "$file" -d "$target_dir" && rm "$file"
# done
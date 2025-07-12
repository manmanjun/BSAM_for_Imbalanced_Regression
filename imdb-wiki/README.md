## A Pytorch Implementation of Balanced Sharpness-Aware Minimization for Imbalanced Regression (ICCV 2025)

### Usage
- Download IMDB-WIKI dataset by 
    ```
    cd data
    python download_imdb_wiki.py
    tar -xvf imdb_crop.tar
    tar -xvf wiki_crop.tar
    ```

- We provide the sqrt_inv weight as `imdb_sqrt_weight.pt` which is obtained from the code of [imbalanced-regression](https://github.com/YyzHarry/imbalanced-regression/blob/a6fdc45d45c04e6f5c40f43925bc66e580911084/imdb-wiki-dir/datasets.py#L64).

- To train the model, run 
    ```
    CUDA_VISIBLE_DEVICES=0 python train.py --rho 0.05
    ```

- To test the model, run 
    ```
    CUDA_VISIBLE_DEVICES=0 python test.py --test_dir xxx.pth
    ```

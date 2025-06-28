## A Pytorch Implementation of Balanced Sharpness-Aware Minimization for Imbalanced Regression (ICCV 2025)

### Usage
Download AgeDB dataset from [here](https://ibug.doc.ic.ac.uk/resources/agedb/) and extract the zip file (you may need to contact the authors of AgeDB dataset for the zip password) to folder `./data`.

- We provide the sqr_inv weight as `agedb_sqrt_weight.pt` which is obtained from the code of [imbalanced-regression](https://github.com/YyzHarry/imbalanced-regression/blob/a6fdc45d45c04e6f5c40f43925bc66e580911084/agedb-dir/datasets.py#L64).

- To train the model, run 
    ```
    CUDA_VISIBLE_DEVICES=0 python train.py --rho 0.2
    ```

- To test the model, run 
    ```
    CUDA_VISIBLE_DEVICES=0 python test.py --test_dir xxx.pth
    ```

# About

This is the code repository for the experiments of the paper ** accepted at the DistShift workshop
of NeurIPS 2022.

## Setting up

`conda create -f environment.yaml` then install pytorch for your CUDA version in this created conda
env (1.12.1 is recommended), such as: `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge`.

Download the DexYCB dataset. Optionally download the ShapeNetCore dataset for the ObMan dataset.

Download my `learn2learn` fork, switch to the branch `derivative_order_annealing` and in your conda
environment you just created, run `python setup.py build && python setup.py install`  in the root of learn2learn.


## Training & testing

Refer to Hydra's documentation for more details.

Run `./train_test.py experiment=<name_of_experiment>`; see the list of YAML files in
`conf/experiments` for experiment names (ommit the `.yaml` extension).
Use the option `test_mode=true` to test, and don't forget to specify a trained model with
`experiment.saved_model=<path>`.

## Analysis
### Procrustes Analysis for the heatmap of grasps

Run the `procrustes_analysis.py` script.

### Analysing the gradient norms during adaptation

```
./analyse_grads.py experiment=anil_cnn_dexycb experiment.hold_out=9 experiment.saved_model=models/handonly/anil_9.tar analyse_tasks=100 hand_only=true
```
Make sure that `experiment.hold_out` matches your trained model.


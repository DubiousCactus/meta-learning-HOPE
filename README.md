# meta-learning-HOPE

## Installation

Run `python setup_hope.py install` to install the HOPE project as a package.


## Training

Refer to Hydra's documentation for more details.

Run `./train.py experiment=<name_of_experiment>`; see the list of YAML files in `conf/experiments`
for experiment names (ommit the `.yaml` extension).

## Analysing the gradient norms during adaptation

```
./analyse_grads.py experiment=anil_cnn_dexycb experiment.hold_out=9 experiment.saved_model=models/handonly/anil_9.tar analyse_tasks=100 hand_only=true
```

Make sure that `experiment.hold_out` matches your trained model.

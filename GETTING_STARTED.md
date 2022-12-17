## Getting Started with AlphAction

We recommend to create a directory `models` inside `HIT/data` to place 
model weights. 

```shell
cd /path/to/HIT
mkdir data/models
ln -s /path/to/models data/models
```

### Training

Download pre-trained models from [MODEL_ZOO.md](MODEL_ZOO.md#pre-trained-models).
Then place pre-trained models in `data/models` directory with following structure:

```
models/
|_ pretrained_models/
|  |_ SlowFast-ResNet50-4x16.pth
```

Train on a single GPU:

```shell
python train_net.py --config-file "config_files/hitnet.yaml"
```
#### For training on AVA and MultiSports, we use multiple GPUs.
We use the launch utility `torch.distributed.launch` to launch multiple 
processes for distributed training on multiple gpus. `GPU_NUM` should be
replaced by the number of gpus to use. Hyper-parameters in the config file
can still be modified in the way used in single-GPU training.

Configs for these datasets are coming soon. The reader can also refer to the supplementary materials and create their config files.

```shell
python -m torch.distributed.launch --nproc_per_node=GPU_NUM \
train_net.py --config-file "path/to/config/file.yaml" \
```

### Inference

Run the following command to perform inference. Note that 
our code first tries to load the `last_checkpoint` in the `OUTPUT_DIR`. If there
 is no such file in `OUTPUT_DIR`, it will then load the model from the 
 path specified in `MODEL.WEIGHT`. To use `MODEL.WEIGHT` to do the inference,
 you need to ensure that there is no `last_checkpoint` in `OUTPUT_DIR`. 
 You can download the model weights from [MODEL_ZOO.md](MODEL_ZOO.md#ava-models).
 
 ```shell
python test_net.py --config-file "hitnet.yaml" \
MODEL.WEIGHT "path/to/model/weight"
 ```
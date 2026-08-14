## Sharded training set

Gadi has limited inodes per NCI project. To save inodes, we shard the training sets into a collection of large zipped zarr files and resemble them in the Pytorch data loader. Each zipped shard contains many small training sample files.

## Training Directory Structure

The following snippet shows the directory structure of the training set. Under the root training data directory, there are two main sub-directories:

* The data sub-directory contains the shards in zipped zarr file format. For example, tas.0.zarr.zip is the first shard for the tas variable.

* The stats sub-directory contains the min/max normalizer for the variable.

```
<training data directory>
├── data
│   ├── tas.0.zarr.zip
│   ├── tas.100.zarr.zip
│     ... 
└── stats
    ├── tas_inp.pt
      ...
```

* For a shard file, the directory within the shard has the following structure. For example, in tas.0.zarr.zip, each file such as 200.0.0.0 is a training sample. input/200.0.0.0 is the ERA5 2t and target/200.0.0.0 is the BARRA2-C2 tas. 

* For a sample file such as 200.0.0.0, it is a numpy array with shape `[latent channels, height, width]`. Please refer to [build_training_set.py](./build_training_set.py) for the implementation details.

```
<var_name>.<shard_id>.zarr.zip
├── input
│   ├── 200.0.0.0
│   └── 2655.0.0.0
│    ...
└── target
    ├── 200.0.0.0
    └── 2655.0.0.0
     ...
```

## Building the Training Set

* Run `bash run_build_sharded_training_set.sh` to build an small example training set with only 20 samples per shard. This is to give a sense what the training set looks like.

* To build the full training set, Run `cd data && bash run_build_training_sets_main.sh`. This script generates PBS jobs for each variable to auto-submit the PBS jobs on your behalf. These scripts combined will use more than 1000 CPUs. Please get familiar with the scripts before running them.

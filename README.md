# E2Efold: RNA Secondary Structure Prediction By Learning Unrolled Algorithms

pytorch implementation for [RNA Secondary Structure Prediction By Learning Unrolled Algorithms](https://openreview.net/forum?id=S1eALyrYDH) [1]

## Setup

### Install the package
The environment that we use is given in `environment.yml`. You can consider to use exactly the same environment by running the following command.
```
Conda env create -f environment.yml
```

Please navigate to the root of this repository, and run the following command to install the package e2efold.
```
source activate rna_ss # activate the enviornment
pip install -e .
```

### Data

Please download the RNA secondary structure [data](https://drive.google.com/open?id=19KPRYJjjMJh1qdMhtmUoYA_ncw3ocAHc) and put all the `.tgz` files in the `/data` folder. Then run:
```
tar -xzf rnastralign_all.tgz
tar -xzf rnastralign_all_600.tgz
tar -xzf archiveII_all.tgz
```
These files contain the processed data. As a reference, the codes for preprocessing the data are also given in this `/data` folder.

### Folder structure

Finally the project should have the following folder structure:

```
e2efold
|___e2efold  # source code
|___data  # data
    |___archiveII_all
    |___rnastralign_all_600
    |___rnastralign_all
    |___preprocess_archiveii.py  # just as a reference. no need to run.
    |......
|___models_ckpt  # trained models
|___results
|___experiment_archiveii
|___experiment_rnastralign
...
```

## Test with trained model
You can download the [pretrained models](https://drive.google.com/open?id=1m038Fw0HBGEzsvhS0mRxd0U7cGXqLAVt) and put the `.pt` files in the folder `/models_ckpt`.

### RNAStralign
You can navigate to the `/experiment_ranstralign` folder and run the following command to test the model on RNAStralign test dataset:
```
python e2e_learning_stage3.py -c config.json --test True
python e2e_learning_stage3_rnastralign_all_long.py -c config_long.json --test True
```

### ArchiveII
You can navigate to the `/experiment_archiveii` folder and run the following command to test the model on ArchiveII data. Note that the saved model is trained on the RNAStralign database.
```
# For sequences shorter than 600
python e2e_learning_stage3.py -c config.json

# For sequences from 600 to 1800, not performing well on long sequence in archiveii
python e2e_learning_stage3_rnastralign_all_long.py -c config_long.json
```

## Reproduce the training process

The model is trained on the RNAstralign training set. To reproduce the training process, you can navigate to the folder
`e2efold_rnastralign` and run:
```
# For sequences shorter than 600
python e2e_learning_stage1.py -c config.json  # pre-train the score network
python e2e_learning_stage3.py -c config.json  # end-to-end training

# For sequences from 600 to 1800 
python e2e_learning_stage1_rnastralign_all_long.py -c config_long.json 
python e2e_learning_stage3_rnastralign_all_long.py -c config_long.json 
```



## Citation
If you found this library useful in your research, please consider citing
```
@article{chen2020rna,
  title={RNA Secondary Structure Prediction By Learning Unrolled Algorithms},
  author={Chen, Xinshi and Li, Yu and Umarov, Ramzan and Gao, Xin and Song, Le},
  journal={arXiv preprint arXiv:2002.05810},
  year={2020}
}
```
## References
[1] Xinshi Chen*, Yu Li*, Ramzan Umarov, Xin Gao, Le Song. "RNA Secondary Structure Prediction By Learning Unrolled Algorithms." *In International Conference on Learning Representations.* 2020.

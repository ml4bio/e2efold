# E2Efold: RNA Secondary Structure Prediction By Learning Unrolled Algorithms

pytorch implementation for [RNA Secondary Structure Prediction By Learning Unrolled Algorithms](https://openreview.net/forum?id=S1eALyrYDH) [1]

[[Paper](https://openreview.net/pdf?id=S1eALyrYDH)] [[Presentation](https://iclr.cc/virtual_2020/poster_S1eALyrYDH.html)] [[Slides](http://xinshi-chen.com/papers/slides/iclr2020-e2efold.pdf)]
[<a href="https://cse.gatech.edu/news/633633/machine-learning-tool-may-help-us-better-understand-rna-viruses">GaTech news</a>]
[<a href="https://www.jiqizhixin.com/articles/2020-02-18-8">Chinese news</a>]
[<a href="https://mp.weixin.qq.com/s/eyWlQdRqrnnxOKv0TF6gmA">Chinese introduction</a>]
[<a href="https://github.com/ml4bio/e2efold/blob/master/slides_and_articles/long_abstract.pdf">Plain explanation</a>]


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
|___e2efold_productive  # productive code for handling new sequences
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
|___slides_and_articles    # slides and articles related to the project 
...
```

## Prediction for user's input sequence

To directly use our trained model to make prediction for any RNA sequence, please refer to the information in `/e2efold_productive` folder.

## Reproduce experimental results in the paper

To reproduce the experiments in our paper, please refer to the following steps:

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

## Reproduce the training process or re-train the model on a new dataset

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

Given the training logic implemented in the above python files, you can modify the data generator to re-train the model on other datasets. Our data generator in defined in `e2efold/data_generator.py`. You could probably choose to define a Sub Class based on the Class `RNASSDataGenerator`.


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

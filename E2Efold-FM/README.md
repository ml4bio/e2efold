# E2Efold-FM 
E2Efold-FM: accurate RNA secondary structure prediction with end-to-end deep neural networks and RNA foundation model

This is the pytorch implementation of E2Efold-FM.


![Overview](./pics/E2Efold-FM_overview.svg)





## System Requirements

The `E2Efold-FM` package is built under the Linux system with the popular softwares [Anaconda](https://www.anaconda.com/) and [Pytorch](https://pytorch.org/). The versions of the software dependencies that the `E2Efold-FM` package has been tested on are provided in the `environment.yml`. Users can conveniently create the same environment by running the command:
```
conda env create -f environment.yml
```




## Installation guide


### Install the package
The environment that we use is given in `environment.yml`. You can create the same environment by running the command:
```
conda env create -f environment.yml
```


Please navigate to the root of this repository, and run the following command to install the package e2efoldFM:
```
conda activate e2efoldFM     # activate the enviornment
pip install -e .     # install the package
```


### Folder structure

The project has the following folder structure:

```
E2Efold-FM
|___e2efoldFM  # source code
|___e2efoldFM_productive  # productive code for handling new sequences
|___data  # data
    |___archiveII_all
    |___rnastralign_all_600
    |___rnastralign_all
    |___preprocess_archiveii.py  # just as a reference. no need to run.
    |......
|___models_ckpt  # trained models
|___results
|___pics
...
```





## Demo and instractions for use

To directly use our trained model to make prediction for any RNA sequence, please refer to the information in `/e2efoldFM_productive` folder.














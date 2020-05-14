# E2Efold productive package

## For short sequences (<600)
```
sh main_short.sh
```
The sample input sequences are in folder `short_seqs`. When use the package, please make sure the prepared data are in the same format. In `config.json`, you can specify the input sequences folder and the output folder. The output results are in the `.ct` format.


## For longer sequences (600<seq<1800)
The code is similar but with some matrix manipulation tricks for saving memory. It will be uploaded be the end of May 2020.
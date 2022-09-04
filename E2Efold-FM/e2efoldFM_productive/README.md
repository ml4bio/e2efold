# e2efoldFM productive package

## For sequences (<500)


For new RNA sequences, first use the [RNA-FM server](https://proj.cse.cuhk.edu.hk/rnafm/#/) to extract the r-ss features and put these `.npy` files in the `./ss-rnafm/r-ss` folder

Run the following command:
```
sh main.sh
```
The sample input sequences are in folder `seqs`. When use the package, please make sure the prepared data are in the same format. In `config.json`, you can specify the input sequences folder and the output folder. The output results are in the `.ct` format.

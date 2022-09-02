#!/bin/bash

# Training
# For sequences shorter than 600
python e2e_learning_stage1.py -c config.json 
python e2e_learning_stage3.py -c config.json 

# For sequences from 600 to 1800 
python e2e_learning_stage1_rnastralign_all_long.py -c config_long.json 
python e2e_learning_stage3_rnastralign_all_long.py -c config_long.json 


# Direct test against the testing data with trained model
python e2e_learning_stage3.py -c config.json --test True
python e2e_learning_stage3_rnastralign_all_long.py -c config_long.json --test True

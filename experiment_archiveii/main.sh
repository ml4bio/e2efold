#!/bin/bash

# Testing
# For sequences shorter than 600
python e2e_learning_stage3.py -c config.json 

# For sequences from 600 to 1800, not performing well on long sequence in archiveii
python e2e_learning_stage3_rnastralign_all_long.py -c config_long.json 

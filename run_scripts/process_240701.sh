#!/bin/bash

wandb sweep --project MLCASim ../configs/240628.yaml > 240628.run
ID = $(grep -oP '(?<=wandb: Creating sweep with ID: )\d+' 240628.run)
# echo $ID
# wandb agent $ID
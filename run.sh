#!/bin/bash

declare -a dataset=("CIFAR10")
decalre -a alpha=(0.5)
declare -a client_num=(10)

decalre -a model=("ConvNet")
declare -a ipc=(10)
declare -a dc_iterations=(1000)
decalre -a model_epochs=(500)

log_path="results/${dataset}_alpha${alpha}_${client_num}clients/${model}_${ipc}ipc_${dc_iterations}dc_${model_epochs}epochs/output.log"
mkdir -p $(dirname "$log_path")
echo "settings: ${dataset} with alpha ${alpha} and ${num_client} clients."
echo "parameters: ${model} with ${ipc} ipc, ${dc_iterations} dc iterations and ${model_epochs} epochs."
nohup python main.py --model ${model}
echo "process complete"
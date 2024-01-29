#!/bin/bash

path=''


while getopts 'f:' flag; do
  case "${flag}" in
    f) path="${OPTARG}" ;;
    *) echo "1" ;;
  esac
done



locomotion=(
    "hopper-random-v2"
    "hopper-medium-v2"
    "hopper-medium-replay-v2"
    "hopper-medium-expert-v2"
    "hopper-expert-v2"
)
 

for env in ${locomotion[*]}
do
    python algorithm/learn_critic.py \
    --env $env\
    --seed $"0"\
    --checkpoint_path $path
done&
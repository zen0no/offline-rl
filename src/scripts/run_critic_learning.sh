#!/bin/bash

PATH=""
ENV_NAME=""
 
while getopts ":f:e:" flag; do
  case "${flag}" in
    f) PATH="${OPTARG}" ;;
    e) ENV_NAME="${OPTARG}" ;;
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
 

for env_type in ${locomotion[*]}
do
    ENV="${ENV_NAME}"-"${env_type}"
    python3 src/algorithm/learn_critic.py \
    --env $ENV\
    --seed $"0"\
    --checkpoint_path $path
done&
#!/bin/bash


while getopts ":f:e:" flag; do
  case "${flag}" in
    f) CHECKPOINT_PATH="${OPTARG}" ;;
    e) ENV_NAME="${OPTARG}" ;;
    *) echo "1" ;;
  esac
done


locomotion=(
    "random-v2"
    "medium-v2"
    "medium-replay-v2"
    "medium-expert-v2"
    "expert-v2"
)

 

for env_type in ${locomotion[*]}
do
    ENV="${ENV_NAME}"-"${env_type}"
    python3 src/algorithm/learn_critic.py \
    --env $ENV\
    --seed $"0"\
    --checkpoint_path $CHECKPOINT_PATH
done&
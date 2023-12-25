#!/bin/bash

locomotion=(
    "hopper-medium-expert-v2"
    "hopper-medium-v2"
    "hopper-medium-replay-v2"
    "hopper-random-v2"
    "hopper-expert-v2")
    
locomotion_2=("walker2d-medium-v2"
    "walker2d-medium-expert-v2"
    "walker2d-medium-replay-v2"
    "walker2d-expert-v2"
    "walker2d-random-v2")


for env in ${locomotion[*]}
do
    python learn_distance_random.py \
    --env $env\
    --seed $"0"
done&

for env in ${locomotion[*]}
do
    python learn_distance_random.py \
    --env $env \
    --seed $"1"
done&

for env in ${locomotion[*]}
do
    python learn_distance_random.py \
    --env $env \
    --seed $"2"
done&

for env in ${locomotion_2[*]}
do
    python learn_distance_random.py \
    --env $env\
    --seed $"0"
done&

for env in ${locomotion_2[*]}
do
    python learn_distance_random.py \
    --env $env \
    --seed $"1"
done&

for env in ${locomotion_2[*]}
do
    python learn_distance_random.py \
    --env $env \
    --seed $"2"
done&

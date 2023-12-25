#!/bin/bash

locomotion=(
    "halfcheetah-medium-expert-v2"
    "halfcheetah-medium-v2"
    "halfcheetah-medium-replay-v2"
    "hopper-medium-expert-v2"
    "hopper-medium-v2"
    "hopper-medium-replay-v2"
    "walker2d-medium-expert-v2")
    
locomotion_2=("walker2d-medium-v2"
    "walker2d-medium-replay-v2"
    "halfcheetah-random-v2"
    "halfcheetah-expert-v2"
    "hopper-random-v2"
    "hopper-expert-v2"
    "walker2d-expert-v2"
    "walker2d-random-v2")

locomotion_3=("bullet-halfcheetah-random-v0")

# for env in ${locomotion[*]}
# do
#     python learn_distance.py \
#     --env $env\
#     --seed $"0"
# done&

# for env in ${locomotion[*]}
# do
#     python learn_distance.py \
#     --env $env \
#     --seed $"1"
# done&

# for env in ${locomotion[*]}
# do
#     python learn_distance.py \
#     --env $env \
#     --seed $"2"
# done&

# for env in ${locomotion_2[*]}
# do
#     python learn_distance.py \
#     --env $env\
#     --seed $"0"
# done&

# for env in ${locomotion_2[*]}
# do
#     python learn_distance.py \
#     --env $env \
#     --seed $"1"
# done&

# for env in ${locomotion_2[*]}
# do
#     python learn_distance.py \
#     --env $env \
#     --seed $"2"
# done&

for env in ${locomotion_3[*]}
do
    python learn_distance.py \
    --env $env\
    --seed $"0"
done&

for env in ${locomotion_3[*]}
do
    python learn_distance.py \
    --env $env \
    --seed $"1"
done&

for env in ${locomotion_3[*]}
do
    python learn_distance.py \
    --env $env \
    --seed $"2"
done&
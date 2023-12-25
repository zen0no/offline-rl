#!/bin/bash

    
antmaze_1=(
    "antmaze-umaze-v0"
    "antmaze-umaze-diverse-v0"
    "antmaze-medium-play-v0"
    "antmaze-medium-diverse-v0"
    "antmaze-large-play-v0"
    "antmaze-large-diverse-v0"
    )
    
# antmaze_2=(
#     "antmaze-medium-play-v0"
#     "antmaze-medium-diverse-v0"
#     )
    
# antmaze_3=(
#     "antmaze-large-play-v0"
#     "antmaze-large-diverse-v0"
#     )
    

for env in ${antmaze_1[*]}
do
    python learn_distance_antmaze.py \
    --env $env\
    --seed $"0"
done&

for env in ${antmaze_1[*]}
do
    python learn_distance_antmaze.py \
    --env $env\
    --seed $"1"
done&

for env in ${antmaze_1[*]}
do
    python learn_distance_antmaze.py \
    --env $env\
    --seed $"2"
done&



# for env in ${antmaze_2[*]}
# do
#     python learn_distance_antmaze.py \
#     --env $env\
#     --seed $"0"
# done&


# for env in ${antmaze_2[*]}
# do
#     python learn_distance_antmaze.py \
#     --env $env\
#     --seed $"1"
# done&

# for env in ${antmaze_2[*]}
# do
#     python learn_distance_antmaze.py \
#     --env $env\
#     --seed $"2"
# done&



# for env in ${antmaze_3[*]}
# do
#     python learn_distance_antmaze.py \
#     --env $env\
#     --seed $"0"
# done&

# for env in ${antmaze_3[*]}
# do
#     python learn_distance_antmaze.py \
#     --env $env\
#     --seed $"1"
# done&

# for env in ${antmaze_3[*]}
# do
#     python learn_distance_antmaze.py \
#     --env $env\
#     --seed $"2"
# done&
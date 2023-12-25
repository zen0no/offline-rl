#!/bin/bash

locomotion_1=(
    "halfcheetah-medium-expert-v2"
    )
    
locomotion_2=(
    "halfcheetah-medium-replay-v2"
    )
    
locomotion_3=(
    "hopper-medium-v2"
    )
    
locomotion_4=(
    "walker2d-medium-expert-v2"
    )
    
locomotion_5=(
    "walker2d-medium-replay-v2"
    )
    
locomotion_6=(
    "halfcheetah-expert-v2"
    )
    
locomotion_7=(
    "hopper-expert-v2"
    )

locomotion_8=(
    "halfcheetah-medium-v2"
    )
    
locomotion_9=(
    "hopper-medium-expert-v2"
    )
    
locomotion_10=(
    "hopper-medium-replay-v2"
    )
    
locomotion_11=(
    "walker2d-medium-v2"
    )
    
locomotion_12=(
    "walker2d-expert-v2"
    )
    
locomotion_13=(
    "hopper-random-v2"
    )
    
locomotion_14=(
    "halfcheetah-random-v2"
    )
    
locomotion_15=(
    "walker2d-random-v2"
    )
    
locomotion_16=(
    "bullet-halfcheetah-random-v0"
    )
    

# for env in ${locomotion_1[*]}
# do
#     python learn_critic.py \
#     --env $env\
#     --seed $"0"
# done&

# for env in ${locomotion_2[*]}
# do
#     python learn_critic.py \
#     --env $env\
#     --seed $"0"
# done&

# for env in ${locomotion_3[*]}
# do
#     python learn_critic.py \
#     --env $env\
#     --seed $"0"
# done&

# for env in ${locomotion_4[*]}
# do
#     python learn_critic.py \
#     --env $env\
#     --seed $"0"
# done&

# for env in ${locomotion_5[*]}
# do
#     python learn_critic.py \
#     --env $env\
#     --seed $"0"
# done&

# for env in ${locomotion_6[*]}
# do
#     python learn_critic.py \
#     --env $env\
#     --seed $"0"
# done&

# for env in ${locomotion_7[*]}
# do
#     python learn_critic.py \
#     --env $env\
#     --seed $"0"
# done&

# for env in ${locomotion_8[*]}
# do
#     python learn_critic.py \
#     --env $env\
#     --seed $"0"
# done&

# for env in ${locomotion_9[*]}
# do
#     python learn_critic.py \
#     --env $env\
#     --seed $"0"
# done&

# for env in ${locomotion_10[*]}
# do
#     python learn_critic.py \
#     --env $env\
#     --seed $"0"
# done&

# for env in ${locomotion_11[*]}
# do
#     python learn_critic.py \
#     --env $env\
#     --seed $"0"
# done&

# for env in ${locomotion_12[*]}
# do
#     python learn_critic.py \
#     --env $env\
#     --seed $"0"
# done&

# for env in ${locomotion_13[*]}
# do
#     python learn_critic.py \
#     --env $env\
#     --seed $"0"
# done&

# for env in ${locomotion_14[*]}
# do
#     python learn_critic.py \
#     --env $env\
#     --seed $"0"
# done&

# for env in ${locomotion_15[*]}
# do
#     python learn_critic.py \
#     --env $env\
#     --seed $"0"
# done&

for env in ${locomotion_16[*]}
do
    python learn_critic.py \
    --env $env\
    --seed $"0"
done&
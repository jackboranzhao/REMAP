#!/bin/bash
pwd_full_path=$(dirname $(readlink -f "$0"))
parent_pwd_full_path=`dirname "${pwd_full_path}"`
parent_fold_name=`basename "${pwd_full_path}"`
#echo ${pwd_full_path}
#echo ${parent_pwd_full_path}
#echo ${parent_fold_name}
cd ${pwd_full_path} 
source /home/zbr/anaconda3/etc/profile.d/conda.sh
conda activate rl_ven
server='mcts' # 252
#server='other'  # cxg
if [ $# = 2 ] && [ "$1" = "gen" -o "$1" = "run" -o "$1" = "rm" ] && [ $2 = "power" -o $2 = "area" -o $2 = "all" ]
then
    if [ "$2" = "power" ]
    then
        ConsName=('power')
    elif [ "$2" = "area" ]
    then
        ConsName=('area')
    elif [ "$2" = "all" ]
    then
        ConsName=('power' 'area')
    fi

    # gen power
    #for opt in rtm egy
    for opt in rtm
    do
        for con in ${ConsName[@]}
        do
            #for re in ult cld iot iox
            for re in cld
            do
            (cd ${pwd_full_path} && ./muti_run.sh $1 ${server} ${opt} ${con} ${re} &)
            done
        done
    done

else
    echo "first parameter should be 'gen' , 'run', 'rm'"
    echo "second parameter should be 'power' , 'area'"
fi

#!/bin/bash
pwd_full_path=$(dirname $(readlink -f "$0"))
parent_fold_name=`basename "${pwd_full_path}"`
parent_pwd_full_path=`dirname "${pwd_full_path}"`

#echo ${pwd_full_path}
#echo ${parent_fold_name}

cd ${parent_pwd_full_path} 
source /home/zbr/anaconda3/etc/profile.d/conda.sh
conda activate rl_ven

NetId="self.NetId                    ="
ConsrainDisPower="self.contr_en_power   = False"
ConsrainDisArea="self.contr_en_area    = False"
ConsrainName="self.contr_nm                    ="
NetStart=0
NetNum=6
inc=1
if [ $# = 5 ] && [ "$1" = "gen" -o "$1" = "run" -o "$1" = "rm" ] && [ "$2" = "mcts" -o "$2" = "other" ] && [ "$3" = "rtm" -o "$3" = "egy" ] && [ "$4" = "power" -o "$4" = "area" ] && [ "$5" = "ult" -o "$5" = "cld" -o "$5" = "iot" -o "$5" = "iox" ]
then
    # $2 test
    if [ "$2" = "mcts" ] 
    then
        MctsEn="self.MODE_MCTS_EN            = True"
    elif [ "$2" = "other" ] 
    then
        MctsEn="self.MODE_MCTS_EN            = False"
    fi

    # $3 test
    RunFileName="$2_$3_$4_$5"
    if [ "$3" = "rtm" ] 
    then
        OptRuntimeEN="self.opt_object_runtime_en    = True"
    elif [ "$3" = "egy" ] 
    then
        OptRuntimeEN="self.opt_object_runtime_en    = False"
    fi

    # $4 test
    if [ "$4" = "power" ] 
    then
        ConsrainEn="self.contr_en_power   = True"
    elif [ "$4" = "area" ] 
    then
        ConsrainEn="self.contr_en_area   = True"
    fi

    # $1 test
    if [ "$1" = "gen" ] 
    then
        echo "OK gen eygx"
        for ((i=${NetStart}; i<${NetNum}; i=`expr $i + ${inc}`))
        do
            cp -r ${parent_pwd_full_path}/mcts_maestro_only ${parent_pwd_full_path}/${RunFileName}${i}
            rm -fr ${parent_pwd_full_path}/${RunFileName}${i}/results/*
            sed -i "20i \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ ${ConsrainEn}"   ${RunFileName}${i}/var.py
            sed -i "20i \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ ${ConsrainDisPower}"   ${RunFileName}${i}/var.py
            sed -i "20i \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ ${ConsrainDisArea}"   ${RunFileName}${i}/var.py
            sed -i "20i \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ ${ConsrainName}'$5'"   ${RunFileName}${i}/var.py
            sed -i "20i \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ ${MctsEn}"   ${RunFileName}${i}/var.py
            sed -i "20i \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ ${NetId}${i}"   ${RunFileName}${i}/var.py
            sed -i "20i \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ ${OptRuntimeEN}"   ${RunFileName}${i}/var.py
        done
    elif [ "$1" = "run" ] 
    then
        echo "OK run ${RunFileName}x/Env_Maestro_mcts.py"

        for ((i=${NetStart}; i<${NetNum}; i=`expr $i + ${inc}`))
        do
            (cd ${parent_pwd_full_path}/${RunFileName}${i} && python3 Env_Maestro_mcts.py)
        done

    elif [ "$1" = "rm" ] 
    then
        echo "OK rm ${RunFileName}x"
        for ((i=${NetStart}; i<${NetNum}; i=`expr $i + ${inc}`))
        do
            (cd ${parent_pwd_full_path} && rm -fr ${RunFileName}${i})
        done
    fi

else
    echo "first parameter should be one of 'gen' , 'run', 'rm'"
    echo "second parameter should be one of 'mcts', 'other' "
    echo "third parameter should be one of 'rtm', 'egy' "
    echo "fourth parameter should be one of 'power', 'area' "
    echo "fifth parameter should be one of 'ult', 'cld', 'iot', 'iox' "
fi


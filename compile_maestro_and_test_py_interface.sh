#!/bin/bash
pwd
#source ../rl_ven/bin/activate
rm -fr *.o *.so
/usr/bin/g++-7 -fPIC -I/usr/include/python3.6m  -c -o maestro-top.o -c -std=c++17  -I. -Ilib/include -Ilib/include/base -Ilib/include/tools -Ilib/include/user-api -Ilib/include/dataflow-analysis -Ilib/include/dataflow-specification-language -Ilib/include/design-space-exploration -Ilib/include/cost-analysis -Ilib/include/abstract-hardware-model -Ilib/src maestro-top.cpp -lboost_program_options -lboost_filesystem -lboost_system -g
/usr/bin/g++-7 -L /lib64 -shared maestro-top.o  -lpython3.6m -lboost_python3 -o maestro.so -lboost_program_options -lboost_filesystem -lboost_system -g
python3 hello.py
#python3 Env_Maestro_mcts.py


import numpy as np
from jinja2 import Environment, FileSystemLoader
import sys

import pandas as pd
import matplotlib.pyplot as plt
import os
import var
import random
import math
import hashlib
import logging
import argparse
import nevergrad as ng
import sys
import time
import maestro
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
import sys

import pandas as pd
import matplotlib.pyplot as plt
import os
import var
import random
import math
import hashlib
import logging
import argparse
import nevergrad as ng
import sys
import time
import maestro
from tqdm import tqdm
#NetFileNameAll = ['test_vgg16.csv', 'test_unet.csv', 'test_shufflenet.csv',  'test_resnet50.csv',  'test_efficientnetb0.csv', 'test_anet.csv', 'test_bnet.csv']
NetFileNameAll = ['test_vgg16.csv', 'test_unet.csv', 'test_shufflenet.csv',  'test_resnet50.csv',  'test_efficientnetb0.csv', 'test_mobilenetv2.csv']
other_algrithm =  ["RandomSearch", "ScrHammersleySearch", "TwoPointsDE",  "CMA", "PSO"]
#NetFileName = ['test_anet.csv', 'test_efficientnetb0.csv', 'test_resnet50.csv', 'test_shufflenet.csv', 'test_unet.csv', 'test_vgg16.csv']
#NetFileName = ['test_vgg16.csv', 'test_efficientnetb0.csv']
#NetFileName = ['test_anet.csv', 'test_bnet.csv']
#NetFileName = ['test_vgg16.csv', 'test_unet.csv', 'test_shufflenet.csv',  'test_resnet50.csv',  'test_efficientnetb0.csv']
#NetFileName = ['test_vgg16.csv', 'test_unet.csv', 'test_shufflenet.csv',  'test_resnet50.csv']
varInst = var.Var(NetFileNameAll[0])
NetFileName = [NetFileNameAll[varInst.NetId]]

#calc other algrithm process time 
#  first gen a 5x5 matrix from OtherAlgPrTimeFactor, copy 5 row:     [1,5] to [5,5]
#  second gen a 5x5 matrix from OtherAlgPrTimeFactor, copy 5 colums: [5,1] to [5,5]
#  dot mult two matrix, [5,5] .* [5,5] = [5,5] 
NetNum = len(NetFileNameAll)
AlgNum = len(other_algrithm)
OtherAlgPrTimeFactor        = [1.2,    1.8,                    1.4,           2.0,   1.6 ]
if varInst.opt_object_runtime_en == True:
    MctsNetPrTime           = (np.genfromtxt('mcts_rtm_prTime.csv', delimiter=',')[:,0]).reshape([NetNum, 1])
else:
    MctsNetPrTime           = (np.genfromtxt('mcts_egy_prTime.csv', delimiter=',')[:,0]).reshape([NetNum, 1])
np.random.seed(124)
net_data_ptime = np.array(MctsNetPrTime).reshape([NetNum, 1])
alg_data_ptime = np.array(OtherAlgPrTimeFactor).reshape([1, AlgNum])
net_data_ptime_extd = np.tile(net_data_ptime, (1, AlgNum))
alg_data_ptime_extd = np.tile(alg_data_ptime, (NetNum, 1))
OtherAlgrithmProcessTime = alg_data_ptime_extd*net_data_ptime_extd


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger') # recored the logger

FLOAT_S_RETURN  = False
DEBUG_PRINT     =  False
DEBUG_PLOT_FIG  =  False
DEBUG_SAVE_CSV  =  True
DEBUG_SAVE_MAP  =  True
DEBUG_SAVE_NPY  =  False

DIM_NAME_LIST   = [ 'K',     'C',    'R',    'S',    'Y',     'X'     ]
DIM_NAME_LIST_ALL = DIM_NAME_LIST * varInst.CLUSTR_LVL_MEM_MAX 

all_lvl_dim_len = len(DIM_NAME_LIST_ALL)

if varInst.MODE_MCTS_EN == True:
    P_V = -0.005
else:
    P_V = sys.maxsize 
PENALTY_DIM_OUT_OF_ORDER  = P_V
PENALTY_VALUE_OUT_RANG    = P_V
PENALTY_TOO_MANY_OR_NO_SPATIAL_MAP = P_V
PENALTY_NO_SPATIAL_MAP = P_V
PENALTY_OUT_OF_RANGE = P_V
PENALTY_MAESTRO_ERROR = P_V
#ACT_TYPE_INDEX = 0
#ACT_SIZE_PERCENT_INDEX = 1
#ACT_OFFSET_PERCENT = 2
#ACT_DIM_INDEX = 3


base_path               = os.getcwd()#'//opt//data//private//work//rl_code//Reinforcement-learning-with-tensorflow//contents//rl_maestro'
template_path           = '//template//dataflow_t.m'
template_other_path     = '//template//dataflow_t_t.m'
dataflow_output_path    = '//data//DFSL_description//test.m' 
cpp_code_run_sh_path    = '//run_example.sh'
output_csr_path         = '//test1.csv'

# need to revised
global constr_name 
global contr_power 
global contr_area
constr_name = ['ult', 'cld', 'iot', 'iox']
contr_power = {'ult':int(varInst.constrain_max_power), 'cld':int(varInst.constrain_max_power*0.5), 'iot':int(varInst.constrain_max_power*0.1), 'iox':int(varInst.constrain_max_power*0.05)} 
contr_area  = {'ult':int(varInst.constrain_max_area),  'cld':int(varInst.constrain_max_area*0.5),  'iot':int(varInst.constrain_max_area*0.1), 'iox':int(varInst.constrain_max_area*0.05)} 

global glb_state
global glb_cnt
global glb_no_valid_times_cnt
global glb_no_valid_times_list

global net_power_mcts 
global net_area_mcts  
global net_power_other 
global net_area_other 

global glb_num_pe_used
global glb_reuse_ifactor
global glb_reuse_ofactor
global glb_reuse_wfactor
global glb_analysis_single_data_tmp_list 


if varInst.opt_object_runtime_en == True: # using throughput as the reward, about [0,100]
    W_throughput_mcts    = 1 
    W_enegy_mcts         = 0 
    W_time_mcts          = 0

    W_time_other_methd   = 1
    W_enegy_other_methd  = 0
else:
    W_throughput_mcts    = 0 
    W_enegy_mcts         = 1 
    W_time_mcts          = 0

    W_time_other_methd   = 0
    W_enegy_other_methd  = 1


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   
        os.makedirs(path)            

def wFile(wStr, fileName):
    with open(fileName, "a") as file_p:
        file_p.write(wStr+'\n')

fileNameYrMthDay    = time.strftime("%Y_%m_%d", time.localtime())
if varInst.MODE_MCTS_EN == True:
    fileName = time.strftime("mcts_%Y_%m_%d___%H_%M_%S", time.localtime()) 
else:
    fileName = time.strftime("other_%Y_%m_%d___%H_%M_%S", time.localtime())
dir_name            = "results//" + fileNameYrMthDay + '//' + fileName
fileNameFull        = dir_name +'//'+ fileName + ".txt"
mkdir(dir_name)

dir_name_fig        = dir_name + '//' + fileName + '_fig'
dir_name_csv        = dir_name + '//' + fileName + '_csv'
dir_name_map        = dir_name + '//' + fileName + '_map'
dir_name_npy        = dir_name + '//' + fileName + '_npy'

mkdir(dir_name_fig)
mkdir(dir_name_csv)
mkdir(dir_name_map)
mkdir(dir_name_npy)

def cur_time():
    c_time = time.strftime("%Y_%m_%d___%H_%M_%S", time.localtime()) + "\n"
    return c_time

def cpTestDataFlowFileExp():
    t = time.time()
    fileName_t = (int(round(t * 1000000))) 
    os.system("cp ./data/DFSL_description/test.m ./data/DFSL_description/tmp/"+'exp_'+str(fileName_t)+'.txt')

def cpTestDataFlowFile_float_false_pos():
    t = time.time()
    fileName_t = (int(round(t * 1000000))) 
    os.system("cp ./data/DFSL_description/test.m ./data/DFSL_description/tmp/"+'fltFalse_pos_'+str(fileName_t)+'.txt')

def cpTestDataFlowFile_float_false_neg():
    t = time.time()
    fileName_t = (int(round(t * 1000000))) 
    os.system("cp ./data/DFSL_description/test.m ./data/DFSL_description/tmp/"+'fltFalse_neg_'+str(fileName_t)+'.txt')

def self_permute(nums):
    if varInst.restrict_dim_order_cross_layer == False:
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        def backtrack(position, end):
            """
            Find possible results using backtrack.
            :param position:
            :param end:
            :return:
            """

            if position == end:
                res.append(nums[:])
                return

            for index in range(position, end):
                nums[index], nums[position] = nums[position], nums[index]
                backtrack(position + 1, end)
                nums[index], nums[position] = nums[position], nums[index]

        res = []
        backtrack(0, len(nums))
        return res
    else:
        res= []
        res.append(['X', 'Y', 'K', 'R', 'S', 'C'])
        res.append(['X', 'Y', 'K', 'R', 'C', 'S'])
        res.append(['X', 'Y', 'K', 'S', 'R', 'C'])
        res.append(['X', 'Y', 'K', 'S', 'C', 'R'])
        res.append(['X', 'Y', 'K', 'C', 'R', 'S'])
        res.append(['X', 'Y', 'K', 'C', 'S', 'R'])

        res.append(['Y', 'X', 'K', 'R', 'S', 'C'])
        res.append(['Y', 'X', 'K', 'R', 'C', 'S'])
        res.append(['Y', 'X', 'K', 'S', 'R', 'C'])
        res.append(['Y', 'X', 'K', 'S', 'C', 'R'])
        res.append(['Y', 'X', 'K', 'C', 'R', 'S'])
        res.append(['Y', 'X', 'K', 'C', 'S', 'R'])
        return res


global mcts_init_done_flag
global mcts_train_rtm


global fig_name
global fig_type
if varInst.opt_object_runtime_en == True:
    #fig_name = ['whole_rtm', 'layer_rtm', 'layer_explore_cnt', 'glb_layer_rtm']
    fig_name = ['whole_rtm', 'glb_layer_rtm', 'no_valid_cnt', 'no',  'num_pe_used',  'reuse_ifactor', 'reuse_ofactor', 'reuse_wfactor']
else:
    #fig_name = ['whole_egy', 'layer_egy', 'layer_explore_cnt', 'glb_layer_egy']
    fig_name = ['whole_egy', 'glb_layer_egy', 'no_valid_cnt',  'no',  'num_pe_used',  'reuse_ifactor', 'reuse_ofactor', 'reuse_wfactor']

fig_type = ['line',      'bar',       'bar',     'bar',         #followed new add: 4 5 6 7
             'bar', 'bar','bar','bar',]
def plt_init():
    plt.ion() 
def plt_plot(data, fig):
    global fig_name
    global fig_type
    data        = np.array(data) 
    data_shape  = np.shape(data)
    data_ndim   = np.ndim(data)
    #fig_index = fig_name.index(fig) 
    fig_index = fig 
    fig_t  = fig_type[fig_index] 
    if fig_t == 'line' and data_ndim == 1:
        plt.figure(fig_index)
        plt.clf() 
        #plt.plot(data, c='r', label='MCTS')
        plt.plot(data, c='r', label=varInst.contr_nm)
        plt.legend(loc='best')
        plt.ylabel(fig_name[fig_index])
        plt.yscale('log')
        plt.xlabel('Episode')
        plt.draw()
        plt.pause(0.0000000000001)
    elif fig_t == 'bar' and data_ndim == 1:
        plt.figure(fig_index)
        plt.clf() 
        x = np.linspace(0, data_shape[0]-1, data_shape[0])
        plt.bar(x, data, label='MCTS')
        plt.legend(loc='best')
        plt.ylabel(fig_name[fig_index])
        #plt.yscale('log')
        plt.xlabel('Episode')
        plt.draw()
        plt.pause(0.0000000000001)
    elif fig_t == 'bar' and data_ndim == 2:
        plt.figure(fig_index)
        plt.clf() 
        x = np.linspace(0, data_shape[0]-1, data_shape[0])
        plt.bar(x, data[:,0], label='MCTS')
        plt.legend(loc='best')
        plt.ylabel(fig_name[fig_index])
        #plt.yscale('log')
        plt.xlabel('Episode')
        plt.draw()
        plt.pause(0.0000000000001)
    else:
        print('plot argument error:')
        xx
def plt_save_fig(fig_index, fig_name):
    plt.figure(fig_index)
    plt.savefig(fig_name)

def save_all_fig(fig_num, alg_str='',):
    global fig_name
    global fileName
    for i in range(fig_num):
        plt_save_fig(i, dir_name_fig+"//"+fileName+"_"+net[5:8]+"_"+alg_str+"_"+fig_name[i]+".jpg")

def save_np(data, alg_str='', np_str=''):
    np_data = np.array(data)
    np.savetxt( dir_name_csv +"//"+fileName+"_"+net[5:8]+"_"+alg_str+"_"+np_str+".csv", np_data)

def save_all_npy():
    for i in range(varInst.LY_LEN): # for each layer gather all rows data, and save it
        len_one_layer_all_data = len(glb_analysis_single_data_tmp_list[i])
        analysis_layer_data = np.ones((len_one_layer_all_data, varInst.analysis_data_num_clm))
        for j in range(len_one_layer_all_data):
            analysis_layer_data[j,:] = glb_analysis_single_data_tmp_list[i][j]# for i layer gather all the data to a numpy data and save it
        np.save(dir_name_npy +"//"+fileName+"_"+net[5:8]+ '_ly' + str("%03d"%i) + ".npy", analysis_layer_data)

def save_npy(ly_idx, data):
    if np.ndim(data) != 1 or len(data) != varInst.analysis_data_num_clm:
        raise Exception('npy data len is not :'+ str(varInst.analysis_data_num_clm))
    glb_analysis_single_data_tmp_list[ly_idx].append(np.array(data))



self_env = Environment(loader=FileSystemLoader(base_path, encoding='utf8'))
self_dim_perm = self_permute(DIM_NAME_LIST)

#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration.  ??
SCALAR=1/math.sqrt(2.0)
global net_min_runtime 
global net_min_energy 
global net_min_energy_cly
global net_min_l2_regy
global net_min_l2_wegy
global net_min_l1_regy
global net_min_l1_wegy
global net_min_mac_egy
global net_min_ddr_egy
net_min_runtime = sys.maxsize
net_min_energy  = sys.maxsize
net_min_energy_cly = sys.maxsize
net_min_l2_regy = sys.maxsize
net_min_l2_wegy = sys.maxsize
net_min_l1_regy = sys.maxsize
net_min_l1_wegy = sys.maxsize
net_min_mac_egy = sys.maxsize
net_min_ddr_egy = sys.maxsize

global other_each_layer_pe_num
global other_each_layer_buf_num
global other_each_layer_rtm
global other_each_layer_egy
global other_each_layer_tpt
global other_each_layer_area
global other_each_layer_power
global other_algrithm_init_done
global other_algrithm_init_cnt
other_algrithm_init_cnt = 0
rtm_egy_init_v = int(sys.maxsize/100000)
class Env_Maestro:
    NUM_TURNS = 3                              # round max,  moves*10, play 10 times 
    num_moves = varInst.A_NUM
    def __init__( 
        self,
        s = np.zeros([varInst.S_LEN,],dtype=np.int), 
        moves = [],
        turn  = NUM_TURNS,
        rw   = .0,
    ):
        self.s = s 
        self.moves = moves
        self.turn=turn                      # number of turn, max is NUM_TURNS
        self.rw  = rw
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference

    def get_1d_max_index_softmax_prob(self, data_1d, idx=[], get_num=1):
        '''input dim value'''
        data_1d = np.array(data_1d).copy()
        d_l=np.size(data_1d)
        d_t = data_1d.reshape(d_l,)
        if len(idx) == 0:
            d_t_idx = np.linspace(0,d_l-1, d_l, dtype=np.int)
        else:
            assert np.size(data_1d) == np.size(idx)
            idx = np.array(idx)
            idx_l = np.size(idx)
            idx_t = idx.reshape(idx_l,)
            d_t_idx = idx_t
        prob = self.softmax(d_t)
        d_max_idx = np.random.choice(d_t_idx, get_num, p=prob)
        return d_max_idx

    def get_1d_min_index_softmax_prob(self, data_1d, idx=[], get_num=1):
        '''input dim value'''
        data_1d = np.array(data_1d).copy()
        d_l=np.size(data_1d)
        d_t = data_1d.reshape(d_l,)
        d_t_max_1 = np.max(d_t) + 1
        d_t_reverse = d_t_max_1 - d_t
        if len(idx) == 0:
            d_t_idx = np.linspace(0,d_l-1, d_l, dtype=np.int)
        else:
            assert np.size(data_1d) == np.size(idx)
            idx = np.array(idx)
            idx_l = np.size(idx)
            idx_t = idx.reshape(idx_l,)
            d_t_idx = idx_t
        prob = self.softmax(d_t_reverse)
        #print(prob, d_t_reverse, d_t_idx)
        d_min_idx = np.random.choice(d_t_idx, get_num, p=prob)
        return d_min_idx
    
    def is_large_equal_than_top_max_n(self, test_data, all_data, n=2):
        '''input dim value'''
        assert np.size(all_data) >= n
        data_t = all_data.reshape(np.size(all_data),).copy()
        data_t_top_max_idx_array = data_t.argsort()[-n:]
        #print(data_t_top_max_idx_array)
        #print(data_t[data_t_top_max_idx_array])
        data_t_top_max = np.min(data_t[data_t_top_max_idx_array])
        if test_data >= data_t_top_max:
            return True
        else:
            return False
    def is_less_equal_than_top_min_n(self, test_data, all_data, n=2):
        '''input dim value'''
        assert np.size(all_data) >= n
        data_t = all_data.reshape(np.size(all_data),).copy()
        data_t_top_min_idx_array = data_t.argsort()[0:n]
        #print(all_data)
        #print(data_t_top_min_idx_array)
        #print(data_t[data_t_top_min_idx_array])
        data_t_top_min = np.max(data_t[data_t_top_min_idx_array])
        if test_data <= data_t_top_min:
            return True
        else:
            return False

    def type_get_1d_equ_idx(self, test_data, all_data):
        '''input dim type'''
        data = np.array(all_data).copy()
        data = data.reshape(np.size(data),)
        data_equ_idx = np.where(data == test_data)[0]
        return data_equ_idx 

    def next_state(self, decay_turn, is_init=False):
        global mcts_init_done_flag
        #act_t_type      = np.random.randint( 0, varInst.act_type_len,       [varInst.act_type_exist,        ], np.int).tolist() 
        #act_t_dim       = np.random.randint( 0, varInst.act_dim_len,        [varInst.act_dim_exist,         ], np.int).tolist() 
        act_t_dim_order = np.random.randint( 0, varInst.act_dim_order_len,  [varInst.act_dim_order_exist,   ], np.int).tolist() 
        act_t_dim_top   = np.random.randint( 0, varInst.act_dim_top_len,    [varInst.act_dim_top_exist,     ], np.int).tolist() 
        #act_t_clt_lvl   = np.random.randint( 0, varInst.act_clt_lvl_len,    [varInst.act_clt_lvl_exist,     ], np.int).tolist() 
        act_t_clt_v     = np.random.randint( 0, varInst.act_clt_v_len,      [varInst.act_clt_v_exist,       ], np.int).tolist() 

        act_t_pe_num    = np.random.randint( 0, varInst.act_pe_num_len,     [varInst.act_pe_num_exist,      ], np.int).tolist() 
        act_t_buf_num   = np.random.randint( 0, varInst.act_buf_num_len,    [varInst.act_buf_num_exist,     ], np.int).tolist() 
        #act_t_layer     = np.random.randint( 0, varInst.act_layer_len,      [varInst.act_layer_exist,       ], np.int).tolist() 

        s_res_rtm       = glb_state[varInst.s_ly_idx_index    : varInst.s_res_rtm_index  ].reshape([varInst.LY_LEN, 1]).astype(np.int)
        s_res_egy       = glb_state[varInst.s_res_rtm_index   : varInst.s_res_egy_index     ].reshape([varInst.LY_LEN, 1]).astype(np.int)
        s_ly_epl_cnt    = glb_state[varInst.s_res_tpt_index   : varInst.s_ly_epl_cnt_index  ].reshape([varInst.LY_LEN, 1])
        if is_init == False:
            s_res_rtm_v = s_res_rtm.copy()
            s_res_egy_v = s_res_egy.copy()
            # calc a ratio of expoler each layer, the max times of expoler each layer = ratio * avg(except_max_rtm_or_egy)times
            s_res_rtm_t = s_res_rtm.copy()
            s_res_egy_t = s_res_egy.copy()

            s_res_rtm_t_except_max_idx = np.where(np.max(s_res_rtm_t) != s_res_rtm_t)
            s_res_rtm_t_except_max_avg = np.average(s_res_rtm_t[s_res_rtm_t_except_max_idx])
            s_res_rtm_epl_ratio        = np.max(s_res_rtm_t) / s_res_rtm_t_except_max_avg

            s_res_egy_t_except_max_idx = np.where(np.max(s_res_egy_t) != s_res_egy_t)
            s_res_egy_t_except_max_avg = np.average(s_res_egy_t[s_res_egy_t_except_max_idx])
            s_res_egy_epl_ratio        = np.max(s_res_egy_t) / s_res_egy_t_except_max_avg

            epl_ratio = s_res_rtm_epl_ratio if varInst.opt_object_runtime_en == True else s_res_egy_epl_ratio
            # get th threshold
            threshold = np.average(s_ly_epl_cnt) * epl_ratio
            # if some layer have exceed the threshold set it's rtm/egy to 1, make it not be selected
            max_epl_ly_idx = np.where(s_ly_epl_cnt > threshold)
            s_res_rtm_v[max_epl_ly_idx] = 1
            s_res_egy_v[max_epl_ly_idx] = 1

        #===================================select parameter poistion 1:layer 2: cluster level 3: dim===============================================#
        #1: set layer id random select 1 index from top N max index
        if varInst.opt_object_runtime_en == True:
                if rtm_egy_init_v in s_res_rtm:
                    pass
                else:
                    mcts_init_done_flag = True

                if is_init:
                    act_t_layer =  np.argmax(s_res_rtm, axis=0).tolist()
                else: 
                    act_t_layer = (self.get_1d_max_index_softmax_prob(s_res_rtm_v)).tolist()
                #s_res_rtm_oneD = s_res_rtm.reshape(np.size(s_res_rtm),)
                #s_res_rtm_oneD_top_max_idx = s_res_rtm_oneD.argsort()[-varInst.TOP_MAX_NUM:]
                #act_t_layer = np.random.choice(s_res_rtm_oneD_top_max_idx, 1).tolist()
        else:

            if rtm_egy_init_v in s_res_egy:
                pass
            else:
                mcts_init_done_flag = True
            if is_init:
                act_t_layer     =  np.argmax(s_res_egy, axis=0).tolist()
            else:
                act_t_layer = self.get_1d_max_index_softmax_prob(s_res_egy_v).tolist()
            #s_res_egy_oneD = s_res_egy.reshape(np.size(s_res_egy),)
            #s_res_egy_oneD_top_max_idx = s_res_egy_oneD.argsort()[-varInst.TOP_MAX_NUM:]
            #act_t_layer = np.random.choice(s_res_egy_oneD_top_max_idx, 1).tolist()


        
        #2-1:act_t_clt_lvl_gen set cluster level of gen, it is just for gen dataflow file test.m 
        act_t_clt_lvl_gen = np.random.randint( 0, varInst.act_clt_lvl_gen_len,    [varInst.act_clt_lvl_gen_exist, ], np.int).tolist() 
        # adjust it  if some dim is 1, then set level is one, to avoid error
        act_t_layer_idx = int(act_t_layer[0])
        dim_max = (varInst.DIM_MAX[act_t_layer_idx,:]).astype(np.int)
        if 1 in dim_max: # if some dim is 1 in 6 dims
            act_t_clt_lvl_gen = np.zeros([varInst.act_clt_lvl_gen_exist,],dtype=np.int).tolist()

        #2-2:act_t_clt_lvl select cluster level from 3, this value is to chanage state value, it should <= act_t_clt_lvl_gen
        act_t_clt_lvl   = np.random.randint( 0, act_t_clt_lvl_gen[0]+1,    [varInst.act_clt_lvl_exist,     ], np.int).tolist() 



        act_t_layer_v   = act_t_layer[0]
        act_t_clt_lvl_v = act_t_clt_lvl[0]
        s_type_all        = self.s[varInst.s_dim_order_index : varInst.s_type_index        ].reshape([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX*varInst.DIM_NUM]).astype(np.int)
        s_type_this_lvl   = s_type_all[act_t_layer_v, act_t_clt_lvl_v*varInst.DIM_NUM : (act_t_clt_lvl_v+1)*varInst.DIM_NUM].copy()
        s_type_this_lvl = s_type_this_lvl.reshape(np.size(s_type_this_lvl),)
        s_type_this_lvl_sum = np.sum(s_type_this_lvl)

        #3: select a dim of 6 dims, simutauniously set the dim type
        # stratgy:
        #3-1: if sp sum >=3, then set it as temporal, which is sp before
        #3-2: elif sp sum <=0, then set it as sp, which is tp before
        #3-3: elif sp sum ==1, set it as sp, which is tp and it's dim value is top max
        #3-3: elif sp sum ==2, assume the sp is X and Y
              #3-3-1: if X is top2 min, then set it as tp
              #3-3-2: elif Y is top2 min, then set it as tp
              #3-3-3: elif X is top2 max, then keep it tp again
              #3-3-3: else Y is top2 max, then keep it tp again
        dim_v = dim_max.copy()

        if varInst.IS_DSCONV[act_t_layer_v]:
            dim_v[0] = 1
        #code is:
        #3-1: if sp sum >=3, then set it as temporal, which is sp before
        if s_type_this_lvl_sum >=3:
            sp_dim_idx = self.type_get_1d_equ_idx(test_data=1, all_data=s_type_this_lvl)
            sp_dim = dim_v[sp_dim_idx]
            sp_dim_min_v_idx = self.get_1d_min_index_softmax_prob(data_1d=sp_dim, idx=sp_dim_idx)
            # get the min value index, which is sp, set it tp
            act_t_dim = sp_dim_min_v_idx.tolist()
            act_t_type = np.zeros([varInst.act_clt_lvl_gen_exist,],dtype=np.int).tolist()

        #3-2: elif sp sum <=0, then set it as sp, which is tp before
        elif s_type_this_lvl_sum <=0:
            tp_dim_max_v_idx = self.get_1d_max_index_softmax_prob(data_1d=dim_v)
            act_t_dim = tp_dim_max_v_idx.tolist()
            # get the max value index, which is tp, set it sp
            act_t_type = np.ones([varInst.act_clt_lvl_gen_exist,],dtype=np.int).tolist()

        #3-3: elif sp sum ==1, set it as sp, which is tp and it's dim value is top max
        elif s_type_this_lvl_sum ==1:
            tp_dim_idx = self.type_get_1d_equ_idx(test_data=0, all_data=s_type_this_lvl)
            tp_dim = dim_v[tp_dim_idx]
            tp_dim_max_v_idx = self.get_1d_max_index_softmax_prob(data_1d=tp_dim, idx=tp_dim_idx)
            act_t_dim = tp_dim_max_v_idx.tolist()
            act_t_type = np.ones([varInst.act_clt_lvl_gen_exist,],dtype=np.int).tolist()
        elif s_type_this_lvl_sum ==2:
            sp_dim_idx = self.type_get_1d_equ_idx(test_data=1, all_data=s_type_this_lvl)
            np.random.shuffle(sp_dim_idx)
            sp_dim = dim_v[sp_dim_idx]
            sp_dim_idx0 = sp_dim_idx[0]* np.ones((1,),dtype=np.int)
            sp_dim_idx1 = sp_dim_idx[1]* np.ones((1,),dtype=np.int)
            sp_dim0 = np.asscalar(sp_dim[0])
            sp_dim1 = np.asscalar(sp_dim[1])
            prob = np.random.randint(0, 100,(1,))
            dim_v_one_idx = self.type_get_1d_equ_idx(test_data=1, all_data=dim_v)
            dim_v_one_sum = np.size(dim_v_one_idx)
            if dim_v_one_sum >= 4:
                prob = 100
            #3-3-1: if X is top2 min, then set it as tp
            if self.is_less_equal_than_top_min_n(test_data=sp_dim0, all_data=dim_v, n=3): 
                act_t_dim = sp_dim_idx0.tolist()
                act_t_type = np.zeros([varInst.act_clt_lvl_gen_exist,],dtype=np.int).tolist()
            #3-3-2: if Y is top2 min, then set it as tp
            elif self.is_less_equal_than_top_min_n(test_data=sp_dim1, all_data=dim_v, n=3):
                act_t_dim = sp_dim_idx1.tolist()
                act_t_type = np.zeros([varInst.act_clt_lvl_gen_exist,],dtype=np.int).tolist()
            #3-3-3 if sp it at top2 25% adjust,
            elif self.is_large_equal_than_top_max_n(test_data=sp_dim0, all_data=dim_v) and (prob > varInst.PROB_TWO_SP_EXPOLER_TP):
                act_t_dim = sp_dim_idx0.tolist()
                act_t_type = np.ones([varInst.act_clt_lvl_gen_exist,],dtype=np.int).tolist()
            #3-3-4 if sp it at top2 25% adjust 
            elif self.is_large_equal_than_top_max_n(test_data=sp_dim1, all_data=dim_v) and (prob > varInst.PROB_TWO_SP_EXPOLER_TP):
                act_t_dim = sp_dim_idx1.tolist()
                act_t_type = np.ones([varInst.act_clt_lvl_gen_exist,],dtype=np.int).tolist()
            else:
                tp_dim_idx = self.type_get_1d_equ_idx(test_data=0, all_data=s_type_this_lvl)
                act_t_dim = np.random.choice(tp_dim_idx, 1).tolist()
                act_t_type = np.zeros([varInst.act_clt_lvl_gen_exist,],dtype=np.int).tolist()
        else:
            idx = np.linspace(0,varInst.DIM_NUM-1, varInst.DIM_NUM, dtype=np.int) # 1-5 jump 0
            act_t_dim = self.get_1d_max_index_softmax_prob(data_1d=dim_v, idx=idx)
            act_t_dim = act_t_dim.tolist()
            act_t_type = np.zeros([varInst.act_clt_lvl_gen_exist,],dtype=np.int).tolist()
        


        dim_name = varInst.DIM_NAME_LIST[int(act_t_dim[0])] 
        if dim_name == "K" and varInst.IS_DSCONV[act_t_layer_v]:
            idx = np.linspace(1,varInst.DIM_NUM-1, varInst.DIM_NUM-1, dtype=np.int) # 1-5 jump 0
            act_t_dim = self.get_1d_max_index_softmax_prob(data_1d=dim_v[1:], idx=idx)
            act_t_dim = act_t_dim.tolist()
            act_t_type = np.ones([varInst.act_clt_lvl_gen_exist,],dtype=np.int).tolist()



        ##3-1: select a dim of 6 dims, condidata one: it is max dim value
        #act_t_dim_v_max       = self.get_1d_max_index_softmax_prob(dim_max).tolist() 
        ## adjust it  if some dim is 1, DO NOT select this dim
        #act_typ_sze_oft_dim_index = int(act_t_dim_v_max[0] + act_t_clt_lvl[0] * varInst.act_dim_len)
        #while(act_dim_max == 1 ):
        #    #act_t_dim_v_max       = np.random.randint( 0, varInst.act_dim_len,        [varInst.act_dim_exist,         ], np.int).tolist() 
        #    act_t_dim_v_max       = self.get_1d_max_index_softmax_prob(dim_max).tolist() 
        #    act_typ_sze_oft_dim_index = int(act_t_dim_v_max[0] + act_t_clt_lvl[0] * varInst.act_dim_len)
        #    act_dim_max = varInst.s_oft_v_max[act_t_layer_idx, act_typ_sze_oft_dim_index]
        ##3-2: select a dim of 6 dims, condidata two: it is spatial previously
        #act_t_dim_sp       = self.get_1d_max_index_softmax_prob(s_type_this_lvl).tolist() 
        #act_t_dim   = np.random.choice(np.array(act_t_dim_v_max+act_t_dim_sp),1 ).tolist()
        #===================================select parameter poistion 1:layer 2: cluster level 3: dim===============================================#

        #===================================change parameter value 1:type 2:size 3:offset ===============================================#
        #1 set type value
        #act_t_dim_v     = act_t_dim[0]
        #s_type_this_lvl_this_value = s_type_this_lvl[act_t_dim_v]
        #if(s_type_this_lvl_sum >= 3):# if spatial sum large or equal to 3, then set this to 0: Temporal
        #    act_t_type = np.zeros([varInst.act_clt_lvl_gen_exist,],dtype=np.int).tolist()
        #elif(s_type_this_lvl_sum <= 1):# if spatial sum smaller or equal to 1, then set this to 1: Spatial
        #    act_t_type = np.ones([varInst.act_clt_lvl_gen_exist,],dtype=np.int).tolist()
        #elif(s_type_this_lvl_sum == 2):
        #    if(s_type_this_lvl_this_value == 1): # random set this value ,it can be: tp or sp
        #        act_t_type      = np.random.randint( 0, varInst.act_type_len,       [varInst.act_type_exist,        ], np.int).tolist() 
        #    elif(s_type_this_lvl_this_value == 0):# if sum==2 and this value is 0/temporal, it also be temporal, to avoid too much sp
        #        act_t_type = np.zeros([varInst.act_clt_lvl_gen_exist,],dtype=np.int).tolist()
        if is_init and varInst.MCTS_INIT_RANDOM: 
            tp_dim_idx = self.type_get_1d_equ_idx(test_data=0, all_data=s_type_this_lvl)
            tp_dim = dim_v[tp_dim_idx]
            tp_dim_max_v_idx = self.get_1d_max_index_softmax_prob(data_1d=tp_dim, idx=tp_dim_idx)
            act_t_dim = tp_dim_max_v_idx.tolist()
            act_t_type = np.zeros([varInst.act_clt_lvl_gen_exist,],dtype=np.int).tolist()
        act_dim_max = dim_max[int(act_t_dim[0])]
        #act_t_size      = np.random.randint( 1, int(2+act_dim_max/500),  [varInst.act_size_step_exist,   ], np.int).tolist()
        act_t_size      = np.random.randint( 1, 2,  [varInst.act_size_step_exist,   ], np.int).tolist()
        # revised value
        #=== X_size >= R_size===#
        #=== Y_size >= S_size===#
        #act_t_offset    = np.random.randint( int(9*act_dim_max/10), act_dim_max, [varInst.act_offset_step_exist, ], np.int).tolist() 
        if is_init and varInst.MCTS_INIT_RANDOM:
            act_t_offset    = np.random.randint( 1, act_dim_max+1, [varInst.act_offset_step_exist, ], np.int).tolist() 
        else:
            act_t_offset    = np.random.randint( int(max(act_dim_max-2,1)), act_dim_max+2, [varInst.act_offset_step_exist, ], np.int).tolist() 
        #===================================change parameter value 1:type 2:size 3:offset ===============================================#

        if varInst.en_global_record_talbe == False:
            act_t_layer     = np.random.randint( 0, varInst.act_layer_len,      [varInst.act_layer_exist,       ], np.int).tolist() 
        

        if varInst.en_heuristic_action_space == False:
            act_t_type      = np.random.randint( 0, varInst.act_type_len,       [varInst.act_type_exist,        ], np.int).tolist() 
            act_t_dim       = np.random.randint( 0, varInst.act_dim_len,        [varInst.act_dim_exist,         ], np.int).tolist() 
            act_t_offset    = np.random.randint( 1, act_dim_max+1, [varInst.act_offset_step_exist, ], np.int).tolist() 
            act_t_size      = np.random.randint( 1, 2,  [varInst.act_size_step_exist,   ], np.int).tolist()

            dim_name = varInst.DIM_NAME_LIST[int(act_t_dim[0])] 
            if dim_name == "K" and varInst.IS_DSCONV[act_t_layer_v]:
                act_t_dim       = np.random.randint( 0, varInst.act_dim_len,        [varInst.act_dim_exist,         ], np.int).tolist() 
                act_t_type = np.ones([varInst.act_clt_lvl_gen_exist,],dtype=np.int).tolist()


        action_t        = act_t_type +act_t_size +act_t_offset + act_t_dim + act_t_dim_order + act_t_dim_top + act_t_clt_lvl +  act_t_clt_v + act_t_pe_num + act_t_buf_num + act_t_layer + act_t_clt_lvl_gen
        action =  np.array(action_t) 


        tmp_s, tmp_rw, tmp_done = self.step(action)
        if decay_turn == True:
            turn_v = self.turn-1
        else:
            turn_v = self.turn
        next = Env_Maestro(s=tmp_s, moves=self.moves+action_t, turn=turn_v, rw=tmp_rw,)
        return next

    def terminal(self):
        if self.turn == 0 :
            return True
        else:
            return False 

    def reward(self,):
        return self.rw
    
    def __hash__(self):
        return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(),16) # us the moves = [-1, 0, ...] gen a hash value, to compare the state
    def __eq__(self,other): # compare if two state is same 
        if hash(self)==hash(other):
            return True
        return False
    def __repr__(self):
        s="Moves: %s"%(self.moves)
        return s

    def reset(
        self,
    ):
        """For reset environment to a initial state"""
        # reset state
        s_dim_order     = np.random.randint( 0, varInst.dim_order_num, [varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX], np.int)
        s_type          = np.array(varInst.DIRCT_TYPE_INT, dtype=np.int)
        s_size          = np.array(varInst.DIRCT_SIZE_INIT, dtype=np.int)  
        s_offset        = np.array(varInst.DIRCT_OFFSET_INIT, dtype=np.int)
        s_cluster_v     = np.random.randint( 1, 1025, [varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX-1], np.int)
        s_pe_num        = np.random.randint( 1, 1025, [varInst.LY_LEN, 1], np.int)
        s_buf_num       = np.random.randint( 16384, 262144, [varInst.LY_LEN, 1], np.int)
        s_ly_idx        = np.random.randint( 0, varInst.LY_LEN, [1, 1], np.int)
        s_res_rtm       = np.ones( [varInst.LY_LEN, 1], np.int) * rtm_egy_init_v# init value: -0.005  1/(-0.005) = -200
        s_res_egy       = s_res_rtm.copy()
        s_res_tpt       = np.ones( [varInst.LY_LEN, 1], np.int) * (P_V) # init value: -0.005 
        s_ly_epl_cnt    = np.zeros( [varInst.LY_LEN, 1], np.int)
        s_res_area      = s_res_rtm.copy()
        s_res_power     = s_res_rtm.copy()
        s_res_egy_ncl   = s_res_rtm.copy()
        s_l2_regy       = s_res_rtm.copy()
        s_l2_wegy       = s_res_rtm.copy()
        s_l1_regy       = s_res_rtm.copy()
        s_l1_wegy       = s_res_rtm.copy()
        s_mac_egy       = s_res_rtm.copy()
        s_ddr_egy       = s_res_rtm.copy()
        s_all_rst       = np.hstack(( s_dim_order.flatten(), s_type.flatten(), s_size.flatten(), 
                                s_offset.flatten(), s_cluster_v.flatten(),s_pe_num.flatten(),
                                s_buf_num.flatten(), s_ly_idx.flatten(), s_res_rtm.flatten(), 
                                s_res_egy.flatten(), s_res_tpt.flatten(), s_ly_epl_cnt.flatten(), 
                                s_res_area.flatten(), s_res_power.flatten(), s_res_egy_ncl.flatten(),
                                s_l2_regy.flatten(), s_l2_wegy.flatten(), s_l1_regy.flatten(),
                                s_l1_wegy.flatten(), s_mac_egy.flatten(), s_ddr_egy.flatten() )) 
        self.s = s_all_rst
        return s_all_rst
    def init_run_all_layer(self, run_num = varInst.LY_LEN): # 
        cnt = run_num
        global mcts_init_done_flag
        state = self.next_state(decay_turn=False, is_init=True)
        while(mcts_init_done_flag == False and cnt>0):
            state = state.next_state(decay_turn=False, is_init=True)
            if varInst.en_global_record_talbe==False or varInst.en_heuristic_action_space==False:
                cnt = cnt -1 
        return state

    def assert_mtx(self, arr,min,max):
        """assert the matrix all value in the min max range"""
        pos_min = arr>=min
        pos_max = arr<=max
        pos_rst = pos_min & pos_max
        pos     = np.where(pos_rst == False)[0].shape[0]==0 
        return pos

    def assert_each_row_vec_of_mtx(self, arr, dim_size_list, start_value):
        """assert the matrix each row in the -dim_size_list[i] and dim_size_list[i] range"""
        row_num = np.shape(arr)[0]
        for i in range(row_num):
            if(self.assert_mtx(arr[i,:], start_value, dim_size_list[i])==False ):
                if DEBUG_PRINT == True:
                    print("Size or Offset not in Dim range: " + DIM_NAME_LIST_ALL[i], arr[i,:])
                return False
        
        return True

    def list_change_pos(self, old_pos, new_pos, content):
        new_list  = []
        for i in range(len(content)):
            pos = old_pos.index(new_pos[i])
            new_list.append(content[pos])
        return new_list

    def array_change_pos(self, old_pos, new_pos, content):
        new_array  = np.array(content)
        cnt = 0
        for i in range(len(content)):
            pos = old_pos.index(new_pos[i])
            new_array[cnt] = content[pos]     
            cnt = cnt + 1
        return new_array

    def is_float(self, val):
        try:
            float(val)
        except ValueError:
            return False
        else:
            return True

    def is_nan(self, val):
        try:
            x = float(val)
        except ValueError:
            return True
        if(math.isnan(x)):
            return True
        else:
            return False

    def dim_order_to_dim_name_list(self, dim_order_mtx, dim_list_std=DIM_NAME_LIST,):
        new_order = np.argmax(dim_order_mtx, axis=0)
        rule = dict(zip(dim_list_std, new_order))
        dim_list_new = sorted(dim_list_std, key=lambda x:rule[x])
        return dim_list_new
        
    def step(self, action):
        """ a ->     s_, r,done, ->"""

        global mcts_init_done_flag 
        global net_power_mcts 
        global net_area_mcts 

        if mcts_init_done_flag == False:
            net_power_mcts = 0 # initial value
            net_area_mcts  = 0
        start = time.time()
        tmp_s = self.s 
        tmp_s_ = self.s            
        global glb_no_valid_times_cnt
        glb_no_valid_times_cnt = glb_no_valid_times_cnt + 1
        #split all the info frome state, 36 dimesion order matrix, 6 type, 6 size, 6 offset
        s_dim_order   = self.s[0                            : varInst.s_dim_order_index   ].reshape([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX]).astype(np.int)
        s_type        = self.s[varInst.s_dim_order_index    : varInst.s_type_index        ].reshape([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX*varInst.DIM_NUM]).astype(np.int)
        s_size        = self.s[varInst.s_type_index         : varInst.s_size_index        ].reshape([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX*varInst.DIM_NUM]).astype(np.int)
        s_offset      = self.s[varInst.s_size_index         : varInst.s_oft_index         ].reshape([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX*varInst.DIM_NUM]).astype(np.int)
        s_cluster_v   = self.s[varInst.s_oft_index          : varInst.s_ctl_v_index       ].reshape([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX-1]).astype(np.int)
        s_pe_num      = self.s[varInst.s_ctl_v_index        : varInst.s_pe_num_index      ].reshape([varInst.LY_LEN, 1]).astype(np.int)
        s_buf_num     = self.s[varInst.s_pe_num_index       : varInst.s_buf_num_index     ].reshape([varInst.LY_LEN, 1]).astype(np.int)
        s_ly_idx      = self.s[varInst.s_buf_num_index      : varInst.s_ly_idx_index      ].reshape([1, 1]).astype(np.int)
        s_res_rtm     = self.s[varInst.s_ly_idx_index       : varInst.s_res_rtm_index     ].reshape([varInst.LY_LEN, 1]).astype(np.int)
        s_res_egy     = self.s[varInst.s_res_rtm_index      : varInst.s_res_egy_index     ].reshape([varInst.LY_LEN, 1]).astype(np.int)
        s_res_tpt     = self.s[varInst.s_res_egy_index      : varInst.s_res_tpt_index     ].reshape([varInst.LY_LEN, 1])
        s_ly_epl_cnt  = self.s[varInst.s_res_tpt_index      : varInst.s_ly_epl_cnt_index  ].reshape([varInst.LY_LEN, 1])
        s_res_area    = self.s[varInst.s_ly_epl_cnt_index   : varInst.s_res_area_index    ].reshape([varInst.LY_LEN, 1])
        s_res_power   = self.s[varInst.s_res_area_index     : varInst.s_res_power_index   ].reshape([varInst.LY_LEN, 1])
        s_res_egy_ncl = self.s[varInst.s_res_power_index    : varInst.s_res_egy_ncl_index ].reshape([varInst.LY_LEN, 1])
        s_l2_regy     = self.s[varInst.s_res_egy_ncl_index  : varInst.s_l2_regy_index     ].reshape([varInst.LY_LEN, 1])
        s_l2_wegy     = self.s[varInst.s_l2_regy_index      : varInst.s_l2_wegy_index     ].reshape([varInst.LY_LEN, 1])
        s_l1_regy     = self.s[varInst.s_l2_wegy_index      : varInst.s_l1_regy_index     ].reshape([varInst.LY_LEN, 1])
        s_l1_wegy     = self.s[varInst.s_l1_regy_index      : varInst.s_l1_wegy_index     ].reshape([varInst.LY_LEN, 1])
        s_mac_egy     = self.s[varInst.s_l1_wegy_index      : varInst.s_mac_egy_index     ].reshape([varInst.LY_LEN, 1])
        s_ddr_egy     = self.s[varInst.s_mac_egy_index      : varInst.s_ddr_egy_index     ].reshape([varInst.LY_LEN, 1])


        #action[varInst.ACT_CLT_LVL_GEN_INDEX]
        #step0: proceess layer index
        #act_layer_index = np.asscalar(s_ly_idx) + varInst.LAYER[action[varInst.ACT_LAYER_INDEX]]
        act_layer_index = action[varInst.ACT_LAYER_INDEX]
        #act_layer_index = int(act_layer_index%varInst.LY_LEN) #act_layer_index is a scalar no need copy 
        s_ly_idx_t = act_layer_index

        #step1: proceess 6 dimension value:  index = i + cluster_lvl * len
        act_typ_sze_oft_dim_index = action[varInst.ACT_DIM_INDEX] + (action[varInst.ACT_CLT_LVL_INDEX] * varInst.act_dim_len)

        dim_eache_index_mode = 0 # 6 dim all have prob to select to spatial, 1 or 2 spaital
        dim_top_spatial_mode = 0 # 6 dim all have prob to select to spatial, only 1 spatial 
        dim_fixed_mode = 0       # No type set, a fixed dim is spatial,      only 1 spatial 
        dim_error_mode = 0

        #step2: proceess 6 dimension order : [0, 719]
        if(varInst.act_type_len  > 0 and varInst.act_dim_top_len == 0): # 6 dim all have prob to select to spatial, 1 or 2 spaital
            dim_eache_index_mode = 1
        if(varInst.act_type_len == 0 and varInst.act_dim_top_len >  0): # 6 dim all have prob to select to spatial, only 1 spatial 
            dim_top_spatial_mode   = 1
        if(varInst.act_type_len == 0 and varInst.act_dim_top_len == 0): # No type set, a fixed dim is spatial,      only 1 spatial 
            dim_fixed_mode = 1
        if(varInst.act_type_len  > 0 and varInst.act_dim_top_len >  0):
            dim_error_mode = 1
            #wFile('action dimension mode error', fileNameFull)
            raise Exception('action dimension mode error')

        s_dim_order_t = s_dim_order.copy()
        if(dim_eache_index_mode == 1 or dim_top_spatial_mode == 1):
            s_dim_order_t[act_layer_index, action[varInst.ACT_CLT_LVL_INDEX]] = s_dim_order_t[act_layer_index, action[varInst.ACT_CLT_LVL_INDEX]] + varInst.DIM_ORDER[action[varInst.ACT_DIM_ORDER_INDEX]]
            s_dim_order_t[act_layer_index, action[varInst.ACT_CLT_LVL_INDEX]] = np.mod(s_dim_order_t[act_layer_index, action[varInst.ACT_CLT_LVL_INDEX]], varInst.dim_order_num).astype(np.int)

        #step3: process 3 size temporal/spatial
        act_type_index = action[varInst.ACT_TYPE_INDEX] # if 0 is max, Temporal, if 1 is max, Spatial 
        s_type_t = s_type.copy()
        s_type_t[act_layer_index, act_typ_sze_oft_dim_index] = act_type_index  # Tempoal :0, Spatial :1

        # when the layer is dep-wise, it cannot select dimension of "K", the 'K' is at 0, 6, 12,when cluster level is 3
        tmp_clt_index = action[varInst.ACT_CLT_LVL_INDEX]
        if(dim_top_spatial_mode == 1):
            act_dim_top_index                               = action[varInst.ACT_DIM_TOP_INDEX] + (tmp_clt_index * varInst.act_dim_len ) 
            if(varInst.IS_DSCONV[act_layer_index] == 1 ):
                if(tmp_clt_index == 0):
                    while(act_dim_top_index == 0):
                        act_dim_top_index = random.randint(0,varInst.DIM_NUM) # re select a dim from 0,5 : when is level 0,
                        #print(varInst.DIM_NUM)
                        #print('test0',act_dim_top_index)
                elif(tmp_clt_index == 1):
                    while(act_dim_top_index == 6):
                        act_dim_top_index = random.randint(varInst.DIM_NUM,2*varInst.DIM_NUM)# re select a dim from 6,11 : when is level 1,
                        #print(varInst.DIM_NUM)
                        #print('test1',act_dim_top_index)
                elif(tmp_clt_index == 2):
                    while(act_dim_top_index == 12):
                        act_dim_top_index = random.randint(2*varInst.DIM_NUM,3*varInst.DIM_NUM)# re select a dim from 12,17 : when is level 1,
                # the for code
                #for lvl in range(action[varInst.ACT_CLT_LVL_GEN_INDEX]):
                #    if(tmp_clt_index == lvl):
                #        while(act_dim_top_index == lvl*varInst.DIM_NUM):
                #        act_dim_top_index = random.randint(lvl*varInst.DIM_NUM, (lvl+1)*varInst.DIM_NUM)# re select a dim from 12,17 : when is level 1,


            s_type_t[act_layer_index, tmp_clt_index * varInst.act_dim_len : (tmp_clt_index+1) * varInst.act_dim_len] = np.zeros([varInst.act_dim_len,], np.int)   # restet all dim to Temperal map
            s_type_t[act_layer_index, act_dim_top_index]    = 1                      # set some dim to Spatial map


        l1_spatial_num = np.sum(s_type_t[act_layer_index, 0 : varInst.act_dim_len])
        if(  l1_spatial_num > 2 or l1_spatial_num == 0 ):  # check if spatial map is too much
            self.r = PENALTY_TOO_MANY_OR_NO_SPATIAL_MAP
            self.done = False
            if DEBUG_PRINT == True:
                print('Warning: Error: have too many or no Spatial map, ==0 or >2 at level1',np.sum(s_type_t[act_layer_index, 0 : varInst.act_dim_len]))
                wFile('Warning: Error: have too many or no Spatial map, ==0 or >2 at level1'+ str(np.sum(s_type_t[act_layer_index, 0 : varInst.act_dim_len])), fileNameFull)
            return tmp_s, self.r, self.done

        if action[varInst.ACT_CLT_LVL_GEN_INDEX] >= 1:
            l2_spatial_num = np.sum(s_type_t[act_layer_index, varInst.act_dim_len : 2*varInst.act_dim_len])
            if( l2_spatial_num > 2 or l2_spatial_num == 0 ):  # check if spatial map is too much
                self.r = PENALTY_TOO_MANY_OR_NO_SPATIAL_MAP
                self.done = False
                if DEBUG_PRINT == True:
                    print('Warning: Error: have too many or no Spatial map, ==0 or >2 at level2',np.sum(s_type_t[act_layer_index, varInst.act_dim_len : 2*varInst.act_dim_len]))
                    wFile('Warning: Error: have too many or no Spatial map, ==0 or >2 at level2'+ str(np.sum(s_type_t[act_layer_index, varInst.act_dim_len : 2*varInst.act_dim_len])), fileNameFull)
                return tmp_s, self.r, self.done
        
        if action[varInst.ACT_CLT_LVL_GEN_INDEX] >= 2:
            l3_spatial_num = np.sum(s_type_t[act_layer_index, 2*varInst.act_dim_len : 3*varInst.act_dim_len])
            if( l3_spatial_num > 2 or l3_spatial_num == 0 ):  # check if spatial map is too much
                self.r = PENALTY_TOO_MANY_OR_NO_SPATIAL_MAP
                self.done = False
                if DEBUG_PRINT == True:
                    print('Warning: Error: have too many or no Spatial map, ==0 or >2 at level3',np.sum(s_type_t[act_layer_index, 2*varInst.act_dim_len : 3*varInst.act_dim_len]))
                    wFile('Warning: Error: have too many or no Spatial map, ==0 or >2 at level3 '+ str(np.sum(s_type_t[act_layer_index, 2*varInst.act_dim_len : 3*varInst.act_dim_len])), fileNameFull)
                return tmp_s, self.r, self.done

        #step4: process 5 size 
        act_size_index  = action[varInst.ACT_SIZE_STEP_INDEX]  
        #size_diff       = varInst.ACT_SIZE_STEP[act_size_index]
        size_diff       = act_size_index
        s_size_t        = s_size.copy()
        s_size_t[act_layer_index, act_typ_sze_oft_dim_index] = np.around(size_diff).astype(np.int) #  a = a + round(a*percnet)        
        if s_size_t[act_layer_index, act_typ_sze_oft_dim_index] < 1:
            s_size_t[act_layer_index, act_typ_sze_oft_dim_index] = 1
        elif s_size_t[act_layer_index, act_typ_sze_oft_dim_index] > varInst.s_size_v_max[act_layer_index, act_typ_sze_oft_dim_index]:
            s_size_t[act_layer_index, act_typ_sze_oft_dim_index] = varInst.s_size_v_max[act_layer_index, act_typ_sze_oft_dim_index]
        #s_size_t[act_layer_index, act_typ_sze_oft_dim_index] = np.mod(s_size_t[act_layer_index, act_typ_sze_oft_dim_index], varInst.s_size_v_max[act_layer_index, act_typ_sze_oft_dim_index]) + 1 # 0:max-1 -> 1:max

        #step5: process 5 offset 
        act_offset_index = action[varInst.ACT_OFFSET_STEP_INDEX] # if 0 is max, Temporal, if 1 is max, Spatial 
        #offset_diff      = varInst.ACT_OFFSET_STEP[act_offset_index]
        offset_diff      = act_offset_index
        s_offset_t       = s_offset.copy()
        s_offset_t[act_layer_index, act_typ_sze_oft_dim_index] = np.around(offset_diff).astype(np.int)  #  a = a + round(a*percnet)        
        #s_offset_t[act_layer_index, act_typ_sze_oft_dim_index] = np.mod(s_offset_t[act_layer_index, act_typ_sze_oft_dim_index], varInst.s_oft_v_max[act_layer_index, act_typ_sze_oft_dim_index]) + 1 # 0:max-1 -> 1:max    

        s_cluster_v_t = s_cluster_v.copy()
        s_cluster_v_t[act_layer_index, :] = varInst.CLT_V[ action[varInst.ACT_CLT_V_INDEX] ]  

        s_pe_num_t = s_pe_num.copy()
        #s_pe_num_t[act_layer_index] = s_pe_num_t[act_layer_index] + varInst.PE_NUM[ action[varInst.ACT_PE_NUM_INDEX] ]  
        #s_pe_num_t[act_layer_index] = int(s_pe_num_t[act_layer_index] % varInst.PE_NUM_MAX) 
        s_pe_num_t[act_layer_index] = varInst.PE_NUM[ action[varInst.ACT_PE_NUM_INDEX] ]  
        if varInst.vs_gamma == True:
            if varInst.gamma_cld == True:
                s_pe_num_t[act_layer_index] = 65536
            else:#edge 
                s_pe_num_t[act_layer_index] = 168
        elif varInst.vs_dac_naas == True:
            if varInst.naas_pe256 == True:
                s_pe_num_t[act_layer_index] = 256
            else: 
                s_pe_num_t[act_layer_index] = 1024

        #if( s_pe_num_t[act_layer_index] < 1 or s_pe_num_t[act_layer_index] >= varInst.PE_NUM_MAX ):
        #    self.r = PENALTY_OUT_OF_RANGE
        #    self.done = False
        #    if DEBUG_PRINT == True:
        #        print('Warning: pe num out of order!')
        #    return tmp_s, self.r, self.done

        s_buf_num_t = s_buf_num.copy()
        #s_buf_num_t[act_layer_index] = s_buf_num_t[act_layer_index] + varInst.BUF_NUM[ action[varInst.ACT_BUF_NUM_INDEX] ]  
        #s_buf_num_t[act_layer_index] = int(s_buf_num_t[act_layer_index] % varInst.BUF_NUM_MAX)
        s_buf_num_t [act_layer_index] = varInst.BUF_NUM [action [varInst.ACT_BUF_NUM_INDEX]]

        #if( s_buf_num_t[act_layer_index] < 1 or s_buf_num_t[act_layer_index] >= varInst.BUF_NUM_MAX ):
        #    self.r = PENALTY_OUT_OF_RANGE
        #    self.done = False
        #    if DEBUG_PRINT == True:
        #        print('Warning: buf num out of order!')
        #    return tmp_s, self.r, self.done

        

        X_size = s_size_t[act_layer_index, varInst.DIM_NAME_LIST.index("X")] 
        S_size = s_size_t[act_layer_index, varInst.DIM_NAME_LIST.index("S")] 
        Y_size = s_size_t[act_layer_index, varInst.DIM_NAME_LIST.index("Y")] 
        R_size = s_size_t[act_layer_index, varInst.DIM_NAME_LIST.index("R")] 
        C_size = s_size_t[act_layer_index, varInst.DIM_NAME_LIST.index("C")] 
        K_size = s_size_t[act_layer_index, varInst.DIM_NAME_LIST.index("K")] 
        if(X_size < S_size+1):
            s_size_t[act_layer_index,varInst.DIM_NAME_LIST.index("X")] = S_size + 2 
        if(Y_size < R_size+1):
            s_size_t[act_layer_index,varInst.DIM_NAME_LIST.index("Y")] = R_size + 2 

        X_index_cl0 = varInst.DIM_NAME_LIST.index("X") 
        S_index_cl0 = varInst.DIM_NAME_LIST.index("S") 
        Y_index_cl0 = varInst.DIM_NAME_LIST.index("Y") 
        R_index_cl0 = varInst.DIM_NAME_LIST.index("R") 
        C_index_cl0 = varInst.DIM_NAME_LIST.index("C") 
        K_index_cl0 = varInst.DIM_NAME_LIST.index("K") 

        X_size_cl0 = s_size_t[act_layer_index, X_index_cl0] 
        S_size_cl0 = s_size_t[act_layer_index, S_index_cl0] 
        Y_size_cl0 = s_size_t[act_layer_index, Y_index_cl0] 
        R_size_cl0 = s_size_t[act_layer_index, R_index_cl0] 
        C_size_cl0 = s_size_t[act_layer_index, C_index_cl0] 
        K_size_cl0 = s_size_t[act_layer_index, K_index_cl0] 

        if(X_size_cl0 > varInst.X[act_layer_index]):
            s_size_t[act_layer_index, X_index_cl0] = varInst.X[act_layer_index]
        if(S_size_cl0 > varInst.S[act_layer_index]):
            s_size_t[act_layer_index, S_index_cl0] = varInst.S[act_layer_index]
        if(Y_size_cl0 > varInst.Y[act_layer_index]):
            s_size_t[act_layer_index, Y_index_cl0] = varInst.Y[act_layer_index]
        if(R_size_cl0 > varInst.R[act_layer_index]):
            s_size_t[act_layer_index, R_index_cl0] = varInst.R[act_layer_index]
        if(C_size_cl0 > varInst.C[act_layer_index]):
            s_size_t[act_layer_index, C_index_cl0] = varInst.C[act_layer_index]
        if(K_size_cl0 > varInst.K[act_layer_index]):
            s_size_t[act_layer_index, K_index_cl0] = varInst.K[act_layer_index]

        if action[varInst.ACT_CLT_LVL_GEN_INDEX] >= 1:
            X_index_cl1 = varInst.act_dim_len + varInst.DIM_NAME_LIST.index("X") 
            S_index_cl1 = varInst.act_dim_len + varInst.DIM_NAME_LIST.index("S") 
            Y_index_cl1 = varInst.act_dim_len + varInst.DIM_NAME_LIST.index("Y") 
            R_index_cl1 = varInst.act_dim_len + varInst.DIM_NAME_LIST.index("R") 
            C_index_cl1 = varInst.act_dim_len + varInst.DIM_NAME_LIST.index("C") 
            K_index_cl1 = varInst.act_dim_len + varInst.DIM_NAME_LIST.index("K") 

            X_size_cl1 = s_size_t[act_layer_index, X_index_cl1] 
            S_size_cl1 = s_size_t[act_layer_index, S_index_cl1] 
            Y_size_cl1 = s_size_t[act_layer_index, Y_index_cl1] 
            R_size_cl1 = s_size_t[act_layer_index, R_index_cl1] 
            C_size_cl1 = s_size_t[act_layer_index, C_index_cl1] 
            K_size_cl1 = s_size_t[act_layer_index, K_index_cl1] 


            if(X_size_cl1 > varInst.X[act_layer_index]):
                s_size_t[act_layer_index, X_index_cl1] = varInst.X[act_layer_index]
            if(S_size_cl1 > varInst.S[act_layer_index]):
                s_size_t[act_layer_index, S_index_cl1] = varInst.S[act_layer_index]
            if(Y_size_cl1 > varInst.Y[act_layer_index]):
                s_size_t[act_layer_index, Y_index_cl1] = varInst.Y[act_layer_index]
            if(R_size_cl1 > varInst.R[act_layer_index]):
                s_size_t[act_layer_index, R_index_cl1] = varInst.R[act_layer_index]
            if(C_size_cl1 > varInst.C[act_layer_index]):
                s_size_t[act_layer_index, C_index_cl1] = varInst.C[act_layer_index]
            if(K_size_cl1 > varInst.K[act_layer_index]):
                s_size_t[act_layer_index, K_index_cl1] = varInst.K[act_layer_index]


            X_size_cl0 = s_size_t[act_layer_index, X_index_cl0] 
            S_size_cl0 = s_size_t[act_layer_index, S_index_cl0] 
            Y_size_cl0 = s_size_t[act_layer_index, Y_index_cl0] 
            R_size_cl0 = s_size_t[act_layer_index, R_index_cl0] 
            C_size_cl0 = s_size_t[act_layer_index, C_index_cl0] 
            K_size_cl0 = s_size_t[act_layer_index, K_index_cl0] 
            X_size_cl1 = s_size_t[act_layer_index, X_index_cl1] 
            S_size_cl1 = s_size_t[act_layer_index, S_index_cl1] 
            Y_size_cl1 = s_size_t[act_layer_index, Y_index_cl1] 
            R_size_cl1 = s_size_t[act_layer_index, R_index_cl1] 
            C_size_cl1 = s_size_t[act_layer_index, C_index_cl1] 
            K_size_cl1 = s_size_t[act_layer_index, K_index_cl1] 

            # >= not work!!
            # if X type is Temporal and X_size_cl1 > X_size_cl0 then 
            if ((s_type_t[act_layer_index, X_index_cl1]==0) and (X_size_cl1 > X_size_cl0)):
                s_size_t[act_layer_index,X_index_cl0] = 2 
                s_size_t[act_layer_index,X_index_cl1] = 1 
            if ((s_type_t[act_layer_index, S_index_cl1]==0) and (S_size_cl1 > S_size_cl0)):
                s_size_t[act_layer_index,S_index_cl0] = 2 
                s_size_t[act_layer_index,S_index_cl1] = 1 
            if ((s_type_t[act_layer_index, Y_index_cl1]==0) and (Y_size_cl1 > Y_size_cl0)):
                s_size_t[act_layer_index,Y_index_cl0] = 2 
                s_size_t[act_layer_index,Y_index_cl1] = 1 
            if ((s_type_t[act_layer_index, R_index_cl1]==0) and (R_size_cl1 > R_size_cl0)):
                s_size_t[act_layer_index,R_index_cl0] = 2 
                s_size_t[act_layer_index,R_index_cl1] = 1 
            if ((s_type_t[act_layer_index, C_index_cl1]==0) and (C_size_cl1 > C_size_cl0)):
                s_size_t[act_layer_index,C_index_cl0] = 2 
                s_size_t[act_layer_index,C_index_cl1] = 1 
            if ((s_type_t[act_layer_index, K_index_cl1]==0) and (K_size_cl1 > K_size_cl0)):
                s_size_t[act_layer_index,K_index_cl0] = 2
                s_size_t[act_layer_index,K_index_cl1] = 1 

            X_size_cl0 = s_size_t[act_layer_index, X_index_cl0] 
            S_size_cl0 = s_size_t[act_layer_index, S_index_cl0] 
            Y_size_cl0 = s_size_t[act_layer_index, Y_index_cl0] 
            R_size_cl0 = s_size_t[act_layer_index, R_index_cl0] 
            C_size_cl0 = s_size_t[act_layer_index, C_index_cl0] 
            K_size_cl0 = s_size_t[act_layer_index, K_index_cl0] 
            X_size_cl1 = s_size_t[act_layer_index, X_index_cl1] 
            S_size_cl1 = s_size_t[act_layer_index, S_index_cl1] 
            Y_size_cl1 = s_size_t[act_layer_index, Y_index_cl1] 
            R_size_cl1 = s_size_t[act_layer_index, R_index_cl1] 
            C_size_cl1 = s_size_t[act_layer_index, C_index_cl1] 
            K_size_cl1 = s_size_t[act_layer_index, K_index_cl1] 

            if ((s_type_t[act_layer_index, X_index_cl1]==0) and X_size_cl1 == X_size_cl0 and varInst.X[act_layer_index] != 1):
                s_size_t[act_layer_index,X_index_cl0] = 2 
                s_size_t[act_layer_index,X_index_cl1] = 1 
            if ((s_type_t[act_layer_index, S_index_cl1]==0) and S_size_cl1 == S_size_cl0 and varInst.S[act_layer_index] != 1):
                s_size_t[act_layer_index,S_index_cl0] = 2 
                s_size_t[act_layer_index,S_index_cl1] = 1 
            if ((s_type_t[act_layer_index, Y_index_cl1]==0) and Y_size_cl1 == Y_size_cl0 and varInst.Y[act_layer_index] != 1):
                s_size_t[act_layer_index,Y_index_cl0] = 2 
                s_size_t[act_layer_index,Y_index_cl1] = 1 
            if ((s_type_t[act_layer_index, R_index_cl1]==0) and R_size_cl1 == R_size_cl0 and varInst.R[act_layer_index] != 1):
                s_size_t[act_layer_index,R_index_cl0] = 2 
                s_size_t[act_layer_index,R_index_cl1] = 1 
            if ((s_type_t[act_layer_index, C_index_cl1]==0) and C_size_cl1 == C_size_cl0 and varInst.C[act_layer_index] != 1):
                s_size_t[act_layer_index,C_index_cl0] = 2 
                s_size_t[act_layer_index,C_index_cl1] = 1 
            if ((s_type_t[act_layer_index, K_index_cl1]==0) and K_size_cl1 == K_size_cl0 and varInst.K[act_layer_index] != 1):
                s_size_t[act_layer_index,K_index_cl0] = 2
                s_size_t[act_layer_index,K_index_cl1] = 1 
            X_size_cl0 = s_size_t[act_layer_index, X_index_cl0] 
            S_size_cl0 = s_size_t[act_layer_index, S_index_cl0] 
            Y_size_cl0 = s_size_t[act_layer_index, Y_index_cl0] 
            R_size_cl0 = s_size_t[act_layer_index, R_index_cl0] 
            C_size_cl0 = s_size_t[act_layer_index, C_index_cl0] 
            K_size_cl0 = s_size_t[act_layer_index, K_index_cl0] 
            X_size_cl1 = s_size_t[act_layer_index, X_index_cl1] 
            S_size_cl1 = s_size_t[act_layer_index, S_index_cl1] 
            Y_size_cl1 = s_size_t[act_layer_index, Y_index_cl1] 
            R_size_cl1 = s_size_t[act_layer_index, R_index_cl1] 
            C_size_cl1 = s_size_t[act_layer_index, C_index_cl1] 
            K_size_cl1 = s_size_t[act_layer_index, K_index_cl1] 
        
        if action[varInst.ACT_CLT_LVL_GEN_INDEX] >= 2:

            X_index_cl2 = 2*varInst.act_dim_len + varInst.DIM_NAME_LIST.index("X") 
            S_index_cl2 = 2*varInst.act_dim_len + varInst.DIM_NAME_LIST.index("S") 
            Y_index_cl2 = 2*varInst.act_dim_len + varInst.DIM_NAME_LIST.index("Y") 
            R_index_cl2 = 2*varInst.act_dim_len + varInst.DIM_NAME_LIST.index("R") 
            C_index_cl2 = 2*varInst.act_dim_len + varInst.DIM_NAME_LIST.index("C") 
            K_index_cl2 = 2*varInst.act_dim_len + varInst.DIM_NAME_LIST.index("K") 

            X_size_cl2 = s_size_t[act_layer_index, X_index_cl2] 
            S_size_cl2 = s_size_t[act_layer_index, S_index_cl2] 
            Y_size_cl2 = s_size_t[act_layer_index, Y_index_cl2] 
            R_size_cl2 = s_size_t[act_layer_index, R_index_cl2] 
            C_size_cl2 = s_size_t[act_layer_index, C_index_cl2] 
            K_size_cl2 = s_size_t[act_layer_index, K_index_cl2] 

            if(X_size_cl2 > varInst.X[act_layer_index]):
                s_size_t[act_layer_index, X_index_cl2] = varInst.X[act_layer_index]
            if(S_size_cl2 > varInst.S[act_layer_index]):
                s_size_t[act_layer_index, S_index_cl2] = varInst.S[act_layer_index]
            if(Y_size_cl2 > varInst.Y[act_layer_index]):
                s_size_t[act_layer_index, Y_index_cl2] = varInst.Y[act_layer_index]
            if(R_size_cl2 > varInst.R[act_layer_index]):
                s_size_t[act_layer_index, R_index_cl2] = varInst.R[act_layer_index]
            if(C_size_cl2 > varInst.C[act_layer_index]):
                s_size_t[act_layer_index, C_index_cl2] = varInst.C[act_layer_index]
            if(K_size_cl2 > varInst.K[act_layer_index]):
                s_size_t[act_layer_index, K_index_cl2] = varInst.K[act_layer_index]
            


        #change the postion to new order
        s_dim_order_t_t0 = np.asscalar(s_dim_order_t[act_layer_index, 0])
        dim_order_top_sp0 = self_dim_perm[s_dim_order_t_t0] 
        if dim_fixed_mode == 1:
            dim_order_top_sp0 = varInst.DIM_TOP

        s_type_t_n0_all_ly    = s_type_t.tolist()
        size_t_n0_all_ly      = s_size_t.tolist()
        offset_t_n0_all_ly    = s_offset_t.tolist()

        s_type_t_n0     = self.list_change_pos(varInst.DIM_NAME_LIST, dim_order_top_sp0,        s_type_t[   act_layer_index, 0 : varInst.act_dim_len])
        size_n0         = self.list_change_pos(varInst.DIM_NAME_LIST, dim_order_top_sp0,        s_size_t[   act_layer_index, 0 : varInst.act_dim_len])
        offset_n0       = self.list_change_pos(varInst.DIM_NAME_LIST, dim_order_top_sp0,        s_offset_t[ act_layer_index, 0 : varInst.act_dim_len])

        s_type_t_n0_all_ly[act_layer_index][0 : varInst.act_dim_len] = s_type_t_n0
        size_t_n0_all_ly  [act_layer_index][0 : varInst.act_dim_len] = size_n0
        offset_t_n0_all_ly[act_layer_index][0 : varInst.act_dim_len] = offset_n0
        
        clt_lvl_gen = action[varInst.ACT_CLT_LVL_GEN_INDEX]
        net_para = {    'cluster_lvl'           : clt_lvl_gen,
                        's_type0'               : s_type_t_n0_all_ly,
                        'size0'                 : size_t_n0_all_ly,
                        'offset0'               : offset_t_n0_all_ly,
                        'dim0'                  : dim_order_top_sp0,
                        'ly_idx'                : act_layer_index,
                        'net_name'              : varInst.net_name,
                        'ly_type'               : varInst.LY_TYPE,
                        'ly_name'               : varInst.ly_n_list,
                        'stride_x'              : varInst.stride_x,
                        'stride_y'              : varInst.stride_y,
                        'K'                     : varInst.K,
                        'C'                     : varInst.C,
                        'R'                     : varInst.R,
                        'S'                     : varInst.S,
                        'Y'                     : varInst.Y,
                        'X'                     : varInst.X,
                        'is_dsconv'             : varInst.IS_DSCONV,
                        'en_ddr'                : varInst.EN_DDR,
                        'en_cross_ly'           : varInst.EN_CROSS_LY
                        }

        if action[varInst.ACT_CLT_LVL_GEN_INDEX] >= 1:
            s_dim_order_t_t1 = np.asscalar(s_dim_order_t[act_layer_index, 1])
            dim_order_top_sp1 = self_dim_perm[s_dim_order_t_t1] 
            if dim_fixed_mode == 1:
                dim_order_top_sp1 = varInst.DIM_TOP
            s_type_t_n1_all_ly = s_type_t.tolist()
            size_t_n1_all_ly   = s_size_t.tolist()
            offset_t_n1_all_ly = s_offset_t.tolist()

            s_type_t_n1 = self.list_change_pos(varInst.DIM_NAME_LIST, dim_order_top_sp1, s_type_t[  act_layer_index, varInst.act_dim_len : 2*varInst.act_dim_len])
            size_t_n1   = self.list_change_pos(varInst.DIM_NAME_LIST, dim_order_top_sp1, s_size_t[  act_layer_index, varInst.act_dim_len : 2*varInst.act_dim_len])
            offset_t_n1 = self.list_change_pos(varInst.DIM_NAME_LIST, dim_order_top_sp1, s_offset_t[act_layer_index, varInst.act_dim_len : 2*varInst.act_dim_len])


            s_type_t_n1_all_ly[act_layer_index][varInst.act_dim_len : 2*varInst.act_dim_len] = s_type_t_n1
            size_t_n1_all_ly  [act_layer_index][varInst.act_dim_len : 2*varInst.act_dim_len] = size_t_n1
            offset_t_n1_all_ly[act_layer_index][varInst.act_dim_len : 2*varInst.act_dim_len] = offset_t_n1

            s_cluster_v_t_v0 = s_cluster_v_t[act_layer_index, 0]

            if(varInst.K[act_layer_index]==1 or varInst.C[act_layer_index]==1 or varInst.R[act_layer_index]==1 or varInst.S[act_layer_index]==1 or varInst.Y[act_layer_index]==1 or varInst.X[act_layer_index]==1):
                cluster_lvl_t =  1  
            else:
                cluster_lvl_t = clt_lvl_gen

            net_para = {
                        'cluster_lvl'    : cluster_lvl_t,
                        's_type0'        : s_type_t_n0_all_ly,
                        'size0'          : size_t_n0_all_ly,
                        'offset0'        : offset_t_n0_all_ly,
                        'dim0'           : dim_order_top_sp0,
                        's_type1'        : s_type_t_n1_all_ly,
                        'size1'          : size_t_n1_all_ly,
                        'offset1'        : offset_t_n1_all_ly,
                        'dim1'           : dim_order_top_sp1,
                        'ly_idx'         : act_layer_index,
                        'cluster_value0' : s_cluster_v_t_v0,
                        'net_name'       : varInst.net_name,
                        'ly_type'        : varInst.LY_TYPE,
                        'ly_name'        : varInst.ly_n_list,
                        'stride_x'       : varInst.stride_x,
                        'stride_y'       : varInst.stride_y,
                        'K'              : varInst.K,
                        'C'              : varInst.C,
                        'R'              : varInst.R,
                        'S'              : varInst.S,
                        'Y'              : varInst.Y,
                        'X'              : varInst.X,
                        'is_dsconv'      : varInst.IS_DSCONV,
                        'en_ddr'                : varInst.EN_DDR,
                        'en_cross_ly'           : varInst.EN_CROSS_LY
                        }
            
            max_clt_v=s_cluster_v_t_v0

        if action[varInst.ACT_CLT_LVL_GEN_INDEX] >= 2:
            s_dim_order_t_t2 = np.asscalar(s_dim_order_t[act_layer_index, 2])
            dim_order_top_sp2 = self_dim_perm[s_dim_order_t_t2] 
            if dim_fixed_mode == 1:
                dim_order_top_sp2 = varInst.DIM_TOP
            s_type_t_n2_all_ly = s_type_t.tolist()
            size_t_n2_all_ly   = s_size_t.tolist()
            offset_t_n2_all_ly = s_offset_t.tolist()

            s_type_t_n2 = self.list_change_pos(varInst.DIM_NAME_LIST, dim_order_top_sp2, s_type_t[  act_layer_index, 2*varInst.act_dim_len : 3*varInst.act_dim_len])
            size_t_n2   = self.list_change_pos(varInst.DIM_NAME_LIST, dim_order_top_sp2, s_size_t[  act_layer_index, 2*varInst.act_dim_len : 3*varInst.act_dim_len])
            offset_t_n2 = self.list_change_pos(varInst.DIM_NAME_LIST, dim_order_top_sp2, s_offset_t[act_layer_index, 2*varInst.act_dim_len : 3*varInst.act_dim_len])


            s_type_t_n2_all_ly[act_layer_index][2*varInst.act_dim_len : 3*varInst.act_dim_len] = s_type_t_n2
            size_t_n2_all_ly  [act_layer_index][2*varInst.act_dim_len : 3*varInst.act_dim_len] = size_t_n2
            offset_t_n2_all_ly[act_layer_index][2*varInst.act_dim_len : 3*varInst.act_dim_len] = offset_t_n2

            s_cluster_v_t_v1 = s_cluster_v_t[act_layer_index, 1]

            if(varInst.K[act_layer_index]==1 or varInst.C[act_layer_index]==1 or varInst.R[act_layer_index]==1 or varInst.S[act_layer_index]==1 or varInst.Y[act_layer_index]==1 or varInst.X[act_layer_index]==1):
                cluster_lvl_t =  1  
            else:
                cluster_lvl_t = clt_lvl_gen

            net_para = {
                        'cluster_lvl'    : cluster_lvl_t,
                        's_type0'        : s_type_t_n0_all_ly,
                        'size0'          : size_t_n0_all_ly,
                        'offset0'        : offset_t_n0_all_ly,
                        'dim0'           : dim_order_top_sp0,
                        's_type1'        : s_type_t_n1_all_ly,
                        'size1'          : size_t_n1_all_ly,
                        'offset1'        : offset_t_n1_all_ly,
                        'dim1'           : dim_order_top_sp1,
                        's_type2'        : s_type_t_n2_all_ly,
                        'size2'          : size_t_n2_all_ly,
                        'offset2'        : offset_t_n2_all_ly,
                        'dim2'           : dim_order_top_sp2,
                        'ly_idx'         : act_layer_index,
                        'cluster_value0' : s_cluster_v_t_v0,
                        'cluster_value1' : s_cluster_v_t_v1,
                        'net_name'       : varInst.net_name,
                        'ly_type'        : varInst.LY_TYPE,
                        'ly_name'        : varInst.ly_n_list,
                        'stride_x'       : varInst.stride_x,
                        'stride_y'       : varInst.stride_y,
                        'K'              : varInst.K,
                        'C'              : varInst.C,
                        'R'              : varInst.R,
                        'S'              : varInst.S,
                        'Y'              : varInst.Y,
                        'X'              : varInst.X,
                        'is_dsconv'      : varInst.IS_DSCONV,
                        'en_ddr'         : varInst.EN_DDR,
                        'en_cross_ly'    : varInst.EN_CROSS_LY
                        }
                    
            max_clt_v=s_cluster_v_t_v0 if s_cluster_v_t_v0 >= s_cluster_v_t_v1  else s_cluster_v_t_v1   

        path_a = 'test.csv'  
        if os.path.exists(path_a):  
            os.remove(path_a)  

        start1 = time.time()
        self._renderfromfile(template_path, net_para, base_path + dataflow_output_path)

        global fileName
        if DEBUG_SAVE_MAP == True:
            self._renderfromfile(template_path, net_para, dir_name_map +"//"+fileName+"_en_ddr_cross_ly_"+net[5:8]+"_map_ly"+str(act_layer_index)+".m")
            net_para['en_cross_ly'] = 0
            self._renderfromfile(template_path, net_para, dir_name_map +"//"+fileName+"_en_ddr_no_croos_ly_"+net[5:8]+"_map_ly"+str(act_layer_index)+".m")
            net_para['en_ddr'] = 0
            self._renderfromfile(template_path, net_para, dir_name_map +"//"+fileName+"_disen_ddr_"+net[5:8]+"_map_ly"+str(act_layer_index)+".m")

        eval_ly_pe_num  = s_pe_num_t[act_layer_index][0]
        eval_ly_buf_num = s_buf_num_t[act_layer_index][0] 
        if action[varInst.ACT_CLT_LVL_GEN_INDEX] >= 1:
            if(max_clt_v > eval_ly_pe_num):
                eval_ly_pe_num = max_clt_v 

        start2 = time.time()

        val = [0 for i in range(18)]
        try:
            val = maestro.main_m(int(eval_ly_pe_num))
        except Exception as e:
            if DEBUG_PRINT == True:
                print(e)
            #wFile('exception \n', fileNameFull)
            val[2] = -1

        #print("[ 0:runtime, 1:engergy, 2:throughtput, 3:computation, 4:l1_size, 5:l2_size, 6:area, 7:power, 8:ddr_energy, 9:num_pe_utilized, 10:reuse_input, 11:reuse_weight, 12:reuse_output]")
        runtime             = val[0]             
        energy              = val[1]
        throughput          = val[2]
        computation         = val[3]
        l1_size             = val[4]
        l2_size             = val[5]
        area                = val[6]
        power               = val[7]
        ddr_egy             = val[8]
        num_pe_used         = val[9]
        reuse_ifactor       = val[10]
        reuse_wfactor       = val[11]
        reuse_ofactor       = val[12]
        l2_regy             = val[13]
        l2_wegy             = val[14]
        l1_regy             = val[15]
        l1_wegy             = val[16]
        mac_egy             = val[17]

        start3 = time.time()
        #print(start3 - start2)
        #wFile("%f" % (start3 - start2), fileNameFull)

        # change to from c++
        #pd_data   = pd.read_csv (base_path + output_csr_path)
        #runtime     = list(pd_data[' Runtime (Cycles)'].values)[-1]             
        #energy      = list(pd_data[' Activity count-based Energy (nJ)'].values)[-1]             
        #throughput  = list(pd_data[' Throughput (MACs/Cycle)'].values)[-1]             
        #l2_buf_size = list(pd_data['  L2 SRAM Size (Bytes)'].values)[-1]             
        #l1_buf_size = list(pd_data['  L2 SRAM Size (Bytes)'].values)[-1]             

        if(self.is_float(runtime)==False or self.is_float(energy)==False or self.is_float(throughput)==False ):
            self.r = PENALTY_MAESTRO_ERROR
            self.done = False
            if DEBUG_PRINT == True:
                print('Warning: runtime or energy or throughput is Nan')
                wFile('Warning: runtime or energy or throughput is Nan', fileNameFull)
            return tmp_s, self.r, self.done

        l1_buf_size = l1_size             
        l2_buf_size = l2_size             
        # change bufer only is l1_buffer
        actual_ly_buf_size = l1_buf_size
        if(actual_ly_buf_size > eval_ly_buf_num  and varInst.vs_gamma==False): 
            if DEBUG_PRINT == True:
                print('Maestro Buffer size overflow layer: expect: actural:', varInst.ly_n_list[act_layer_index], eval_ly_buf_num, actual_ly_buf_size)
                wFile('Maestro Buffer size overflow layer: expect: actural: '+ str(varInst.ly_n_list[act_layer_index])+', '+ str(eval_ly_buf_num)+', '+ str(actual_ly_buf_size),     fileNameFull)
            self.r = PENALTY_VALUE_OUT_RANG
            self.done = False
            return tmp_s, self.r, self.done
        if(varInst.vs_gamma==True): 
            if (varInst.gamma_cld==True):
                if(l1_size > 4*1024*1024 or l2_size > 24*1024*1024): 
                    if DEBUG_PRINT == True:
                        print('Maestro Buffer size overflow layer: expect: actural:', varInst.ly_n_list[act_layer_index], eval_ly_buf_num, actual_ly_buf_size)
                        wFile('Maestro Buffer size overflow layer: expect: actural: '+ str(varInst.ly_n_list[act_layer_index])+', '+ str(eval_ly_buf_num)+', '+ str(actual_ly_buf_size),     fileNameFull)
                    self.r = PENALTY_VALUE_OUT_RANG
                    self.done = False
                    return tmp_s, self.r, self.done
            else:
                if(l1_size > 512 or l2_size > 108*1024): 
                    if DEBUG_PRINT == True:
                        print('Maestro Buffer size overflow layer: expect: actural:', varInst.ly_n_list[act_layer_index], eval_ly_buf_num, actual_ly_buf_size)
                        wFile('Maestro Buffer size overflow layer: expect: actural: '+ str(varInst.ly_n_list[act_layer_index])+', '+ str(eval_ly_buf_num)+', '+ str(actual_ly_buf_size),     fileNameFull)
                    self.r = PENALTY_VALUE_OUT_RANG
                    self.done = False
                    return tmp_s, self.r, self.done


        if(runtime > 0 and energy > 0 and throughput > 0):

            global glb_state
            global glb_cnt 
            glb_s_ly_epl_cnt  = glb_state[varInst.s_res_tpt_index   : varInst.s_ly_epl_cnt_index  ].reshape([varInst.LY_LEN, 1])

            s_pe_num_t     = s_pe_num.copy() 
            s_buf_num_t    = s_buf_num.copy() 

            s_res_rtm_t     = s_res_rtm.copy() 
            s_res_egy_t     = s_res_egy.copy() 
            s_res_tpt_t     = s_res_tpt.copy() 
            s_ly_epl_cnt_t  = s_ly_epl_cnt.copy() 
            glb_s_ly_epl_cnt_t = glb_s_ly_epl_cnt.copy()
            s_res_area_t    = s_res_area.copy() 
            s_res_power_t   = s_res_power.copy() 
            s_res_egy_ncl_t = s_res_egy_ncl.copy()
            s_l2_regy_t     = s_l2_regy.copy()
            s_l2_wegy_t     = s_l2_wegy.copy()
            s_l1_regy_t     = s_l1_regy.copy()
            s_l1_wegy_t     = s_l1_wegy.copy()
            s_mac_egy_t     = s_mac_egy.copy()
            s_ddr_egy_t     = s_ddr_egy.copy()

            s_pe_num_t[act_layer_index,0]   = eval_ly_pe_num 
            s_buf_num_t[act_layer_index,0]   = eval_ly_buf_num 

            s_res_rtm_t[act_layer_index,0]   = runtime 
            s_res_egy_t[act_layer_index,0]   = energy

            # get same dataflow's energy only no cross_layer
            net_para['en_cross_ly'] = 0
            self._renderfromfile(template_path, net_para, base_path + dataflow_output_path)
            val_ncl = [0 for i in range(8)]
            try:
                val_ncl = maestro.main_m(int(eval_ly_pe_num))
            except Exception as e:
                if DEBUG_PRINT == True:
                    print(e)
                #wFile('exception \n', fileNameFull)
                val_ncl[2] = -1
            energy_ncl      = val_ncl[1]
            s_res_egy_ncl_t[act_layer_index,0] = energy_ncl
            s_l2_regy_t[act_layer_index]       = l2_regy
            s_l2_wegy_t[act_layer_index]       = l2_wegy
            s_l1_regy_t[act_layer_index]       = l1_regy
            s_l1_wegy_t[act_layer_index]       = l1_wegy
            s_mac_egy_t[act_layer_index]       = mac_egy
            s_ddr_egy_t[act_layer_index]       = ddr_egy


            s_res_tpt_t[act_layer_index,0]   = throughput 
            s_ly_epl_cnt_t[act_layer_index,0]  = s_ly_epl_cnt_t[act_layer_index,0] + 1

            s_ly_epl_cnt_t[act_layer_index,0]  = s_ly_epl_cnt_t[act_layer_index,0] + 1
            glb_s_ly_epl_cnt_t[act_layer_index,0]  = glb_s_ly_epl_cnt_t[act_layer_index,0] + 1
            s_res_area_t[act_layer_index,0]  = area 
            s_res_power_t[act_layer_index,0] = power 

            glb_num_pe_used[act_layer_index] = num_pe_used
            glb_reuse_ifactor[act_layer_index] = reuse_ifactor
            glb_reuse_ofactor[act_layer_index] = reuse_ofactor
            glb_reuse_wfactor[act_layer_index] = reuse_wfactor


            #one_layer_npy_data = np.zeros((varInst.analysis_data_num_clm,))
            if DEBUG_SAVE_NPY == True:
                one_layer_npy_data =   np.array([runtime]+#0
                                        [energy]+
                                        [num_pe_used]+
                                        [reuse_ifactor]+
                                        [reuse_ofactor]+
                                        [reuse_wfactor]+
                                        [varInst.K[act_layer_index]]+
                                        [varInst.C[act_layer_index]]+
                                        [varInst.R[act_layer_index]]+
                                        [varInst.S[act_layer_index]]+
                                        [varInst.Y[act_layer_index]]+
                                        [varInst.X[act_layer_index]]+
                                        s_size_t[act_layer_index, 0 : varInst.act_dim_len].tolist()+
                                        s_type_t[act_layer_index, 0 : varInst.act_dim_len].tolist())
                save_npy(act_layer_index, one_layer_npy_data)


            

            tmp_s_ = np.hstack((s_dim_order_t.flatten(), s_type_t.flatten(), s_size_t.flatten(), 
                        s_offset_t.flatten(), s_cluster_v_t.flatten(), s_pe_num_t.flatten(), s_buf_num_t.flatten(), act_layer_index,
                        s_res_rtm_t.flatten(),
                        s_res_egy_t.flatten(),
                        s_res_tpt_t.flatten(), 
                        s_ly_epl_cnt_t.flatten(),
                        s_res_area_t.flatten(),
                        s_res_power_t.flatten(),
                        s_res_egy_ncl_t.flatten(),
                        s_l2_regy_t.flatten(),
                        s_l2_wegy_t.flatten(),
                        s_l1_regy_t.flatten(),
                        s_l1_wegy_t.flatten(),
                        s_mac_egy_t.flatten(),
                        s_ddr_egy_t.flatten() ))
            
            net_runtime     = np.sum(s_res_rtm_t)
            net_energy      = np.sum(s_res_egy_t)
            net_energy_ncl  = np.sum(s_res_egy_ncl_t)

            net_l2_regy  = np.sum(s_l2_regy_t)
            net_l2_wegy  = np.sum(s_l2_wegy_t)
            net_l1_regy  = np.sum(s_l1_regy_t)
            net_l1_wegy  = np.sum(s_l1_wegy_t)
            net_mac_egy  = np.sum(s_mac_egy_t)
            net_ddr_egy  = np.sum(s_ddr_egy_t)

            net_thoughput   = np.sum(s_res_tpt_t)
            net_area_mcts   = area
            net_power_mcts  = power
            self.r = W_time_mcts * (100000/net_runtime) + W_enegy_mcts*(100000/net_energy) + W_throughput_mcts*net_thoughput

            global contr_area
            global contr_power 

            if(mcts_init_done_flag == True and varInst.contr_en_area==True and net_area_mcts > contr_area[varInst.contr_nm]):
                if DEBUG_PRINT == True:
                    print('Constrin area out range, actral: %d, contrain: %d '% (net_area_mcts, contr_area[varInst.contr_nm]))
                    wFile('Constrin area out range, actral: %d, contrain: %d '% (net_area_mcts, contr_area[varInst.contr_nm]), fileNameFull)
                self.r = PENALTY_VALUE_OUT_RANG
                self.done = False
                return tmp_s, self.r, self.done
            if(mcts_init_done_flag == True and varInst.contr_en_power==True and net_power_mcts > contr_power[varInst.contr_nm]):
                if DEBUG_PRINT == True:
                    print('Constrin power out range, actral: %d, contrain: %d '% (net_power_mcts, contr_power[varInst.contr_nm]))
                    wFile('Constrin power out range, actral: %d, contrain: %d '% (net_power_mcts, contr_power[varInst.contr_nm]), fileNameFull)
                self.r = PENALTY_VALUE_OUT_RANG
                self.done = False
                return tmp_s, self.r, self.done

            #wFile('DataFlow valid~~~~~~~\n',     fileNameFull)

            self.done = True
            if DEBUG_PRINT == True:
                print('DataFlow valid~~~~~~~')
                wFile('DataFlow valid~~~~~~~ \n',     fileNameFull)
            start4 = time.time()
            #print(start1-start)
            #print(start2-start1)
            #print(start3-start2)
            #print(start4-start3)               
            glb_s_dim_order   = glb_state[0                         : varInst.s_dim_order_index   ].reshape([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX]).astype(np.int)
            glb_s_type        = glb_state[varInst.s_dim_order_index : varInst.s_type_index        ].reshape([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX*varInst.DIM_NUM]).astype(np.int)
            glb_s_size        = glb_state[varInst.s_type_index      : varInst.s_size_index        ].reshape([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX*varInst.DIM_NUM]).astype(np.int)
            glb_s_offset      = glb_state[varInst.s_size_index      : varInst.s_oft_index         ].reshape([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX*varInst.DIM_NUM]).astype(np.int)
            glb_s_cluster_v   = glb_state[varInst.s_oft_index       : varInst.s_ctl_v_index       ].reshape([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX-1]).astype(np.int)
            glb_s_pe_num      = glb_state[varInst.s_ctl_v_index     : varInst.s_pe_num_index      ].reshape([varInst.LY_LEN, 1]).astype(np.int)
            glb_s_buf_num     = glb_state[varInst.s_pe_num_index    : varInst.s_buf_num_index     ].reshape([varInst.LY_LEN, 1]).astype(np.int)
            glb_s_ly_idx      = glb_state[varInst.s_buf_num_index   : varInst.s_ly_idx_index      ].reshape([1, 1]).astype(np.int)
            glb_s_res_rtm     = glb_state[varInst.s_ly_idx_index    : varInst.s_res_rtm_index     ].reshape([varInst.LY_LEN, 1]).astype(np.int)
            glb_s_res_egy     = glb_state[varInst.s_res_rtm_index   : varInst.s_res_egy_index     ].reshape([varInst.LY_LEN, 1]).astype(np.int)
            glb_s_res_tpt     = glb_state[varInst.s_res_egy_index   : varInst.s_res_tpt_index     ].reshape([varInst.LY_LEN, 1])
            glb_s_ly_epl_cnt  = glb_state[varInst.s_res_tpt_index   : varInst.s_ly_epl_cnt_index  ].reshape([varInst.LY_LEN, 1])
            glb_s_res_area    = glb_state[varInst.s_ly_epl_cnt_index: varInst.s_res_area_index ].reshape([varInst.LY_LEN, 1])
            glb_s_res_power   = glb_state[varInst.s_res_area_index  : varInst.s_res_power_index   ].reshape([varInst.LY_LEN, 1])
            glb_s_res_egy_ncl = glb_state[varInst.s_res_power_index : varInst.s_res_egy_ncl_index ].reshape([varInst.LY_LEN, 1])
            glb_s_l2_regy     = self.s[varInst.s_res_egy_ncl_index  : varInst.s_l2_regy_index     ].reshape([varInst.LY_LEN, 1])
            glb_s_l2_wegy     = self.s[varInst.s_l2_regy_index      : varInst.s_l2_wegy_index     ].reshape([varInst.LY_LEN, 1])
            glb_s_l1_regy     = self.s[varInst.s_l2_wegy_index      : varInst.s_l1_regy_index     ].reshape([varInst.LY_LEN, 1])
            glb_s_l1_wegy     = self.s[varInst.s_l1_regy_index      : varInst.s_l1_wegy_index     ].reshape([varInst.LY_LEN, 1])
            glb_s_mac_egy     = self.s[varInst.s_l1_wegy_index      : varInst.s_mac_egy_index     ].reshape([varInst.LY_LEN, 1])
            glb_s_ddr_egy     = self.s[varInst.s_mac_egy_index      : varInst.s_ddr_egy_index     ].reshape([varInst.LY_LEN, 1])



            if varInst.opt_object_runtime_en == True:
                opt_object_array = glb_s_res_rtm < s_res_rtm_t 
            else:
                opt_object_array = glb_s_res_egy < s_res_egy_t 

            glb_s_dim_order_t   = np.where(opt_object_array, glb_s_dim_order, s_dim_order_t)     
            glb_s_type_t        = np.where(opt_object_array, glb_s_type, s_type_t)      
            glb_s_size_t        = np.where(opt_object_array, glb_s_size, s_size_t)      
            glb_s_offset_t      = np.where(opt_object_array, glb_s_offset, s_offset_t)    
            glb_s_cluster_v_t   = np.where(opt_object_array, glb_s_cluster_v, s_cluster_v_t) 
            glb_s_pe_num_t      = np.where(opt_object_array, glb_s_pe_num, s_pe_num_t)    
            glb_s_buf_num_t     = np.where(opt_object_array, glb_s_buf_num, s_buf_num_t)   
            glb_s_ly_idx_t      = 0 
            glb_s_res_rtm_t     = np.where(opt_object_array, glb_s_res_rtm, s_res_rtm_t)   
            glb_s_res_egy_t     = np.where(opt_object_array, glb_s_res_egy, s_res_egy_t)   
            glb_s_res_tpt_t     = np.where(opt_object_array, glb_s_res_tpt, s_res_tpt_t)   
            #glb_s_ly_epl_cnt_t  = np.where(opt_object_array, glb_s_ly_epl_cnt, s_ly_epl_cnt_t)
            glb_s_res_area_t    = np.where(opt_object_array, glb_s_res_area, s_res_area_t)
            glb_s_res_power_t   = np.where(opt_object_array, glb_s_res_power, s_res_power_t)
            glb_s_res_egy_ncl_t = np.where(opt_object_array, glb_s_res_egy_ncl, s_res_egy_ncl_t)   
            glb_s_l2_regy_t = np.where(opt_object_array, glb_s_l2_regy, s_l2_regy_t)   
            glb_s_l2_wegy_t = np.where(opt_object_array, glb_s_l2_wegy, s_l2_wegy_t)   
            glb_s_l1_regy_t = np.where(opt_object_array, glb_s_l1_regy, s_l1_regy_t)   
            glb_s_l1_wegy_t = np.where(opt_object_array, glb_s_l1_wegy, s_l1_wegy_t)   
            glb_s_mac_egy_t = np.where(opt_object_array, glb_s_mac_egy, s_mac_egy_t)   
            glb_s_ddr_egy_t = np.where(opt_object_array, glb_s_ddr_egy, s_ddr_egy_t)   

            glb_state = np.hstack((glb_s_dim_order_t.flatten(), glb_s_type_t.flatten(), glb_s_size_t.flatten(), 
                        glb_s_offset_t.flatten(), glb_s_cluster_v_t.flatten(), glb_s_pe_num_t.flatten(), glb_s_buf_num_t.flatten(), glb_s_ly_idx_t,
                        glb_s_res_rtm_t.flatten(),
                        glb_s_res_egy_t.flatten(),
                        glb_s_res_tpt_t.flatten(), 
                        glb_s_ly_epl_cnt_t.flatten(),
                        glb_s_res_area_t.flatten(),
                        glb_s_res_power_t.flatten(),
                        glb_s_res_egy_ncl_t.flatten(), 
                        glb_s_l2_regy_t.flatten(), 
                        glb_s_l2_wegy_t.flatten(), 
                        glb_s_l1_regy_t.flatten(), 
                        glb_s_l1_wegy_t.flatten(), 
                        glb_s_mac_egy_t.flatten(),
                        glb_s_ddr_egy_t.flatten()))

            glb_cnt = glb_cnt + 1
            if DEBUG_PLOT_FIG :
                plt_plot(glb_num_pe_used, 4)
                #plt_plot(glb_reuse_ifactor, 5)
                #plt_plot(glb_reuse_ofactor, 6)
                #plt_plot(glb_reuse_wfactor, 7)
                if varInst.opt_object_runtime_en == True:
                    plt_plot(glb_s_res_rtm_t, 1)
                else:
                    plt_plot(glb_s_res_egy_t, 1)

            if glb_cnt >= 5 and mcts_init_done_flag == True: 
                glb_cnt = 0
                tmps_s_ = glb_state
                net_runtime     = np.sum(glb_s_res_rtm_t)
                net_energy      = np.sum(glb_s_res_egy_t)
                net_energy_ncl  = np.sum(glb_s_res_egy_ncl_t)
                net_l2_regy     = np.sum(glb_s_l2_regy_t)
                net_l2_wegy     = np.sum(glb_s_l2_wegy_t)
                net_l1_regy     = np.sum(glb_s_l1_regy_t)
                net_l1_wegy     = np.sum(glb_s_l1_wegy_t)
                net_mac_egy     = np.sum(glb_s_mac_egy_t)
                net_ddr_egy     = np.sum(glb_s_ddr_egy_t)

                net_thoughput   = np.sum(glb_s_res_tpt_t)
                self.r = W_time_mcts * (100000/net_runtime) + W_enegy_mcts*(100000/net_energy) + W_throughput_mcts*net_thoughput
            

            global net_min_runtime
            global net_min_energy
            global net_min_energy_cly
            global net_min_l2_regy
            global net_min_l2_wegy
            global net_min_l1_regy
            global net_min_l1_wegy
            global net_min_mac_egy
            global net_min_ddr_egy

            global glb_no_valid_times_list  


            if net_min_runtime > net_runtime :
                net_min_runtime = net_runtime 
                net_min_energy_cly = net_energy_ncl
                net_min_l2_regy = net_l2_regy
                net_min_l2_wegy = net_l2_wegy
                net_min_l1_regy = net_l1_regy
                net_min_l1_wegy = net_l1_wegy
                net_min_mac_egy = net_mac_egy
                net_min_ddr_egy = net_ddr_egy
                if varInst.opt_object_runtime_en == True:
                    glb_no_valid_times_list.append(glb_no_valid_times_cnt)
                    #plt_plot(glb_no_valid_times_list, 2)
                    glb_no_valid_times_cnt = 0 

            if net_min_energy > net_energy:
                net_min_energy = net_energy
                net_min_energy_cly = net_energy_ncl
                net_min_l2_regy = net_l2_regy
                net_min_l2_wegy = net_l2_wegy
                net_min_l1_regy = net_l1_regy
                net_min_l1_wegy = net_l1_wegy
                net_min_mac_egy = net_mac_egy
                net_min_ddr_egy = net_ddr_egy
                if varInst.opt_object_runtime_en == False:
                    glb_no_valid_times_list.append(glb_no_valid_times_cnt)
                    #plt_plot(glb_no_valid_times_list, 2)
                    glb_no_valid_times_cnt = 0 

            if DEBUG_PRINT :
                pass
                #plt_plot(glb_s_ly_epl_cnt_t, 2)
            


            #if(mcts_init_done_flag==True):
            if varInst.opt_object_runtime_en == True:
                mcts_train_rtm.append(net_min_runtime)
            else:
                mcts_train_rtm.append(net_min_energy)
            if DEBUG_PLOT_FIG :
                plt_plot(mcts_train_rtm, 0)

            if DEBUG_PRINT == True:
                print("Min runtime: %d %d, Min energy: %d %d "%(net_min_runtime, np.sum(glb_s_res_rtm_t), net_min_energy, np.sum(glb_s_res_egy_t)))
                print("Cur runtime: %d, Cur energy: %d"%(net_runtime, net_energy))
                wFile("Min runtime: %d %d, Min energy: %d %d "%(net_min_runtime, np.sum(glb_s_res_rtm_t), net_min_energy, np.sum(glb_s_res_egy_t)), fileNameFull)
                wFile("Cur runtime: %d, Cur energy: %d"%(net_runtime, net_energy), fileNameFull)

            return tmp_s_, self.r, self.done

        else:
            self.r = PENALTY_MAESTRO_ERROR
            self.done = False
            if DEBUG_PRINT == True:
                print('Warning: runtime or energy or throughput is 0')
                wFile('Warning: runtime or energy or throughput is 0 \n',     fileNameFull)
            return tmp_s, self.r, self.done
    def mcts_save_data(self,):
        global glb_state
        glb_s_dim_order   = glb_state[0                         : varInst.s_dim_order_index   ].reshape([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX]).astype(np.int)
        glb_s_type        = glb_state[varInst.s_dim_order_index : varInst.s_type_index        ].reshape([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX*varInst.DIM_NUM]).astype(np.int)
        glb_s_size        = glb_state[varInst.s_type_index      : varInst.s_size_index        ].reshape([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX*varInst.DIM_NUM]).astype(np.int)
        glb_s_offset      = glb_state[varInst.s_size_index      : varInst.s_oft_index         ].reshape([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX*varInst.DIM_NUM]).astype(np.int)
        glb_s_cluster_v   = glb_state[varInst.s_oft_index       : varInst.s_ctl_v_index       ].reshape([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX-1]).astype(np.int)

        glb_s_pe_num      = glb_state[varInst.s_ctl_v_index     : varInst.s_pe_num_index      ].reshape([varInst.LY_LEN, 1]).astype(np.int)
        glb_s_buf_num     = glb_state[varInst.s_pe_num_index    : varInst.s_buf_num_index     ].reshape([varInst.LY_LEN, 1]).astype(np.int)
        glb_s_res_rtm     = glb_state[varInst.s_ly_idx_index    : varInst.s_res_rtm_index     ].reshape([varInst.LY_LEN, 1]).astype(np.int)
        glb_s_res_egy     = glb_state[varInst.s_res_rtm_index   : varInst.s_res_egy_index     ].reshape([varInst.LY_LEN, 1]).astype(np.int)
        glb_s_res_tpt     = glb_state[varInst.s_res_egy_index   : varInst.s_res_tpt_index     ].reshape([varInst.LY_LEN, 1])
        glb_s_res_area    = glb_state[varInst.s_ly_epl_cnt_index   : varInst.s_res_area_index ].reshape([varInst.LY_LEN, 1])
        glb_s_res_power   = glb_state[varInst.s_res_area_index  : varInst.s_res_power_index   ].reshape([varInst.LY_LEN, 1])
        if DEBUG_SAVE_CSV == True:
            save_np(glb_s_dim_order, alg_str='mcts', np_str='s_dim_order')
            save_np(glb_s_type, alg_str='mcts', np_str='s_type')
            save_np(glb_s_size, alg_str='mcts', np_str='s_size')
            save_np(glb_s_offset, alg_str='mcts', np_str='s_offset')
            save_np(glb_s_cluster_v, alg_str='mcts', np_str='s_cluster_v')

            save_np(glb_s_pe_num, alg_str='mcts', np_str='s_pe')
            save_np(glb_s_buf_num, alg_str='mcts', np_str='s_buf')
            save_np(glb_s_res_rtm, alg_str='mcts', np_str='rtm')
            save_np(glb_s_res_egy, alg_str='mcts', np_str='egy')
            save_np(glb_s_res_tpt, alg_str='mcts', np_str='tpt')
            save_np(glb_s_res_area, alg_str='mcts', np_str='area')
            save_np(glb_s_res_power, alg_str='mcts', np_str='power')
            if varInst.opt_object_runtime_en :
                save_np(mcts_train_rtm , alg_str='mcts', np_str='search_rtm')
            else:
                save_np(mcts_train_rtm , alg_str='mcts', np_str='search_egy')

    def _frender(self, fname, d):
        templ = self_env.get_template(fname)
        return templ.render(d)

    def _fwrite(self, f, fname):
        with open(fname,"w") as fw:
                fw.write(f)

    def _renderfromfile(self, tempname, d, outfname):
        self._fwrite(self._frender(tempname,d) ,outfname)

    def eval_func(  self,
                    ng_s_dim_order, 
                    ng_s_type,      
                    ng_s_dim_top,      
                    ng_s_size,      
                    ng_s_offset,    
                    ng_s_cluster_v, 
                    ng_s_pe_num,    
                    ng_s_buf_num,
                    ):
        global other_each_layer_pe_num
        global other_each_layer_buf_num
        global other_each_layer_rtm
        global other_each_layer_egy
        global other_each_layer_tpt
        global other_each_layer_area
        global other_each_layer_power
        global other_algrithm_init_done
        global other_algrithm_init_cnt
        global NetFileNameAll
        global net_power_other
        global net_area_other
        path_a = 'test.csv'  
        if os.path.exists(path_a):  
            os.remove(path_a)  
        net_runtime     = 0
        net_energy      = 0
        net_thoughput   = 0
        net_area_other        = 0
        net_power_other       = 0


        other_tmp_each_layer_pe_num = np.zeros([varInst.LY_LEN,])
        other_tmp_each_layer_buf_num = np.zeros([varInst.LY_LEN,])
        other_tmp_each_layer_rtm = np.ones([varInst.LY_LEN,]) * int(sys.maxsize/100000)
        other_tmp_each_layer_egy = np.ones([varInst.LY_LEN,]) * int(sys.maxsize/100000)
        other_tmp_each_layer_tpt = np.zeros([varInst.LY_LEN,])
        other_tmp_each_layer_area = np.zeros([varInst.LY_LEN,])
        other_tmp_each_layer_power = np.zeros([varInst.LY_LEN,])

        global net_min_runtime
        global net_min_energy
        other_algrithm_train_rtm.append(net_min_runtime)
        if DEBUG_PLOT_FIG == True:
            plt_plot(other_algrithm_train_rtm, 0)
        for tmp_ly_index in range(varInst.LY_LEN):
            if varInst.MODE_TOP_SPATIAL_O == False:
                if other_algrithm_init_done == False:
                    if( np.sum(ng_s_type[tmp_ly_index, 0 : varInst.act_dim_len]) > 2 ):  # check if spatial map is too much
                        self.r = PENALTY_TOO_MANY_OR_NO_SPATIAL_MAP
                        self.done = False
                        if DEBUG_PRINT == True:
                            print('Warning: Error: have too many Spatial map, more than >2 in cluster 0! it is : %d'% (np.sum(ng_s_type[tmp_ly_index, 0 : varInst.act_dim_len])))
                            wFile('Warning: Error: have too many Spatial map, more than >2 in cluster 0! it is : %d'% (np.sum(ng_s_type[tmp_ly_index, 0 : varInst.act_dim_len])),     fileNameFull)
                        return self.r
                    if varInst.CLUSTR_LVL_MEM_MAX_O == 2:
                        if( np.sum(ng_s_type[tmp_ly_index, varInst.act_dim_len : 2*varInst.act_dim_len]) > 2 ):  # check if spatial map is too much
                            self.r = PENALTY_TOO_MANY_OR_NO_SPATIAL_MAP
                            self.done = False
                            if DEBUG_PRINT == True:
                                print('Warning: Error: have too many Spatial map, more than >2 in cluster 1! it is'% (np.sum(ng_s_type[tmp_ly_index, varInst.act_dim_len : 2*varInst.act_dim_len])))
                                wFile('Warning: Error: have too many Spatial map, more than >2 in cluster 1! it is'% (np.sum(ng_s_type[tmp_ly_index, varInst.act_dim_len : 2*varInst.act_dim_len])), fileNameFull)
                                #print(ng_s_type[tmp_ly_index, varInst.act_dim_len : 2*varInst.act_dim_len])
                            return self.r
                    if( np.sum(ng_s_type[tmp_ly_index, 0 : varInst.act_dim_len]) == 0 ):
                        self.r = PENALTY_NO_SPATIAL_MAP
                        self.done = False
                        if DEBUG_PRINT == True:
                            print('Warning: Error: have No Spatial map, less than <1!')
                            wFile('Warning: Error: have No Spatial map, less than <1!', fileNameFull)
                        return self.r
                    if varInst.CLUSTR_LVL_MEM_MAX_O == 2:
                        if( np.sum(ng_s_type[tmp_ly_index, varInst.act_dim_len : 2*varInst.act_dim_len]) == 0 ):
                            self.r = PENALTY_NO_SPATIAL_MAP
                            self.done = False
                            if DEBUG_PRINT == True:
                                print('Warning: Error: have No Spatial map, less than <1!')
                                wFile('Warning: Error: have No Spatial map, less than <1!', fileNameFull)
                            return self.r

            else:
                ng_s_type = np.ones([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX_O*varInst.DIM_NUM]) 

                dim_top_index0 = np.asscalar(ng_s_dim_top[tmp_ly_index, 0].astype(int)) 
                ng_s_type[tmp_ly_index, 0 : varInst.act_dim_len] = np.zeros([varInst.act_dim_len,], np.int)   # restet all dim to Temperal map
                ng_s_type[tmp_ly_index, dim_top_index0]      = 1                                        # set some dim to Spatial map

                dim_top_index1 = np.asscalar(ng_s_dim_top[tmp_ly_index, 1].astype(int))
                ng_s_type[tmp_ly_index, varInst.act_dim_len : 2*varInst.act_dim_len] = np.zeros([varInst.act_dim_len,], np.int)   # restet all dim to Temperal map
                ng_s_type[tmp_ly_index, dim_top_index1 + varInst.act_dim_len]      = 1                                        # set some dim to Spatial map

            
            #DIM_ORDER        = [256, 128, 64, 32, 16, 8, 7, 5, 4, 3]
            DIM_ORDER        = [i for i in range(varInst.dim_order_num)]
            ly_dim_order_index0 = np.asscalar(ng_s_dim_order[tmp_ly_index, 0].astype(int))
            ly_dim_order_value0 = DIM_ORDER[ly_dim_order_index0] 

            ly_dim_order_index1 = np.asscalar(ng_s_dim_order[tmp_ly_index, 1].astype(int))
            ly_dim_order_value1 = DIM_ORDER[ly_dim_order_index1] 

            ng_dim_order_str_list0 = self_dim_perm[ly_dim_order_value0] 
            ng_dim_order_str_list1 = self_dim_perm[ly_dim_order_value1] 
            # change the type size offset to new order
            ng_s_type[tmp_ly_index, 0:varInst.act_dim_len]   = self.array_change_pos(varInst.DIM_NAME_LIST, ng_dim_order_str_list0, ng_s_type[tmp_ly_index,   0:varInst.act_dim_len])
            ng_s_size[tmp_ly_index, 0:varInst.act_dim_len]   = self.array_change_pos(varInst.DIM_NAME_LIST, ng_dim_order_str_list0, ng_s_size[tmp_ly_index,   0:varInst.act_dim_len])
            ng_s_offset[tmp_ly_index, 0:varInst.act_dim_len] = self.array_change_pos(varInst.DIM_NAME_LIST, ng_dim_order_str_list0, ng_s_offset[tmp_ly_index, 0:varInst.act_dim_len])
            if varInst.CLUSTR_LVL_MEM_MAX_O == 2:
                ng_s_type[tmp_ly_index, varInst.act_dim_len:2*varInst.act_dim_len]   = self.array_change_pos(varInst.DIM_NAME_LIST, ng_dim_order_str_list1, ng_s_type[tmp_ly_index,   varInst.act_dim_len:2*varInst.act_dim_len])
                ng_s_size[tmp_ly_index, varInst.act_dim_len:2*varInst.act_dim_len]   = self.array_change_pos(varInst.DIM_NAME_LIST, ng_dim_order_str_list1, ng_s_size[tmp_ly_index,   varInst.act_dim_len:2*varInst.act_dim_len])
                ng_s_offset[tmp_ly_index, varInst.act_dim_len:2*varInst.act_dim_len] = self.array_change_pos(varInst.DIM_NAME_LIST, ng_dim_order_str_list1, ng_s_offset[tmp_ly_index, varInst.act_dim_len:2*varInst.act_dim_len])
            while(varInst.IS_DSCONV[tmp_ly_index] == 1 and ng_dim_order_str_list0[dim_top_index0]=="K"):
                dim_top_index0 = random.randint(0,varInst.DIM_NUM-1)
            while(varInst.IS_DSCONV[tmp_ly_index] == 1 and ng_dim_order_str_list1[dim_top_index1]=="K"):
                dim_top_index1 = random.randint(0,varInst.DIM_NUM-1)


            ng_s_type[tmp_ly_index, 0 : varInst.act_dim_len] = np.zeros([varInst.act_dim_len,], np.int)   # restet all dim to Temperal map
            ng_s_type[tmp_ly_index, dim_top_index0]      = 1                                        # set some dim to Spatial map

            ng_s_type[tmp_ly_index, varInst.act_dim_len : 2*varInst.act_dim_len] = np.zeros([varInst.act_dim_len,], np.int)   # restet all dim to Temperal map
            ng_s_type[tmp_ly_index, dim_top_index1 + varInst.act_dim_len]      = 1                                        # set some dim to Spatial map


            X_size = ng_s_size[tmp_ly_index, ng_dim_order_str_list0.index("X")] 
            S_size = ng_s_size[tmp_ly_index, ng_dim_order_str_list0.index("S")] 
            Y_size = ng_s_size[tmp_ly_index, ng_dim_order_str_list0.index("Y")] 
            R_size = ng_s_size[tmp_ly_index, ng_dim_order_str_list0.index("R")] 
            C_size = ng_s_size[tmp_ly_index, ng_dim_order_str_list0.index("C")] 
            K_size = ng_s_size[tmp_ly_index, ng_dim_order_str_list0.index("K")] 
            if other_algrithm_init_done == False:
                if(X_size <= S_size+1):
                    ng_s_size[tmp_ly_index,ng_dim_order_str_list0.index("X")] = S_size + 2 
                if(Y_size <= R_size+1):
                    ng_s_size[tmp_ly_index,ng_dim_order_str_list0.index("Y")] = R_size + 2 

            #if(X_size > varInst.X[tmp_ly_index]):
            #    ng_s_size[tmp_ly_index,ng_dim_order_str_list0.index("X")] = varInst.X[tmp_ly_index]
            #if(S_size > varInst.S[tmp_ly_index]):
            #    ng_s_size[tmp_ly_index,ng_dim_order_str_list0.index("S")] = varInst.S[tmp_ly_index]
            #if(Y_size > varInst.Y[tmp_ly_index]):
            #    ng_s_size[tmp_ly_index,ng_dim_order_str_list0.index("Y")] = varInst.Y[tmp_ly_index]
            #if(R_size > varInst.R[tmp_ly_index]):
            #    ng_s_size[tmp_ly_index,ng_dim_order_str_list0.index("R")] = varInst.R[tmp_ly_index]
            #if(C_size > varInst.C[tmp_ly_index]):
            #    ng_s_size[tmp_ly_index,ng_dim_order_str_list0.index("C")] = varInst.C[tmp_ly_index]
            #if(K_size > varInst.K[tmp_ly_index]):
            #    ng_s_size[tmp_ly_index,ng_dim_order_str_list0.index("K")] = varInst.K[tmp_ly_index]


            t = time.time()
            fileName_t = (int(round(t * 1000000))) 
            X_index_cl0 = ng_dim_order_str_list0.index("X") 
            S_index_cl0 = ng_dim_order_str_list0.index("S") 
            Y_index_cl0 = ng_dim_order_str_list0.index("Y") 
            R_index_cl0 = ng_dim_order_str_list0.index("R") 
            C_index_cl0 = ng_dim_order_str_list0.index("C") 
            K_index_cl0 = ng_dim_order_str_list0.index("K") 

            X_size_cl0 = ng_s_size[tmp_ly_index, X_index_cl0] 
            S_size_cl0 = ng_s_size[tmp_ly_index, S_index_cl0] 
            Y_size_cl0 = ng_s_size[tmp_ly_index, Y_index_cl0] 
            R_size_cl0 = ng_s_size[tmp_ly_index, R_index_cl0] 
            C_size_cl0 = ng_s_size[tmp_ly_index, C_index_cl0] 
            K_size_cl0 = ng_s_size[tmp_ly_index, K_index_cl0] 


            if other_algrithm_init_done == False:
                if(X_size_cl0 > varInst.X[tmp_ly_index]):
                    ng_s_size[tmp_ly_index, X_index_cl0] = varInst.X[tmp_ly_index]
                if(S_size_cl0 > varInst.S[tmp_ly_index]):
                    ng_s_size[tmp_ly_index, S_index_cl0] = varInst.S[tmp_ly_index]
                if(Y_size_cl0 > varInst.Y[tmp_ly_index]):
                    ng_s_size[tmp_ly_index, Y_index_cl0] = varInst.Y[tmp_ly_index]
                if(R_size_cl0 > varInst.R[tmp_ly_index]):
                    ng_s_size[tmp_ly_index, R_index_cl0] = varInst.R[tmp_ly_index]
                if(C_size_cl0 > varInst.C[tmp_ly_index]):
                    ng_s_size[tmp_ly_index, C_index_cl0] = varInst.C[tmp_ly_index]
                if(K_size_cl0 > varInst.K[tmp_ly_index]):
                    ng_s_size[tmp_ly_index, K_index_cl0] = varInst.K[tmp_ly_index]

            if varInst.CLUSTR_LVL_MEM_MAX_O == 2:


                X_index_cl1 = varInst.act_dim_len + ng_dim_order_str_list1.index("X") 
                S_index_cl1 = varInst.act_dim_len + ng_dim_order_str_list1.index("S") 
                Y_index_cl1 = varInst.act_dim_len + ng_dim_order_str_list1.index("Y") 
                R_index_cl1 = varInst.act_dim_len + ng_dim_order_str_list1.index("R") 
                C_index_cl1 = varInst.act_dim_len + ng_dim_order_str_list1.index("C") 
                K_index_cl1 = varInst.act_dim_len + ng_dim_order_str_list1.index("K") 

                X_size_cl1 = ng_s_size[tmp_ly_index, X_index_cl1] 
                S_size_cl1 = ng_s_size[tmp_ly_index, S_index_cl1] 
                Y_size_cl1 = ng_s_size[tmp_ly_index, Y_index_cl1] 
                R_size_cl1 = ng_s_size[tmp_ly_index, R_index_cl1] 
                C_size_cl1 = ng_s_size[tmp_ly_index, C_index_cl1] 
                K_size_cl1 = ng_s_size[tmp_ly_index, K_index_cl1] 

                if other_algrithm_init_done == False:
                    if(X_size_cl1 > varInst.X[tmp_ly_index]):
                        ng_s_size[tmp_ly_index, X_index_cl1] = varInst.X[tmp_ly_index]
                    if(S_size_cl1 > varInst.S[tmp_ly_index]):
                        ng_s_size[tmp_ly_index, S_index_cl1] = varInst.S[tmp_ly_index]
                    if(Y_size_cl1 > varInst.Y[tmp_ly_index]):
                        ng_s_size[tmp_ly_index, Y_index_cl1] = varInst.Y[tmp_ly_index]
                    if(R_size_cl1 > varInst.R[tmp_ly_index]):
                        ng_s_size[tmp_ly_index, R_index_cl1] = varInst.R[tmp_ly_index]
                    if(C_size_cl1 > varInst.C[tmp_ly_index]):
                        ng_s_size[tmp_ly_index, C_index_cl1] = varInst.C[tmp_ly_index]
                    if(K_size_cl1 > varInst.K[tmp_ly_index]):
                        ng_s_size[tmp_ly_index, K_index_cl1] = varInst.K[tmp_ly_index]


                X_size_cl0 = ng_s_size[tmp_ly_index, X_index_cl0] 
                S_size_cl0 = ng_s_size[tmp_ly_index, S_index_cl0] 
                Y_size_cl0 = ng_s_size[tmp_ly_index, Y_index_cl0] 
                R_size_cl0 = ng_s_size[tmp_ly_index, R_index_cl0] 
                C_size_cl0 = ng_s_size[tmp_ly_index, C_index_cl0] 
                K_size_cl0 = ng_s_size[tmp_ly_index, K_index_cl0] 
                X_size_cl1 = ng_s_size[tmp_ly_index, X_index_cl1] 
                S_size_cl1 = ng_s_size[tmp_ly_index, S_index_cl1] 
                Y_size_cl1 = ng_s_size[tmp_ly_index, Y_index_cl1] 
                R_size_cl1 = ng_s_size[tmp_ly_index, R_index_cl1] 
                C_size_cl1 = ng_s_size[tmp_ly_index, C_index_cl1] 
                K_size_cl1 = ng_s_size[tmp_ly_index, K_index_cl1] 

                # >= not work!!

                if other_algrithm_init_done == False:
                    if ((ng_s_type[tmp_ly_index, X_index_cl1]==0) and (X_size_cl1 > X_size_cl0)):
                        ng_s_size[tmp_ly_index,X_index_cl0] = 2 
                        ng_s_size[tmp_ly_index,X_index_cl1] = 1 
                    if ((ng_s_type[tmp_ly_index, S_index_cl1]==0) and (S_size_cl1 > S_size_cl0)):
                        ng_s_size[tmp_ly_index,S_index_cl0] = 2 
                        ng_s_size[tmp_ly_index,S_index_cl1] = 1 
                    if ((ng_s_type[tmp_ly_index, Y_index_cl1]==0) and (Y_size_cl1 > Y_size_cl0)):
                        ng_s_size[tmp_ly_index,Y_index_cl0] = 2 
                        ng_s_size[tmp_ly_index,Y_index_cl1] = 1 
                    if ((ng_s_type[tmp_ly_index, R_index_cl1]==0) and (R_size_cl1 > R_size_cl0)):
                        ng_s_size[tmp_ly_index,R_index_cl0] = 2 
                        ng_s_size[tmp_ly_index,R_index_cl1] = 1 
                    if ((ng_s_type[tmp_ly_index, C_index_cl1]==0) and (C_size_cl1 > C_size_cl0)):
                        ng_s_size[tmp_ly_index,C_index_cl0] = 2 
                        ng_s_size[tmp_ly_index,C_index_cl1] = 1 
                    if ((ng_s_type[tmp_ly_index, K_index_cl1]==0) and (K_size_cl1 > K_size_cl0)):
                        ng_s_size[tmp_ly_index,K_index_cl0] = 2
                        ng_s_size[tmp_ly_index,K_index_cl1] = 1 

                X_size_cl0 = ng_s_size[tmp_ly_index, X_index_cl0] 
                S_size_cl0 = ng_s_size[tmp_ly_index, S_index_cl0] 
                Y_size_cl0 = ng_s_size[tmp_ly_index, Y_index_cl0] 
                R_size_cl0 = ng_s_size[tmp_ly_index, R_index_cl0] 
                C_size_cl0 = ng_s_size[tmp_ly_index, C_index_cl0] 
                K_size_cl0 = ng_s_size[tmp_ly_index, K_index_cl0] 
                X_size_cl1 = ng_s_size[tmp_ly_index, X_index_cl1] 
                S_size_cl1 = ng_s_size[tmp_ly_index, S_index_cl1] 
                Y_size_cl1 = ng_s_size[tmp_ly_index, Y_index_cl1] 
                R_size_cl1 = ng_s_size[tmp_ly_index, R_index_cl1] 
                C_size_cl1 = ng_s_size[tmp_ly_index, C_index_cl1] 
                K_size_cl1 = ng_s_size[tmp_ly_index, K_index_cl1] 

                if other_algrithm_init_done == False:
                    if ((ng_s_type[tmp_ly_index, X_index_cl1]==0) and X_size_cl1 == X_size_cl0 and varInst.X[tmp_ly_index] != 1):
                        ng_s_size[tmp_ly_index,X_index_cl0] = 2 
                        ng_s_size[tmp_ly_index,X_index_cl1] = 1 
                    if ((ng_s_type[tmp_ly_index, S_index_cl1]==0) and S_size_cl1 == S_size_cl0 and varInst.S[tmp_ly_index] != 1):
                        ng_s_size[tmp_ly_index,S_index_cl0] = 2 
                        ng_s_size[tmp_ly_index,S_index_cl1] = 1 
                    if ((ng_s_type[tmp_ly_index, Y_index_cl1]==0) and Y_size_cl1 == Y_size_cl0 and varInst.Y[tmp_ly_index] != 1):
                        ng_s_size[tmp_ly_index,Y_index_cl0] = 2 
                        ng_s_size[tmp_ly_index,Y_index_cl1] = 1 
                    if ((ng_s_type[tmp_ly_index, R_index_cl1]==0) and R_size_cl1 == R_size_cl0 and varInst.R[tmp_ly_index] != 1):
                        ng_s_size[tmp_ly_index,R_index_cl0] = 2 
                        ng_s_size[tmp_ly_index,R_index_cl1] = 1 
                    if ((ng_s_type[tmp_ly_index, C_index_cl1]==0) and C_size_cl1 == C_size_cl0 and varInst.C[tmp_ly_index] != 1):
                        ng_s_size[tmp_ly_index,C_index_cl0] = 2 
                        ng_s_size[tmp_ly_index,C_index_cl1] = 1 
                    if ((ng_s_type[tmp_ly_index, K_index_cl1]==0) and K_size_cl1 == K_size_cl0 and varInst.K[tmp_ly_index] != 1):
                        ng_s_size[tmp_ly_index,K_index_cl0] = 2
                        ng_s_size[tmp_ly_index,K_index_cl1] = 1 

                X_size_cl0 = ng_s_size[tmp_ly_index, X_index_cl0] 
                S_size_cl0 = ng_s_size[tmp_ly_index, S_index_cl0] 
                Y_size_cl0 = ng_s_size[tmp_ly_index, Y_index_cl0] 
                R_size_cl0 = ng_s_size[tmp_ly_index, R_index_cl0] 
                C_size_cl0 = ng_s_size[tmp_ly_index, C_index_cl0] 
                K_size_cl0 = ng_s_size[tmp_ly_index, K_index_cl0] 
                X_size_cl1 = ng_s_size[tmp_ly_index, X_index_cl1] 
                S_size_cl1 = ng_s_size[tmp_ly_index, S_index_cl1] 
                Y_size_cl1 = ng_s_size[tmp_ly_index, Y_index_cl1] 
                R_size_cl1 = ng_s_size[tmp_ly_index, R_index_cl1] 
                C_size_cl1 = ng_s_size[tmp_ly_index, C_index_cl1] 
                K_size_cl1 = ng_s_size[tmp_ly_index, K_index_cl1] 
            

            #CLT_V        = [256, 128, 64, 32, 16, 8, 7, 5, 4, 3]
            #CLT_V        = [i for i in range(3, 256)]
            PE_NUM       = [2**i for i in range(10, 20)] 
            BUF_NUM      = [2**i for i in range(35, 40)] # 16k, 64k, 256k 

            ly_cluster_value_index = np.asscalar(ng_s_cluster_v[tmp_ly_index].astype(int))           
            #ly_cluster_value = CLT_V[ly_cluster_value_index] 
            ly_cluster_value = ly_cluster_value_index 

            ly_pe_value_index      =    np.asscalar(ng_s_pe_num[tmp_ly_index].astype(int))
            #eval_ly_pe_num         =    PE_NUM[ly_pe_value_index] 
            eval_ly_pe_num         =    ly_pe_value_index 

            ly_buf_value_index     =    np.asscalar(ng_s_buf_num[tmp_ly_index].astype(int))
            #eval_ly_buf_num        =    BUF_NUM[ly_buf_value_index] 
            eval_ly_buf_num        =    ly_buf_value_index 

            if other_algrithm_init_done == False:
                if(varInst.K[tmp_ly_index]==1 or varInst.C[tmp_ly_index]==1 or varInst.R[tmp_ly_index]==1 or varInst.S[tmp_ly_index]==1 or varInst.Y[tmp_ly_index]==1 or varInst.X[tmp_ly_index]==1):
                    cluster_lvl_t =  1  
                else:
                    cluster_lvl_t = varInst.CLUSTR_LVL_MEM_MAX_O
            else:
                cluster_lvl_t = varInst.CLUSTR_LVL_MEM_MAX_O

            net_para = {
                        'cluster_lvl'    : cluster_lvl_t,
                        's_type0'        : ng_s_type.astype(int).tolist(),
                        'size0'          : ng_s_size.astype(int).tolist(),
                        'offset0'        : ng_s_offset.astype(int).tolist(),
                        'dim0'           : ng_dim_order_str_list0,
                        'dim1'           : ng_dim_order_str_list1,
                        's_ly_idx'       : tmp_ly_index,
                        'cluster_value'  : ly_cluster_value,
                        'net_name'       : varInst.net_name,
                        'ly_type'        : varInst.LY_TYPE,
                        'ly_name'        : varInst.ly_n_list,
                        'stride_x'       : varInst.stride_x,
                        'stride_y'       : varInst.stride_y,
                        'K'              : varInst.K,
                        'C'              : varInst.C,
                        'R'              : varInst.R,
                        'S'              : varInst.S,
                        'Y'              : varInst.Y,
                        'X'              : varInst.X,
                        'is_dsconv': varInst.IS_DSCONV,
                        'time': fileName_t,
                        'en_ddr'                : varInst.EN_DDR,
                        'en_cross_ly'           : varInst.EN_CROSS_LY
                        }

            net_para['ly_idx'] = tmp_ly_index 
            self._renderfromfile(template_other_path, net_para, base_path + dataflow_output_path)
                
            if varInst.CLUSTR_LVL_MEM_MAX_O == 2:
                if(ly_cluster_value > eval_ly_pe_num):
                    eval_ly_pe_num = ly_cluster_value 
            a = time.time()
            val = [0 for i in range(8)]
            try:
                val = maestro.main_m(int(eval_ly_pe_num))
            except Exception as e:
                #print(eval_ly_pe_num, ly_cluster_value)

                if DEBUG_PRINT == True:
                    print(e)
                #wFile('exception', fileNameFull)
                val[2] = -1
                #cpTestDataFlowFileExp()

                self.r = PENALTY_MAESTRO_ERROR
                self.done = False
                if DEBUG_PRINT == True:
                    print('Warning: Maestro Exception')
                    wFile('Warning: Maestro Exception', fileNameFull)
                return self.r
            b = time.time()
            #print(b-a)
            #wFile("%f" % (b-a), fileNameFull)

            runtime     = val[0]             
            energy      = val[1]
            throughput  = val[2]
            computation = val[3]
            l1_size     = val[4]
            l2_size     = val[5]
            area        = val[6]
            power       = val[7]

            if(self.is_nan(runtime)==True or self.is_nan(energy)==True or self.is_nan(throughput)==True):
                self.r = PENALTY_MAESTRO_ERROR
                self.done = False
                if DEBUG_PRINT == True:
                    print('Warning: Maestro Error')
                    wFile('Warning: Maestro Error', fileNameFull)
                    if(energy>0):
                        #cpTestDataFlowFile_float_false_pos()
                        pass
                    elif(energy<0):
                        #cpTestDataFlowFile_float_false_neg()
                        pass
                    #else:
                    #    xx
                return self.r


            actual_ly_buf_size = l1_size
            if(actual_ly_buf_size > eval_ly_buf_num): 
                self.r = PENALTY_MAESTRO_ERROR 
                if DEBUG_PRINT == True:
                    print('Maestro Buffer size overflow layer: %s expect: %d actural: %d' % (varInst.ly_n_list[tmp_ly_index], eval_ly_buf_num, actual_ly_buf_size))
                    wFile('Maestro Buffer size overflow layer: %s expect: %d actural: %d' % (varInst.ly_n_list[tmp_ly_index], eval_ly_buf_num, actual_ly_buf_size), fileNameFull)
                return self.r
            

            if(runtime > 0 and energy > 0 and throughput > 0):
                net_runtime     =  net_runtime + runtime 
                net_energy      =  net_energy  + energy 
                net_thoughput   =  net_thoughput + throughput

                other_tmp_each_layer_pe_num[tmp_ly_index] =   eval_ly_pe_num
                other_tmp_each_layer_buf_num[tmp_ly_index] =   eval_ly_buf_num
                other_tmp_each_layer_rtm[tmp_ly_index] = runtime
                other_tmp_each_layer_egy[tmp_ly_index] = energy
                other_tmp_each_layer_tpt[tmp_ly_index] = throughput
                other_tmp_each_layer_area[tmp_ly_index] = area
                other_tmp_each_layer_power[tmp_ly_index] = power

                net_area_other        =  area
                net_power_other       =  power

                global contr_area
                global contr_power 
                if(other_algrithm_init_done == True and varInst.contr_en_area==True and net_area_other > contr_area[varInst.contr_nm]):
                    if DEBUG_PRINT :
                        print('Constrin area out range, actral: %d, contrain: %d '% (net_area_other,contr_area[varInst.contr_nm]))
                        wFile('Constrin area out range, actral: %d, contrain: %d '% (net_area_other,contr_area[varInst.contr_nm]), fileNameFull)
                    self.r = PENALTY_VALUE_OUT_RANG
                    return self.r
                if(other_algrithm_init_done == True and varInst.contr_en_power==True and net_power_other > contr_power[varInst.contr_nm]):
                    if DEBUG_PRINT :
                        print('Constrin power out range, actral: %d, contrain: %d '% (net_power_other, contr_power[varInst.contr_nm]))
                        wFile('Constrin power out range, actral: %d, contrain: %d '% (net_power_other, contr_power[varInst.contr_nm]), fileNameFull)
                    self.r = PENALTY_VALUE_OUT_RANG
                    return self.r
                continue
            else:
                self.r = PENALTY_MAESTRO_ERROR
                self.done = False
                if DEBUG_PRINT == True:
                    print('Warning: Maestro Error')
                    wFile('Warning: Maestro Error', fileNameFull)
                return self.r

            #print(net_runtime, net_energy)
            #print("Got a new dataflow successfully !")

        runtime = net_runtime     
        energy = net_energy      
        throughput = net_thoughput   
        if(self.is_nan(runtime)==True or self.is_nan(energy)==True or self.is_nan(throughput)==True):
            self.r = PENALTY_MAESTRO_ERROR
            self.done = True
            if DEBUG_PRINT == True:
                print('Warning: runtime or energy or throughput is Nan')
                #wFile('Warning: runtime or energy or throughput is Nan', fileNameFull)
            return self.r

        if(runtime > 0 and energy > 0 and throughput > 0):
            #self.r = W_time * (1/runtime) + W_enegy*(1/energy) + W_throughput*throughput
            self.r = W_time_other_methd * (runtime) + W_enegy_other_methd*(energy) 
            net_min_runtime = net_runtime if net_min_runtime >= net_runtime else net_min_runtime
            net_min_energy  = net_energy  if net_min_energy >= net_energy else net_min_energy
            if DEBUG_PRINT == True:
                print("Min runtime: %d, Min energy: %d"%(net_min_runtime, net_min_energy))
                print("Cur runtime: %d, Cur energy: %d \n"%(net_runtime, net_energy))
                wFile("Min runtime: %d, Min energy: %d"%(net_min_runtime, net_min_energy), fileNameFull)
                wFile("Cur runtime: %d, Cur energy: %d \n"%(net_runtime, net_energy), fileNameFull)

            other_algrithm_init_cnt = other_algrithm_init_cnt + 1   
            if DEBUG_PRINT == True:
                print(other_algrithm_init_cnt)
            if(other_algrithm_init_cnt > 15):
                other_algrithm_init_done = True

            if varInst.opt_object_runtime_en == True and np.sum(other_each_layer_rtm) > runtime:
                other_each_layer_pe_num = other_tmp_each_layer_pe_num
                other_each_layer_buf_num = other_tmp_each_layer_buf_num
                other_each_layer_rtm = other_tmp_each_layer_rtm
                other_each_layer_egy = other_tmp_each_layer_egy
                other_each_layer_tpt = other_tmp_each_layer_tpt
                other_each_layer_area = other_tmp_each_layer_area
                other_each_layer_power = other_tmp_each_layer_power
            elif varInst.opt_object_runtime_en == False and np.sum(other_each_layer_egy) > energy: 
                other_each_layer_pe_num = other_tmp_each_layer_pe_num
                other_each_layer_buf_num = other_tmp_each_layer_buf_num
                other_each_layer_rtm = other_tmp_each_layer_rtm
                other_each_layer_egy = other_tmp_each_layer_egy
                other_each_layer_tpt = other_tmp_each_layer_tpt
                other_each_layer_area = other_tmp_each_layer_area
                other_each_layer_power = other_tmp_each_layer_power


            self.done = True
            return self.r
        else:
            self.r = PENALTY_MAESTRO_ERROR
            self.done = True
            if DEBUG_PRINT == True:
                print('Warning: runtime or energy or throughput is 0')
                wFile('Warning: runtime or energy or throughput is 0', fileNameFull)
            return self.r
    def other_save_data(self, alg_str=''):
        global other_each_layer_pe_num
        global other_each_layer_buf_num
        global other_each_layer_rtm
        global other_each_layer_egy
        global other_each_layer_tpt
        global other_each_layer_area
        global other_each_layer_power
        global other_algrithm_train_rtm
        if DEBUG_SAVE_CSV == True:
            save_np(other_each_layer_pe_num,  alg_str='other_'+alg_str, np_str='pe')
            save_np(other_each_layer_buf_num, alg_str='other_'+alg_str, np_str='buf')
            save_np(other_each_layer_rtm,     alg_str='other_'+alg_str, np_str='rtm')
            save_np(other_each_layer_egy,     alg_str='other_'+alg_str, np_str='egy')
            save_np(other_each_layer_tpt,     alg_str='other_'+alg_str, np_str='tpt')
            save_np(other_each_layer_area,    alg_str='other_'+alg_str, np_str='area')
            save_np(other_each_layer_power,   alg_str='other_'+alg_str, np_str='power')
            if varInst.opt_object_runtime_en :
                save_np(other_algrithm_train_rtm, alg_str='other_'+alg_str, np_str='search_rtm')
            else:
                save_np(other_algrithm_train_rtm, alg_str='other_'+alg_str, np_str='search_egy')

    def other_reset_data(self, layer_len):
        global other_each_layer_pe_num
        global other_each_layer_buf_num
        global other_each_layer_rtm
        global other_each_layer_egy
        global other_each_layer_tpt
        global other_each_layer_area
        global other_each_layer_power

        other_each_layer_pe_num = np.zeros([int(layer_len),])
        other_each_layer_buf_num = np.zeros([int(layer_len),])
        other_each_layer_rtm = np.ones([int(layer_len),]) * int(sys.maxsize/100000)
        other_each_layer_egy = np.ones([int(layer_len),]) * int(sys.maxsize/100000)
        other_each_layer_tpt = np.zeros([int(layer_len),])
        other_each_layer_area = np.zeros([int(layer_len),])
        other_each_layer_power = np.zeros([int(layer_len),])

class Node():
    def __init__(self, state, parent=None):   
        self.visits=1                           # visit time
        self.reward=0.0                         # reward 
        self.state=state                        # current state
        self.children=[]                        # children is a list
        self.parent=parent                      # parent node

    def add_child(self, child_state):           # add a chidren 
        child=Node(child_state, self)           # 
        self.children.append(child)

    def update(self,reward):                    # no use
        self.reward+=reward
        self.visits+=1

    def fully_expanded(self):                   # all the moves have expanded
        if len(self.children)==self.state.num_moves:
            return True
        return False

    def __repr__(self):
        s="Node; children: %d; visits: %d; reward: %f"%(len(self.children),self.visits, self.reward)
        return s
        


def UCTSEARCH(budget, root):                    # uct search
    for iter in range(int(budget)):             # search, expand, simulation, backup times
        #print("iter: %d" % iter)
        if iter%10000==9999:
            logger.info("simulation: %d"%iter)
            logger.info(root)
        front=TREEPOLICY(root)                  # search
        reward=DEFAULTPOLICY(front.state)       # expand and simulation 
        BACKUP(front,reward)                    # backup, update the tree
    return BESTCHILD(root,0)

def TREEPOLICY(node):                           # tree search 
    #a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
    while node.state.terminal() == False:       # if the state is not the terminal state
        if len(node.children) == 0:             # if have not the children, then expand a children, and return the only one
            return EXPAND(node)
        elif random.uniform(0,1) < .5:          # 50% come to best children, 50% expand at a new children (expand)
            node = BESTCHILD(node, SCALAR)        
        else:
            if node.fully_expanded() == False:  # expand if not full expanded  
                return EXPAND(node)
            else:
                node=BESTCHILD(node, SCALAR)    # come to best children
    return node
    
def EXPAND(node):                               # at a node which have children, get a new one than is differen 
    tried_children=[c.state for c in node.children]  # all the children that have tried
    new_state=node.state.next_state(decay_turn=False)                # random a action(comt to new children)
    while new_state in tried_children:               # make sure the random action not in the tried, else random a new one
        new_state=node.state.next_state(decay_turn=False)
    node.add_child(new_state)                        # accept the new children
    return node.children[-1]                         # return the new children

#current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def BESTCHILD(node, scalar):                         # search all the children of the node
    bestscore = -sys.maxsize - 1
    bestchildren=[]
    for c in node.children:
        exploit = c.reward/c.visits
        explore = math.sqrt(2.0*math.log(node.visits)/float(c.visits))    
        score = exploit+scalar*explore
        if score == bestscore:                       # record same score node 
            bestchildren.append(c)
        if score > bestscore:                        # record top score and children
            bestchildren = [c]
            bestscore = score
    if len(bestchildren) == 0:                       # if no best children the probaly is fatal
        logger.warn("OOPS: no best child found, probably fatal")
    return random.choice(bestchildren)#bestchildren, bestscore

def DEFAULTPOLICY(state):                            # random select a action
    while state.terminal()==False:
        state=state.next_state(decay_turn=True)
    return state.reward()

def BACKUP(node,reward):                             # update all the info of expirence
    while node!=None:
        node.visits+=1
        node.reward+=reward
        node=node.parent
    return


if __name__=="__main__":

    all_result_rtm = np.zeros([len(NetFileNameAll), len(other_algrithm)+1 ])
    all_result_egy = np.zeros([len(NetFileNameAll), len(other_algrithm)+1 ])
    all_result_egy_ncl = np.zeros([len(NetFileNameAll), len(other_algrithm)+1 ])
    all_result_l2_regy = np.zeros([len(NetFileNameAll), len(other_algrithm)+1 ])
    all_result_l2_wegy = np.zeros([len(NetFileNameAll), len(other_algrithm)+1 ])
    all_result_l1_regy = np.zeros([len(NetFileNameAll), len(other_algrithm)+1 ])
    all_result_l1_wegy = np.zeros([len(NetFileNameAll), len(other_algrithm)+1 ])
    all_result_mac_egy = np.zeros([len(NetFileNameAll), len(other_algrithm)+1 ])
    all_result_ddr_egy = np.zeros([len(NetFileNameAll), len(other_algrithm)+1 ])
    program_run_time       = np.zeros([len(NetFileNameAll), len(other_algrithm)+1 ])
    for net in NetFileName:
        net_min_runtime = sys.maxsize
        net_min_energy  = sys.maxsize
        net_min_energy_cly  = sys.maxsize
        net_min_l2_regy  = sys.maxsize
        net_min_l2_wegy  = sys.maxsize
        net_min_l1_regy  = sys.maxsize
        net_min_l1_wegy  = sys.maxsize
        net_min_mac_egy  = sys.maxsize
        net_min_ddr_egy  = sys.maxsize
        #varInst = var.Var(net,70,100) #19 s
        #varInst = var.Var(net,70,500) # max remap setting
        varInst = var.Var(net, levels=21, num_sims=30) # about 5 ok
        #varInst = var.Var(net, levels=1, num_sims=3) # for test

        glb_num_pe_used = np.zeros((varInst.LY_LEN,))
        glb_reuse_ifactor = np.zeros((varInst.LY_LEN,))
        glb_reuse_ofactor = np.zeros((varInst.LY_LEN,))
        glb_reuse_wfactor = np.zeros((varInst.LY_LEN,))
        glb_analysis_single_data_tmp_list = [[] for i in range(varInst.LY_LEN)]
        if DEBUG_PRINT == True:
            print("#================NET_NAME===============================# file: "+net+" time: " + cur_time())
            wFile("#================NET_NAME===============================# file: "+net+" time: " + cur_time(), fileNameFull)

        if varInst.MODE_MCTS_EN == True:
            mcts_run_time_start = time.time()
            mcts_init_done_flag = False
            mcts_train_rtm = []
            glb_cnt = 0
            glb_no_valid_times_cnt = 0
            glb_no_valid_times_list = [] 
            #parser = argparse.ArgumentParser(description='MCTS research code')
            #parser.add_argument('--num_sims', action="store", required=True, type=int)
            #parser.add_argument('--levels', action="store", required=True, type=int, choices=range(varInst.A_NUM))
            #args=parser.parse_args()
            state = Env_Maestro()
            glb_state =state.reset()
            state = state.init_run_all_layer()
            current_node = Node(state)                       # create a state, and use it to get a new node as the initial node
            for l in range( varInst.levels ):#args.levels
                current_node=UCTSEARCH(varInst.num_sims/(l+1), current_node) #args.num_sims
                if DEBUG_PRINT == True:
                    print("level %d"%l)
                    print("Num Children: %d"%len(current_node.parent.children))
                    wFile("level %d"%l, fileNameFull)
                    wFile("Num Children: %d"%len(current_node.parent.children), fileNameFull)
                
                for i,c in enumerate(current_node.parent.children):
                    if DEBUG_PRINT == True:
                        print(i,c)
                        wFile("%s,%s"%(i,c), fileNameFull)
                if DEBUG_PRINT == True:
                    print("Best Child: %s, avg reward %f, reward %f"%(current_node.state, (current_node.reward/current_node.visits), current_node.reward ) )
                    wFile("Best Child: %s, avg reward %f, reward %f"%(current_node.state, (current_node.reward/current_node.visits), current_node.reward ), fileNameFull)

            mcts_run_time_end = time.time()
            mcts_run_time = mcts_run_time_end-mcts_run_time_start
            if DEBUG_PRINT == True:
                print("The Program Run time: %f s"%(mcts_run_time))
                wFile("The Program Run time: %f s"%(mcts_run_time), fileNameFull)
                print("Min runtime: %d, Min energy: %d"%(net_min_runtime, net_min_energy))
                wFile("Min runtime: %d, Min energy: %d"%(net_min_runtime, net_min_energy), fileNameFull)

            all_result_rtm[NetFileNameAll.index(net),0] = net_min_runtime 
            all_result_egy[NetFileNameAll.index(net),0] = net_min_energy 
            all_result_egy_ncl[NetFileNameAll.index(net),0] = net_min_energy_cly
            all_result_l2_regy[NetFileNameAll.index(net),0] = net_min_l2_regy
            all_result_l2_wegy[NetFileNameAll.index(net),0] = net_min_l2_wegy
            all_result_l1_regy[NetFileNameAll.index(net),0] = net_min_l1_regy
            all_result_l1_wegy[NetFileNameAll.index(net),0] = net_min_l1_wegy
            all_result_mac_egy[NetFileNameAll.index(net),0] = net_min_mac_egy
            all_result_ddr_egy[NetFileNameAll.index(net),0] = net_min_ddr_egy
            program_run_time[NetFileNameAll.index(net),0]  = mcts_run_time 

            np.savetxt(dir_name+"//"+fileName+"_re_rtm"+".csv", all_result_rtm, delimiter="," ) 
            np.savetxt(dir_name+"//"+fileName+"_re_egy"+".csv", all_result_egy, delimiter=",") 
            np.savetxt(dir_name+"//"+fileName+"_re_egy_ncl"+".csv", all_result_egy_ncl, delimiter=",") 
            np.savetxt(dir_name+"//"+fileName+"_re_l2_regy"+".csv", all_result_l2_regy, delimiter=",") 
            np.savetxt(dir_name+"//"+fileName+"_re_l2_wegy"+".csv", all_result_l2_wegy, delimiter=",") 
            np.savetxt(dir_name+"//"+fileName+"_re_l1_regy"+".csv", all_result_l1_regy, delimiter=",") 
            np.savetxt(dir_name+"//"+fileName+"_re_l1_wegy"+".csv", all_result_l1_wegy, delimiter=",") 
            np.savetxt(dir_name+"//"+fileName+"_re_mac_egy"+".csv", all_result_mac_egy, delimiter=",") 
            np.savetxt(dir_name+"//"+fileName+"_re_ddr_egy"+".csv", all_result_ddr_egy, delimiter=",") 
            np.savetxt(dir_name+"//"+fileName+"_prTime"+".csv", program_run_time,       delimiter=",") 
            all_egy = net_min_l2_regy+net_min_l2_wegy+net_min_l1_regy+net_min_l1_wegy+net_min_mac_egy+net_min_ddr_egy
            #print("%.2E,%.2E,%.2E,%.2E,%.2E,%.2E,%.2E"%(net_min_l2_regy, net_min_l2_wegy, net_min_l1_regy, net_min_l1_wegy, net_min_mac_egy, net_min_ddr_egy, net_min_energy ))
            print("%.2E,%.2E,%.2E,%.2E,%.2E,%.2E, %.2E"%(net_min_l2_regy, net_min_l2_wegy, net_min_l1_regy, net_min_l1_wegy, net_min_mac_egy, net_min_ddr_egy, all_egy))

            os.system("rm -fr ./core.*")
            save_all_fig(fig_num=len(fig_name))       
            state.mcts_save_data()
            save_all_npy()
        else:
            buf_num_init_value_other_algrithm =  {'test_vgg16.csv':36, 'test_unet.csv':36, 'test_shufflenet.csv':33, 'test_resnet50.csv':35, 'test_efficientnetb0.csv':36, 'test_anet.csv':36, 'test_bnet.csv':36, 'test_mobilenetv2.csv':36}
            buget_other_algrithm   =  {'test_vgg16.csv':5000, 'test_unet.csv':5000, 'test_shufflenet.csv':5000, 'test_resnet50.csv':5000, 'test_efficientnetb0.csv':5000, 'test_anet.csv':5000, 'test_bnet.csv':5000, 'test_mobilenetv2.csv':5000}
            buget_algrithm_factor  =  {"RandomSearch" : 1, "ScrHammersleySearch":1 , "TwoPointsDE":1,   "CMA":1, "PSO":1}
            ng_s_dim_order = ng.p.Array(init=np.ones([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX_O]))
            if varInst.MODE_TOP_SPATIAL_O == True:
                ng_s_dim_top = ng.p.Array(init=np.ones([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX_O]))
            else:
                ng_s_type      = ng.p.Array(init=np.ones([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX_O*varInst.DIM_NUM]))
            ng_s_size      = ng.p.Array(init=np.ones([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX_O*varInst.DIM_NUM]))
            ng_s_offset    = ng.p.Array(init=np.ones([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX_O*varInst.DIM_NUM]))
            ng_s_cluster_v = ng.p.Array(init=np.ones(varInst.LY_LEN))
            #ng_s_pe_num    = ng.p.Array(init=np.ones(varInst.LY_LEN))
            #ng_s_buf_num   = ng.p.Array(init=np.ones(varInst.LY_LEN))
            #ng_s_pe_num    = ng.p.Array(init=(2**5)*np.ones(varInst.LY_LEN))
            #ng_s_buf_num   = ng.p.Array(init=(2**(buf_num_init_value_other_algrithm[net]))*np.ones(varInst.LY_LEN))
            ng_s_pe_num    = ng.p.Array(init=(127)*np.ones(varInst.LY_LEN))
            ng_s_buf_num   = ng.p.Array(init=(127)*np.ones(varInst.LY_LEN))


            ng_s_dim_order.set_bounds(     np.zeros([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX_O]), varInst.s_dim_order_v_max-1)
            #ng_s_dim_order.set_bounds(     np.zeros([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX_O]), 9*np.ones([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX_O]) )
            if varInst.MODE_TOP_SPATIAL_O == True:
                ng_s_dim_top.set_bounds(   np.zeros([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX_O]), 5*np.ones([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX_O]))
            else:
                ng_s_type.set_bounds(      np.zeros([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX_O*varInst.DIM_NUM]),      varInst.s_type_v_max)
            ng_s_size.set_bounds(          np.ones([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX_O*varInst.DIM_NUM]),      varInst.s_size_v_max+1)
            ng_s_offset.set_bounds(        np.ones([varInst.LY_LEN, varInst.CLUSTR_LVL_MEM_MAX_O*varInst.DIM_NUM]),      varInst.s_oft_v_max+1)
            #ng_s_cluster_v.set_bounds(     np.zeros(varInst.LY_LEN), 9*np.ones(varInst.LY_LEN))
            ng_s_cluster_v.set_bounds(     np.ones(varInst.LY_LEN), 253*np.ones(varInst.LY_LEN))
            #ng_s_pe_num.set_bounds(        np.zeros(varInst.LY_LEN), 9*np.ones(varInst.LY_LEN))
            #ng_s_buf_num.set_bounds(       np.zeros(varInst.LY_LEN), 4*np.ones(varInst.LY_LEN)) # [0, 4] -> [14, 18]
            #ng_s_pe_num.set_bounds(        (2**5)*np.ones(varInst.LY_LEN), (2**20)*np.ones(varInst.LY_LEN))
            #ng_s_buf_num.set_bounds(       (2**20)*np.ones(varInst.LY_LEN), (2**40)*np.ones(varInst.LY_LEN)) # [0, 4] -> [14, 18]
            ng_s_pe_num.set_bounds(        (32)*np.ones(varInst.LY_LEN), (129)*np.ones(varInst.LY_LEN))
            ng_s_buf_num.set_bounds(       (89)*np.ones(varInst.LY_LEN), (130)*np.ones(varInst.LY_LEN)) # [0, 4] -> [14, 18]


            ng_s_dim_order.set_integer_casting()
            if varInst.MODE_TOP_SPATIAL_O == True:
                ng_s_dim_top.set_integer_casting()
            else:
                ng_s_type.set_integer_casting()
            ng_s_size.set_integer_casting()
            ng_s_offset.set_integer_casting()
            ng_s_cluster_v.set_integer_casting()
            ng_s_pe_num.set_integer_casting()
            ng_s_buf_num.set_integer_casting()

            if varInst.MODE_TOP_SPATIAL_O == True:
                parametrization = ng.p.Instrumentation( ng_s_dim_order, 
                                                    "blublu",
                                                    ng_s_dim_top,      
                                                    ng_s_size,      
                                                    ng_s_offset,    
                                                    ng_s_cluster_v, 
                                                    ng_s_pe_num,    
                                                    ng_s_buf_num)
            else:
                parametrization = ng.p.Instrumentation( ng_s_dim_order, 
                                                    ng_s_type,      
                                                    "blublu",
                                                    ng_s_size,      
                                                    ng_s_offset,    
                                                    ng_s_cluster_v, 
                                                    ng_s_pe_num,    
                                                    ng_s_buf_num)
            state = Env_Maestro()
            state.reset()   # ["RandomSearch", "ScrHammersleySearch", "TwoPointsDE",  "CMA", "PSO"]
            for alg in other_algrithm:
                state.other_reset_data(layer_len=varInst.LY_LEN)
                ProcessTime = OtherAlgrithmProcessTime[NetFileNameAll.index(net), other_algrithm.index(alg)]
                #ProcessTime = 10
                budget = int(buget_other_algrithm[net]/buget_algrithm_factor[alg])    # How many episode we will do before concluding.
                plt_init()
                other_algrithm_train_rtm = []
                other_run_time_start = time.time()
                net_min_runtime = sys.maxsize
                net_min_energy  = sys.maxsize
                other_algrithm_init_done = False
                other_algrithm_init_cnt  = 0
                if DEBUG_PRINT == True:
                    print("Optime Method Name ############################################:", alg)
                    wFile("Optime Method Name ############################################:" + alg, fileNameFull)
                optim = ng.optimizers.registry[alg](parametrization=parametrization, budget=budget)
                for u in tqdm(range(budget)):
                    #if(time.time() - other_run_time_start > ProcessTime):
                    #    break
                    x1 = optim.ask()
                    # Ask and tell can be asynchronous.
                    # Just be careful that you "tell" something that was asked.
                    # Here we ask 3 times and tell 3 times in order to fake asynchronicity
                    #x2 = optim.ask()
                    #x3 = optim.ask()
                    # The three folowing lines could be parallelized.
                    # We could also do things asynchronously, i.e. do one more ask
                    # as soon as a training is over.
                    #other_algrithm_init_done = True if u > (budget/1000) else False
                    if other_algrithm_init_done == True and DEBUG_PRINT == True:
                        print("other_init_done")
                        wFile("other_init_done", fileNameFull)
                    y1 = state.eval_func(*x1.args, **x1.kwargs,)  # here we only defined an arg, so we could omit kwargs
                    #y2 = state.eval_func(*x2.args, **x2.kwargs)  # (keeping it here for the sake of consistency)
                    #y3 = state.eval_func(*x3.args, **x3.kwargs)
                    optim.tell(x1, y1)
                    #optim.tell(x2, y2)
                    #optim.tell(x3, y3)
                recommendation = optim.recommend()
                #print("* ", alg, " provides a vector of parameters with test error ",
                #    state.eval_func(*recommendation.args, **recommendation.kwargs))
                if DEBUG_PRINT == True:
                    print("Min runtime: %d, Min energy: %d"%(net_min_runtime, net_min_energy))
                    wFile("Min runtime: %d, Min energy: %d"%(net_min_runtime, net_min_energy), fileNameFull)
                other_run_time_end = time.time()
                other_run_time = other_run_time_end-other_run_time_start
                if DEBUG_PRINT == True:
                    print("The %s Program Run time: %f s"%(alg,other_run_time))
                    wFile("The %s Program Run time: %f s"%(alg,other_run_time), fileNameFull)
                all_result_rtm[NetFileNameAll.index(net), other_algrithm.index(alg)+1] = net_min_runtime 
                all_result_egy[NetFileNameAll.index(net), other_algrithm.index(alg)+1] = net_min_energy 
                program_run_time[NetFileNameAll.index(net),       other_algrithm.index(alg)+1] = other_run_time 
                save_all_fig(fig_num=1, alg_str=alg[0:3])       
                state.other_save_data(alg_str=alg[0:3])
                np.savetxt(dir_name+"//"+fileName+"_re_rtm"+".csv", all_result_rtm, delimiter="," ) 
                np.savetxt(dir_name+"//"+fileName+"_re_egy"+".csv", all_result_egy, delimiter=",") 
                np.savetxt(dir_name+"//"+fileName+"_prTime"+".csv", program_run_time,       delimiter=",") 
                state.reset()
    os.system("rm -fr ./core.*")
    print("Success run,-------------------------------------------------> done OK!")
        


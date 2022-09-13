import numpy as np 
import pandas as pd
class Var:
        def __init__(self, NetFileName, levels=1, num_sims=5):
                self.MODE_MCTS_EN            = True
                self.en_global_record_talbe = True
                self.en_heuristic_action_space = False
                #self.MODE_MCTS_EN            = False
                self.NetId = 2
                self.opt_object_runtime_en = True
                self.contr_nm         = 'ult'
                self.contr_en_area    = False
                self.contr_en_power   = False
                self.vs_gamma         = False
                self.gamma_cld        = False




                # shell can append at 20 above code DO NOT MOVE !! else the shell not work!!
                self.vs_dac_naas       = False
                self.naas_pe256        = False # pe=256 or pe1024

                self.EN_DDR        = 1
                self.EN_CROSS_LY   = 1
                self.restrict_dim_order_cross_layer = True
                self.NetFileName = NetFileName
                max_clm_name = ['power_rtm', 'power_egy', 'area_rtm', 'area_egy']
                max_value = np.genfromtxt('max_power_area_mcts_0305_real_all.txt', delimiter=',')

                max_idx_str0 = 'rtm' if self.opt_object_runtime_en else 'egy'
                max_pidx  = max_clm_name.index('power_'+max_idx_str0)
                max_aidx  = max_clm_name.index('area_'+max_idx_str0)
                
                max_pvalue  = max_value[self.NetId, max_pidx]
                max_avalue  = max_value[self.NetId, max_aidx]
                
                self.constrain_max_power = max_pvalue
                self.constrain_max_area =  max_avalue

                self.normal_dim_order_num = 720
                self.cross_layer_restrict_dim_order_num = 12
                self.dim_order_num = self.cross_layer_restrict_dim_order_num if self.restrict_dim_order_cross_layer==True else self.normal_dim_order_num
                #print('%e %e'%(max_pvalue, max_avalue))

                #constr_name = ['ult', 'cld', 'edge', 'iot']
                #---------------------get the network structure from csv to some lists-----------------------------#
                csv_data_start_columns   = 0  #  0 
                csv_data_end_columns     = 12  #  0 
                #pd_data = pd.read_csv ('test_anet.csv') #85
                #pd_data = pd.read_csv ('test_EfficientNetB0.csv') #86 line
                #pd_data = pd.read_csv ('test_resnet50.csv') # 67 line
                #pd_data = pd.read_csv ('test_shufflenet.csv') # 71  line 
                #pd_data = pd.read_csv ('test_unet.csv') # 23 line 
                #pd_data = pd.read_csv ('test_vgg16.csv') # 13 line 
                pd_data = pd.read_csv (self.NetFileName) # 13 line 
                self.ly_n_list   = list(pd_data['layer_name'].values)             # row name index dic ->  layer name list :   blk1_conv1_3s1 , blk2_conv1_3s2 ...
                self.LY_LEN      = len(list(pd_data['index'].values))             # row len            ->  layer number 98
                ly_name_i_dic = {}                                           # dic                ->  key=layer_name   value=layer_index 0:97
                for i in range(self.LY_LEN):
                    ly_name_i_dic[self.ly_n_list[i]] = i

                clm_para_i_dic      = {}
                clm_para_list     = list(pd_data.columns[csv_data_start_columns: csv_data_end_columns])     # column index dic 
                for i in range(len(clm_para_list)):
                    clm_para_i_dic[clm_para_list[i]] = i  

                np_data = np.array(pd_data)

                self.net_name  = list(np_data[:, clm_para_i_dic['net_name']])[0]
                self.stride_x  = list(np_data[:, clm_para_i_dic['stride_x']].astype(int))
                self.stride_y  = list(np_data[:, clm_para_i_dic['stride_y']].astype(int))

                self.LY_TYPE   = list(np_data[:, clm_para_i_dic['type']])
                self.IS_DSCONV = list(np.zeros([len(self.LY_TYPE)], dtype=np.int))
                "change K to max int, if type is depth-wise"
                for i in range(len(self.LY_TYPE)):
                        if self.LY_TYPE[i] == "DSCONV":
                                self.IS_DSCONV[i] = 1
                                np_data[i, clm_para_i_dic['K']] = 2**20

                self.K         = list(np_data[:, clm_para_i_dic['K']])
                self.C         = list(np_data[:, clm_para_i_dic['C']])
                self.R         = list(np_data[:, clm_para_i_dic['R']])
                self.S         = list(np_data[:, clm_para_i_dic['S']])
                self.Y         = list(np_data[:, clm_para_i_dic['Y']])
                self.X         = list(np_data[:, clm_para_i_dic['X']])

                dim_data = np_data[: , 3:9]
                self.DIM_MAX = np.array(dim_data) 


                all_ly_dim_max_v_plus_one = np.max(dim_data) + 1
                self.MODE_TOP_SPATIAL        = False
                self.MODE_TOP_SPATIAL_O      = True
                self.CLUSTR_LVL_MEM_MAX      = 2             # two level of cluster
                self.CLUSTR_LVL_MEM_MAX_O    = 2             # two level of cluster
                self.DIM_NUM                 = 6             #X self.Y self.Cself.self.K number 
                self.TOP_MAX_NUM             = 2
                self.PROB_TWO_SP_EXPOLER_TP  = 75            # 0-99
                self.LY_MAX_TIMES_AVG        = 2            # >=10 mob
                #BAD_LAYER = 1 if self.LY_LEN < 10 else 0
                BAD_LAYER = int(self.LY_LEN/4) if self.NetId==3  else int(self.LY_LEN/1.5)
                #BAD_LAYER = self.LY_LEN
                INI_V_CUT_RATO = 0.5
                self.MCTS_INIT_RANDOM        = False

                self.PE_NUM_MAX = 2**20
                # TODO change to l1 
                self.BUF_NUM_MAX = 2**40# 262144 #256*1024

                s_dim_order_len   = self.CLUSTR_LVL_MEM_MAX * self.LY_LEN 
                s_type_len        = self.DIM_NUM * self.CLUSTR_LVL_MEM_MAX * self.LY_LEN
                s_size_len        = self.DIM_NUM * self.CLUSTR_LVL_MEM_MAX * self.LY_LEN
                s_oft_len         = self.DIM_NUM * self.CLUSTR_LVL_MEM_MAX * self.LY_LEN
                s_ctl_v_len       = (self.CLUSTR_LVL_MEM_MAX-1) * self.LY_LEN
                s_pe_num_len      = self.LY_LEN
                s_buf_num_len     = self.LY_LEN
                s_ly_idx_len      = 1
                s_res_rtm_len     = self.LY_LEN
                s_res_egy_len     = self.LY_LEN
                s_res_tpt_len     = self.LY_LEN
                s_ly_epl_cnt_len  = self.LY_LEN
                s_res_area_len    = self.LY_LEN
                s_res_power_len   = self.LY_LEN
                s_res_egy_ncl_len = self.LY_LEN
                s_l2_regy_len     = self.LY_LEN
                s_l2_wegy_len     = self.LY_LEN
                s_l1_regy_len     = self.LY_LEN
                s_l1_wegy_len     = self.LY_LEN
                s_mac_egy_len     = self.LY_LEN
                s_ddr_egy_len     = self.LY_LEN


                self.s_dim_order_index  = s_dim_order_len
                self.s_type_index       = s_dim_order_len + s_type_len
                self.s_size_index       = s_dim_order_len + s_type_len + s_size_len
                self.s_oft_index        = s_dim_order_len + s_type_len + s_size_len + s_oft_len
                self.s_ctl_v_index      = s_dim_order_len + s_type_len + s_size_len + s_oft_len + s_ctl_v_len
                self.s_pe_num_index     = s_dim_order_len + s_type_len + s_size_len + s_oft_len + s_ctl_v_len + s_pe_num_len
                self.s_buf_num_index    = s_dim_order_len + s_type_len + s_size_len + s_oft_len + s_ctl_v_len + s_pe_num_len + s_buf_num_len 
                self.s_ly_idx_index     = s_dim_order_len + s_type_len + s_size_len + s_oft_len + s_ctl_v_len + s_pe_num_len + s_buf_num_len + s_ly_idx_len
                self.s_res_rtm_index    = s_dim_order_len + s_type_len + s_size_len + s_oft_len + s_ctl_v_len + s_pe_num_len + s_buf_num_len + s_ly_idx_len +  s_res_rtm_len
                self.s_res_egy_index    = s_dim_order_len + s_type_len + s_size_len + s_oft_len + s_ctl_v_len + s_pe_num_len + s_buf_num_len + s_ly_idx_len +  s_res_rtm_len +  s_res_egy_len
                self.s_res_tpt_index    = s_dim_order_len + s_type_len + s_size_len + s_oft_len + s_ctl_v_len + s_pe_num_len + s_buf_num_len + s_ly_idx_len +  s_res_rtm_len +  s_res_egy_len + s_res_tpt_len 
                self.s_ly_epl_cnt_index = s_dim_order_len + s_type_len + s_size_len + s_oft_len + s_ctl_v_len + s_pe_num_len + s_buf_num_len + s_ly_idx_len +  s_res_rtm_len +  s_res_egy_len + s_res_tpt_len  + s_ly_epl_cnt_len
                self.s_res_area_index   = s_dim_order_len + s_type_len + s_size_len + s_oft_len + s_ctl_v_len + s_pe_num_len + s_buf_num_len + s_ly_idx_len +  s_res_rtm_len +  s_res_egy_len + s_res_tpt_len  + s_ly_epl_cnt_len + s_res_area_len
                self.s_res_power_index  = s_dim_order_len + s_type_len + s_size_len + s_oft_len + s_ctl_v_len + s_pe_num_len + s_buf_num_len + s_ly_idx_len +  s_res_rtm_len +  s_res_egy_len + s_res_tpt_len  + s_ly_epl_cnt_len + s_res_area_len + s_res_power_len
                self.s_res_egy_ncl_index= s_dim_order_len + s_type_len + s_size_len + s_oft_len + s_ctl_v_len + s_pe_num_len + s_buf_num_len + s_ly_idx_len +  s_res_rtm_len +  s_res_egy_len + s_res_tpt_len  + s_ly_epl_cnt_len + s_res_area_len + s_res_power_len + s_res_egy_ncl_len
                self.s_l2_regy_index    = s_dim_order_len + s_type_len + s_size_len + s_oft_len + s_ctl_v_len + s_pe_num_len + s_buf_num_len + s_ly_idx_len +  s_res_rtm_len +  s_res_egy_len + s_res_tpt_len  + s_ly_epl_cnt_len + s_res_area_len + s_res_power_len + s_res_egy_ncl_len + s_l2_regy_len
                self.s_l2_wegy_index    = s_dim_order_len + s_type_len + s_size_len + s_oft_len + s_ctl_v_len + s_pe_num_len + s_buf_num_len + s_ly_idx_len +  s_res_rtm_len +  s_res_egy_len + s_res_tpt_len  + s_ly_epl_cnt_len + s_res_area_len + s_res_power_len + s_res_egy_ncl_len + s_l2_regy_len + s_l2_wegy_len 
                self.s_l1_regy_index    = s_dim_order_len + s_type_len + s_size_len + s_oft_len + s_ctl_v_len + s_pe_num_len + s_buf_num_len + s_ly_idx_len +  s_res_rtm_len +  s_res_egy_len + s_res_tpt_len  + s_ly_epl_cnt_len + s_res_area_len + s_res_power_len + s_res_egy_ncl_len + s_l2_regy_len + s_l2_wegy_len + s_l1_regy_len
                self.s_l1_wegy_index    = s_dim_order_len + s_type_len + s_size_len + s_oft_len + s_ctl_v_len + s_pe_num_len + s_buf_num_len + s_ly_idx_len +  s_res_rtm_len +  s_res_egy_len + s_res_tpt_len  + s_ly_epl_cnt_len + s_res_area_len + s_res_power_len + s_res_egy_ncl_len + s_l2_regy_len + s_l2_wegy_len + s_l1_regy_len + s_l1_wegy_len
                self.s_mac_egy_index    = s_dim_order_len + s_type_len + s_size_len + s_oft_len + s_ctl_v_len + s_pe_num_len + s_buf_num_len + s_ly_idx_len +  s_res_rtm_len +  s_res_egy_len + s_res_tpt_len  + s_ly_epl_cnt_len + s_res_area_len + s_res_power_len + s_res_egy_ncl_len + s_l2_regy_len + s_l2_wegy_len + s_l1_regy_len + s_l1_wegy_len + s_mac_egy_len
                self.s_ddr_egy_index    = s_dim_order_len + s_type_len + s_size_len + s_oft_len + s_ctl_v_len + s_pe_num_len + s_buf_num_len + s_ly_idx_len +  s_res_rtm_len +  s_res_egy_len + s_res_tpt_len  + s_ly_epl_cnt_len + s_res_area_len + s_res_power_len + s_res_egy_ncl_len + s_l2_regy_len + s_l2_wegy_len + s_l1_regy_len + s_l1_wegy_len + s_mac_egy_len + s_ddr_egy_len


                self.S_LEN = self.s_ly_idx_index 

                #============================================Index: layer index[0 : self.LY_LEN-1] ==============================================================================#
                self.LAYER           = [2**i for i in range(5)]


                #============================================Index: cluster level 0/1 or 0==============================================================================#
                self.CLT_LVL      = [i for i in range(self.CLUSTR_LVL_MEM_MAX)]

                #============================================Index: which dim to change it's type size offset==============================================================================#
                self.DIM              = [ 'K',     'C',    'R',    'S',    'Y', 'X'     ]


                #============================================Value: self.DIM order [0:719]=============================================================================#
                self.DIM_ORDER       = [0, -1, 0, 1]


                #============================================Value: type: temperal/spatial 0/1============================================================================#
                if (self.MODE_TOP_SPATIAL==True) : # only one dim can be spatial
                        self.ACT_TYPE        = []  # 0/1 : temperal/spatial
                        self.DIM_TOP         = [ 'K',     'C',    'R',    'S',    'Y', 'X'     ]
                else:                         # all dim can be spatial
                        self.ACT_TYPE        = [0, 1] 
                        self.DIM_TOP         = []


                #============================================Value: size: [1,dim_max]============================================================================#
                #ACT_SIZE_STEP  = [-1024, -256, -64, -16, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 16, 64, 256, 1024]
                self.ACT_SIZE_STEP  = [0, 1, 2, 3, 5, 7, 16, 64, 256, 1024]
                #ACT_SIZE_STEP = [-16, -1, 0, 1, 16]


                #============================================Value: offset: [1,dim_max]============================================================================#
                #self.ACT_OFFSET_STEP  = [-1024, -256, -64, -16, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 16, 64, 256, 1024]
                self.ACT_OFFSET_STEP  = [0, 1, 2, 3, 5, 7, 16, 64, 256, 1024]
                #self.ACT_OFFSET_STEP = [-16, -1, 0, 1, 16]


                #============================================Value: self.Cluster(P, xx)===========================================================================#
                if self.CLUSTR_LVL_MEM_MAX == 1 :
                        self.CLT_V        = []
                else:
                        self.CLT_V        = [256, 128, 64, 32, 16, 8, 7, 5, 4, 3]

                #============================================Value: PE Buffer size===========================================================================#
                #self.PE_NUM       = [2**i for i in range(10, 20)] 
                self.PE_NUM       = [1024] 
                #self.PE_NUM       = [32, 48, 64, 96, 128, 256, 512, 1024] 
                #self.PE_NUM          = [-512, -128, -32, 0, 32, 128, 512] 
                # 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128

                # TODO l1
                #self.BUF_NUM      = [2**i for i in range(35, 40)] # 16k, 64k, 256k 

                self.BUF_NUM      = [89, 99, 109, 119, 129] # 16k, 64k, 256k 
                #self.BUF_NUM         = [-262144, -65536, -16384, 0, 16384, 65536, 262144] # 0 16k, 64k, 256k 

                #============================================Value: Cluster Level gen max======================================================================#
                self.CLT_LVL_GEN      = [i for i in range(self.CLUSTR_LVL_MEM_MAX)]

                self.act_type_len            = len(self.ACT_TYPE)
                self.act_size_step_len       = len(self.ACT_SIZE_STEP)
                self.act_offset_step_len     = len(self.ACT_OFFSET_STEP)
                self.act_dim_len             = len(self.DIM)
                self.act_dim_order_len       = len(self.DIM_ORDER)
                self.act_dim_top_len         = len(self.DIM_TOP)
                self.act_clt_lvl_len         = len(self.CLT_LVL)
                self.act_clt_v_len           = len(self.CLT_V)
                self.act_pe_num_len          = len(self.PE_NUM)
                self.act_buf_num_len         = len(self.BUF_NUM)
                self.act_layer_len           = len(self.LAYER)
                self.act_clt_lvl_gen_len     = len(self.CLT_LVL_GEN)

                self.act_type_exist            =  1 if self.act_type_len > 0 else 0    
                self.act_size_step_exist       =  1 if self.act_size_step_len > 0 else 0    
                self.act_offset_step_exist     =  1 if self.act_offset_step_len > 0 else 0    
                self.act_dim_exist             =  1 if self.act_dim_len > 0 else 0    
                self.act_dim_order_exist       =  1 if self.act_dim_order_len > 0 else 0    
                self.act_dim_top_exist         =  1 if self.act_dim_top_len > 0 else 0    
                self.act_clt_lvl_exist         =  1 if self.act_clt_lvl_len > 0 else 0    
                self.act_clt_v_exist           =  1 if self.act_clt_v_len > 0 else 0    
                self.act_pe_num_exist          =  1 if self.act_pe_num_len > 0 else 0    
                self.act_buf_num_exist         =  1 if self.act_buf_num_len > 0 else 0    
                self.act_layer_exist           =  1 if self.act_layer_len else 0    
                self.act_clt_lvl_gen_exist     =  1 if self.act_clt_lvl_gen_len else 0    

                act_type_len_l          =    [] if self.act_type_exist == 0 else [self.act_type_len]
                act_size_step_len_l     =    [] if self.act_size_step_exist == 0 else [self.act_size_step_len]
                act_offset_step_len_l   =    [] if self.act_offset_step_exist == 0 else [self.act_offset_step_len]
                act_dim_len_l           =    [] if self.act_dim_exist == 0 else [self.act_dim_len]
                act_dim_order_len_l     =    [] if self.act_dim_order_exist == 0 else [self.act_dim_order_len]
                act_dim_top_len_l       =    [] if self.act_dim_top_exist == 0 else [self.act_dim_top_len]
                act_clt_lvl_len_l       =    [] if self.act_clt_lvl_exist == 0 else [self.act_clt_lvl_len]
                act_clt_v_len_l         =    [] if self.act_clt_v_exist == 0 else [self.act_clt_v_len]
                act_pe_num_len_l        =    [] if self.act_pe_num_exist == 0 else [self.act_pe_num_len]
                act_buf_num_len_l       =    [] if self.act_buf_num_exist == 0 else [self.act_buf_num_len]
                act_layer_len_l         =    [] if self.act_layer_exist == 0 else [self.act_layer_len]
                act_clt_lvl_gen_len_l       =    [] if self.act_clt_lvl_gen_exist == 0 else [self.act_clt_lvl_gen_len]

                act_each_len            = act_type_len_l + act_size_step_len_l + act_offset_step_len_l + act_dim_len_l + act_dim_order_len_l + act_dim_top_len_l + act_clt_lvl_len_l + act_clt_v_len_l + act_pe_num_len_l + act_buf_num_len_l + act_layer_len_l + act_clt_lvl_gen_len_l
                A_NET_OUT_LEN           = self.act_type_len   + self.act_size_step_len   + self.act_offset_step_len   + self.act_dim_len   + self.act_dim_order_len   + self.act_dim_top_len + self.act_clt_lvl_len + self.act_clt_v_len + self.act_pe_num_len + self.act_buf_num_len + self.act_layer_len + self.act_clt_lvl_gen_len

                act_type_tmp            =  1 if self.act_type_len == 0 else self.act_type_len   
                act_size_step_tmp       =  1 if self.act_size_step_len == 0 else self.act_size_step_len   
                act_offset_step_tmp     =  1 if self.act_offset_step_len == 0 else self.act_offset_step_len   
                act_dim_tmp             =  1 if self.act_dim_len == 0 else self.act_dim_len   
                act_dim_order_tmp       =  1 if self.act_dim_order_len == 0 else self.act_dim_order_len   
                act_dim_top_tmp         =  1 if self.act_dim_top_len == 0 else self.act_dim_top_len   
                act_clt_lvl_tmp         =  1 if self.act_clt_lvl_len == 0 else self.act_clt_lvl_len   
                act_clt_v_tmp           =  1 if self.act_clt_v_len == 0 else self.act_clt_v_len   
                act_pe_num_tmp          =  1 if self.act_pe_num_len == 0 else self.act_pe_num_len   
                act_buf_num_tmp         =  1 if self.act_buf_num_len == 0 else self.act_buf_num_len   
                act_layer_tmp           =  1 if self.act_layer_len == 0 else self.act_layer_len   
                act_clt_lvl_gen_tmp     =  1 if self.act_clt_lvl_gen_len == 0 else self.act_clt_lvl_gen_len   

                self.A_NUM                   = act_type_tmp   * act_size_step_tmp   * act_offset_step_tmp   * act_dim_tmp   * act_dim_order_tmp   * act_dim_top_tmp * act_clt_lvl_tmp * act_clt_v_tmp * act_pe_num_tmp * act_buf_num_tmp * act_layer_tmp * act_clt_lvl_gen_tmp

                self.ACT_TYPE_INDEX          = self.act_type_exist   -1
                self.ACT_SIZE_STEP_INDEX     = self.act_type_exist   + self.act_size_step_exist   -1
                self.ACT_OFFSET_STEP_INDEX   = self.act_type_exist   + self.act_size_step_exist   + self.act_offset_step_exist   -1
                self.ACT_DIM_INDEX           = self.act_type_exist   + self.act_size_step_exist   + self.act_offset_step_exist   + self.act_dim_exist   -1
                self.ACT_DIM_ORDER_INDEX     = self.act_type_exist   + self.act_size_step_exist   + self.act_offset_step_exist   + self.act_dim_exist   + self.act_dim_order_exist  -1
                self.ACT_DIM_TOP_INDEX       = self.act_type_exist   + self.act_size_step_exist   + self.act_offset_step_exist   + self.act_dim_exist   + self.act_dim_order_exist   + self.act_dim_top_exist-1
                self.ACT_CLT_LVL_INDEX       = self.act_type_exist   + self.act_size_step_exist   + self.act_offset_step_exist   + self.act_dim_exist   + self.act_dim_order_exist   + self.act_dim_top_exist + self.act_clt_lvl_exist-1
                self.ACT_CLT_V_INDEX         = self.act_type_exist   + self.act_size_step_exist   + self.act_offset_step_exist   + self.act_dim_exist   + self.act_dim_order_exist   + self.act_dim_top_exist + self.act_clt_lvl_exist + self.act_clt_v_exist-1
                self.ACT_PE_NUM_INDEX        = self.act_type_exist   + self.act_size_step_exist   + self.act_offset_step_exist   + self.act_dim_exist   + self.act_dim_order_exist   + self.act_dim_top_exist + self.act_clt_lvl_exist + self.act_clt_v_exist + self.act_pe_num_exist-1
                self.ACT_BUF_NUM_INDEX       = self.act_type_exist   + self.act_size_step_exist   + self.act_offset_step_exist   + self.act_dim_exist   + self.act_dim_order_exist   + self.act_dim_top_exist + self.act_clt_lvl_exist + self.act_clt_v_exist + self.act_pe_num_exist + self.act_buf_num_exist -1
                self.ACT_LAYER_INDEX         = self.act_type_exist   + self.act_size_step_exist   + self.act_offset_step_exist   + self.act_dim_exist   + self.act_dim_order_exist   + self.act_dim_top_exist + self.act_clt_lvl_exist + self.act_clt_v_exist + self.act_pe_num_exist + self.act_buf_num_exist + self.act_layer_exist-1
                self.ACT_CLT_LVL_GEN_INDEX   = self.act_type_exist   + self.act_size_step_exist   + self.act_offset_step_exist   + self.act_dim_exist   + self.act_dim_order_exist   + self.act_dim_top_exist + self.act_clt_lvl_exist + self.act_clt_v_exist + self.act_pe_num_exist + self.act_buf_num_exist + self.act_layer_exist + + self.act_clt_lvl_gen_exist-1

                A_LEN                   = len(act_each_len) 

                '''
                gen size and offset value init value, if self.R or self.S  <= X self.Y seself.self.K, then self.R self.S, elself.Xlf.C self.self.K
                '''
                h_shape = dim_data.shape[0]
                v_shape = dim_data.shape[1]
                init_data_oft = np.zeros([h_shape, v_shape], dtype=np.int)
                init_data_size  = np.ones([h_shape, v_shape], dtype=np.int)
                for h in range(h_shape):
                        min = self.R[h] if self.R[h] <= self.S[h] else self.S[h]
                        for v in range(v_shape):
                                init_data_oft[h, v] = min if min <= dim_data[h, v] else dim_data[h, v]

                self.DIM_NAME_LIST   = [ 'K',     'C',    'R',    'S',    'Y', 'X'     ]
                DIRCT_TYPE_INT_     = [0,1,0,0,0,0]
                self.DIRCT_TYPE_INT      = [DIRCT_TYPE_INT_      * self.CLUSTR_LVL_MEM_MAX ] * self.LY_LEN # one row 6*2                          * layer_num #DIRCT_SIZE_INIT   = [ [i for col in range(DIM_NUM * CLUSTR_LVL_MEM_MAX) ] for i in (R) ] # 3*12
                self.DIRCT_SIZE_INIT   = np.tile(init_data_size.astype(int), (1,self.CLUSTR_LVL_MEM_MAX)).tolist() 
                oft_init_bad_v= np.tile(init_data_oft.astype(int), (1,self.CLUSTR_LVL_MEM_MAX))
                oft_init_good_v = np.tile(dim_data.astype(int), (1,self.CLUSTR_LVL_MEM_MAX))
                rows = oft_init_good_v.shape[0]
                clms = oft_init_good_v.shape[1]
                oft_init_max_each_row = np.max(oft_init_good_v, axis=1)
                #oft_init_bad_v = oft_init_good_v.copy()
                for r in range(rows):
                    max_idx_row = np.where(oft_init_good_v[r,:] == oft_init_max_each_row[r])[0]
                    max_idx_row0 = max_idx_row[0]
                    oft_init_bad_v[r, max_idx_row0] = int(oft_init_max_each_row[r]*INI_V_CUT_RATO)
                #print(oft_init_bad_v)
                layer_good_flag = [False]*BAD_LAYER + [True]*(rows-BAD_LAYER)
                np.random.shuffle(layer_good_flag)
                layer_good_flag = np.array(layer_good_flag)
                layer_good_flag = layer_good_flag.reshape(np.size(layer_good_flag),1)
                #print(layer_good_flag)
                oft_init = np.where(layer_good_flag, oft_init_good_v, oft_init_bad_v)
                #print(oft_init)
                self.DIRCT_OFFSET_INIT = oft_init


                    #oft_in
                #print(self.DIRCT_OFFSET_INIT)

                DIM_ORDER_MTX_RANGE = 1
                DIRCT_TYPE_INT_RANGE = 1

                self.s_dim_order_v_max = np.array([[self.dim_order_num for col in range(self.CLUSTR_LVL_MEM_MAX)] for row in range(self.LY_LEN)])
                self.s_type_v_max      = np.array([[1 for col in range(self.DIM_NUM * self.CLUSTR_LVL_MEM_MAX)] for row in range(self.LY_LEN)])    
                self.s_size_v_max      = np.tile(dim_data.astype(int), (1,self.CLUSTR_LVL_MEM_MAX))
                self.s_oft_v_max       = np.tile(dim_data.astype(int), (1,self.CLUSTR_LVL_MEM_MAX))
                self.s_clt_v_v_max     = np.array([1024 for i in range(self.LY_LEN)])
                self.s_pe_num_v_max    = np.array([1024 for i in range(self.LY_LEN)])
                self.s_buf_num_v_max   = np.array([262144 for i in range(self.LY_LEN)])
                self.s_ly_idx_v_max    = np.array([self.LY_LEN])

                LY_PIP = False
                self.levels = levels
                self.num_sims = num_sims
                #np.savetxt("shu_dim_size.csv",dim_data,fmt='%d',delimiter=',')
                #np.savetxt("eff_dim_size.csv",dim_data,fmt='%d',delimiter=',')

                self.analysis_data_name_list = ['latency', 'energy', 'num_pe_used', 'reuse_ifactor', 'reuse_ofactor', 'reuse_wfactor', 
                      'dim_size_k', #6
                      'dim_size_c', 
                      'dim_size_r', 
                      'dim_size_s', 
                      'dim_size_y', 
                      'dim_size_x', #11 

                      'size_k', #12 
                      'size_c', 
                      'size_r', 
                      'size_s', 
                      'size_y', 
                      'size_x', #17

                      'type_k',#18 
                      'type_c', 
                      'type_r', 
                      'type_s', 
                      'type_y', 
                      'type_x',#23 
                      ]

                self.analysis_data_num_clm = len(self.analysis_data_name_list)

        #Var('test_shufflenet.csv', levels=1, num_sims=5)
#Var('test_efficientnetb0.csv', levels=1, num_sims=5)
'''
#=========================================================================
#===============MCTS STATE AND ACTION=====================================
#=========================================================================

STATE:
1:dim order matrix: 1x1 range: all the permitation 6! = 620 
        
2:three arbitute: type,size,offset: 6x3
range  0 or 1    1->max    1->max 
       type       size     offset
    X:  0          0         0
    Y:  0          0         0
    R:  0          0         0
    S:  0          0         0
    C:  0          0         0
    K:  0          0         0
total: 1+18=19

1:
ACTION value (network output):
1 type:         Temporal, Saptial: ->  [0, 1] 1+1=2
2 size:         -1024, -256, -64, -16, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 16, 64, 256, 1024  ->  19
3 offset:       -1024, -256, -64, -16, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 16, 64, 256, 1024  ->  19
4 dim value     X Y R S C K       ->  6
5 dim order     [0, -1, 0, 1]     ->  4
6 dim top       []                ->  0

7 cluster lvl   [0, 1]            ->  1
6 cluster value [256, 128, 64, 32, 16, 8, 7, 5, 4, 3]   ->  10
total: 2+19+19+6+4 = 50

2: fixed dim order
ACTION value (network output):
1 type:     Temporal, Saptial: ->  [0, 1] 1+1=2
2 size:     -1024, -256, -64, -16, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 16, 64, 256, 1024  ->  19
3 offset:   -1024, -256, -64, -16, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 16, 64, 256, 1024  ->  19
4 dim value X Y R S C K       ->  6
5 dim order []                ->  0
6 dim top   []                ->  0
7 cluster lvl   [0, 1]            ->  1
8 cluster value [256, 128, 64, 32, 16, 8, 7, 5, 4, 3]   ->  10
total: 2+19+19+6 = 46

3: 1: fixed dim order  2: no offset
ACTION value (network output):
1 type:     Temporal, Saptial: ->  [0, 1] 1+1=2
2 size:     -1024, -256, -64, -16, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 16, 64, 256, 1024  ->  19
3 offset:   []  -> 0
4 dim value X Y R S C K       ->  6
5 dim order []                ->  0
6 dim top   []                ->  0
7 cluster lvl   [0, 1]            ->  1
8 cluster value [256, 128, 64, 32, 16, 8, 7, 5, 4, 3]   ->  10
total: 2+19+6  = 27

3-1: 1: fixed dim order  2: no offset 3 add dim top
ACTION value (network output):
1 type:     Temporal, Saptial: ->  [] 0
2 size:     -1024, -256, -64, -16, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 16, 64, 256, 1024  ->  19
3 offset:   []  -> 0
4 dim value X Y R S C K       ->  6
5 dim order []                ->  0
6 dim top   X Y R S C K       ->  6
7 cluster lvl   [0, 1]            ->  1
8 cluster value [256, 128, 64, 32, 16, 8, 7, 5, 4, 3]   ->  10
total: 19+6+6  = 31

4: 1: fixed dim order  2: no offset 3: decrease actions
ACTION value (network output):
1 type:     Temporal, Saptial: ->  [0, 1] 1+1=2
2 size:     -16, -1, 0, 1, 16  ->  5
3 offset:   []                 -> 0
4 dim value X Y R S C K        ->  6
5 dim order []                 ->  0
6 dim top   []                 ->  0
7 cluster lvl   [0, 1]            ->  1
8 cluster value [256, 128, 64, 32, 16, 8, 7, 5, 4, 3]   ->  10
total: 2+5+6  = 13

4-1: 1: fixed dim order  2: no offset 3: decrease actions  4 add dim top
ACTION value (network output):
1 type:     Temporal, Saptial: ->  [] 0
2 size:     -16, -1, 0, 1, 16  ->  5
3 offset:   []                 -> 0
4 dim value X Y R S C K        ->  6
5 dim order []                 ->  0
6 dim top   X Y R S C K        ->  6
7 cluster lvl   [0, 1]            ->  1
8 cluster value [256, 128, 64, 32, 16, 8, 7, 5, 4, 3]   ->  10
total: 5+6+6  = 17

#======================================================
#======================================================
'''

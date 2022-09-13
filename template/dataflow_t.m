

Network {{net_name}} {                                              
        Layer {{ly_name[ly_idx]}} {                                                       
                Type: {{ly_type[ly_idx]}}                                                  

{%- if  (en_ddr == 1) and (en_cross_ly == 1) -%}  

    {{'\n'}}
        {%- if  0 == ly_idx -%}  
    {{'\t'}}        DDR2RAM_WT_EN : True
    {{'\t'}}        DDR2RAM_IFM_EN: True
    {{'\t'}}        RAM2DDR_OFM_EN: False
    {#    {%- elif  ly_name|length == ly_idx -%}  #}
    //{{'\t'}}        DDR2RAM_WT_EN : True
    //{{'\t'}}        DDR2RAM_IFM_EN: False
    //{{'\t'}}        RAM2DDR_OFM_EN: True
        {%- else -%} 
    {{'\t'}}        DDR2RAM_WT_EN : True
    {{'\t'}}        DDR2RAM_IFM_EN: False
    {{'\t'}}        RAM2DDR_OFM_EN: False
        {%- endif  -%}
    {{'\n'}}


{%- elif (en_ddr == 1) and (en_cross_ly == 0) -%} 

    {{'\n'}}
        {%- if  0 == ly_idx -%}  
    {{'\t'}}        DDR2RAM_WT_EN : True
    {{'\t'}}        DDR2RAM_IFM_EN: True
    {{'\t'}}        RAM2DDR_OFM_EN: True
    {#    {%- elif  ly_name|length == ly_idx -%}  #}
    //{{'\t'}}        DDR2RAM_WT_EN : True
    //{{'\t'}}        DDR2RAM_IFM_EN: False
    //{{'\t'}}        RAM2DDR_OFM_EN: True
        {%- else -%} 
    {{'\t'}}        DDR2RAM_WT_EN : True
    {{'\t'}}        DDR2RAM_IFM_EN: True
    {{'\t'}}        RAM2DDR_OFM_EN: True
        {%- endif  -%}
    {{'\n'}}

{%- endif  -%}{{"\n"}}
                Stride { X: {{stride_x[ly_idx]}}, Y: {{stride_x[ly_idx]}} }                                       
                Dimensions { K: {{K[ly_idx]}}, C: {{C[ly_idx]}}, R: {{R[ly_idx]}}, S: {{S[ly_idx]}}, Y: {{Y[ly_idx]}}, X: {{X[ly_idx]}} }        
                Dataflow {                                                  
{% for i in range((6)) %}

    {%- if  (dim0[i] == "K") and (is_dsconv[ly_idx] == 1) -%}  
    {%- else -%} 
        {%- if  s_type0[ly_idx][i] > 0  -%}  
                        {{'\t\t\t\t'}}  SpatialMap({{size0[ly_idx][i]}},{{offset0[ly_idx][i]}}) {{dim0[i]}}; {{'\n'}}                                 
        {%- else -%} 
                        {{'\t\t\t\t'}} TemporalMap({{size0[ly_idx][i]}},{{offset0[ly_idx][i]}}) {{dim0[i]}}; {{'\n'}}                                 
        {%- endif  -%}
    {%- endif  -%}
{%- endfor %} 

{%- if  cluster_lvl >= 1  -%}  
	              {{'\t\t\t\t'}} Cluster({{cluster_value0}}, P);
{% for i in range((6)) %}
    {%- if  (dim1[i] == "K") and (is_dsconv[ly_idx] == 1) -%}  
    {%- else -%} 
        {%- if  s_type1[ly_idx][i+6] > 0  -%}  
                            {{'\t\t\t\t'}}  SpatialMap({{size1[ly_idx][i+6]}},{{offset1[ly_idx][i+6]}}) {{dim1[i]}}; {{'\n'}}                                 
        {%- else -%} 
                            {{'\t\t\t\t'}} TemporalMap({{size1[ly_idx][i+6]}},{{offset1[ly_idx][i+6]}}) {{dim1[i]}}; {{'\n'}}                                 
        {%- endif  -%}
    {%- endif  -%}
{%- endfor -%} 

{%- endif  -%}


{%- if  cluster_lvl >= 2  -%}  
	              {{'\t\t\t\t'}} Cluster({{cluster_value1}}, P);
{% for i in range((6)) %}
    {%- if  (dim2[i] == "K") and (is_dsconv[ly_idx] == 1) -%}  
    {%- else -%} 
        {%- if  s_type2[ly_idx][i+12] > 0  -%}  
                            {{'\t\t\t\t'}}  SpatialMap({{size2[ly_idx][i+12]}},{{offset2[ly_idx][i+12]}}) {{dim2[i]}}; {{'\n'}}                                 
        {%- else -%} 
                            {{'\t\t\t\t'}} TemporalMap({{size2[ly_idx][i+12]}},{{offset2[ly_idx][i+12]}}) {{dim2[i]}}; {{'\n'}}                                 
        {%- endif  -%}
    {%- endif  -%}
{%- endfor -%} 

{%- endif  -%}
            {{'\t\t\t'}}    }                                                           
        }                                                                   
} 
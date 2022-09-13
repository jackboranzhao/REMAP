

Network {{net_name}} {                                              
        Layer {{ly_name[ly_idx]}} {                                                       
                Type: {{ly_type[ly_idx]}}                                                  
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

{%- if  cluster_lvl == 2  -%}  
	              {{'\t\t\t\t'}} Cluster({{cluster_value}}, P);
{% for i in range((6)) %}
    {%- if  (dim1[i] == "K") and (is_dsconv[ly_idx] == 1) -%}  
    {%- else -%} 
        {%- if  s_type0[ly_idx][i+6] > 0  -%}  
                            {{'\t\t\t\t'}}  SpatialMap({{size0[ly_idx][i+6]}},{{offset0[ly_idx][i+6]}}) {{dim1[i]}}; {{'\n'}}                                 
        {%- else -%} 
                            {{'\t\t\t\t'}} TemporalMap({{size0[ly_idx][i+6]}},{{offset0[ly_idx][i+6]}}) {{dim1[i]}}; {{'\n'}}                                 
        {%- endif  -%}
    {%- endif  -%}
{%- endfor -%} 

{%- endif  -%}
            {{'\t\t\t'}}    }                                                           
        }                                                                   
} 
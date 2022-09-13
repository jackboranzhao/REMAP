

Network ShuffleNet {                                              
        Layer stage_4_6_right_a {                                                       
                Type: CONV

                Stride { X: 1, Y: 1 }                                       
                Dimensions { K: 128, C: 256, R: 1, S: 1, Y: 23, X: 40 }        
                Dataflow {                                                  
				 TemporalMap(3,139) Y; 
				 TemporalMap(3,255) X; 
				 TemporalMap(1,1) K; 
				 TemporalMap(1,1) R; 
				  SpatialMap(1,14) S; 
				 TemporalMap(1,1) C; 
			    }                                                           
        }                                                                   
} 
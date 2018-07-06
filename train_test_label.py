# Sentiment analysis of collection of 100000 reviews of IMDB
# Goal: prediction of user's opinion based on her comments



# -*- coding: utf-8 -*-

import numpy as np
test_label=np.zeros(shape=len(test_data))
for i in range(0,len(test_data)):
    test_label[i]=int(test_data.id[i][len(test_data.id[i])-1])
    
    if test_label[i]==0 or test_label[i]>=7:
        test_label[i]=1
    else:
        test_label[i]=0
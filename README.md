# group_Schatten_p

For the paper:

    Xiaoqian Wang, Dinggang Shen, Heng Huang. Prediction of Memory Impairment with MRI Data: 
    A Longitudinal Study of Alzheimer's Disease. MICCAI 2016

This function optimizes the following problem:

    \min_{W,Qi} sum_{t=1}^c||X_t'*W_t-Y_t||_F^2 + gamma*(\sum_i^numG||W*Qi||_Sp^p)^k

Format of input:
    
    n: number of samples
    dim: number of features
    c: number of tasks
    T: number of time points
    inX: dim*(n*T) SNP data
    inY: n*(c*T) phenotype matrix
    numG: number of groups

Simply run the code in matlab as below:

    [outW, outQ, outObj, outNumIter, fea_rank] = L2QTrSquaretrain(inX, inY, numG, T);

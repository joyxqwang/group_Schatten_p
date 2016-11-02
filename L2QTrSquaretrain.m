%% function L2QTrSquaretrain for MICCAI2016
% Author: Xiaoqian Wang
% \min_{W,Qi} sum_{t=1}^c||X_t'*W_t-Y_t||_F^2 + gamma*(\sum_i^numG||W*Qi||_Sp^p)^k
% Xiaoqian Wang, Dinggang Shen, Heng Huang.
% Prediction of Memory Impairment with MRI Data: A Longitudinal Study of Alzheimer’s Disease.
% MICCAI 2016
function [outW, outQ, outObj, outNumIter, fea_rank] = L2QTrSquaretrain(inX, inY, numG, T, p, k, gamma, inMaxIter, inThresh)
% Input:
%   n: number of samples
%   dim: number of features
%   c: number of tasks
%   T: number of time points
%   inX: dim*(n*T) SNP data
%   inY: n*(c*T) phenotype matrix
%   numG: number of groups
% Output:
%   outW: dim*(c*T) weight matrix
%   outQ: cT*numG group indicator
%   outObj: objective function value
%   outNumIter: number of iterations
%   fea_rank: rank of the features

%% Initialization
%
if nargin < 9
    inThresh = 10^-8;
end
if nargin < 8
    inMaxIter = 5000;
end
if nargin < 7
    gamma = 1;
end
if nargin < 6
    k = 3;
end
if nargin < 5
    p = 0.1;
end
delta = 10^-12;

%
c = size(inY, 2) / T;
dim = size(inX, 2) / T;

%
s = RandStream.create('mt19937ar','seed',7);  %% seed, "457" can be changed
RandStream.setDefaultStream(s);
A = rand(c, numG);
Q = A./(sum(A,2)*ones(1,numG));

%
for ii = 1 : T
    X(:, :, ii) = inX(:, (ii-1)*dim+1:ii*dim)';    
    Y(:, :, ii) = inY(:, (ii-1)*c+1:ii*c);
    XY(:, :, ii) = X(:, :, ii)*inY(:, (ii-1)*c+1:ii*c);
    XX(:, :, ii) = X(:, :, ii)*X(:, :, ii)';
    W(:, :, ii) = (XX(:, :, ii) + gamma*eye(dim))\XY(:, :, ii);
end

%% Main Code
obj = zeros(inMaxIter, 1);

for iter = 1: inMaxIter

    % fix W, Q, update D
    [D, Qtr] = getQtraceSquare(W, Q, delta, p, k, T);
 
    % fix D, W, update Q
    for g = 1: numG        
        A(:, g) = zeros(c,1);
        for ii = 1 : T
            A(:, g) = A(:, g) + diag(W(:, :, ii)'*D{g}*W(:, :, ii));
        end
    end
    for t = 1: c
        a = A(t, :);
        Q(t,:) = 1/sum(1./a)*1./a;
    end

    % fix D, Q, update W
    for t = 1: c
        tmp = zeros(dim, dim);
        for g = 1: numG
            tmp = tmp + D{g}*(Q(t, g)^2);
        end       
        for ii = 1 : T
            W(:, t, ii) = (XX(:, :, ii) + gamma*tmp)\XY(:,t,ii);
        end
    end    

    % calculate obj
    %
    Qtr = 0;
    if dim > c
        z = zeros(dim-c,1);
    else
        z = [];
    end  
    for g = 1:numG        
        WQ = zeros(dim, c);
        for ii = 1 : T
            WQ = WQ + W(:, :, ii)*diag(Q(:,g));
        end
        [U,S,V] = svd(WQ);
        s = [diag(S); z];         
        s0 = sqrt(s.^2+delta).^p;
        Qtrg = sum(s0);
        Qtr = Qtr + Qtrg^k;
    end
    
    %
    Loss = 0;
    for ii = 1 : T
        Loss  =  norm(Y(:,:,ii) - X(:,:,ii)'*W(:,:,ii), 'fro')^2;   
    end    
    obj(iter) = Loss + gamma*Qtr;

    if(iter > 1)
        if((obj(iter-1) - obj(iter))/obj(iter-1) < inThresh)
            break;
        end
    end
    if mode(iter, 50) == 0
        fprintf('process iteration %d, the obj is %d ...\n', iter, obj(iter)); 
    end

end

%% Output
%
outNumIter = iter;
outObj = obj(1:iter);

%
[min_val, qMat] = max(Q, [], 2);
outQ = zeros(c,numG);
for t = 1:c     
    outQ(t, qMat(t)) = 1;
end

%
for ii = 1 : T
    outW(:, (ii-1)*c+1:ii*c) = W(:,:,ii);
end

%
w = sum(outW.*outW, 2);
[~, fea_rank] = sort(w, 'descend');

end


% Di = k*p/2 * (||W*Qi||_Sp^p)^(k-1) * (W*Qi*W'+delta*eye(dim))^(0.5*p-1)
function [D, Qtr] = getQtraceSquare(W, Q, delta, p, k, T)
% W: dim*(c*T) weight matrix
% Q: cT*numG group indicator

%% Initialization
%
numG = size(Q, 2);
[dim, c, ~] = size(W);
Qtr = 0;
WQ = zeros(dim, c);

if dim > c
    z = zeros(dim-c,1);
else
    z = [];
end       

%% Main code
for g = 1:numG
    
    for ii = 1 : T
        WQ = WQ + W(:, :, ii)*diag(Q(:,g));
    end
    [U,S,V] = svd(WQ);
    s = [diag(S); z]; 
    sv(:,g)=[sum(s); sum(s.^p)^k; s];
    s0 = sqrt(s.^2+delta).^p;
    Qtrg = sum(s0);
    Qtr = Qtr + Qtrg^k;
    d = sqrt(s.^2+delta).^(p-2);
    D{g} = Qtrg^(k-1)*U*diag(d)*U';

end

end
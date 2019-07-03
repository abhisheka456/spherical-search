%% Spherical Search Algorithm 
%% This package is a MATLAB/Octave source code of Spherical Search Algorithm
%% Please see the following paper:
%% (Under publication) Abhishek Kumar, Rakesh Kumar Misra, Devender Singh, 
%% Sujeet Mishra, and Swagatam Das:The Spherical Search Algorithm for Bound
%% -constrained Global Optimization problem
%% Developed in MATLAB R2018a
%% Author and programmer: Abhishek Kumar
%% Email ID: abhishek.kumar.eee13@iitbhu.ac.in, abhisheka456@gmail.com
%% Program Starts here
function [BciFitVar, BciSolution, BciIndex] = SSunopf(FHD, LU, PopSize, MaxNfes, PbestRate, rd, c, varargin)
%% Input:
% 1. FHD = objective function handle, example: @(x) x^2
% 2. LU = lower and upper limit of Search spaace, example LU = [-10 -10;10
% 10]
% 3. MaxNFes = maximum allowed function evaluation
% 4. PbestRate = parameter for towards-best
% 5. PopSize = population size
% 6. rd = parameter for rank of diagonal matrix
% 7. cmin = minimum scale of step-size
% 8. cmax = maximum scale of step-size
%% Output:
% BciFitVar = objective function value of best solution
% BciSolution = best solution.
% BciIndex = history of objective function value.
rand('seed', sum(100 * clock));
ProblemSize = size(LU,2);
%% default values
if isempty(MaxNfes)
   MaxNfes = 10000 * ProblemSize;
end
if isempty(PbestRate)
   PbestRate = 0.1;
end
if isempty(PopSize)
   PopSize = 100;
end
if isempty(rd)
   rd = 0.95;
end
if isempty(c)
    c = [0.5 0.7];
end
%% program start here
gg = 0;    
Nfes = 0;
%% Initialize the main Population
PopOld = repmat(LU(1, :), PopSize, 1) + rand(PopSize, ProblemSize) .* (repmat(LU(2, :) - LU(1, :), PopSize, 1));
Pop = PopOld; % the old Population becomes the current Population
PopOld2 = repmat(LU(1, :), PopSize, 1) + rand(PopSize, ProblemSize) .* (repmat(LU(2, :) - LU(1, :), PopSize, 1));
Fitness = feval(FHD,Pop',varargin{1});
Fitness = Fitness';
%==========================================================================
BciFitVar = 1e+30;
BciIndex = [];
BciSolution = zeros(1, ProblemSize);
for i = 1 : PopSize
    Nfes = Nfes + 1;
    if Fitness(i) < BciFitVar
       BciFitVar = Fitness(i);
       BciSolution = Pop(i, :);
    end
    if Nfes > MaxNfes; break; end
end
%==========================================================================
while Nfes < MaxNfes
    gg = gg+1;
    %% calculation of z for every individuals
    n = floor(PopSize/2);
    PopOld2 = PopOld2(randperm(PopSize),:);
    Pop = PopOld; 
    [~, SortedIndex] = sort(Fitness, 'ascend');
    ks = SortedIndex(1:n);
    ci = c(1)+(c(2)-c(1))*rand(PopSize,1);
    r0 = 1 : PopSize;
    PopAll = Pop;
    [r1, r2, r3] = genR1R2R3(PopSize, size(PopAll, 1), r0);
    pNP = max(round(PbestRate * PopSize), 2); 
    randindex = ceil(rand(1, PopSize) .* pNP); 
    randindex = max(1, randindex); 
    pbest = Pop(SortedIndex(randindex), :); 
    tep = pbest;tep(ks,:)=Pop(r3(ks),:);PopA = PopOld2(r1,:);PopA(ks,:)=PopAll(r2(ks),:);
    zi = tep - Pop(r0,:) + Pop(r1, :) - PopA;
    zi = BoundConstraint(zi, Pop, LU);
    J_= mod(floor(rand(PopSize, 1)*ProblemSize), ProblemSize) + 1;
    J = (J_-1)*PopSize + (1:PopSize)';
    bi = rand(PopSize, ProblemSize) < rd(:, ones(1, ProblemSize));
    %% calculation of Orthogonal matrix
    A = RandOrthMat(ProblemSize);
    %% calculation of yi = Pop + ci.A.diag(bi).A'zi in parallel
      TM = A;
      TM_= A';
      zi = zi*TM;
      Ur = zeros(PopSize, ProblemSize);
      Ur(J) = zi(J);
      Ur(bi) = zi(bi);
      yi = Pop(r0,:) + ci(:, ones(1, ProblemSize)) .* Ur*TM_;    
      yi = BoundConstraint(yi, Pop, LU);
      yiFitness = feval(FHD, yi', varargin{1});
      yiFitness = yiFitness';
      for i = 1 : PopSize
          Nfes = Nfes + 1;
          if yiFitness(i) < BciFitVar
              BciFitVar = yiFitness(i);
              BciSolution = yi(i, :); 
          end
          
          if Nfes > MaxNfes; break; end
      end      
      [Fitness, I] = min([Fitness, yiFitness], [], 2);
      PopOld = Pop;
      PopOld(I == 2, :) = yi(I == 2, :);
      PopOld2(I == 2, :) = Pop(I == 2, :);
      if rem(gg,10) == 1
         fprintf('best-so-far objective function at %d th iteration = %1.8e\n',gg,BciFitVar);
      end
      BciIndex(gg) = BciFitVar;
end 
end
function [r1, r2, r3] = genR1R2R3(NP1, NP2, r0)
NP0 = length(r0);
r3 = randperm(NP0);
for i = 1: 99999999
    pos = (r3 == r0);
    if sum(pos) == 0
        break;
    else
        r3 = randperm(NP0);
    end
     if i > 1000
        error('Can not genrate r3 in 1000 iterations');
     end
end
r1 = floor(rand(1, NP0) * NP1) + 1;

for i = 1 : 99999999
    pos = (r1 == r0)|(r3 == r0);
    if sum(pos) == 0
        break;
    else 
        r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end
    if i > 1000
        error('Can not genrate r1 in 1000 iterations');
    end
end

r2 = floor(rand(1, NP0) * NP2) + 1;

for i = 1 : 99999999
    pos = ((r2 == r1) | (r2 == r0)) | (r2 == r0);
    if sum(pos)==0
        break;
    else 
        r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
    end
    if i > 1000
        error('Can not genrate r2 in 1000 iterations');
    end
end
end
function zi = BoundConstraint (zi, Pop, LU)
[NP, ~] = size(Pop); 
xl = repmat(LU(1, :), NP, 1);
pos = zi < xl;
zi(pos) = (Pop(pos) + xl(pos)) / 2;
xu = repmat(LU(2, :), NP, 1);
pos = zi > xu;
zi(pos) = (Pop(pos) + xu(pos)) / 2;
end
function M=RandOrthMat(n, tol)
    if nargin==1
	  tol=1e-6;
    end
    M = zeros(n); 
    %% gram-schmidt on random column vectors
    vi = randn(n,1); 
    [~,i] = sort(abs(vi),'descend');
    vi = vi(i,:);
    M(:,1) = vi ./ norm(vi);
    
    for i=2:n
	  nrm = 0;
	  while nrm<tol
		vi = randn(n,1);
		vi = vi -  M(:,1:i-1)  * ( M(:,1:i-1).' * vi )  ;
		nrm = norm(vi);
      end
      [~,j] = sort(abs(vi(i:end,:)),'descend');
      j = j+i-1;
      M(i:end,:)=M(j,:);
      vi(i:end,:)=vi(j,:);
	  M(:,i) = vi ./ nrm;

    end        
end 
    
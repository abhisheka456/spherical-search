%% Self-Adaptive Spherical Search Algorithm 
%% This package is a MATLAB/Octave source code of Spherical Search Algorithm
%% Please see the following paper:
%% (Under publication) Abhishek Kumar, Rakesh Kumar Misra, Devender Singh, 
%% Sujeet Mishra, and Swagatam Das:The Spherical Search Algorithm for Bound
%% -constrained Global Optimization problem
%% Developed in MATLAB R2018a
%% Author and programmer: Abhishek Kumar
%% Email ID: abhishek.kumar.eee13@iitbhu.ac.in, abhisheka456@gmail.com
%% Program Starts here
function [BciFitVar, BciSolution, BciIndex] = SASSunopf(FHD, LU, PopSize, MaxNfes, PbestRate, rd, c, Ar, Ms, varargin)
%% Input:
% 1. FHD = objective function handle, example: @(x) x^2
% 2. LU = lower and upper limit of Search spaace, example LU = [-10 -10;10
% 10]
% 3. MaxNFes = maximum allowed function evaluation
% 4. PbestRate = parameter for towards-best
% 5. PopSize = maximum population size
% 6. rd = parameter for rank of diagonal matrix
% 7. c = parameter for scale of step-size
% 8. Ar = Size of Arch
% 9. Ms = Memory size of History.
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
   PbestRate = 0.11;
end
if isempty(PopSize)
   PopSize = 18*ProblemSize;
end
if isempty(rd)
   rd = 0.5;
end
if isempty(c)
    c = 0.7;
end
if isempty(Ar)
    Ar = 1.4;
end
if isempty(Ms)
    Ms = 5;
end
%% program start here
MAX_PopSize = PopSize;
MIN_PopSize = 4.0;
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
MemCI = c .* ones(Ms, 1);
MemRD = rd .* ones(Ms, 1);
MemI = 1;
Arch.NP = Ar * PopSize; 
Arch.Pop = zeros(0, ProblemSize); 
Arch.OF = zeros(0, 1);
%==========================================================================
while Nfes < MaxNfes
    gg = gg+1;
    %% calculation of z for every individuals
    n = floor(PopSize/2);
    PopOld2 = PopOld2(randperm(PopSize),:);
    Pop = PopOld; 
    [~, SortedIndex] = sort(Fitness, 'ascend');
    ks = SortedIndex(1:n);
    mem_rand_index = ceil(Ms * rand(PopSize, 1));
    MUci = MemCI(mem_rand_index);
    MUrd = MemRD(mem_rand_index);
    rd = normrnd(MUrd, 0.1);
    Term_Pos = find(MUrd == -1);
    rd(Term_Pos) = 0;
    rd = min(rd, 1);
    rd = max(rd, 0);
    ci = MUci + 0.1 * tan(pi * (rand(PopSize, 1) - 0.5));
    Pos = find(ci <= 0);
    while ~ isempty(Pos)
	     ci(Pos) = MUci(Pos) + 0.1 * tan(pi * (rand(length(Pos), 1) - 0.5));
	     Pos = find(ci <= 0);
    end
    ci = min(ci, 1);
    r0 = 1 : PopSize;
    PopAll = [Pop; Arch.Pop];
    [r1, r2, r3] = genR1R2R3(PopSize, size(PopAll, 1), r0);
    pNP = max(round(PbestRate * PopSize), 2); 
    randindex = ceil(rand(1, PopSize) .* pNP); 
    randindex = max(1, randindex); 
    pbest = Pop(SortedIndex(randindex), :); 
    tep = pbest;tep(ks,:)=Pop(r3(ks),:);PopA = PopOld2(r1,:);PopA(ks,:)=PopAll(r2(ks),:);
    zi = tep - Pop(r0,:) + Pop(r1, :) - PopA;
    %% calculation of Orthogonal matrix
    A = RandOrthMat(ProblemSize,1e-12);
    %% calculation of yi = Pop + ci.A.diag(bi).A'zi in parallel
      zi = zi*A;
      Ur = zeros(PopSize, ProblemSize);
      J = (mod(floor(rand(PopSize, 1)*ProblemSize), ProblemSize))*PopSize + (1:PopSize)';
      bi = rand(PopSize, ProblemSize) < rd(:, ones(1, ProblemSize));
      Ur(J) = zi(J);
      Ur(bi) = zi(bi);
      yi = Pop(r0,:) + ci(:, ones(1, ProblemSize)) .* Ur*A';    
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
      %%===================================================================
      dif = abs(Fitness - yiFitness);
      I = (Fitness > yiFitness);
      GdRD = rd(I == 1);  
      GdCI = ci(I == 1);
      DiffVal = dif(I == 1);  
      Arch = UpdArch(Arch, PopOld(I == 1, :), Fitness(I == 1));
      %%===================================================================
      [Fitness, I] = min([Fitness, yiFitness], [], 2);
      PopOld = Pop;
      PopOld(I == 2, :) = yi(I == 2, :);
      PopOld2(I == 2, :) = Pop(I == 2, :);
      %%===================================================================
      NumSucc = numel(GdRD);
      if NumSucc > 0 
	     SumDif = sum(DiffVal);
	     DiffVal = DiffVal / SumDif;
	     MemCI(MemI) = (DiffVal' * (GdCI .^ 2)) / (DiffVal' * GdCI);
	     if max(GdRD) == 0 || MemRD(MemI)  == -1
	        MemRD(MemI)  = -1;
         else
	        MemRD(MemI) = (DiffVal' * (GdRD .^ 2)) / (DiffVal' * GdRD);
         end
         MemI = MemI + 1;
	     if MemI > Ms 
            MemI = 1; 
         end
      end
      %%===================================================================
      Plan_PopSize = round((((MIN_PopSize - MAX_PopSize) /MaxNfes) * Nfes) + MAX_PopSize);
      if PopSize > Plan_PopSize
	     RedPop = PopSize - Plan_PopSize;
	     if PopSize - RedPop <  MIN_PopSize
            RedPop = PopSize - MIN_PopSize;
         end
         PopSize = PopSize - RedPop;
	     for r = 1 : RedPop
	         [valBest indBest] = sort(Fitness, 'ascend');
	         worst_ind = indBest(end);
	         PopOld(worst_ind,:) = [];
             PopOld2(worst_ind,:) = [];
	         Pop(worst_ind,:) = [];
	         Fitness(worst_ind,:) = [];
         end
	     Arch.NP = round(Ar * PopSize); 
         if size(Arch.Pop, 1) > Arch.NP 
	        rndPos = randperm(size(Arch.Pop, 1));
	        rndPos = rndPos(1 : Arch.NP);
	        Arch.Pop = Arch.Pop(rndPos, :);
	     end
      end

    
      if rem(gg,10) == 1
%          fprintf('best-so-far objective function at %d th iteration = %1.8e\n',gg,BciFitVar);
      end
      BciIndex(gg) = BciFitVar;
end 
end
function [r1, r2, r3] = genR1R2R3(NP1, NP2, r0)
NP0 = length(r0);
r3 = randperm(NP0);
for i = 1: 99999999
    Pos = (r3 == r0);
    if sum(Pos) == 0
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
    Pos = (r1 == r0)|(r3 == r0);
    if sum(Pos) == 0
        break;
    else 
        r1(Pos) = floor(rand(1, sum(Pos)) * NP1) + 1;
    end
    if i > 1000
        error('Can not genrate r1 in 1000 iterations');
    end
end

r2 = floor(rand(1, NP0) * NP2) + 1;

for i = 1 : 99999999
    Pos = ((r2 == r1) | (r2 == r0)) | (r2 == r0);
    if sum(Pos)==0
        break;
    else 
        r2(Pos) = floor(rand(1, sum(Pos)) * NP2) + 1;
    end
    if i > 1000
        error('Can not genrate r2 in 1000 iterations');
    end
end
end
function zi = BoundConstraint (zi, Pop, LU)
[NP, ~] = size(Pop); 
xl = repmat(LU(1, :), NP, 1);
Pos = zi < xl;
zi(Pos) = (Pop(Pos) + xl(Pos)) / 2;
xu = repmat(LU(2, :), NP, 1);
Pos = zi > xu;
zi(Pos) = (Pop(Pos) + xu(Pos)) / 2;
end
function R = RandOrthMat(n, t)
    % orthogonal matrix approx
    R = eye(n);
    l = randperm(n);
    % t = 1e-8;
    for ii = 1:floor(n/2)
        i = 2*(ii-1)+1;
        R(l(i),l(i)) = sin(t);
        R(l(i+1),l(i+1)) = sin(t);
        R(l(i),l(i+1)) = cos(t);
        R(l(i+1),l(i)) = -cos(t);
    end 
end 
function Arch = UpdArch(Arch, pop, funvalue)
if Arch.NP == 0, return; end
if size(pop, 1) ~= size(funvalue,1), error('check it'); end
popAll = [Arch.Pop; pop ];
funvalues = [Arch.OF; funvalue ];
[dummy IX]= unique(popAll, 'rows');
if length(IX) < size(popAll, 1) 
  popAll = popAll(IX, :);
  funvalues = funvalues(IX, :);
end
if size(popAll, 1) <= Arch.NP   
  Arch.Pop = popAll;
  Arch.OF = funvalues;
else                
  rndPos = randperm(size(popAll, 1)); 
  rndPos = rndPos(1 : Arch.NP);
  Arch.Pop = popAll  (rndPos, :);
  Arch.OF = funvalues(rndPos, :);
end
end   
clc;
clear all;
format long;
format compact;
ProblemSize = 30;
FF = zeros(51,30); 
Val2Reach = 10^(-8);
MaxRegion = 100.0;
MinRegion = -100.0;
LU = [-100 * ones(1, ProblemSize); 100 * ones(1, ProblemSize)];
FHD=@cec14_func;
NumPrbs = 30;
Runs = 51;
PopSize = 100;
MaxNfes = 10000*ProblemSize;
PbestRate = 0.1;
rd = 0.95;
c = [0.5,0.7];
fprintf('Running SS algorithm on D= %d\n', ProblemSize) 
for Func = 1 : NumPrbs
    Optimum = Func * 100.0;
    fprintf('\n-------------------------------------------------------\n')
    fprintf('Function = %d, Dimension size = %d\n', Func, ProblemSize);
    for run_id = 1 : Runs
%         [BciFitVar, BciSolution, BciIndex] = SSunopf(FHD, LU, PopSize, MaxNfes, PbestRate, rd, c, Func);
         [BciFitVar, BciSolution, BciIndex] = SASSunopf(FHD, LU, [], MaxNfes, [], [], [],[],[], Func);
         bci_error_val = BciFitVar - Optimum;
         if bci_error_val < Val2Reach
            bci_error_val = 0;
         end
         FF(run_id,Func)= bci_error_val;
         fprintf('%d th run, best-so-far error vaLUe = %1.8e\n', run_id , bci_error_val);
    end
fprintf('\n')
fprintf('min error vaLUe = %1.8e, max = %1.8e, median = %1.8e, mean = %1.8e, std = %1.8e\n', min(FF(:,Func)), max(FF(:,Func)), median(FF(:,Func)), mean(FF(:,Func)), std(FF(:,Func)))
end %% end 1 Function run
xlswrite('Population.xlsx',FF,1);

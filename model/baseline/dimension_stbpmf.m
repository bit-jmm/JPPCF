%%Bayesian TMF using sparse matrices stored on the disk
%% arguments
%9, 9801, 2113, 10,'movie_input.txt', 5,1
%10, 66051, 33387, 10,'beer_input.txt', 5,1
%10, 27385, 7590, 10,'foods_input.txt', 5,1
%10, 96291, 14077, 10,'epinions_input.txt', 5,1
%10, 1623, 36658, 10,'eachmovie_input.txt', 5,1
%10, 48277, 36492, 10,'flixster_input.txt', 5,1
%%
function dimension_stbpmf(num_Time, num_m, num_p, num_feat, dataset, repeat_times, continue_flag)
fprintf(1,'Running Sparse Temporal Bayesian Probabilistic Matrix Factorization (TBPMF) \n');
split_data;
%clear U V B;
sparse = 1;
  clear count
    for i = 1:num_Time
    count = makematrix(train(1:lengthofvec(i),:,i),num_m,num_p,sparse);
    path = sprintf('count%d.mat',i);
    delete(path);
    save( path, 'count'); 
    end
for randomfactor = 0:(repeat_times-1)
tic  
    fprintf(1,'Feat %4i \n',num_feat);
    if continue_flag == 1
        path = sprintf('Matrix_TPMF_time_%d_feat_%d_%d.mat',num_Time,num_feat,randomfactor+1);
        load( path, 'U', 'V', 'B');
    end
        tbayes;


    U=U_sample;
    V=V_sample;
    fprintf(1 ,'\n FEAT %4i', num_feat);
 path = sprintf('model.mat');
 delete(path);
 save( path, 'U', 'V', 'B');

clear U V B;
toc
end
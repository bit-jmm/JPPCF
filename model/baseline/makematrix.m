% Version 0.100
%
%% Create a matrix of size num_p by num_m from triplets {user_id, movie_id, rating_id}

function [count] = makematrix(train_vec,num_m,num_p,S)
if S == 0
    count = zeros(num_p,num_m,'single');
else
    count = sparse(num_p,num_m);
end
for mm=1:num_m
    ff= find(train_vec(:,2)==mm);
    count(train_vec(ff,1),mm) = train_vec(ff,3);
end
count = count';



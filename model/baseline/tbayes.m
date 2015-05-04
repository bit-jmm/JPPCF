% Version 0.100
%
% Random Initialization 0 /  Continue 1

rand('state',randomfactor);
randn('state',randomfactor);

    epoch=1;
    maxepoch=20;
    
    iter=0;
    num_V = num_m;
    num_U = num_p;
    
    % Initialize hierarchical priors
    beta=2; % observation noise (precision)
    mu_u = zeros(num_feat,1);
    mu_m = zeros(num_feat,1);
    
    alpha_u = eye(num_feat);
    alpha_m = eye(num_feat);
    
    % parameters of Inv-Whishart distribution (see paper for details)
    WI_u = eye(num_feat);
    b0_u = 2;
    df_u = num_feat;
    mu0_u = zeros(num_feat,1);
    
    WI_m = eye(num_feat);
    b0_m = 2;
    df_m = num_feat;
    mu0_m = zeros(num_feat,1);
    
    B_inc = zeros(num_feat, num_feat, num_U);
    M0 = eye(num_feat);
    
    mean_rating = mean(train_vec(:,3));
    
    pairs_tr = length(train_vec);    
 
    err_test = cell(maxepoch,1);
    if continue_flag == 0
        fprintf(1,'Initializing Bayesian TBPMF Randomly\n');
        V_sample     = 0.1*randn(num_V, num_feat,num_Time); % Movie feature vectors
        B     = zeros(num_feat,num_feat,num_U);
        U_sample(:,:,1)  = 0.1*randn(num_U, num_feat); % User feature vecators
        for i = 2:num_Time
            U_sample(:,:,i)= U_sample(:,:,1)+0.1*randn(num_U, num_feat);
        end
        for i = 1: num_U
            B(:,:,i) = eye(num_feat);
        end
    else
       fprintf(1,'Initializing Bayesian TBPMF with TPMF result\n');
       V_sample = V;
       U_sample = U;
    end
    
    counter_prob=1;

for epoch = epoch:maxepoch
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Sample from movie hyperparams (see paper for details)
    N = size(V_sample,1);
    vvariance = zeros(num_feat,num_feat,num_Time);
    vmean = zeros(num_feat,1,num_Time);
    for i = 1: num_Time
        x_bar = mean(V_sample(:,:,i))';
        
        S_bar =cov(V_sample(:,:,i));
        
        WI_post = inv(inv(WI_m) + N/1*S_bar + ...
            N*b0_m*(mu0_m - x_bar)*(mu0_m - x_bar)'/(1*(b0_m+N)));
        WI_post = (WI_post + WI_post')/2;
        df_mpost = df_m+N;
        alpha_m = wishrnd(WI_post,df_mpost);
        mu_temp = (b0_m*mu0_m + N*x_bar)/(b0_m+N);
        lam = chol( inv((b0_m+N)*alpha_m) ); lam=lam';
        mu_m = lam*randn(num_feat,1)+mu_temp;
        vvariance(:,:,i) =alpha_m;
        vmean(:,:,i) = mu_m;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Sample from user hyperparams
    N = size(U_sample,1);
    pvariance = zeros(num_feat,num_feat,num_Time);
    
    for i = 1: num_Time
        sum_u = zeros(num_feat,num_feat);
        for j = 1: N
            if i == 1
                sum_u = sum_u + U_sample(j,:,i)'*U_sample(j,:,i);
            else
                sum_u = sum_u +  (U_sample(j,:,i) - U_sample(j,:,i-1)*B(:,:,j))'*( U_sample(j,:,i) - U_sample(j,:,i-1)*B(:,:,j));
            end
        end
        WI_post = inv(inv(WI_u)+ sum_u);
        WI_post = (WI_post + WI_post')/2;
        df_mpost = df_u+N;
        alpha_u = wishrnd(WI_post,df_mpost);
        pvariance(:,:,i) = alpha_u;
     
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Sample from B hyperparams
    sum_B = zeros(num_feat,num_feat);
    for i = 1:N
        sum_B =sum_B + B(:,:,i);
    end
    B_bar =sum_B/N;
    M0_star = (M0 + N*B_bar)/(1+N);
    M = M0_star ;%+ 0.1*randn(num_feat,num_feat);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Start doing Gibbs updates over user and
    % movie feature vectors given hyperparams.
    
    for gibbs=1:2
        for time = 1:num_Time
            % fprintf(1,'\t\t Gibbs sampling %d \r', gibbs);
            temp_mean_rating = mean(train(1:lengthofvec(time),3,time));
            %%% Infer posterior distribution over all movie feature vectors
            path = sprintf('count%d.mat',time);
            load (path);
            count_temp = count';
            for mm=1:num_V
                ff = find(count_temp(:,mm)>0);
                temp_mat = U_sample(ff,:,time);
                rr = count_temp(ff,mm)-temp_mean_rating;
                covar = inv((vvariance(:,:,time)+beta*temp_mat'*temp_mat));
                mean_m = covar * (beta*temp_mat'*rr+vvariance(:,:,time)*vmean(:,:,time));
                %       lam = chol(covar); lam=lam';
                %  V_sample(mm,:,time) = 0.1*lam*randn(num_feat,1)+mean_m;
                V_sample(mm,:,time) = mean_m;
            end
            
            %%% Infer posterior distribution over all user feature vectors
            count_temp=count;
            for uu=1:num_U
                ff = find(count_temp(:,uu)>0);
                temp_mat = V_sample(ff,:,time);
                rr = count_temp(ff,uu)-temp_mean_rating;
                covar = inv((pvariance(:,:,time)+beta*temp_mat'*temp_mat));
                if time > 1
                    mean_u = covar * (beta*temp_mat'*rr+pvariance(:,:,time)*(U_sample(uu,:,time-1)*B(:,:,uu))');
                else
                    mean_u = covar * (beta*temp_mat'*rr);%+alpha_u*U_sample(uu,:,time-1)*B(:,:,uu));
                end
                %  lam = chol(covar); lam=lam';
                %   U_sample(uu,:,time) = 0.1*lam*randn(num_feat,1)+mean_u;
                if time == 1 && isempty(temp_mat)
                    U_sample(uu,:,time) =  U_sample(uu,:,time);
                else
                    U_sample(uu,:,time) = mean_u;
                end
            end
        end
        
    end
    
    for uu=1:num_U
        temp1=0;
        temp2=0;
        for bb = 2:num_Time
            temp1 = temp1+ U_sample(uu,:,bb-1)'*U_sample(uu,:,bb-1);
            temp2 = temp2+ U_sample(uu,:,bb-1)'*U_sample(uu,:,bb);           
        end
        M_star = inv(temp1+eye(num_feat))*(temp2+M);
        B(:,:,uu) = M_star;% + 0.1* randn(num_feat,num_feat);
    end
        
    fprintf(1, '\nEpoch %d\n', epoch);    
end



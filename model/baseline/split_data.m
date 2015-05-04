%% prepare data for training. %%
input = load(dataset);
Base = min(input(:,4));
End = max(input(:,4));

for timeyear = 1 : num_Time
    lengthofvec(timeyear) =  length(input(input(:,4)== (Base -1 + timeyear),:));
end
longest = max(lengthofvec);
train = zeros(longest, 4 , num_Time);
for timeyear = 1 : num_Time
    temp = input(input(:,4)== (Base -1 + timeyear),:);
    train(1:lengthofvec(timeyear) ,:,timeyear) = temp;
    
    if timeyear == 1
        train_vec =      train(1:lengthofvec(timeyear) ,:,timeyear);
    else
        train_vec = vertcat(train_vec,  train(1:lengthofvec(timeyear) ,:,timeyear));
    end
end





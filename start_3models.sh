#for (( i=15; i < 30; i++))
for k in 200
do
        for i in 0.01 0.1 100 1000 10000
        do
            nohup python demo/three_models.py $k $i >> ./log/3models_redirect_log_k_"$k"_lambda_$i.log 2>&1 &
        done
done

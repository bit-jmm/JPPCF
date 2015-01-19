#for (( i=15; i < 30; i++))
for k in 200
do
        for i in 1 10
        do
            for eta in 0.01 0.05 0.1 0.3 0.5 0.7 0.9
            do
                nohup python demo/all_models.py $k $i $eta >> ./log/all_models_log_k_"$k"_lambda_"$i"_eta_"$eta".log 2>&1 &
            done
        done
done

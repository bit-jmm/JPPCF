#for (( i=15; i < 30; i++))
for k in 200
do
        for i in 1 10
        do
            for eta in 0.01 0.03 0.05 0.07 0.1 0.3 0.5 0.7 0.9
            do
                nohup python demo/jppcf_with_topic.py $k $i $eta >> ./log/redirect_log_k_"$k"_lambda_"$i"_eta_"$eta"_predict_score_with_topic.log 2>&1 &
            done
        done
done

#!/usr/bin/env bash
#for (( i=15; i < 30; i++))
for k in 100 200
do
        for i in 1 5 10 15 20 50 100 300 500
        do
            nohup python ../model/dynamic_user_doc_num.py $k $i >> ../log/redirect_log_k_"$k"_lambda_$i.log 2>&1 &
        done
done

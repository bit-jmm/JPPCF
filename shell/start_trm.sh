#!/usr/bin/env bash
#for (( i=15; i < 30; i++))
for k in 200
do
        for i in 0.0000001 0.0001 30 50 70 90 100000 10000000
        do
            nohup python ../model/trm.py $k $i >> ../log/trm_redirect_log_k_"$k"_lambda_$i.log 2>&1 &
        done
done

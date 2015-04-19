#!/usr/bin/env bash
for k in 200
do
        for i in 10
        do
            for eta in 0.0 0.4 0.6 0.7 0.8 0.9 1.0
            do
                nohup python ../model/ttarm.py $k $i $eta >> ../log/ttarm_eta_"$eta"_redirect_log_k_"$k"_lambda_"$i".log 2>&1 &
            done
        done
done

k=200
#for (( i=15; i < 30; i++))
for i in 5 8 12 15 20 50 100 500 1000 100000 10000000
do
    nohup python demo/demo_dynamic_user_item_num.py $k $i >> ./log/redirect_log_k_"$k"_lambda_$i.log 2>&1 &
done

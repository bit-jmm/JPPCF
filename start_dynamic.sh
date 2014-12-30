k=200
#for (( i=15; i < 30; i++))
for i in 1 5 10 15 20 50 100 300 500
do
    nohup python demo/dynamic_user_doc_num.py $k $i >> ./log/redirect_log_k_"$k"_lambda_$i.log 2>&1 &
done

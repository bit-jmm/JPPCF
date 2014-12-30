k=200
#for (( i=15; i < 30; i++))
for i in 0.01 0.001 0.0001 0.00001 0.1 1000 10000 100000 1000000
do
    nohup python demo/demo_at_tsinghua_filter_by_10.py $k $i >> ./log/tsinghua_server/filter_by_10/status_log/k_"$k"_lambda_$i.txt 2>&1 &
done

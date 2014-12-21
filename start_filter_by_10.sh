#nohup python demo/demo_at_tsinghua_filter_by_10.py 100 0.5 >> ./log/tsinghua_server/filter_by_10/status_log/k_100_lambda_0.5.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 100 1 >> ./log/tsinghua_server/filter_by_10/status_log/k_100_lambda_1.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 100 5 >> ./log/tsinghua_server/filter_by_10/status_log/k_100_lambda_5.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 100 10 >> ./log/tsinghua_server/filter_by_10/status_log/k_100_lambda_10.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 100 50 >> ./log/tsinghua_server/filter_by_10/status_log/k_100_lambda_50.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 100 100 >> ./log/tsinghua_server/filter_by_10/status_log/k_100_lambda_100.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 100 500 >> ./log/tsinghua_server/filter_by_10/status_log/k_100_lambda_500.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 100 6 >> ./log/tsinghua_server/filter_by_10/status_log/k_100_lambda_6.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 100 7 >> ./log/tsinghua_server/filter_by_10/status_log/k_100_lambda_7.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 100 8 >> ./log/tsinghua_server/filter_by_10/status_log/k_100_lambda_8.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 100 9 >> ./log/tsinghua_server/filter_by_10/status_log/k_100_lambda_9.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 100 11 >> ./log/tsinghua_server/filter_by_10/status_log/k_100_lambda_11.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 100 12 >> ./log/tsinghua_server/filter_by_10/status_log/k_100_lambda_12.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 100 13 >> ./log/tsinghua_server/filter_by_10/status_log/k_100_lambda_13.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 100 14 >> ./log/tsinghua_server/filter_by_10/status_log/k_100_lambda_14.txt 2>&1 &
k=200
#for (( i=15; i < 30; i++))
for i in 2 3 4 6 7 9 11 12 13 14 16 17 18 19
do
    nohup python demo/demo_at_tsinghua_filter_by_10.py $k $i >> ./log/tsinghua_server/filter_by_10/status_log/k_"$k"_lambda_$i.txt 2>&1 &
done
#nohup python demo/demo_at_tsinghua_filter_by_10.py 200 0.5 >> ./log/tsinghua_server/filter_by_10/status_log/k_200_lambda_0.5.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 200 1 >> ./log/tsinghua_server/filter_by_10/status_log/k_200_lambda_1.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 200 5 >> ./log/tsinghua_server/filter_by_10/status_log/k_200_lambda_5.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 200 8 >> ./log/tsinghua_server/filter_by_10/status_log/k_200_lambda_8.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 200 10 >> ./log/tsinghua_server/filter_by_10/status_log/k_200_lambda_10.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 200 15 >> ./log/tsinghua_server/filter_by_10/status_log/k_200_lambda_15.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 200 20 >> ./log/tsinghua_server/filter_by_10/status_log/k_200_lambda_20.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 200 25 >> ./log/tsinghua_server/filter_by_10/status_log/k_200_lambda_25.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 200 30 >> ./log/tsinghua_server/filter_by_10/status_log/k_200_lambda_30.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 200 35 >> ./log/tsinghua_server/filter_by_10/status_log/k_200_lambda_35.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 200 40 >> ./log/tsinghua_server/filter_by_10/status_log/k_200_lambda_40.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 200 45 >> ./log/tsinghua_server/filter_by_10/status_log/k_200_lambda_45.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 200 50 >> ./log/tsinghua_server/filter_by_10/status_log/k_200_lambda_50.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 200 100 >> ./log/tsinghua_server/filter_by_10/status_log/k_200_lambda_100.txt 2>&1 &
#nohup python demo/demo_at_tsinghua_filter_by_10.py 200 500 >> ./log/tsinghua_server/filter_by_10/status_log/k_200_lambda_500.txt 2>&1 &

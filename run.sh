for i in 1 2 3 4 5
do
  #for model in trm timesvdpp wals tendsorals pmf
  for model in pmf
  do
    nohup python demo.py $model $i >> log/stdout_"$model"_k_20_"$i"_citeulike.log 2>&1 &
  done
done

for i in 1
do
  for model in trm timesvdpp wals tendsorals pmf
  do
    nohup python demo.py $model $i >> log/stdout_"$model"_k_20_"$i"_eachmovie.log 2>&1 &
  done
done

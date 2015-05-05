for i in 1
do
  #for model in trm timesvdpp wals tensorals pmf
  for model in timesvdpp tensorals
  do
    nohup python demo.py $model $i >> log/stdout_"$model"_k_20_"$i"_MovieLens.log 2>&1 &
  done
done

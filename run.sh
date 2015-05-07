for i in 5
do
  #for model in trm timesvdpp wals tensorals pmf
  for model in timesvdpp wals tensorals pmf
  do
    nohup python demo.py $model $i >> log/stdout_"$model"_k_20_"$i"_MovieLens2.log 2>&1 &
  done
done

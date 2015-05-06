for i in 1 2 3 4 5
do
  #for model in trm timesvdpp wals tensorals pmf
  for model in trm
  #for model in trm timesvdpp
  do
    nohup python demo.py $model $i >> log/stdout_"$model"_k_20_"$i"_CiteUlike2.log 2>&1 &
  done
done

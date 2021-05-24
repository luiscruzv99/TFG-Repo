for i in `seq 0 9`
do
  nvprof --system-profiling on --profile-child-processes python3 benchmark.py $i 2> perfilado$i
done
python3 agrupa_resultados.py

mv perfilado* Resultado-*/

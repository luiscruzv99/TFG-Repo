# Script que lanza 10 ejecuciones del benchmark y perfila su uso de las gpus, agrupando
# y guardando los resultados en un directorio con la fecha de la ejecución de los benchmarks

# Ejecuciones del benchmark
for i in `seq 0 1`
do
  nvprof --system-profiling on python3 benchmark.py $i 2> perfilado$i
  rm /tmp/.nvprof/*
  # sleep 5m
done

# Agrupado de los resultados de las runs
python3 agrupa_resultados.py 2> log 

# Agrupado de los perfilados en el directorio de los resultados
directorio=$(ls | grep Resultado*)
mv perfilado* $directorio/

mv $directorio ~/TFG-Resultados/

cd ~/TFG_Resultados

git add *

git commit -m "BOT: Anhadidos $directorio"

git push

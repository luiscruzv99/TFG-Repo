# Script que lanza 10 ejecuciones del benchmark y perfila su uso de las gpus, agrupando
# y guardando los resultados en un directorio con la fecha de la ejecución de los benchmarks

# Ejecuciones del benchmark
for i in `seq 0 1`
do
  nvidia-smi --query-gpu=power.draw --format=csv --filename=energia$i.csv --loop=300 & #Tomamos medicion de la energia cada 5 mins
  PID=$!
  nvprof --system-profiling on --profile-child-processes python3 benchmark.py $i 2> perfilado$i
  kill -2 $PID
  rm /tmp/.nvprof/*
  #sleep 5m
done

# Agrupado de los resultados de las runs
python3 agrupa_resultados.py

# Agrupado de los perfilados en el directorio de los resultados
directorio=$(ls | grep Resultado*)
mv perfilado* $directorio/
mv energia* $directorio/

#mv $directorio ~/TFG-Resultados/

#cd ~/TFG-Resultados

#git add *

#git commit -m "BOT: Anhadidos $directorio"

#git push

# Script que lanza 1 ejecucion del benchmark y perfila su uso del cpu, agrupando
# y guardando los resultados en un directorio con la fecha de la ejecución del benchmark

# Ejecucion del benchmark
#nvidia-smi --query-gpu=power.draw --format=csv -i 0 --filename=energia"$i".csv --loop=300 & #Tomamos medicion de la energia cada 5 mins
#PID=$!
#nvprof --system-profiling on
python3 benchmark_no_dist.py 0 cpu
#kill -2 $PID
#rm /tmp/.nvprof/*

# Agrupado de los resultados de las runs
python3 agrupa_resultados.py 1

# Agrupado de los perfilados en el directorio de los resultados
directorio=$(ls | grep "^Resultado")
mv perfilado* "$directorio/"
mv energia* "$directorio/"

#mv $directorio ~/TFG-Resultados/

#cd ~/TFG-Resultados

#git add *

#git commit -m "BOT: Anhadidos $directorio CPU solo"

#git push

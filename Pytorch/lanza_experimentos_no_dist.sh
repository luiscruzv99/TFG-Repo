# Script que lanza 10 ejecuciones del benchmark y perfila su uso de las gpus, agrupando
# y guardando los resultados en un directorio con la fecha de la ejecución de los benchmarks

# Ejecuciones del benchmark
sauna -oenergia0.dat -t python3 benchmark_no_dist.py e cuda:0 
cut energia0.dat -d ' ' -f 3,4,21,22,74,75 > tmp.dat
head -n -1 tmp.dat > energia0.dat
rm tmp.dat
sleep 1m

for i in `seq 0 4`
do
  python3 benchmark_no_dist.py $i cuda:0
  sleep 1m
done

sauna -oenergia1.dat -t python3 benchmark_no_dist.py e cuda:0
cut energia1.dat -d ' ' -f 3,4,21,22,74,75 > tmp.dat
head -n -1 tmp.dat > energia1.dat
rm tmp.dat

for i in `seq 5 9`
do
  sleep 1m
  python3 benchmark_no_dist.py $i cuda:0
done

sleep 1m
sauna -oenergia2.dat -t python3 benchmark_no_dist.py e cuda:0
cut energia2.dat -d ' ' -f 3,4,21,22,74,75 > tmp.dat
head -n -1 tmp.dat > energia2.dat
rm tmp.dat
rm e

python3 agrupa_resultados.py 10

# Agrupado de los perfilados en el directorio de los resultados
directorio=$(ls | grep Resultado*)
mv energia* $directorio/

mv $directorio ~/TFG-Resultados/

cd ~/TFG-Resultados

git add *

git commit -m "BOT: Anhadidos $directorio"

git push

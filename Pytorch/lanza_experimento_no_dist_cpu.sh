# Script que lanza 1 ejecucion del benchmark y perfila su uso del cpu, agrupando
# y guardando los resultados en un directorio con la fecha de la ejecución del benchmark

# Ejecucion del benchmark
sauna -oenergia.dat -t python3 benchmark_no_dist.py e cpu 
cut energia.dat -d ' ' -f 3,4,21,22,74,75 > tmp.dat
head -n -1 tmp.dat > energia.dat
rm tmp.dat
rm e
sleep 1m

python3 benchmark_no_dist.py 0 cpu

# Agrupado de los resultados de las runs
python3 agrupa_resultados.py 1

# Agrupado de los perfilados en el directorio de los resultados
directorio=$(ls | grep "^Resultado")

mv energia* "$directorio/"

mv $directorio ~/TFG-Resultados/

cd ~/TFG-Resultados

git add *

git commit -m "BOT: Anhadidos $directorio CPU solo"

git push

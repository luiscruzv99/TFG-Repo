# Script que lanza 1 ejecucion del benchmark , agrupando y guardando los resultados en un directorio con
# la fecha de la ejecución del benchmark

# Ejecuciones del benchmark
python3 benchmark.py 0

# Agrupado de los resultados de las runs
python3 agrupa_resultados.py 1

# Agrupado de los perfilados en el directorio de los resultados
directorio=$(ls | grep Resultado*)
mv perfilado* $directorio/
mv energia* $directorio/

#mv $directorio ~/TFG-Resultados/

#cd ~/TFG-Resultados

#git add *

#git commit -m "BOT: Anhadidos $directorio Hibrido"

#git push

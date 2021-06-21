# Script que lanza 1 ejecucion del benchmark , agrupando y guardando los resultados en un directorio con
# la fecha de la ejecución del benchmark

# Ejecuciones del benchmark
python3 benchmark_dist_hibrido.py 0

# Agrupado de los resultados de las runs
python3 agrupa_resultados.py 1

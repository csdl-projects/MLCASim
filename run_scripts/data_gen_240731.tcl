module load license/license
module load hspice/2021
module load python/3.9.5
module list

#TRAIN
python3 ../src/genCircuitData.py -s 1920 -d 2 -x 100 -y 100 -v 2.7 &
python3 ../src/genCircuitData.py -s 1920 -d 2 -x 100 -y 100 -v 2.9 &
python3 ../src/genCircuitData.py -s 1920 -d 2 -x 100 -y 100 -v 3.1 &
python3 ../src/genCircuitData.py -s 2 -d 1080 -x 100 -y 100 -v 2.7 &
python3 ../src/genCircuitData.py -s 2 -d 1080 -x 100 -y 100 -v 2.9 &
python3 ../src/genCircuitData.py -s 2 -d 1080 -x 100 -y 100 -v 3.1 &

#TEST
#python3 ../src/genCircuitData.py -s 1920 -d 2 -x 100 -y 100 -v 3.0 &


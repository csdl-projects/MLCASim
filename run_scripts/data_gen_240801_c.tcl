module load license/license
module load hspice/2021
module load python/3.9.5
module list

#TRAIN
python3 ../src/genCircuitData.py -s 64 -d 36 -x 8 -y 6 -v 2.7 -c 0.016 &
python3 ../src/genCircuitData.py -s 64 -d 36 -x 8 -y 6 -v 2.9 -c 0.016 &
python3 ../src/genCircuitData.py -s 64 -d 36 -x 8 -y 6 -v 3.1 -c 0.016 &
python3 ../src/genCircuitData.py -s 64 -d 2 -x 8 -y 6 -v 2.7 -c 0.016 &
python3 ../src/genCircuitData.py -s 64 -d 2 -x 8 -y 6 -v 2.9 -c 0.016 &
python3 ../src/genCircuitData.py -s 64 -d 2 -x 8 -y 6 -v 3.1 -c 0.016 &
python3 ../src/genCircuitData.py -s 2 -d 36 -x 8 -y 6 -v 2.7 -c 0.016 &
python3 ../src/genCircuitData.py -s 2 -d 36 -x 8 -y 6 -v 2.9 -c 0.016 &
python3 ../src/genCircuitData.py -s 2 -d 36 -x 8 -y 6 -v 3.1 -c 0.016 &

#TEST
python3 ../src/genCircuitData.py -s 64 -d 36 -x 8 -y 6 -v 3.0 -c 0.016 &
python3 ../src/genCircuitData.py -s 64 -d 36 -x 8 -y 6 -v 4.0 -c 0.016 &

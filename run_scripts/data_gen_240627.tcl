module load license/license
module load hspice/2021
module list

#TRAIN
#python3 ../src/generate_circuitarray.py -s 64 -d 36 -x 8 -y 6 -v 2.7 &
#python3 ../src/generate_circuitarray.py -s 64 -d 36 -x 8 -y 6 -v 2.9 &
#python3 ../src/generate_circuitarray.py -s 64 -d 36 -x 8 -y 6 -v 3.1 &
#python3 ../src/generate_circuitarray.py -s 64 -d 2 -x 8 -y 6 -v 2.7 &
#python3 ../src/generate_circuitarray.py -s 64 -d 2 -x 8 -y 6 -v 2.9 &
#python3 ../src/generate_circuitarray.py -s 64 -d 2 -x 8 -y 6 -v 3.1 &
#python3 ../src/generate_circuitarray.py -s 2 -d 36 -x 8 -y 6 -v 2.7 &
#python3 ../src/generate_circuitarray.py -s 2 -d 36 -x 8 -y 6 -v 2.9 &
#python3 ../src/generate_circuitarray.py -s 2 -d 36 -x 8 -y 6 -v 3.1 &

#TEST
python3 ../src/generate_circuitarray.py -s 64 -d 36 -x 8 -y 6 -v 3.0 &

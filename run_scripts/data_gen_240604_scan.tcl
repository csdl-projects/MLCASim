module load license/license
module load hspice/2021
module list

python3 ../src/generate_circuitarray.py -s 64 -d 2 -x 10 -y 10 -v 4.5 &
python3 ../src/generate_circuitarray.py -s 64 -d 2 -x 10 -y 10 -v 4.8 &
python3 ../src/generate_circuitarray.py -s 64 -d 2 -x 10 -y 10 -v 3.9 &
python3 ../src/generate_circuitarray.py -s 64 -d 2 -x 10 -y 10 -v 3.1 &
python3 ../src/generate_circuitarray.py -s 64 -d 2 -x 10 -y 10 -v 3.0 &
python3 ../src/generate_circuitarray.py -s 64 -d 2 -x 10 -y 10 -v 2.7 &
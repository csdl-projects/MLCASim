module load license/license
module load hspice/2021
module list

python3 ../src/generate_circuitarray.py -s 2 -d 1080 -x 2 -y 50 -v 4.5 &
python3 ../src/generate_circuitarray.py -s 2 -d 1080 -x 2 -y 50 -v 4.8 &
python3 ../src/generate_circuitarray.py -s 2 -d 1080 -x 2 -y 50 -v 3.9 &
python3 ../src/generate_circuitarray.py -s 2 -d 1080 -x 2 -y 50 -v 3.1 &
python3 ../src/generate_circuitarray.py -s 2 -d 1080 -x 2 -y 50 -v 3.0 &
python3 ../src/generate_circuitarray.py -s 2 -d 1080 -x 2 -y 50 -v 2.7 &

#!/bin/bash

echo $1
input_file="$1"
mkdir elec 
cd elec
cp ../$input_file .
python ../elec.py $input_file
cd ..

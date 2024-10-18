
#!/bin/bash



echo $1
input_file="$1"


mkdir spms 
cd spms
cp ../$input_file .
python ../../script/spms.py $input_file
cd ..


#!/bin/bash


Multiwfn="/root/App/Multiwfn_3.8_dev_bin_Linux/Multiwfn"

echo $1
input_file="$1"
## 
chk_file="${input_file/.xyz/.chk}"
gjf_file="${input_file/.xyz/.gjf}"

mkdir gaulog
cd gaulog
cp ../$input_file .

# >>>>> #
cat <<EOF > "$gjf_file"
%chk=${chk_file}
#p b3lyp/6-31g(d) em=gd3  pop=NBO6

Title Card Required

0 1
$(tail -n +3 "$input_file")



EOF


# >>>>> #
cat <<EOF > N.gjf
%chk=N.chk
#P B3LYP/6-31G* em=gd3 out=wfn

Title Card Required

0 1
$(tail -n +3 "$input_file")

N.wfn

EOF


# >>>>> #
cat <<EOF > N+1.gjf
%chk=N+1.chk
#P B3LYP/6-31G* em=gd3 out=wfn

Title Card Required

-1 2
$(tail -n +3 "$input_file")

N+1.wfn

EOF

# >>>>> #
cat <<EOF > N-1.gjf
%chk=N-1.chk
#P B3LYP/6-31G* em=gd3 out=wfn

Title Card Required

1 2
$(tail -n +3 "$input_file")

N-1.wfn

EOF
# >>>>> #
##


g16 N.gjf
g16 N+1.gjf
g16 N-1.gjf
g16 $gjf_file


formchk $chk_file
echo -e "22\n2\n0\nq" | $Multiwfn N.wfn >wfn.out
echo -e "9\n1\ny\n0\nq" | $Multiwfn ${chk_file/.chk/.fchk} >> wfn.out

python ../../script/gaulog.py

cd ..

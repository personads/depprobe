# Get data
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4611/ud-treebanks-v2.9.tgz
tar -zxvf ud-treebanks-v2.9.tgz
TARGETS=( 'UD_Ancient_Greek-PROIEL' 'UD_Arabic-PADT' 'UD_English-EWT' 'UD_Finnish-TDT' 'UD_Chinese-GSD' 'UD_Hebrew-HTB' 'UD_Korean-GSD' 'UD_Russian-GSD' 'UD_Swedish-Talbanken')
UD_PATH=data/
EXP_PATH=preds/
mkdir -p $UD_PATH
mkdir -p $EXP_PATH
mkdir configs

for TARGET in ${TARGETS[@]}; do
    mv ud-treebanks-v2.9/$TARGET $UD_PATH
done

rm -rf ud-treebanks-v2.9 ud-treebanks-v2.9.tgz

# Get MaChAmp
git clone https://bitbucket.org/ahmetustunn/mtp
cd mtp
git reset --hard 0b441bf93c555c8a60e803e46e0b45a3689f9132
python3 scripts/misc/cleanconl.py ../data/*/*conllu
cd ..



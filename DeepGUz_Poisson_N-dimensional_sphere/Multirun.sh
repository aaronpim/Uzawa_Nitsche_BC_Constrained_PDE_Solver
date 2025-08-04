for dim in 2 3 4 5 6 7 8 9 10;
do

for seed in 1 2 3;
do

python Ndpoisson-autograd-uzawa.py --SEED $seed --dim $dim
python Ndpoisson-autograd-hard.py --SEED $seed --dim $dim
python Ndpoisson-autograd.py --SEED $seed --dim $dim

done
done

# Training using Random settings
## FCN
python main.py --dataset ucihar --backbone FCN --batch_size 128 --cases random
python main.py --dataset shar --backbone FCN --batch_size 128 --cases random
python main.py --dataset hhar --backbone FCN --batch_size 128 --cases random
## SFCN
python main.py --dataset ucihar --backbone SFCN --lr 1e-3 --batch_size 128 --cases random --tau 0.75 --thresh 0.5
python main.py --dataset shar --backbone SFCN --lr 1e-3 --batch_size 128 --cases random --tau 0.25 --thresh 0.5
python main.py --dataset hhar --backbone SFCN --lr 1e-3 --batch_size 128 --cases random --tau 0.75 --thresh 0.5

# TABLE II. continuous dataset evaluation
## UCIHAR
python main.py --dataset ucihar --backbone FCN --batch_size 128 --eval --cases subject_large --target_domain 1-5
python main.py --dataset ucihar --backbone SFCN --lr 1e-3 --batch_size 128 --tau 0.75 --thresh 0.5 --eval --cases subject_large --target_domain 1
python main.py --dataset ucihar --backbone CASNN --lr 1e-3 --batch_size 128 --tau 0.75 --thresh 0.5 --eval --cases subject_large --target_domain 1
## SHAR
python main.py --dataset shar --backbone FCN --batch_size 128 --eval --cases subject_large --target_domain 2,3,5,6,9
python main.py --dataset shar --backbone SFCN --lr 1e-3 --batch_size 128 --tau 0.25 --thresh 0.5 --eval --cases subject_large --target_domain 2
python main.py --dataset shar --backbone CASNN --lr 1e-3 --batch_size 128 --tau 0.75 --thresh 0.5 --eval --cases subject_large --target_domain 2
## HHAR
python main.py --dataset hhar --backbone FCN --batch_size 128 --eval --cases subject_large --target_domain a,c
python main.py --dataset hhar --backbone SFCN --lr 1e-3 --batch_size 128 --tau 0.75 --thresh 0.5 --eval --cases subject --target_domain a
python main.py --dataset hhar --backbone CASNN --lr 1e-3 --batch_size 128 --tau 0.75 --thresh 0.5 --eval --cases subject --target_domain a

# Training using Cross-person settings
python main.py --dataset ucihar --backbone FCN --batch_size 128 --cases subject_large --target_domain 1
python main.py --dataset ucihar --backbone SFCN --batch_size 128 --cases subject_large --target_domain 8 --tau 0.75 --thresh 0.5
python main.py --dataset ucihar --backbone CASNN --batch_size 128 --tau 0.75 --thresh 0.5 --eval --cases subject_large --target_domain 1 

# EE Threshold Scan
python main.py --dataset ucihar --backbone CASNN --batch_size 128 --tau 0.75 --thresh 0.5 --eval --eescan --cases subject_large --target_domain 1
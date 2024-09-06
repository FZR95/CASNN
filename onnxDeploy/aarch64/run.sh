model_path=$1
repeats=${2:-5} # 5 repeats by default
loader_path="./testloader/"
sample_file="samples_ucihar_1.txt"
target_file="targets_ucihar_1.txt"
./buildx64/mtest $model_path $loader_path$sample_file $loader_path$target_file $repeats
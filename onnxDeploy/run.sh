model_path=$1
repeats=${3:-5} # 5 repeats by default
loader_path="./aarch64/testloader/"
sample_file="samples_ucihar_"+$2+".txt"
target_file="targets_ucihar_"+$2+".txt"
./buildx86/mtest $model_path $loader_path$sample_file $loader_path$target_file $repeats

./buildx86/mtest ~/nfs/aarch64/ucihar_lod/FCN_ucihar_lr0.0001_bs128_sw30_lod15.onnx ~/nfs/aarch64/testloader/samples_ucihar_15.txt ~/nfs/aarch64/testloader/targets_ucihar_15.txt 1

./buildx86/mtest ~/nfs/aarch64/ucihar_lod/SFCN_ucihar_lr0.0001_bs128_sw30_lod3_tau0.75_thresh0.5.onnx ~/nfs/aarch64/testloader/samples_ucihar_3.txt ~/nfs/aarch64/testloader/targets_ucihar_3.txt 1

./buildx86/mtest ~/nfs/aarch64/hhar_lod/FCN_hhar_lr0.001_bs128_sw30_lodc.onnx ~/nfs/aarch64/testloader/samples_hhar_c.txt ~/nfs/aarch64/testloader/targets_hhar_c.txt 1
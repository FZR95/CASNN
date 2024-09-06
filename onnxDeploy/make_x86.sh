rm -rf buildx86/*
cd buildx86
cmake -DBUILD_x64=ON ..
make
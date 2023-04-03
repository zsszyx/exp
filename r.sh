CUDA_VISIBLE_DEVICES=1,2 python main.py -b 128 -d market1501 \
--iters 200 --momentum 0.1

#CUDA_VISIBLE_DEVICES=1,2 python main.py -b 256 -d msmt17 \
#--iters 400 --momentum 0.1
###
#CUDA_VISIBLE_DEVICES=1,2 python main.py -b 256 -d veri \
#--iters 200 --momentum 0.1
###
#CUDA_VISIBLE_DEVICES=1,2 python main.py -b 256 -d dukemtmc \
#--iters 200 --momentum 0.1
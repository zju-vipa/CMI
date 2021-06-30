python vanilla_kd.py \
--teacher wrn40_2 \
--student wrn16_1 \
--dataset cifar10 \
--transfer_set run/cmi-preinverted-wrn402 \
--beta 0 \
--batch_size 128 \
--lr 0.1 \
--epoch 200 \
--gpu 0
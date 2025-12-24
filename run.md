```shell
CUDA_VISIBLE_DEVICES=0 nohup sh scripts/train.sh -g 1 -d wild_places -c pt-ptv3_SALAD_unet -n pt-ptv3_SALAD_unet > train_pt-ptv3_SALAD_unet.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup sh scripts/test.sh -g 1 -d wild_places -c pt-ptv3_SALAD_unet -n pt-ptv3_SALAD_unet -w model_best > test_pt-ptv3_SALAD_unet.log 2>&1 &
```
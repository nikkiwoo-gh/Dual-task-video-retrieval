trainCollection=tgif-msrvtt10k
valCollection=tv2016train
testCollection=iacc.3
rootpath=/vireo00/nikki/AVS_data
visual_feature=pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os
n_caption=2
lr=0.0001
overwrite=1
epoch=50
direction=all
cost_style=sum
lambda=0.2
postfix=run_dual_task_run0

# training
gpu=0

echo "CUDA_VISIBLE_DEVICES=$gpu python train_dual_task.py $trainCollection $valCollection $testCollection --rootpath $rootpath --overwrite $overwrite \
	--multiclass_loss_lamda ${lambda} --max_violation --learning_rate $lr --num_epochs $epoch --text_norm --visual_norm --visual_feature $visual_feature --n_caption $n_caption --direction $direction --postfix $postfix --cost_style $cost_style > output/$postfix.out"

CUDA_VISIBLE_DEVICES=$gpu python train_dual_task.py $trainCollection $valCollection $testCollection --rootpath $rootpath --overwrite $overwrite \
	--multiclass_loss_lamda ${lambda} --max_violation --learning_rate $lr --num_epochs $epoch --text_norm --visual_norm --visual_feature $visual_feature --n_caption $n_caption --direction $direction --postfix $postfix --cost_style $cost_style > output/$postfix.out



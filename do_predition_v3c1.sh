rootpath=/vireo00/nikki/AVS_data
testCollection=v3c1


logger_name=/vireo00/nikki/AVS_data/tgif-msrvtt10k/ACMMM2020/tv2016train/dual_task_concate_full_dp_0.2_measure_cosine_lambda_0.2/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/run_dual_task_0


checkpoint_name=model_best.pth.match.tar
#checkpoint_name=model_best.pth.class.tar
overwrite=0
query_sets=tv19.avs.txt,tv20.avs.txt
query_num_all=50
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python predictor_batch_querysets.py  $testCollection --checkpoint_name $checkpoint_name --query_num_all $query_num_all --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name --query_sets $query_sets


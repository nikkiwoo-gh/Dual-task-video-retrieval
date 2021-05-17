rootpath=/vireo00/nikki/AVS_data
testCollection=v3c1
topic_set=tv19
overwrite=1
topk=1000

declare -a arr=(0 3 5)
for theta in ${arr[@]}
do

    score_file=/vireo00/nikki/AVS_data/v3c1/results/${topic_set}.avs.txt/tgif-msrvtt10k/tv2016train/dual_task_concate_full_dp_0.2_measure_cosine_lambda_0.2/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/run_dual_task_0/model_best.pth.match.tar/id.sent.combined_theta0_${theta}_score.txt

    bash do_txt2xml.sh $testCollection $score_file $topic_set $topk $overwrite
    python trec_eval.py ${score_file}.xml --topk $topk --rootpath $rootpath --collection $testCollection --edition $topic_set --overwrite $overwrite

done

score_file=/vireo00/nikki/AVS_data/v3c1/results/${topic_set}.avs.txt/tgif-msrvtt10k/tv2016train/dual_task_concate_full_dp_0.2_measure_cosine_lambda_0.2/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/run_dual_task_0/model_best.pth.match.tar/id.sent.combined_theta1_0_score.txt

bash do_txt2xml.sh $testCollection $score_file $topic_set $topk $overwrite
python trec_eval.py ${score_file}.xml --topk $topk --rootpath $rootpath --collection $testCollection --edition $topic_set --overwrite $overwrite



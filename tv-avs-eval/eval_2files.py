import os
import pdb

def parse_result(res):
    resp = {}
    lines = res.split('\n')
    for line in lines:
        elems = line.split()
        if 'infAP' in elems:
            print line
        if 'infAP' == elems[0] and 'all' in line:
            return float(elems[-1])

rootpath = '/vireo00/nikki/AVS_data/'
collection = 'v3c1'
query_set='tv19'
# gt_file = os.path.join(rootpath, collection, 'TextData', 'neg.qrels.vireo')
gt_file = os.path.join(rootpath, collection, 'TextData', 'avs.qrels.'+query_set)
treceval_file = '/vireo00/nikki/AVS_data/v3c1/concept_system_boolean_query_result.txt'
# treceval_file = '/vireo00/nikki/AVS_data/v3c1/duak_task.bool.treceval'
#treceval_file='/vireo00/nikki/AVS_data/v3c1/dual_coding_boolean_query_result.txt'
#treceval_file = '/vireo00/nikki/AVS_data/v3c1/SimilarityIndex/vireo.boolean.topics.txt/tgif-msrvtt10k/tv2016train/setA/w2vvpp_resnext101-resnet152_subspace/runs_0/id.sent.score.txt.treceval'

#treceval_file = '/vireo00/nikki/AVS_data/v3c1/SimilarityIndex/vireo.boolean.split.topics.txt/tgif-msrvtt10k/tv2016train/setA/w2vvpp_resnext101-resnet152_subspace/runs_0/id.sent.score.combined.txt.treceval'
#treceval_file = '/vireo00/nikki/AVS_data/v3c1/results/vireo.boolean.split.topics.txt/tgif-msrvtt10k/tv2016train/dual_encoding_concate_full_dp_0.2_measure_cosine/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/runs_0/model_best.pth.tar/id.sent.score.combined.txt.treceval'
#treceval_file='/vireo00/nikki/AVS_data/v3c1/results/vireo.boolean.split.topics.txt/tgif-msrvtt10k/tv2016train/dual_task_concate_full_dp_0.2_measure_cosine_lambda_0.2/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/run_dual_task_res101_152_33kdata_lambda0_2_bk/model_best.pth.tar/emb.id.sent.score.txt.treceval'
#treceval_file='/vireo00/nikki/AVS_data/v3c1/results/vireo.boolean.topics.txt/tgif-msrvtt10k/tv2016train/dual_task_concate_full_dp_0.2_measure_cosine_lambda_0.2/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/run_dual_task_res101_152_33kdata_lambda0_2/model_best.pth.tar/id.sent.combined_theta0_0_score.txt.treceval'
#treceval_file='/vireo00/nikki/AVS_data/v3c1/results/vireo.boolean.topics.txt/tgif-msrvtt10k/tv2016train/dual_task_concate_full_dp_0.2_measure_cosine_lambda_0.2/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/run_dual_task_res101_152_33kdata_lambda0_2_bk/model_best.pth.tar/emb.id.sent.score.txt.treceval'

###------------new -------------

#treceval_file='/vireo00/nikki/AVS_data/v3c1/results/vireo.boolean.topics.0421.txt/tgif-msrvtt10k/tv2016train/dual_task_concate_full_dp_0.2_measure_cosine_lambda_0.2/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/run_dual_task_res101_152_33kdata_lambda0_2_bk/model_best.pth.tar/emb.id.sent.score.txt.new.treceval'
#treceval_file='/vireo00/nikki/AVS_data/v3c1/results/vireo.boolean.topics.0421.txt/tgif-msrvtt10k/tv2016train/dual_encoding_concate_full_dp_0.2_measure_cosine/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/run_ori_htgc2/model_best.pth.tar/id.sent.score.txt.treceval'

#treceval_file='/vireo00/nikki/AVS_data/v3c1/results/vireo.boolean.split.topics.0421.txt/tgif-msrvtt10k/tv2016train/dual_encoding_concate_full_dp_0.2_measure_cosine/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/run_ori_htgc2/model_best.pth.tar/id.sent.score.txt.treceval'
#treceval_file='/vireo00/nikki/AVS_data/v3c1/results/vireo.boolean.split.topics.0421.txt/tgif-msrvtt10k/tv2016train/dual_task_concate_full_dp_0.2_measure_cosine_lambda_0.2/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/run_dual_task_res101_152_33kdata_lambda0_2_bk/model_best.pth.tar/emb.id.sent.score.txt.treceval'

#treceval_file='/vireo00/nikki/AVS_data/v3c1/results/vireo.boolean.split.topics.0421.txt/tgif-msrvtt10k/tv2016train/dual_task_concate_full_dp_0.2_measure_cosine_lambda_0.2/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/run_dual_task_res101_152_33kdata_lambda0_2_bk/model_best.pth.tar/concept.id.sent.score.txt.treceval'


##for anh
treceval_file='/vireo00/nikki/AVS_data/v3c1/Anh_approach2_eval.'+query_set+'.txt'
print(treceval_file)
res_file = treceval_file +'.perf'

cmd = 'perl sample_eval.pl -q %s %s' % (gt_file, treceval_file)
res = os.popen(cmd).read()
with open(res_file, 'w') as fw:
    fw.write(res)

resp = parse_result(res)

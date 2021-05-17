from __future__ import print_function
import os
import pickle
import pdb
import numpy
import time
import logging
import numpy as np
from scipy.spatial import distance
import torch
from torch.autograd import Variable
from basic.metric import getScorer
from basic.util import AverageMeter, LogCollector
import re
from collections import Counter
from basic.generic_utils import Progbar
import tensorboard_logger as tb_logger
def l2norm(X):
    """L2-normalize columns of X
    """
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 * X / norm

def cal_error(videos, captions, measure='cosine'):
    if measure == 'cosine':
        captions = l2norm(captions)
        videos = l2norm(videos)
        errors = -1*numpy.dot(captions, videos.T)
    elif measure == 'euclidean':
        errors = distance.cdist(captions, videos, 'euclidean')
    return errors

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()


def encode_dualtask_data(model, data_loader, log_step=10, logging=print, return_ids=True):
    """Encode all videos and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    end = time.time()

    # numpy array to keep all the embeddings
    video_embs = None
    cap_embs = None
    concept_vectors = None
    class_outs  = None
    video_ids = ['']*len(data_loader.dataset)
    caption_ids = ['']*len(data_loader.dataset)
    captions_ori = [''] * len(data_loader.dataset)
    for i, (videos, captions,concept_bows,caption_ori, idxs, cap_ids, vid_ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        vid_emb, cap_emb = model.forward_matching(videos, captions, True)
        class_out = model.forward_classification(vid_emb, True)
        # initialize the numpy arrays given the size of the embeddings
        if video_embs is None:
            video_embs = np.zeros((len(data_loader.dataset), vid_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
            concept_vectors = np.zeros((len(data_loader.dataset), concept_bows.size(1)))
            class_outs  = np.zeros((len(data_loader.dataset), concept_bows.size(1)))
        # preserve the embeddings by copying from gpu and converting to numpy
        video_embs[idxs] = vid_emb.data.cpu().numpy().copy()
        cap_embs[idxs] = cap_emb.data.cpu().numpy().copy()
        concept_vectors[idxs] = concept_bows.data.cpu().numpy().copy()
        class_outs[idxs] = class_out.data.cpu().numpy().copy()

        for j, idx in enumerate(idxs):
            caption_ids[idx] = cap_ids[j]
            video_ids[idx] = vid_ids[j]
            captions_ori[idx] = caption_ori[j]
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % log_step == 0:
        #     logging('Test: [{0:2d}/{1:2d}]\t'
        #             '{e_log}\t'
        #             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #             .format(
        #                 i, len(data_loader), batch_time=batch_time,
        #                 e_log=str(model.logger)))

        del videos, captions

    if return_ids == True:
        return video_embs, cap_embs,class_outs, concept_vectors,captions_ori,video_ids, caption_ids
    else:
        return video_embs, cap_embs,class_outs,concept_vectors,captions_ori



def encode_vis_data(model, data_loader, log_step=10, logging=print, return_ids=True,sigmoid_len=11147):
    """Encode all videos and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()
    end = time.time()

    # numpy array to keep all the embeddings
    video_embs = None
    video_sigmoids = None
    video_ids = ['']*len(data_loader.dataset)

    for i, (videos, text_data, concept_bows,caption_ori,idxs, cap_ids,vid_ids) in enumerate(data_loader):

        # make sure val logger is used
        model.logger = val_logger
        # compute the embeddings
        vid_emb, sigmoid_out  = model.embed_vis(videos, False)  ##need to check it is True or False for this parameter
        # initialize the numpy arrays given the size of the embeddings
        if video_embs is None:
            video_embs = np.zeros((len(data_loader.dataset), vid_emb.size(1)))
            cap_concepts = np.zeros((len(data_loader.dataset), concept_bows.size(1)))
            captions_list = np.array([None]*len(data_loader.dataset))
            video_sigmoids = np.zeros((len(data_loader.dataset), sigmoid_len))
        # preserve the embeddings by copying from gpu and converting to numpy
        video_embs[idxs] = vid_emb.data.cpu().numpy().copy()
        video_sigmoids[idxs] = sigmoid_out.data.cpu().numpy().copy()
        cap_concepts[idxs] = concept_bows.cpu().numpy().copy()
        captions_list[idxs] = np.array(caption_ori).copy()
        for j, idx in enumerate(idxs):
            video_ids[idx] = vid_ids[j]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0:2d}/{1:2d}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del videos,text_data

    if return_ids == True:
        return video_embs,cap_concepts, video_sigmoids,captions_list,video_ids
    else:
        return video_embs,cap_concepts


def encode_vis_data_one_video(model, data_loader, log_step=10, logging=print, return_ids=True, sigmoid_len=11147):
    """Encode all videos and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()
    end = time.time()

    # numpy array to keep all the embeddings
    video_embs = None
    video_sigmoids = None
    video_ids = [''] * len(data_loader.dataset)

    for i, (videos, text_data, concept_bows, caption_ori, idxs, vid_ids) in enumerate(data_loader):

        # make sure val logger is used
        model.logger = val_logger
        # compute the embeddings
        vid_emb, sigmoid_out = model.embed_vis(videos, False)  ##need to check it is True or False for this parameter
        # initialize the numpy arrays given the size of the embeddings
        if video_embs is None:
            video_embs = np.zeros((len(data_loader.dataset), vid_emb.size(1)))
            cap_concepts = np.zeros((len(data_loader.dataset), concept_bows.size(1)))
            captions_list = np.array([None] * len(data_loader.dataset))
            video_sigmoids = np.zeros((len(data_loader.dataset), sigmoid_len))
        # preserve the embeddings by copying from gpu and converting to numpy
        video_embs[idxs] = vid_emb.data.cpu().numpy().copy()
        video_sigmoids[idxs] = sigmoid_out.data.cpu().numpy().copy()
        cap_concepts[idxs] = concept_bows.cpu().numpy().copy()
        captions_list[idxs] = np.array(caption_ori).copy()
        for j, idx in enumerate(idxs):
            video_ids[idx] = vid_ids[j]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0:2d}/{1:2d}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                i, len(data_loader), batch_time=batch_time,
                e_log=str(model.logger)))
        del videos, text_data

    if return_ids == True:
        return video_embs, cap_concepts, video_sigmoids, captions_list, video_ids
    else:
        return video_embs, cap_concepts


def get_concept_vector(query_file,concept2vec):
    """map all captions into concept vector
    """
    captions = {}
    cap_ids = []
    with open(query_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            captions[cap_id] = caption
            cap_ids.append(cap_id)

    concept_vectors = np.zeros([len(cap_ids),concept2vec.ndims])
    index = 0
    for cap_id in cap_ids:
        caption = captions[cap_id]
        if concept2vec is not None:
            concept_vector = concept2vec.mapping_exist(caption)
            if concept_vector is None:
                concept_vector = np.zeros(concept2vec.ndims)

        concept_vectors[index] = concept_vector
        index = index+1

    return concept_vectors,cap_ids


# recall@k, Med r, Mean r for Text-to-Video Retrieval
def t2i(c2i, vis_details=False, n_caption=5):
    """
    Text->Videos (Text-to-Video Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    vis_details: if true, return a dictionary for ROC visualization purposes
    """
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
    ranks = np.zeros(c2i.shape[0])

    for i in range(len(ranks)):
        d_i = c2i[i]
        inds = np.argsort(d_i)

        rank = np.where(inds == int(i/n_caption))[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return map(float, [r1, r5, r10, medr, meanr])

# recall@k, Med r, Mean r for Video-to-Text Retrieval
def i2t(c2i, n_caption=5):
    """
    Videos->Text (Video-to-Text Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    """
    #remove duplicate videos
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
    ranks = np.zeros(c2i.shape[1])

    for i in range(len(ranks)):
        d_i = c2i[:, i]
        inds = np.argsort(d_i)

        rank = np.where((inds/n_caption) == i)[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return map(float, [r1, r5, r10, medr, meanr])


# mAP for Text-to-Video Retrieval
def t2i_map(c2i, n_caption=5):
    """
    Text->Videos (Text-to-Video Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    """
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape

    scorer = getScorer('AP')
    perf_list = []
    for i in range(c2i.shape[0]):
        d_i = c2i[i, :]
        labels = [0]*len(d_i)
        labels[i/n_caption] = 1

        sorted_labels = [labels[x] for x in np.argsort(d_i)]
        current_score = scorer.score(sorted_labels)
        perf_list.append(current_score)

    return np.mean(perf_list)


# mAP for Video-to-Text Retrieval
def i2t_map(c2i, n_caption=5):
    """
    Videos->Text (Video-to-Text Retrieval)
    c2i: (5N, N) matrix of caption to video errors
    """
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape

    scorer = getScorer('AP')
    perf_list = []
    for i in range(c2i.shape[1]):
        d_i = c2i[:, i]
        labels = [0]*len(d_i)
        labels[i*n_caption:(i+1)*n_caption] = [1]*n_caption

        sorted_labels = [labels[x] for x in np.argsort(d_i)]
        current_score = scorer.score(sorted_labels)
        perf_list.append(current_score)

    return np.mean(perf_list)


def multi_label_classifiction_perf(outs,labels):
    recall = []
    pred_class_num = []
    gt_num = []
    match_num_sum = []
    topn=10
    for i in range(outs.shape[0]):
        topn_pred_class = numpy.zeros([outs.shape[1]])
        outs_i = numpy.squeeze(outs[i, :])
        label_i = numpy.squeeze(labels[i, :])

        ##get the performance of top10
        rankidx = np.argsort(outs_i)[::-1]
        rankidx = rankidx[0:topn]

        ##get the number of predicted class
        outs_pred_class_index = numpy.where(outs_i > 0.5)[0]
        outs_pred_num = outs_pred_class_index.size
        pred_class_num.append(outs_pred_num)
        i_gt_number = numpy.sum(label_i)
        gt_num.append(i_gt_number)

        recallAtk_i = 0.0
        if i_gt_number == 0:
            print('i_gt_number==0')

        topn_pred_class[rankidx] = 1
        match_candidate = numpy.multiply(topn_pred_class, label_i)
        match_index = numpy.where(match_candidate > 0)[0]
        match_num = len(match_index) * 1.0

        if (match_num > 0):
            if i_gt_number > topn:
                recallAtk_i = match_num * 1.0 / topn
            else:
                recallAtk_i = match_num * 1.0 / i_gt_number

        match_num_sum.append(match_num)
        recall.append(recallAtk_i)


    return match_num_sum,recall,gt_num,pred_class_num


def dual_task_eval(opt, val_loader, model, concept2vec,measure='cosine'):

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    # compute the encoding for all the validation video and captions


    video_embs, cap_embs, class_outs,concept_vectors,captions_list,video_ids, caption_ids = encode_dualtask_data(model, val_loader, opt.log_step,
                                                                                    logging.info)

    ##first evaluation encoder
    # we load data as video-sentence pairs
    # but we only need to forward each video once for evaluation
    # so we get the video set and mask out same videos with feature_mask
    feature_mask = []
    evaluate_videos = set()
    for video_id in video_ids:
        feature_mask.append(video_id not in evaluate_videos)
        evaluate_videos.add(video_id)
    video_embs = video_embs[feature_mask]
    class_outs = class_outs[feature_mask]
    concept_vectors = concept_vectors[feature_mask]
    captions_list = list(np.array(captions_list)[feature_mask])
    # video_ids = [x for idx, x in enumerate(video_ids) if feature_mask[idx] is True]

    ##first evaluation encoder
    c2i_all_errors = cal_error(video_embs, cap_embs, measure)
    if opt.val_metric == "recall":

        # video retrieval
        (r1i, r5i, r10i, medri, meanri) = t2i(c2i_all_errors, n_caption=opt.n_caption)
        print(" * Text to video:")
        print(" * r_1_5_10: {}".format([round(r1i, 3), round(r5i, 3), round(r10i, 3)]))
        print(" * medr, meanr: {}".format([round(medri, 3), round(meanri, 3)]))
        print(" * " + '-' * 10)

        # caption retrieval
        (r1, r5, r10, medr, meanr) = i2t(c2i_all_errors, n_caption=opt.n_caption)
        print(" * Video to text:")
        print(" * r_1_5_10: {}".format([round(r1, 3), round(r5, 3), round(r10, 3)]))
        print(" * medr, meanr: {}".format([round(medr, 3), round(meanr, 3)]))
        print(" * " + '-' * 10)

        # record metrics in tensorboard
        tb_logger.log_value('r1', r1, step=model.Eiters)
        tb_logger.log_value('r5', r5, step=model.Eiters)
        tb_logger.log_value('r10', r10, step=model.Eiters)
        tb_logger.log_value('medr', medr, step=model.Eiters)
        tb_logger.log_value('meanr', meanr, step=model.Eiters)
        tb_logger.log_value('r1i', r1i, step=model.Eiters)
        tb_logger.log_value('r5i', r5i, step=model.Eiters)
        tb_logger.log_value('r10i', r10i, step=model.Eiters)
        tb_logger.log_value('medri', medri, step=model.Eiters)
        tb_logger.log_value('meanri', meanri, step=model.Eiters)


    elif opt.val_metric == "map":
        i2t_map_score =i2t_map(c2i_all_errors, n_caption=opt.n_caption)
        t2i_map_score = t2i_map(c2i_all_errors, n_caption=opt.n_caption)
        tb_logger.log_value('i2t_map', i2t_map_score, step=model.Eiters)
        tb_logger.log_value('t2i_map', t2i_map_score, step=model.Eiters)
        print('i2t_map', i2t_map_score)
        print('t2i_map', t2i_map_score)

    encoder_currscore = 0
    if opt.val_metric == "recall":
        if opt.direction == 'i2t' or opt.direction == 'all':
            encoder_currscore += (r1 + r5 + r10)
        if opt.direction == 't2i' or opt.direction == 'all':
            encoder_currscore += (r1i + r5i + r10i)
    elif opt.val_metric == "map":
        if opt.direction == 'i2t' or opt.direction == 'all':
            encoder_currscore += i2t_map_score
        if opt.direction == 't2i' or opt.direction == 'all':
            encoder_currscore += t2i_map_score

    match_num_sum,recall,gt_num,pred_class_num = multi_label_classifiction_perf(class_outs,concept_vectors)
    print(" * Video: multi-label classification:")
    print(" * average matchnum@10: {}".format(np.average(match_num_sum)))
    print(" * average recall@10: {}".format(np.average(recall)))
    print(" * average #sigmoid>0.5: {}".format(np.average(pred_class_num)))
    print(" * average gt_num: {}".format(np.average(gt_num)))
    print(" * "+'-'*10)

    # record metrics in tensorboard
    tb_logger.log_value('class_matchnum@10_vid', np.average(match_num_sum), step=model.Eiters)
    tb_logger.log_value('class_recall@10_vid', np.average(recall), step=model.Eiters)
    tb_logger.log_value('class_#sigmoid>0.5_vid', np.average(pred_class_num), step=model.Eiters)
    tb_logger.log_value('class_gt_num', np.average(gt_num), step=model.Eiters)
    tb_logger.log_value('rsum', encoder_currscore, step=model.Eiters)

    decoder_cur_recall = np.average(recall)

    return encoder_currscore, decoder_cur_recall


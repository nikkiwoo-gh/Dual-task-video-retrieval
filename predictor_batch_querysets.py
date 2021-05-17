from __future__ import print_function
import pickle
import time
import os
import sys
sys.path.append('./util')
from util.build_concept import Concept
import torch
from torch.autograd import Variable
import evaluation
from model import Dual_Task
import util.data_provider as data
from util.vocab import Vocabulary
from util.text2vec import get_text_encoder
import h5py
import logging
import json
import numpy as np
import argparse
from basic.util import read_dict
from basic.constant import ROOT_PATH
from basic.bigfile import BigFile
from basic.common import makedirsforfile, checkToSkip
from basic.generic_utils import Progbar
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('testCollection', type=str, help='test collection')
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],  help='overwrite existed file. (default: 0)')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=10, type=int, help='Number of data loader workers.')
    parser.add_argument('--logger_name', default='runs', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth.tar', type=str, help='name of checkpoint (default: model_best.pth.tar)')
    parser.add_argument('--query_sets', type=str, default='tv19.avs.txt',
                        help='test query sets,  tv16.avs.txt,tv17.avs.txt,tv18.avs.txt for TRECVID 16/17/18.')
    parser.add_argument('--query_num_all', type=int, default=90,
                        help='number of querys for test.')
    args = parser.parse_args()
    return args


def encode_data(model, data_loader, return_ids=True,sigmoid=True,dim=11147):
    """Encode all videos and captions loadable by `data_loader`
    """
    # numpy array to keep all the embeddings
    embeddings = None
    sigmoid_outs = None
    ids = ['']*len(data_loader.dataset)
    pbar = Progbar(len(data_loader.dataset))
    for i, (datas, idxs, data_ids) in enumerate(data_loader):

        # compute the embeddings
        if sigmoid:
            emb,sigmoid_out = model(datas)
            prob_mask = sigmoid_out<=0.5
            sigmoid_out[prob_mask]=0
        else:
            emb = model(datas)
        # initialize the numpy arrays given the size of the embeddings
        if embeddings is None:
            embeddings = np.zeros((len(data_loader.dataset), emb.size(1)))
            sigmoid_outs = np.zeros((len(data_loader.dataset),dim))
        # preserve the embeddings by copying from gpu and converting to numpy
        embeddings[idxs] = emb.data.cpu().numpy().copy()
        if sigmoid:
            sigmoid_outs[idxs] = sigmoid_out.data.cpu().numpy().copy()
        for j, idx in enumerate(idxs):
            ids[idx] = data_ids[j]

        del datas
        pbar.add(len(idxs))

    if sigmoid:
        return embeddings, sigmoid_outs,ids
    else:
        if return_ids == True:
            return embeddings, ids,
        else:
            return embeddings

def compute_distances(model, data_loader,query_embs,concept_vectors, return_ids=True,sigmoid=True,dim=11147):
    """Encode all videos and captions loadable by `data_loader`
    """
    # numpy array to keep all the embeddings
    embedding_matrix = None
    concept_matrix = None
    ids = ['']*len(data_loader.dataset)
    pbar = Progbar(len(data_loader.dataset))

    for i, (datas, idxs, data_ids) in enumerate(data_loader):

        # compute the embeddings
        if sigmoid:
            emb,sigmoid_out = model(datas)
            prob_mask = sigmoid_out<=0.5
            sigmoid_out[prob_mask]=0
        else:
            emb = model(datas)
        # initialize the numpy arrays given the size of the embeddings
        if embedding_matrix is None:
            embedding_matrix = np.zeros([query_embs.shape[0],len(data_loader.dataset)])
            concept_matrix = np.zeros([query_embs.shape[0],len(data_loader.dataset)])
        # preserve the embeddings by copying from gpu and converting to numpy
        embedding_matrix[:,idxs] = query_embs.dot(emb.data.cpu().numpy().copy().T)

        if sigmoid:
            concept_matrix[:, idxs] = concept_vectors.dot(sigmoid_out.data.cpu().numpy().copy().T)
        for j, idx in enumerate(idxs):
            ids[idx] = data_ids[j]

        del datas
        pbar.add(len(idxs))

    if sigmoid:
        return embedding_matrix, concept_matrix,ids
    else:
        if return_ids == True:
            return embedding_matrix, ids,
        else:
            return embedding_matrix

def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    testCollection = opt.testCollection
    resume = os.path.join(opt.logger_name, opt.checkpoint_name)
    # encoder_resume_name = os.path.join(opt.encoder_resume_name, opt.checkpoint_name)
    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)

    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    matching_best_rsum = checkpoint['matching_best_rsum']
    classification_best_rsum = checkpoint['classification_best_rsum']

    print("=> loaded checkpoint '{}' (epoch {}, matching_best_rsum {},classification_best_rsum {})"
          .format(resume, start_epoch, matching_best_rsum, classification_best_rsum))

    options = checkpoint['opt']
    if not hasattr(options, 'concate'):
        setattr(options, "concate", "full")
    if not hasattr(options, 'loss_type'):
        setattr(options, "loss_type", "favorBCEloss")
    model = Dual_Task(options)

    model.load_state_dict(checkpoint['model'])

    model.vid_network.eval()
    model.text_encoding.eval()
    trainCollection = options.trainCollection
    valCollection = options.valCollection

    visual_feat_file = BigFile(os.path.join(rootpath, testCollection, 'FeatureData', options.visual_feature))
    assert options.visual_feat_dim == visual_feat_file.ndims
    video2frames = read_dict(os.path.join(rootpath, testCollection, 'FeatureData', 'video2frames.txt'))

    ## set bow vocabulary and encoding
    bow_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'bow',
                                  options.vocab + '.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)
    options.bow_vocab_size = len(bow_vocab)

    ## set rnn vocabulary
    rnn_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'rnn',
                                  options.vocab + '.pkl')
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    options.vocab_size = len(rnn_vocab)

    # set concept list for multi-label classification
    concept_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'concept', 'concept_list_gt5.pkl')
    concept_list = pickle.load(open(concept_file, 'rb'))
    concept2vec = get_text_encoder('bow')(concept_list, istimes=0)
    options.concept_list_size = len(concept_list)


    visual_loader = data.get_vis_data_loader(visual_feat_file, opt.batch_size, opt.workers, video2frames)

    query_num = opt.query_num_all
    concept_dim = len(concept_list)
    thetas = [0.0,0.3,0.5,1.0]
    output_dir = resume.replace(trainCollection, testCollection)
    query_sets = []
    queryset2queryidxs = {}
    queryidxstart = 0
    query_embs_all = np.zeros([query_num,options.visual_mapping_layers[1]])
    concept_vectors_all= np.zeros([query_num,options.concept_list_size])
    query_ids_all = []
    for query_set in opt.query_sets.strip().split(','):
        query_sets.append(query_set)
        output_dir_tmp = output_dir.replace(valCollection, '%s/%s/%s' % (query_set, trainCollection, valCollection))
        output_dir_tmp = output_dir_tmp.replace('/%s/' % options.cv_name, '/results/')
        pred_result_file = os.path.join(output_dir_tmp, 'id.sent.onematch.score.txt')

        print(pred_result_file)
        if checkToSkip(pred_result_file, opt.overwrite):
            continue
        try:
            makedirsforfile(pred_result_file)
        except Exception as e:
            print(e)

        # data loader prepare

        query_file = os.path.join(rootpath, testCollection, 'TextData', query_set)

        # set data loader
        query_loader = data.get_txt_data_loader(query_file, rnn_vocab, bow2vec, opt.batch_size, opt.workers)

        start = time.time()
        query_embs, query_ids = encode_data(model.embed_txt, query_loader,sigmoid=False,dim=concept_dim)
        print("encode text time: %.3f s" % (time.time() - start))

        query_file = os.path.join(rootpath, testCollection, 'TextData', query_set)
        concept_vectors, query_ids2 = evaluation.get_concept_vector(query_file, concept2vec)

        queryidxs =  np.arange(queryidxstart, queryidxstart + len(query_ids))
        queryset2queryidxs[query_set] =queryidxs
        query_embs_all[queryidxs,:]=query_embs
        concept_vectors_all[queryidxs,:]=concept_vectors

        queryidxstart=queryidxstart+len(query_ids)
        query_ids_all = query_ids_all+query_ids
        ##make sure query_ids and query_ids2 are the same
    embedding_matrix_all = None
    concept_matrix_all = None
    start = time.time()
    if embedding_matrix_all is None:
        embedding_matrix_all, concept_matrix_all, vis_ids = compute_distances(model.embed_vis, visual_loader,
                                                                              query_embs_all, concept_vectors_all,
                                                                              sigmoid=True, dim=concept_dim)
        print("encode image time: %.3f s" % (time.time() - start))

    for query_set in query_sets:
        output_dir_tmp = output_dir.replace(valCollection, '%s/%s/%s' % (query_set, trainCollection, valCollection))
        output_dir_tmp = output_dir_tmp.replace('/%s/' % options.cv_name, '/results/')
        query_idx  =[]
        for i,sample in enumerate(queryset2queryidxs[query_set]):
            query_idx.append(int(sample))
        query_ids = np.array(query_ids_all)[query_idx]
        embedding_matrix = embedding_matrix_all[query_idx,:]
        rows_min = np.min(embedding_matrix, 1)[:, np.newaxis]
        rows_max = np.max(embedding_matrix, 1)[:, np.newaxis]
        embedding_matrix_norm = (embedding_matrix - rows_min) / ((rows_max - rows_min))
        del embedding_matrix
        ##compute the distance of two embedding
        concept_matrix= concept_matrix_all[query_idx,:]
        rows_min = np.min(concept_matrix, 1)[:, np.newaxis]
        rows_max = np.max(concept_matrix, 1)[:, np.newaxis]
        concept_matrix_norm = (concept_matrix - rows_min) / ((rows_max - rows_min))
        del concept_matrix
        print("mapping concept time: %.3f s" % (time.time() - start))

        for theta in thetas:
            pred_result_file = os.path.join(output_dir_tmp,
                                            'id.sent.combined_theta' + str(theta).replace('.', '_') + '_score.txt')
            print(pred_result_file)
            combined_matrix = (1 - theta) * embedding_matrix_norm + (theta) * concept_matrix_norm
            combined_inds = np.argsort(combined_matrix, axis=1)
            with open(pred_result_file, 'w') as fout:
                for index in range(combined_inds.shape[0]):
                    ind = combined_inds[index][::-1]
                    fout.write(query_ids[index] + ' ' + ' '.join(
                        [vis_ids[i] + ' %s' % combined_matrix[index][i] for i in ind]) + '\n')
        del combined_matrix, embedding_matrix_norm, concept_matrix_norm


if __name__ == '__main__':
    main()

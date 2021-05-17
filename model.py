import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm  # clip_grad_norm_ for 0.4.0, clip_grad_norm for 0.3.1
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from loss import TripletLoss,favorpositiveBCEloss, normalBCEloss
from basic.bigfile import BigFile



def get_we_parameter(vocab, w2v_file):
    w2v_reader = BigFile(w2v_file)
    ndims = w2v_reader.ndims

    we = []
    # we.append([0]*ndims)
    for i in range(len(vocab)):
        try:
            vec = w2v_reader.read_one(vocab.idx2word[i])
        except:
            vec = np.random.uniform(-1, 1, ndims)
        we.append(vec)
    print('getting pre-trained parameter for word embedding initialization', np.shape(we)) 
    return np.array(we)

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                             fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)



class MFC(nn.Module):
    """
    Multi Fully Connected Layers
    """
    def __init__(self, fc_layers, dropout, have_dp=True, have_bn=False, have_last_bn=False):
        super(MFC, self).__init__()
        # fc layers
        self.n_fc = len(fc_layers)
        if self.n_fc > 1:
            if self.n_fc > 1:
                self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])

            # dropout
            self.have_dp = have_dp
            if self.have_dp:
                self.dropout = nn.Dropout(p=dropout)

            # batch normalization
            self.have_bn = have_bn
            self.have_last_bn = have_last_bn
            if self.have_bn:
                if self.n_fc == 2 and self.have_last_bn:
                    self.bn_1 = nn.BatchNorm1d(fc_layers[1])

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        if self.n_fc > 1:
            xavier_init_fc(self.fc1)

    def forward(self, inputs):

        if self.n_fc <= 1:
            features = inputs

        elif self.n_fc == 2:
            features = self.fc1(inputs)
            # batch noarmalization
            if self.have_bn and self.have_last_bn:
                features = self.bn_1(features)
            if self.have_dp:
                features = self.dropout(features)

        return features



class Video_netowrk(nn.Module):
    """
    video encoder and decoder
    """

    def __init__(self, opt):
        super(Video_netowrk, self).__init__()
        ##encoder setting
        self.rnn_output_size = opt.visual_rnn_size * 2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.visual_norm = opt.visual_norm
        self.vconcate = opt.vconcate
        # visual bidirectional rnn encoder
        self.rnn = nn.GRU(opt.visual_feat_dim, opt.visual_rnn_size, batch_first=True, bidirectional=True)
        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.visual_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0))
            for window_size in opt.visual_kernel_sizes
        ])
        # visual mapping
        self.visual_mapping = MFC(opt.visual_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)
        ##decoder setting
        self.decode_num_classes = opt.concept_list_size
        self.fc_layer = MFC(opt.decoder_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)
        self.sigmod = nn.Sigmoid()

    def encoding(self, videos):
        """Extract video feature vectors."""

        videos, videos_origin, lengths, vidoes_mask = videos

        # Level 1. Global Encoding by Mean Pooling According
        org_out = videos_origin

        # Level 2. Temporal-Aware Encoding by biGRU
        gru_init_out, _ = self.rnn(videos)
        mean_gru = Variable(torch.zeros(gru_init_out.size(0), self.rnn_output_size)).cuda()
        for i, batch in enumerate(gru_init_out):
            mean_gru[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = mean_gru
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        vidoes_mask = vidoes_mask.unsqueeze(2).expand(-1, -1, gru_init_out.size(2))  # (N,C,F1)
        gru_init_out = gru_init_out * vidoes_mask
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        if self.vconcate == 'full':  # level 1+2+3
            features = torch.cat((gru_out, con_out, org_out), 1)
        elif self.vconcate == 'conv':  # level 1
            features = org_out
        elif self.vconcate == 'gru':  # level 2
            features = gru_out
        elif self.vconcate == 'cnn_on_gru':  # l3
            features = con_out
        elif self.vconcate == 'gru_plus_cnn_on_gru':  # level 2+3
            features = torch.cat((gru_out, con_out), 1)
        elif self.vconcate == 'gru_plus_cnn':  # level 1+3
            features = torch.cat((gru_out, org_out), 1)
        elif self.vconcate == 'cnn_plus_cnn_on_gru':  # level 1+2
            features = torch.cat((con_out, org_out), 1)

        # mapping to common space
        features = self.visual_mapping(features)
        if self.visual_norm:
            features = l2norm(features)

        return features
    
    def decoding(self, visual_embs):
        """decode video feature."""

        features = self.fc_layer(visual_embs)
        outs = self.sigmod(features)
        return outs
    
    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(Video_netowrk, self).load_state_dict(new_state)


class Text_multilevel_encoding(nn.Module):
    """
    text Encoder
    """
    def __init__(self, opt):
        super(Text_multilevel_encoding, self).__init__()
        self.text_norm = opt.text_norm
        self.word_dim = opt.word_dim
        self.we_parameter = opt.we_parameter
        self.rnn_output_size = opt.text_rnn_size*2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.tconcate = opt.tconcate
        
        # visual bidirectional rnn encoder
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.rnn = nn.GRU(opt.word_dim, opt.text_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.text_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
            for window_size in opt.text_kernel_sizes
            ])
        
        # multi fc layers
        self.text_mapping = MFC(opt.text_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)

        self.init_weights()

    def init_weights(self):
        if self.word_dim == 500 and self.we_parameter is not None:
            self.embed.weight.data.copy_(torch.from_numpy(self.we_parameter))
        else:
            self.embed.weight.data.uniform_(-0.1, 0.1)


    def forward(self, text, *args):
        # Embed word ids to vectors
        # cap_wids, cap_w2vs, cap_bows, cap_mask = x
        cap_wids, cap_bows, lengths, cap_mask = text

        # Level 1. Global Encoding by Mean Pooling According
        org_out = cap_bows

        # Level 2. Temporal-Aware Encoding by biGRU
        cap_wids = self.embed(cap_wids)
        packed = pack_padded_sequence(cap_wids, lengths, batch_first=True)
        gru_init_out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(gru_init_out, batch_first=True)
        gru_init_out = padded[0]
        gru_out = Variable(torch.zeros(padded[0].size(0), self.rnn_output_size)).cuda()
        for i, batch in enumerate(padded[0]):
            gru_out[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        if self.tconcate == 'full': # level 1+2+3
            features = torch.cat((gru_out,con_out,org_out), 1)
        elif self.tconcate == 'conv': # level 1
            features = org_out
        elif self.tconcate == 'gru': # level 2
            features = gru_out
        elif self.tconcate == 'cnn_on_gru': # l3
            features = con_out
        elif self.tconcate == 'gru_plus_cnn_on_gru': # level 2+3
            features = torch.cat((gru_out,con_out), 1)
        elif self.tconcate == 'gru_plus_cnn': # level 1+3
            features = torch.cat((gru_out,org_out), 1)
        elif self.tconcate == 'cnn_plus_cnn_on_gru': # level 1+2
            features = torch.cat((con_out,org_out), 1)
        # mapping to common space
        features = self.text_mapping(features)
        if self.text_norm:
            features = l2norm(features)

        return features




class BaseModel(object):

    def state_dict(self):
        state_dict = [self.vid_network.state_dict(), self.text_encoding.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.vid_network.load_state_dict(state_dict[0])
        self.text_encoding.load_state_dict(state_dict[1])



    def forward_loss(self, cap_emb, vid_emb,pred_class,class_label, *agrs, **kwargs):
        """Compute the loss given pairs of video and caption embeddings
        """
        matching_loss = self.matching_loss(cap_emb, vid_emb)
        labels = Variable(class_label, requires_grad=False)  ##cap_bow may have value larger than 1
        if torch.cuda.is_available():
            labels = labels.cuda()
        classification_loss = self.classification_loss(pred_class,labels)
        loss  = matching_loss+classification_loss

        if torch.__version__ == '0.3.1':  # loss.item() for 0.4.0, loss.data[0] for 0.3.1
            self.logger.update('Le', loss.data[0], vid_emb.size(0)) 
        else:
            self.logger.update('Le', loss.item(), vid_emb.size(0))
        return loss, matching_loss, classification_loss

    def train_emb(self, videos, captions, class_label,captions_text,lengths, *args):
        """One training step given videos and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        vid_emb, cap_emb = self.forward_matching(videos, captions, False)
        pred_class = self.forward_classification(vid_emb, False)
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss, loss_matching, loss_classification  = self.forward_loss(cap_emb, vid_emb,pred_class,class_label)


        if torch.__version__ == '0.3.1':
            loss_value = loss.data[0]
            loss_matching_value = loss_matching.data[0]
            loss_classification_value = loss_classification.data[0]
        else:
            loss_value = loss.item()
            loss_matching_value = loss_matching.item()
            loss_classification_value = loss_classification.item()


        # compute gradient and do SGD step
        loss.backward()

        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

        return vid_emb.size(0), loss_value,loss_matching_value,loss_classification_value


class Dual_Task(BaseModel):
    """
    dual task network
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.vid_network = Video_netowrk(opt)
        self.text_encoding = Text_multilevel_encoding(opt)
        self.classification_loss_type = opt.classification_loss_type
        self.matching_loss_type = opt.matching_loss_type
        self.modelname = opt.postfix
        print(self.vid_network)
        print(self.text_encoding)
        if torch.cuda.is_available():
            self.vid_network.cuda()
            self.text_encoding.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        if self.matching_loss_type =='Tripletloss':
            self.matching_loss = TripletLoss(margin=opt.margin,
                                             measure=opt.measure,
                                             max_violation=opt.max_violation,
                                             cost_style=opt.cost_style,
                                             direction=opt.direction)
        elif self.matching_loss_type == 'MultipleNegativesRankingLoss':
            self.matching_loss = MultipleNegativesRankingLoss(measure=opt.measure)


        if self.classification_loss_type== 'favorBCEloss':
            self.classification_loss = favorpositiveBCEloss(opt.multiclass_loss_lamda,opt.cost_style)
        elif self.classification_loss_type== 'normalBCEloss':
            self.classification_loss = normalBCEloss(opt.cost_style)

        params_text = list(self.text_encoding.parameters())
        params_vid = list(self.vid_network.parameters())
        self.params_text = params_text
        self.params_vid = params_vid

        params= params_text+params_vid
        self.params = params


        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=opt.learning_rate)

        self.Eiters = 0


    def forward_matching(self, videos, targets, volatile=False, *args):
        """Compute the video and caption embeddings
        """
        # video data
        frames, mean_origin, video_lengths, vidoes_mask = videos
        frames = Variable(frames, volatile=volatile)
        if torch.cuda.is_available():
            frames = frames.cuda()

        mean_origin = Variable(mean_origin, volatile=volatile)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        vidoes_mask = Variable(vidoes_mask, volatile=volatile)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        videos_data = (frames, mean_origin, video_lengths, vidoes_mask)

        # text data
        captions, cap_bows, lengths, cap_masks = targets
        if captions is not None:
            captions = Variable(captions, volatile=volatile)
            if torch.cuda.is_available():
                captions = captions.cuda()

        if cap_bows is not None:
            cap_bows = Variable(cap_bows, volatile=volatile)
            if torch.cuda.is_available():
                cap_bows = cap_bows.cuda()

        if cap_masks is not None:
            cap_masks = Variable(cap_masks, volatile=volatile)
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()
        text_data = (captions, cap_bows, lengths, cap_masks)


        vid_emb = self.vid_network.encoding(videos_data)
        cap_emb = self.text_encoding(text_data)
        return vid_emb, cap_emb

    def forward_classification(self, vid_embs, volatile=False, *args):
        """Compute the video and caption embeddings
        """

        pred_class = self.vid_network.decoding(vid_embs)

        return pred_class

    def embed_vis(self, vis_data, volatile=True):
        # video data
        frames, mean_origin, video_lengths, vidoes_mask = vis_data
        frames = Variable(frames, volatile=volatile)
        if torch.cuda.is_available():
            frames = frames.cuda()

        mean_origin = Variable(mean_origin, volatile=volatile)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        vidoes_mask = Variable(vidoes_mask, volatile=volatile)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        vis_data = (frames, mean_origin, video_lengths, vidoes_mask)
        embs = self.vid_network.encoding(vis_data)
        sigmoid_out = self.vid_network.decoding(embs)
        return embs,sigmoid_out

    def embed_vis_emb_only(self, vis_data, volatile=True):
        # video data
        frames, mean_origin, video_lengths, vidoes_mask = vis_data
        frames = Variable(frames, volatile=volatile)
        if torch.cuda.is_available():
            frames = frames.cuda()

        mean_origin = Variable(mean_origin, volatile=volatile)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        vidoes_mask = Variable(vidoes_mask, volatile=volatile)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        vis_data = (frames, mean_origin, video_lengths, vidoes_mask)
        embs = self.vid_network.encoding(vis_data)
        return embs

    def embed_vis_concept_only(self, vis_data, volatile=True):
        # video data
        frames, mean_origin, video_lengths, vidoes_mask = vis_data
        frames = Variable(frames, volatile=volatile)
        if torch.cuda.is_available():
            frames = frames.cuda()

        mean_origin = Variable(mean_origin, volatile=volatile)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        vidoes_mask = Variable(vidoes_mask, volatile=volatile)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        vis_data = (frames, mean_origin, video_lengths, vidoes_mask)
        embs = self.vid_network.encoding(vis_data)
        sigmoid_out = self.vid_network.decoding(embs)
        return sigmoid_out


    def embed_txt(self, txt_data, volatile=True):
        # text data
        captions, cap_bows, lengths, cap_masks = txt_data
        if captions is not None:
            captions = Variable(captions, volatile=volatile)
            if torch.cuda.is_available():
                captions = captions.cuda()

        if cap_bows is not None:
            cap_bows = Variable(cap_bows, volatile=volatile)
            if torch.cuda.is_available():
                cap_bows = cap_bows.cuda()

        if cap_masks is not None:
            cap_masks = Variable(cap_masks, volatile=volatile)
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()
        txt_data = (captions, cap_bows, lengths, cap_masks)

        return self.text_encoding(txt_data)



NAME_TO_MODELS = {'dual_task': Dual_Task}

def get_model(name):
    assert name in NAME_TO_MODELS, '%s not supported.'%name
    return NAME_TO_MODELS[name]

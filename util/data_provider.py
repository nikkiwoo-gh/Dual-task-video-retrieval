import torch
import torch.utils.data as data
import numpy as np
import sys
sys.path.append('.')
from basic.util import getVideoId
from vocab import clean_str

VIDEO_MAX_LEN=64




def collate_dualtask_frame_gru_fn(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    videos, captions, cap_bows,concept_bows, caption_ori,idxs, cap_ids, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN, len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
        end = video_lengths[i]
        vidoes[i, :end, :] = frames[:end, :]
        videos_origin[i, :] = torch.mean(frames, 0)
        vidoes_mask[i, :end] = 1.0

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths)).long()
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None

    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

    concept_bows = torch.stack(concept_bows, 0) if concept_bows[0] is not None else None

    video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)
    text_data = (target, cap_bows, lengths, words_mask)

    return video_data, text_data,concept_bows, caption_ori,idxs, cap_ids, video_ids

def collate_frame(data):

    videos, idxs, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
            end = video_lengths[i]
            vidoes[i, :end, :] = frames[:end,:]
            videos_origin[i,:] = torch.mean(frames,0)
            vidoes_mask[i,:end] = 1.0

    video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)

    return video_data, idxs, video_ids


def collate_text(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, cap_bows, idxs, cap_ids = zip(*data)

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths)).long()
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None


    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

    text_data = (target, cap_bows, lengths, words_mask)

    return text_data, idxs, cap_ids

class Dataset4DualTask(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, visual_feat, bow2vec, concept2vec,vocab, n_caption=None, video2frames=None):
        # Captions
        self.captions = {}
        self.captions_all = {}
        self.cap_ids = []
        self.video_ids = set()
        self.video2frames = video2frames
        videoid_list = []
        with open(cap_file, 'rb') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.decode().strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                self.video_ids.add(video_id)

                if not video_id in videoid_list:
                    videoid_list.append(video_id)
                    caption_list= []
                    caption_list.append(caption)
                    self.captions_all[video_id] = caption_list
                else:
                    caption_list = self.captions_all[video_id]
                    caption_list.append(caption)
                    self.captions_all[video_id] = caption_list
        self.visual_feat = visual_feat
        self.bow2vec = bow2vec
        self.concept2vec = concept2vec
        self.vocab = vocab
        self.length = len(self.cap_ids)
        if n_caption is not None:
            assert len(self.video_ids) * n_caption == self.length, "%d != %d" % (
            len(self.video_ids) * n_caption, self.length)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        video_id = getVideoId(cap_id)

        # video
        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)

        # text
        caption_ori = self.captions[cap_id]
        caption_all = self.captions_all[video_id]
        if self.bow2vec is not None:
            cap_bow = self.bow2vec.mapping(caption_ori)
            if cap_bow is None:
                cap_bow = torch.zeros(self.bow2vec.ndims)
            else:
                cap_bow = torch.Tensor(cap_bow)
        else:
            cap_bow = None

        if self.concept2vec is not None:
            cap_concept = self.concept2vec.mapping_exist(' '.join(caption_all))
            if cap_concept is None:
                cap_concept = torch.zeros(self.concept2vec.ndims)
            else:
                cap_concept = torch.Tensor(cap_concept)
        else:
            cap_concept = None

        if self.vocab is not None:
            tokens = clean_str(caption_ori)
            caption = []
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            cap_tensor = torch.Tensor(caption)
        else:
            cap_tensor = None

        return frames_tensor, cap_tensor, cap_bow,cap_concept,' '.join(caption_all), index, cap_id, video_id

    def __len__(self):
        return self.length


class VisDataSet4VideoEncoding(data.Dataset):
    """
    Load video frame features by pre-trained CNN model.
    """
    def __init__(self, visual_feat, video2frames=None):
        self.visual_feat = visual_feat
        self.video2frames = video2frames

        self.video_ids = list(video2frames.keys())
        self.length = len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            #print(frame_id)
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)

        return frames_tensor, index, video_id

    def __len__(self):
        return self.length


class TxtDataSet4TextEncoding(data.Dataset):
    """
    Load captions
    """
    def __init__(self, cap_file, bow2vec, vocab):
        # Captions
        self.captions = {}
        self.cap_ids = []
        if isinstance(cap_file,list):
            for line in cap_file:
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        else:
            with open(cap_file, 'r') as cap_reader:
                for line in cap_reader.readlines():
                    cap_id, caption = line.strip().split(' ', 1)
                    self.captions[cap_id] = caption
                    self.cap_ids.append(cap_id)
        self.bow2vec = bow2vec
        self.vocab = vocab
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]

        caption = self.captions[cap_id]
        if self.bow2vec is not None:
            cap_bow = self.bow2vec.mapping(caption)
            if cap_bow is None:
                cap_bow = torch.zeros(self.bow2vec.ndims)
            else:
                cap_bow = torch.Tensor(cap_bow)
        else:
            cap_bow = None

        if self.vocab is not None:
            tokens = clean_str(caption)
            caption = []
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            cap_tensor = torch.Tensor(caption)
        else:
            cap_tensor = None

        return cap_tensor, cap_bow, index, cap_id

    def __len__(self):
        return self.length


def get_dualtask_data_loaders(cap_files, visual_feats, vocab, bow2vec,concept2vec, batch_size=100, num_workers=2, n_caption=2, video2frames=None):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    dset = {'train': Dataset4DualTask(cap_files['train'], visual_feats['train'], bow2vec,concept2vec, vocab, video2frames=video2frames['train']),
            'val': Dataset4DualTask(cap_files['val'], visual_feats['val'], bow2vec, concept2vec,vocab, n_caption, video2frames=video2frames['val']) }

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=batch_size,
                                    shuffle=(x=='train'),
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_dualtask_frame_gru_fn)
                        for x in cap_files }
    return data_loaders


def get_dualtask_test_data_loaders(cap_files, visual_feats, vocab, bow2vec,concept2vec, batch_size=100, num_workers=2, n_caption=2, video2frames=None):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    dset = {'test': Dataset4DualTask(cap_files, visual_feats, bow2vec,concept2vec, vocab, video2frames=video2frames)}
    data_loaders = torch.utils.data.DataLoader(dataset=dset['test'],
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_dualtask_frame_gru_fn)
    return data_loaders



def get_vis_data_loader(vis_feat, batch_size=100, num_workers=2, video2frames=None):
    dset = VisDataSet4VideoEncoding(vis_feat, video2frames)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_frame)
    return data_loader



def get_txt_data_loader(cap_file, vocab, bow2vec, batch_size=100, num_workers=2):
    dset = TxtDataSet4TextEncoding(cap_file, bow2vec, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_text)
    return data_loader


if __name__ == '__main__':
    pass

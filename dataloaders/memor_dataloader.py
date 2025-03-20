import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
import torch
from dataloaders.base_dataloader import BaseDataLoader
from utils import read_json
import numpy as np
import pandas as pd
from tqdm import tqdm
from features import AudioFeatureExtractor, TextFeatureExtractor, VisualFeatureExtractor, PersonalityFeatureExtractor
import pdb

EMOTIONS = ["neutral","joy","anger","disgust","sadness","surprise","fear","anticipation","trust","serenity","interest","annoyance","boredom","distraction"]

class MEmoRDataset(data.Dataset):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        annos = read_json(config['anno_file'])[config['emo_type']]
        ids = []
        tmp_annos = [] # only keep the annotations with the ids(train/test) in the id_file
        with open(config['id_file']) as fin:
            for line in fin.readlines():
                ids.append(int(line.strip()))
        
        for jj, anno in enumerate(annos):
            if jj in ids:
                tmp_annos.append(anno)
        annos = tmp_annos
        print(f"initialize {config['id_file']} and the length is {len(annos)}")
        
            
        emo_num = 9 if config['emo_type'] == 'primary' else 14
        self.emotion_classes = EMOTIONS[:emo_num]
        
        data = read_json(config['data_file'])
        self.visual_features, self.audio_features, self.text_features = [], [], []
        self.visual_valids, self.audio_valids, self.text_valids, self.personality_valid = [], [], [], []
        self.labels = []
        self.characters_seq = []
        self.time_seq = []
        self.target_loc = []
        self.seg_len = [] 
        self.n_character = []
        vfe = VisualFeatureExtractor(config)
        afe = AudioFeatureExtractor(config)
        tfe = TextFeatureExtractor(config)
        pfe = PersonalityFeatureExtractor(config)
        self.personality_list = pfe.get_features()
        self.personality_features = []
        # self.personality_features_valid = []
        
        # Temporary storage for clips
        clip_storage = {}

        for jj, anno in enumerate(tqdm(annos)):
            clip = anno['clip']
            target_character = anno['character']
            target_moment = anno['moment']

            if clip not in clip_storage:
                clip_storage[clip] = {
                    'characters_seq': [],
                    'time_seq': [],
                    'target_loc': [],
                    'personality_seq': [],
                    'visual_features': [],
                    'audio_features': [],
                    'text_features': [],
                    'personality_valid': [],
                    'visual_valids': [],
                    'audio_valids': [],
                    'text_valids': [],
                    'labels': [],
                    'n_character': 0,
                    'seg_len': 0
                }
                
            
            on_characters = data[clip]['on_character']
            if target_character not in on_characters:
                on_characters.append(target_character)
            on_characters = sorted(on_characters) # all characters in the video clip
            
            # personality_seq, personality_valid = [], []
            
            for character in on_characters:
                for ii in range(len(data[clip]['seg_start'])):
                    clip_storage[clip]['characters_seq'].append([0 if character != i else 1 for i in range(len(config['speakers']))]) 
                    clip_storage[clip]['time_seq'].append(ii)
                    clip_storage[clip]['personality_seq'].append(self.personality_list[character])
                    clip_storage[clip]['personality_valid'].append(1)
                    if character == target_character and data[clip]['seg_start'][ii] <= target_moment < data[clip]['seg_end'][ii]:
                        clip_storage[clip]['target_loc'].append(1)
                    else:
                        clip_storage[clip]['target_loc'].append(0)
            
            vf, v_valid = vfe.get_feature(anno['clip'], target_character)
            af, a_valid = afe.get_feature(anno['clip'], target_character)
            tf, t_valid = tfe.get_feature(anno['clip'], target_character)
            # print(f"before append****,clip_storage[clip]['personality_seq']:{clip_storage[clip]['personality_seq']}, personality valid:{clip_storage[clip]['personality_valid']}***")

            clip_storage[clip]['visual_features'].append(vf)
            clip_storage[clip]['audio_features'].append(af)
            clip_storage[clip]['text_features'].append(tf)
            clip_storage[clip]['visual_valids'].append(v_valid)
            clip_storage[clip]['audio_valids'].append(a_valid)
            clip_storage[clip]['text_valids'].append(t_valid)
            # visual_features=torch.cat(clip_storage[clip]['visual_features'],dim=0)
            # audio_features=torch.cat(clip_storage[clip]['audio_features'],dim=0)
            # text_features=torch.cat(clip_storage[clip]['text_features'],dim=0)
            # visual_valids=torch.cat(clip_storage[clip]['visual_valids'],dim=0)
            # audio_valids=torch.cat(clip_storage[clip]['audio_valids'],dim=0)
            # text_valids=torch.cat(clip_storage[clip]['text_valids'],dim=0)
            clip_storage[clip]['labels'].append(self.emotion_classes.index(anno['emotion']))
            clip_storage[clip]['n_character'] = len(on_characters)
            clip_storage[clip]['seg_len'] = len(data[clip]['seg_start'])
            # clip_storage[clip]['n_character'] += len(on_characters)
            # clip_storage[clip]['seg_len'] += len(data[clip]['seg_start'])
            # print(f"current clip ={clip}, n_character={clip_storage[clip]['n_character']}, seg_len={clip_storage[clip]['seg_len']}, v_valid={clip_storage[clip]['visual_valids']}")
            # if isinstance(visual_features, torch.Tensor):
            #     print("tensor")
            # elif isinstance(visual_features, list):
            #     print("list")
            # else:
            #     print("other type")
            # if jj == 2:
            # print(f" clip={clip}, visual features:{visual_features}, visual features shape:{visual_features.shape}, v_valid:{visual_valids},v_valid len:{len(visual_valids)},\
            #        audio features shape:{audio_features.shape}, a_valid:{len(audio_valids)}, text features shape:{text_features.shape}, t_valid:{len(text_valids)}, \
            #        labels:{clip_storage[clip]['labels']}, n_character:{clip_storage[clip]['n_character']}, seg_len:{clip_storage[clip]['seg_len']}, personality_valid:{clip_storage[clip]['personality_valid']}, personality_seq shape:{len(clip_storage[clip]['personality_valid'])}")
            # print(f"time_seq={len(clip_storage[clip]['time_seq'])}")
            total_length = sum(tensor.shape[0] for tensor in clip_storage[clip]['personality_seq'])
            total_length_personality_valid = len(clip_storage[clip]['personality_valid'])
            # print(f"len(characters_seq)={len(clip_storage[clip]['characters_seq'])},len(time_seq)={len(clip_storage[clip]['time_seq'])},len(target_loc)={len(clip_storage[clip]['target_loc'])}")
            # print('target_loc =',clip_storage[clip]['target_loc'],  'personality_seq feature len=', total_length, 'personality_valid len=', total_length_personality_valid)
            self.umask = torch.tensor([1]*len(clip_storage[clip]['time_seq']), dtype=torch.int8)
            # print(f"len(self.umask)={len(self.umask),self.umask}")
            seq_lengths = [(self.umask[j] == 1).nonzero(as_tuple=True)[-1].item()+1 for j in range(len(self.umask))]
            # print(f"seq_lengths={len(seq_lengths)}, current label={clip_storage[clip]['labels']}")
            # pdb.set_trace()
            # self.personality_features.append(torch.stack(personality_seq))
            # self.personality_valid.append(torch.tensor(personality_valid))
            

        # Combine features and sequences for each clip
        for clip in clip_storage:


            self.n_character.append(clip_storage[clip]['n_character'])
            self.seg_len.append(clip_storage[clip]['seg_len'])
            self.personality_features.append(torch.stack(clip_storage[clip]['personality_seq']))
            self.characters_seq.append(torch.tensor(clip_storage[clip]['characters_seq']))
            self.time_seq.append(torch.tensor(clip_storage[clip]['time_seq']))
            self.target_loc.append(torch.tensor(clip_storage[clip]['target_loc'], dtype=torch.int8))
            self.visual_features.append(torch.cat(clip_storage[clip]['visual_features'],dim=0))  # List of visual features for this clip
            self.audio_features.append(torch.cat(clip_storage[clip]['audio_features'],dim=0))
            self.text_features.append(torch.cat(clip_storage[clip]['text_features'],dim=0))
            self.visual_valids.append(torch.cat(clip_storage[clip]['visual_valids'],dim=0))
            self.audio_valids.append(torch.cat(clip_storage[clip]['audio_valids'],dim=0))
            self.text_valids.append(torch.cat(clip_storage[clip]['text_valids'],dim=0))
            self.personality_valid.append(torch.tensor(clip_storage[clip]['personality_valid']))
            self.labels.append(torch.tensor(clip_storage[clip]['labels']))  # List of labels for this clip
            # if isinstance(self.personality_features, torch.Tensor):
            #     print("tensor")
            # elif isinstance(self.personality_features, list):
            #     print("list")
            # else:
            #     print("other type")
            # print(f'n_character={self.n_character}, seg_len={self.seg_len},  target_loc={self.target_loc}, visual_valids shape={self.visual_valids}, audio_valids shape={self.audio_valids},labels={self.labels}')
            # pdb.set_trace()
        # print(f"visual_features shape={len(self.visual_features)}, audio_features shape={len(self.audio_features)}, text_features shape={len(self.text_features)}")
        # print(f'target_loc={self.target_loc}, n_character={self.n_character}, seg_len={self.seg_len}, charcaters_seq={self.characters_seq}, time_seq={self.time_seq}')
        # pdb.set_trace()
            # self.n_character.append(len(on_characters))
            # self.seg_len.append(len(data[clip]['seg_start'])) # number of segments in the video clip
    
            # self.personality_features.append(torch.stack(personality_seq))
            # self.characters_seq.append(torch.tensor(characters_seq))
            # self.time_seq.append(torch.tensor(time_seq)) # time sequence
            # self.target_loc.append(torch.tensor(target_loc, dtype=torch.int8))
            # self.visual_features.append(vf)
            # self.audio_features.append(af)
            # self.text_features.append(tf)
            # self.visual_valids.append(v_valid)
            # self.audio_valids.append(a_valid)
            # self.text_valids.append(t_valid)
            # self.personality_features_valid.append(torch.tensor(personality_valid)) #  personality mask
            # self.labels.append(self.emotion_classes.index(anno['emotion']))            
    
    # def _get_labels(self, index):
    #     label_list = self.labels[index]
    #     label = np.zeros(6, dtype=np.float32)
    #     filter_label = label_list[1:-1]
    #     for emo in filter_label:
    #         label[emotion_dict[emo]] = 1
    #     return label

    def __getitem__(self, index):
        
        return self.labels[index], \
            self.visual_features[index], \
            self.audio_features[index], \
            self.text_features[index], \
            self.personality_features[index], \
            self.visual_valids[index], \
            self.audio_valids[index], \
            self.text_valids[index], \
            self.personality_valid[index], \
            self.target_loc[index], \
            torch.tensor([1]*len(self.time_seq[index]), dtype=torch.int8), \
            torch.tensor([self.seg_len[index]], dtype=torch.int8), \
            torch.tensor([self.n_character[index]], dtype=torch.int8)
            

    def __len__(self):
        return len(self.visual_features)

    def collate_fn(self, data):
        # print(f"data={data}")
        dat = pd.DataFrame(data)
        try:
            return [pad_sequence(dat[i], batch_first=True, padding_value=-1) if i == 0 else pad_sequence(dat[i], batch_first=True) for i in dat]
        except Exception as e:
            print(f"Error: {e}, data={data}")
        

    def statistics(self):
        all_emotion = [0] * len(self.emotion_classes)
        for emotion_list in self.labels:
            for emotion in emotion_list:
                emotion = emotion.item()
                if isinstance(emotion, int):  # ensure emotion in int
                    all_emotion[emotion] += 1
                else:
                    raise ValueError(f"Expected an integer, but got {type(emotion).__name__}: {emotion}")
        return all_emotion


class MEmoRDataLoader(BaseDataLoader):
    def __init__(self, config, training=True):
        data_loader_config = config['data_loader']['args']
        self.seed = data_loader_config['seed']
        self.dataset = MEmoRDataset(config)
        print(f"the length of dataset is {len(self.dataset)},{self.dataset.__len__()}")
        self.emotion_nums = self.dataset.statistics()
        super().__init__(self.dataset, data_loader_config['batch_size'], data_loader_config['shuffle'], data_loader_config['validation_split'], data_loader_config['num_workers'], collate_fn=self.dataset.collate_fn)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(self.seed)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        weights_per_class = 1. / torch.tensor(self.emotion_nums, dtype=torch.float)
        # print(f"weights_per_class={weights_per_class}, self.n_samples={self.n_samples}")
        weights = [0] * self.n_samples
        for idx in range(self.n_samples):
            if idx in valid_idx:
                weights[idx] = 0.
            else:
                labels = self.dataset[idx][0]
                # print(f"label={labels}")
                if torch.is_tensor(labels):
                    labels = labels.flatten().tolist()  # 转换为列表
                sample_weights = [weights_per_class[label] for label in labels]
                weights[idx] = torch.tensor(round(np.mean(sample_weights),4))# mean value as weight，max：max(sample_weights).item()
        # weights = torch.tensor(weights, dtype=torch.float)
        train_sampler = data.WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    
        valid_sampler = data.SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)
        print(f"train_idx={len(train_idx)},valid_idx={len(valid_idx)}")

        return train_sampler, valid_sampler

import csv
import librosa
import os
import torch
import torchaudio


class Dataset(torch.utils.data.Dataset):
    '''
    Return a dataset from Magnatagatune data.
    
    Parameters :
    root :   Path where the folders are with the audio files. 
    subset : Name of the subset to be load. Options:
             {'train', 'test', 'val'}
             'train': Load the train subset.
             'test':  Load the test subset.
             'val':   Load the validation subset.
    trans:   List whose items are 2-lenght-tuples that indicates the Data Augmentations to aply and its probability respectively.
    '''
    
    def __init__(self, root, subset, trans):
        assert subset in {'train', 'test', 'val'}, '''
        Invalid subset argument; the options are:
        {'train', 'test', 'val'}.
        '''
        self.root = root
        self.indexes = self.read_index()
        self.subset = subset
        self.data, self.labels = self.read_files()
        self.trans = trans
        
    def __len__(self):
        return len(self.data)
        
    def read_index(self):
        index_tsv_file = open(os.path.join(self.root, "index_mtt.tsv"))
        index_tsv = csv.reader(index_tsv_file, delimiter="\t")
        idx_dict = {row[0]:row[1] for row in index_tsv }
        return idx_dict
    
    def read_files(self):
        data, labels = [], []
        tsv_file = open(os.path.join(self.root, "{}_gt_mtt.tsv".format(self.subset)))
        tsv = csv.reader(tsv_file, delimiter="\t")
        for row in tsv:
            data.append((row[0], self.indexes[row[0]]))
            labels.append((row[0], row[1]))
        return data, labels 
    
    def __getitem__(self, n):
        label = self.labels[n][1]
        label = torch.FloatTensor(eval(label))
        
        audio_path = self.root + self.data[n][1]
        audio, sample_rate = librosa.load(audio_path)
        audio = torch.from_numpy(audio).float()
        audio = audio.view(1, audio.shape[0])
        if self.trans:
            audio = self.augmentations(audio)
        return audio, label
            
    def augmentations(self, audio):
        augmented_samples = 0
        while augmented_samples !=1:
            for tr in self.trans:
                if tr[1] < torch.rand(1):
                    pass
                audio = tr[0](audio)
            augmented_samples+=1
        return audio
from reader.preprocess import open_file, encode, get_batch, pad_remain
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import os

train_test_ratio = float(os.getenv('TRAIN_TEST_RATIO', '0.9'))
torch.manual_seed(1337)
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '4'))
CONTEXT_WINDOW_SIZE = int(os.getenv('CONTEXT_WINDOW_SIZE', '8'))

class TextDataset(Dataset):
    def __init__(self, data_directories: Path, device):
        self.data_dir = data_directories
        self.device = device
        self.preprocess()

    def vocab_builder(self, full_corpus: str):
        self.vocab = sorted(list(set(full_corpus)))
        self.vocab_size = len(self.vocab)
        self.stoi = {c:i for i,c in enumerate(self.vocab)}
        self.itos = {i:c for i,c in enumerate(self.vocab)}

    def preprocess(self):
        self.train = []
        self.test = []
        text_dir = os.listdir(self.data_dir)
        full_texts = ''
        for dir in text_dir:
            data_dir = Path(self.data_dir) / dir
            file_paths = os.listdir(data_dir)
            for file_path in tqdm(file_paths):
                file_path = Path(data_dir) / file_path
                text = open_file(file_path)
                full_texts += '\n' + text 
                sentence_train = text[:int(len(text)*train_test_ratio)]
                sentence_test = text[int(len(text)*train_test_ratio):]

                self.train += [sentence_train]
                self.test += [sentence_test]

        self.vocab_builder(full_texts)
        self.train_text = self.train.copy()
        self.test_text = self.test.copy()
        for i, (train_chunk, test_chunk) in enumerate(zip(self.train, self.test)):
            self.train[i] = encode(self.stoi, train_chunk)
            self.test[i] = encode(self.stoi, test_chunk)
        
        
    def load_train(self, batch_size: int = 8):
        self.x_train, self.y_train = get_batch(self.train, batch_size)
        self.x_train, self.y_train = self.x_train.to(self.device), self.y_train.to(self.device)
        
        return self.x_train, self.y_train

    def load_test(self, batch_size: int = 8):
        self.x_test, self.y_test = get_batch(self.test, batch_size)
        self.x_test, self.y_test = self.x_test.to(self.device), self.y_test.to(self.device)
        
        return self.x_test, self.y_test



from torch.utils.data import Dataset, DataLoader
import torch
import os
import warnings
warnings.filterwarnings("ignore")


class Training_dataset(Dataset):
    def __init__(self, root_dir):
        self.audio_dir = os.path.join(root_dir, 'audio')
        self.landmark_dir = os.path.join(root_dir, 'landmark')
        self.data_length = len(os.listdir(self.audio_dir))

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        data_length_range = range(self.data_length)
        index = data_length_range[idx]
        output_audio = torch.load(os.path.join(self.audio_dir, f'{index}.pt'))
        output_landmark = torch.load(os.path.join(self.landmark_dir, f'{index}.pt'))

        return output_audio, output_landmark


class Validation_dataset(Dataset):
    def __init__(self, root_dir):
        self.audio_dir = os.path.join(root_dir, 'audio')
        self.landmark_dir = os.path.join(root_dir, 'landmark')
        self.data_length = len(os.listdir(self.audio_dir))

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        data_length_range = range(self.data_length)
        index = data_length_range[idx]
        output_audio = torch.load(os.path.join(self.audio_dir, f'{index}.pt'))
        output_landmark = torch.load(os.path.join(self.landmark_dir, f'{index}.pt'))

        return output_audio, output_landmark.tolist()


def train_dataloader(root_dir, batch_size):
    expression_dataset = Training_dataset(root_dir)
    train_loader = DataLoader(dataset=expression_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader


def val_dataloader(root_dir):
    expression_dataset = Validation_dataset(root_dir)
    test_loader = DataLoader(dataset=expression_dataset, batch_size=1, shuffle=True)
    return test_loader


if __name__ == '__main__':
    train_root_dir = './violin_feature_extraction_scale01_train'
    val_root_dir = './violin_feature_extraction_scale01_val'
    train_data = train_dataloader(train_root_dir, 2)
    val_data = val_dataloader(val_root_dir)

    for audio_sequence, landmark in train_data:
        print(audio_sequence.size(), landmark.size())
        break

    for audio_sequence, landmark in val_data:
        print(audio_sequence.size(), len(landmark))
        break


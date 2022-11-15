import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import time
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from feature_extraction_dataloader import *
from network import *
from tqdm import tqdm
import numpy as np


# settings
epochs = 50
batch_size = 2
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_3dcnn_res_atten(epochs):

    train_root_dir = './violin_feature_extraction_scale01_train'
    val_root_dir = './violin_feature_extraction_scale01_val'
    os.makedirs('./model_save', exist_ok=True)
    save_dir = './model_save/3dcnn_res_atten_scale01.pkl'

    data = train_dataloader(train_root_dir, batch_size)
    model = network_3dcnn_res_attention().to(device)

    model_params = list(model.parameters())
    optimizer = optim.Adam(model_params, lr=learning_rate)
    criterion = nn.MSELoss()
    pre_val_score = np.inf
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        for i, (audio, landmark) in enumerate(data):
            audio = audio.to(device)
            audio = audio.transpose(1, 2)
            landmark = landmark.to(device)
            optimizer.zero_grad()
            output = model(audio)

            loss = criterion(output, landmark)
            eye_brow_loss = criterion(output[:, 34:54], landmark[:, 34:54])
            eye_loss = criterion(output[:, 72:96], landmark[:, 72:96])
            total_loss = loss + eye_loss + eye_brow_loss

            total_loss.backward()
            optimizer.step()
            print('Epoch: {}, batch: {}/{} {:.3f}%, Loss: {}, time:{:.3f}'.format(epoch + 1, i + 1, len(data),
                                                                                  100 * (i + 1) / len(data),
                                                                                  total_loss,
                                                                                  time.time() - start_time))

        model.eval()
        val_data = val_dataloader(val_root_dir)
        val_score = 0.0
        sample = 1000
        for i, (audio, landmark) in tqdm(enumerate(val_data)):
            if i == sample:
                break
            audio = audio.to(device)
            audio = audio.transpose(1, 2)
            output_landmark = model(audio)
            output_landmark.to('cpu')
            output_landmark = output_landmark.tolist()[0]
            score = mean_squared_error(landmark, output_landmark, squared=False)
            val_score += score

        val_score /= sample
        if val_score < pre_val_score:
            torch.save({
                'epoch': epoch,
                'music2land_net': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_dir)
            pre_val_score = val_score


if __name__ == '__main__':
    train_3dcnn_res_atten(epochs)



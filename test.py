import torch
import numpy as np
from torch.utils.data import DataLoader
from Net2_3 import Net

import os
from dataloader1 import Datases_loader as dataloader
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsz = 2
model = Net().to(device)
from contrast.Segnet.SegNet2_0 import SegNet

savedir = r''   # npy
#
imgdir = r''  # test_img
labdir = r''  # test_lab

#
imgsz = 512

resultsdir = r''


dataset = dataloader(imgdir, labdir, imgsz, imgsz)
testsets = DataLoader(dataset, batch_size=batchsz, shuffle=False)


def test():
    model.load_state_dict(torch.load(savedir))
    exist = os.path.exists(resultsdir)
    if not exist:
        os.makedirs(resultsdir)

    counter = 1

    for idx, samples in enumerate(testsets):
        img, lab = samples['image'], samples['mask']
        img, lab = img.to(device), lab.to(device)

        pred = model(img)

        for i, output in enumerate(pred):
            binary_img = np.where(output[0].cpu().detach().numpy() < 0.5, 255, 0)
            plt.imshow(binary_img, cmap='binary')
            plt.axis('off')
            plt.savefig(f'.../{counter}.png', bbox_inches='tight', pad_inches=0)  # Save image

            plt.show()
            plt.close()
            counter += 1

        np.save(resultsdir + f'/pred{idx + 1}.npy', pred.detach().cpu().numpy())
        np.save(resultsdir + f'/label{idx + 1}.npy', lab.detach().cpu().numpy())


if __name__ == '__main__':
    with torch.no_grad():
        test()


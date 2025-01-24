import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.pylab as plt
import random
import numpy as np

class Datases_loader(Dataset):
    def __init__(self, root_images, root_masks, h, w):
        super().__init__()
        self.root_images = root_images
        self.root_masks = root_masks
        self.h = h
        self.w = w
        self.images = []
        self.labels = []

        files = sorted(os.listdir(self.root_images))
        sfiles = sorted(os.listdir(self.root_masks))
        for i in range(len(sfiles)):
            img_file = os.path.join(self.root_images, files[i])
            mask_file = os.path.join(self.root_masks, sfiles[i])
            # print(img_file, mask_file)
            self.images.append(img_file)
            self.labels.append(mask_file)

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image = self.images[idx]
            mask = self.labels[idx]
        else:
            image = self.images[idx]
            mask = self.labels[idx]
        image = Image.open(image)
        mask = Image.open(mask)
        tf = transforms.Compose([
            transforms.Resize((int(self.h * 1.25), int(self.w * 1.25))),
            # The following three lines of code are turned on during training and commented out during testing.
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(16),
            transforms.CenterCrop((self.h, self.w)),
            transforms.ToTensor()
        ])

        # image = image.convert('L')
        image = image.convert('RGB')
        # image = image.filter(ImageFilter.SHARPEN)
        # mask = mask.convert('L')
        norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        seed = np.random.randint(1459343089)

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        img = tf(image)
        img = norm(img)

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # print(np.max(mask))
        mask = tf(mask)
        mask[mask>0] = 1.
        # mask = (mask==13).float()  #kouyifeigai,duobiaoqianweidanbiaoqian
        sample = {'image': img, 'mask': mask, } #kouyifeigai,duobiaoqianweidanbiaoqian
        # sample = {'image': torch.Tensor(img), 'mask': torch.Tensor(mask)}

        return sample

def imshow_image(mydata_loader):
    plt.figure()
    for (cnt, i) in enumerate(mydata_loader):
        image = i['image']
        label = i['mask']
        # print(image.shape, label.shape)

        for j in range(8):
            # ax = plt.subplot(2, 4, j + 1)
            # ax.axis('off')
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)

            # print(image[j].permute(1, 2, 0).shape)
            # print(image.shape)
            # print(label.shape)
            ax1.imshow(image[j].permute(1, 2, 0), cmap='gray')
            ax1.set_title('image')
            ax2.imshow(label[j].permute(1, 2, 0), cmap='gray')
            ax2.set_title('mask')
            # plt.pause(0.005)
            plt.show()
        if cnt == 6:
            break
    plt.pause(0.005)


if __name__ == '__main__':

    d = Datases_loader(r'/root/autodl-tmp/dataset/Deepcrack/CrackTree260/test_img',
                       r'/root/autodl-tmp/dataset/Deepcrack/CrackTree260/test_lab',
                       512, 512)

    mydata_loader = DataLoader(d, batch_size=8, shuffle=False)
    imshow_image(mydata_loader)

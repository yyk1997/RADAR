"""
Implementation of blend attack
[1] Chen, Xinyun, et al. "Targeted backdoor attacks on deep learning systems using data poisoning." arXiv preprint arXiv:1712.05526 (2017).
"""

import os
import torch
import random
from torchvision.utils import save_image

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, trigger, path, target_class = 0, alpha = 0.1):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.trigger = trigger
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0
        self.alpha = alpha
        # self.delta = delta

        # number of images
        self.num_img = len(dataset)

    def generate_poisoned_training_set(self):

        # random sampling
        id_set = list(range(0,self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort() # increasing order

        label_set = []
        pt = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if pt < num_poison and poison_indices[pt] == i:
                gt = self.target_class

                img_temp = (1 - self.alpha) * img + self.alpha *  self.trigger

                perturbation = img_temp - img
                # perturbation_norm = torch.norm(perturbation.view(-1))

                # if perturbation_norm > self.delta:
                #     scaling_factor = self.delta / perturbation_norm
                #     scaled_perturbation = scaling_factor * perturbation  # 确保 scaled_perturbation 的二范数等于 self.delta
                # else:
                #     scaled_perturbation = perturbation

                img = img + perturbation
                # img = torch.clamp(img, 0, 1)

                pt+=1

            img_file_name = '%d.png' % i
            img_file_path = os.path.join(self.path, img_file_name)
            save_image(img, img_file_path)
            #print('[Generate Poisoned Set] Save %s' % img_file_path)
            label_set.append(gt)

        label_set = torch.LongTensor(label_set)

        return poison_indices, label_set



class poison_transform():
    def __init__(self, img_size, trigger, target_class = 0, alpha = 0.2):
        self.img_size = img_size
        self.trigger = trigger
        self.target_class = target_class # by default : target_class = 0
        self.alpha = alpha
        # self.delta = delta
        
    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()
        # transform clean samples to poison samples

        data_temp = (1 - self.alpha) * data + self.alpha * self.trigger

        perturbation = data_temp - data
        # perturbation_norm = torch.norm(perturbation.view(-1))
        #
        # if perturbation_norm > self.delta:
        #     scaling_factor = self.delta / perturbation_norm
        #     scaled_perturbation = scaling_factor * perturbation  # 确保 scaled_perturbation 的二范数等于 self.delta
        # else:
        #     scaled_perturbation = perturbation

        data = data + perturbation


        if labels.dim() == 0:  # 检查是否是0-dim张量
            labels = torch.tensor(self.target_class, dtype=labels.dtype, device=labels.device)
        else:  # 如果是多维张量
            labels[:] = torch.tensor(self.target_class, dtype=labels.dtype, device=labels.device)

        # debug
        # from torchvision.utils import save_image
        # from torchvision import transforms
        # preprocess = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # reverse_preprocess = transforms.Normalize([-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], [1/0.247, 1/0.243, 1/0.261])
        # save_image(reverse_preprocess(data)[-7], 'a.png')

        return data, labels
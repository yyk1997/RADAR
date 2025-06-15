'''codes used to train models on cleansed dataset with pretrained backbone under RADAR

'''
import argparse
import os, sys
from tqdm import tqdm
import config
from torchvision import datasets, transforms
from torch import nn
import torch
from utils import default_args, supervisor, tools, imagenet
import time
from pathlib import Path
from lightly.models.modules import heads
from torch.cuda.amp import autocast, GradScaler
from opacus.validators import ModuleValidator
from collections import OrderedDict


class SimCLR(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=512,  # Resnet18 features have 512 dimensions.
            hidden_dim=512,
            output_dim=128,
        )

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=False,
                    default='none',
                    choices=default_args.parser_choices['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float, required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-ember_options', type=str, required=False,
                    choices=['constrained', 'unconstrained', 'none'],
                    default='unconstrained')
parser.add_argument('-alpha', type=float, required=False,
                    default=default_args.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float, required=False, default=None)
parser.add_argument('-trigger', type=str, required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-devices', type=str, default='2')
parser.add_argument('-cleanser', type=str, choices=['SCAn','AC','SS', 'CT', 'SPECTRE', 'Strip', 'SentiNet'], default='CT')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)

args = parser.parse_args()
args.dataset = "cifar10"
args.poison_type = "badnet"
args.poison_rate = 0.01
# args.cover_rate = 0.005
# name = 'poisoned_train_set/cifar10/trojan_0.005_poison_seed=0'
pretrain_name = 'pretrain_model_100_epochs_p_1_epoch500vs1000_0.5.pth'
name1 = 'DP_CT_model_noise_aug_0.05_2.pt'
# args.alpha = 0.2

os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
tools.setup_seed(args.seed)

if args.trigger is None:
    if args.dataset != 'imagenette' and args.dataset != 'imagenet':
        args.trigger = config.trigger_default[args.poison_type]
    elif args.dataset == 'imagenet':
        args.trigger = imagenet.triggers[args.poison_type]
    else:
        if args.poison_type == 'badnet':
            args.trigger = 'badnet_high_res.png'
        else:
            raise NotImplementedError('%s not implemented for imagenette' % args.poison_type)


all_to_all = False
if args.poison_type == 'badnet_all_to_all':
    all_to_all = True

if args.log:
    out_path = 'logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, args.cleanser)
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_aug.out' % (supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed)))
    fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr

batch_size = 128



if args.dataset == 'cifar10':

    num_classes = 10

    data_transform_aug = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
                                ])

    data_transform_no_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    ])

    trigger_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])

    momentum = 0.9
    weight_decay = 1e-4
    milestones = [15]
    epochs = 25
    learning_rate = 0.01

elif args.dataset == 'gtsrb':

    num_classes = 43

    data_transform_aug = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    data_transform_no_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    trigger_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    momentum = 0.9
    weight_decay = 1e-4
    epochs = 20
    milestones = [5,10]
    learning_rate = 0.01

elif args.dataset == 'imagenet':

    num_classes = 1000
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 90
    milestones = torch.tensor([30, 60])
    learning_rate = 0.1
    batch_size = 256

elif args.dataset == 'ember':

    num_classes = 2
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-6
    epochs = 10
    learning_rate = 0.1
    milestones = torch.tensor([])
    batch_size = 512

    print('[Non-image Dataset] Amber')

else:
    raise Exception("Invalid Dataset")


if args.dataset == 'imagenet':
    kwargs = {'num_workers': 32, 'pin_memory': True}
else:
    kwargs = {'num_workers': 4, 'pin_memory': True}



if args.dataset != 'ember' and args.dataset != 'imagenet':

    poison_set_dir = supervisor.get_poison_set_dir(args)
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
    poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                         label_path=poisoned_set_label_path, transforms=data_transform_aug)
    #cleansed_set_indices_dir = supervisor.get_cleansed_set_indices_dir(args)
    cleansed_set_indices_dir = os.path.join(poison_set_dir, 'DP_ct_cleansed_set_indices_seed=%d_noise_aug_0.05_2' % args.seed)

    print('load : %s' % cleansed_set_indices_dir)
    cleansed_set_indices = torch.load(cleansed_set_indices_dir)


elif args.dataset == 'imagenet':

    poison_set_dir = supervisor.get_poison_set_dir(args)
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    print('dataset : %s' % poison_set_dir)

    poison_indices = torch.load(poison_indices_path)

    root_dir = '/path_to_imagenet/'
    train_set_dir = os.path.join(root_dir, 'train')
    test_set_dir = os.path.join(root_dir, 'val')

    from utils import imagenet
    poisoned_set = imagenet.imagenet_dataset(directory=train_set_dir, poison_directory=poisoned_set_img_dir,
                                             poison_indices = poison_indices, target_class=imagenet.target_class,
                                             num_classes=1000)

    cleansed_set_indices_dir = supervisor.get_cleansed_set_indices_dir(args)
    print('load : %s' % cleansed_set_indices_dir)
    cleansed_set_indices = torch.load(cleansed_set_indices_dir)


else:
    poison_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

    # stats_path = os.path.join('data', 'ember', 'stats')
    poisoned_set = tools.EMBER_Dataset(x_path=os.path.join(poison_set_dir, 'watermarked_X.npy'),
                                       y_path=os.path.join(poison_set_dir, 'watermarked_y.npy'))
    cleansed_set_indices_dir = os.path.join(poison_set_dir, 'cleansed_set_indices_seed=%d' % args.seed)
    print('load : %s' % cleansed_set_indices_dir)
    cleansed_set_indices = torch.load(cleansed_set_indices_dir)


poisoned_indices = torch.load(os.path.join(poison_set_dir, 'poison_indices'))
cleansed_set_indices.sort()
poisoned_indices.sort()

tot_poison = len(poisoned_indices)
num_poison = 0

if tot_poison > 0:
    pt = 0
    for pid in cleansed_set_indices:
        while poisoned_indices[pt] < pid and pt + 1 < tot_poison: pt += 1
        if poisoned_indices[pt] == pid:
            num_poison += 1

print('remaining poison samples in cleansed set : ', num_poison)



cleansed_set = torch.utils.data.Subset(poisoned_set, cleansed_set_indices)
train_set = cleansed_set


if args.dataset != 'ember' and args.dataset != 'imagenet':

    # Set Up Test Set for Debug & Evaluation
    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')
    test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                 label_path=test_set_label_path, transforms=data_transform_no_aug)
    print('with no aug...')
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    # Poison Transform for Testing
    poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                       target_class=config.target_class[args.dataset], trigger_transform=trigger_transform,
                                                       is_normalized_input=True,
                                                       alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                       trigger_name=args.trigger, args=args)


elif args.dataset == 'imagenet':

    poison_transform = imagenet.get_poison_transform_for_imagenet(args.poison_type)

    test_set = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, aug=False,
                 label_file=imagenet.test_set_labels, num_classes=1000)
    test_set_backdoor = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, aug=False,
                 label_file=imagenet.test_set_labels, num_classes=1000, poison_transform=poison_transform)

    test_split_meta_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_indices = torch.load(os.path.join(test_split_meta_dir, 'test_indices'))

    test_set = torch.utils.data.Subset(test_set, test_indices)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    test_set_backdoor = torch.utils.data.Subset(test_set_backdoor, test_indices)
    test_set_backdoor_loader = torch.utils.data.DataLoader(
        test_set_backdoor,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

else:
    normalizer = poisoned_set.normal

    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')

    test_set = tools.EMBER_Dataset(x_path=os.path.join(test_set_dir, 'X.npy'),
                                   y_path=os.path.join(test_set_dir, 'Y.npy'),
                                   normalizer = normalizer)

    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)


    backdoor_test_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
    backdoor_test_set = tools.EMBER_Dataset(x_path=os.path.join(poison_set_dir, 'watermarked_X_test.npy'),
                                       y_path=None, normalizer = normalizer)
    backdoor_test_set_loader = torch.utils.data.DataLoader(
        backdoor_test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)




train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

arch = config.arch[args.dataset]


if args.poison_type == 'TaCT':
    source_classes = config.source_class
else:
    source_classes = None

device = "cuda:0" if torch.cuda.is_available() else "cpu"
#milestones = milestones.tolist()
resnet = arch(num_classes=num_classes)
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SimCLR(backbone)
model_file = os.path.join(supervisor.get_poison_set_dir(args), f"{pretrain_name}")
model.load_state_dict(torch.load(model_file))


model.projection_head = nn.Linear(512, 10)
projection_head_file = poison_set_dir + f"/{name1}"
checkpoint = torch.load(projection_head_file)


# 检查是否有 "_module." 前缀
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    if k.startswith("_module."):
        new_state_dict[k[len("_module."):]] = v  # 去掉 "_module." 前缀
    else:
        new_state_dict[k] = v

# 去掉 "_module." 前缀并仅提取backbone的权重
new_state_dict_backbone = OrderedDict()
for k, v in checkpoint.items():
    if k.startswith("_module.backbone."):
        # 只保留projection_head的参数，并去除前缀
        new_key = k[len("_module.backbone."):]
        new_state_dict_backbone[new_key] = v
    elif k.startswith("backbone."):
        # 如果没有 "_module." 但有 projection_head 前缀
        new_key = k[len("backbone."):]
        new_state_dict_backbone[new_key] = v

# 去掉 "_module." 前缀并仅提取projection_head的权重
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    if k.startswith("_module.projection_head."):
        # 只保留projection_head的参数，并去除前缀
        new_key = k[len("_module.projection_head."):]
        new_state_dict[new_key] = v
    elif k.startswith("projection_head."):
        # 如果没有 "_module." 但有 projection_head 前缀
        new_key = k[len("projection_head."):]
        new_state_dict[new_key] = v
#
#
# model.backbone.load_state_dict(new_state_dict_backbone, strict=False)
model.projection_head.load_state_dict(new_state_dict)





# #
# # 检查是否有 "_module." 前缀
# new_state_dict = OrderedDict()
# for k, v in checkpoint.items():
#     if k.startswith("_module."):
#         new_state_dict[k[len("_module."):]] = v  # 去掉 "_module." 前缀
#     else:
#         new_state_dict[k] = v
#
# model.projection_head.load_state_dict(new_state_dict)



# # 去掉 "_module." 前缀并仅提取projection_head的权重
# new_state_dict = OrderedDict()
# for k, v in checkpoint.items():
#     if k.startswith("_module.projection_head."):
#         # 只保留projection_head的参数，并去除前缀
#         new_key = k[len("_module.projection_head."):]
#         new_state_dict[new_key] = v
#     elif k.startswith("projection_head."):
#         # 如果没有 "_module." 但有 projection_head 前缀
#         new_key = k[len("projection_head."):]
#         new_state_dict[new_key] = v
#
# model.projection_head.load_state_dict(new_state_dict)



# # 设置 backbone 中的所有参数不进行梯度更新
# for param in model.backbone.parameters():
#     param.requires_grad = False
#
# # 设置 projection_head 中的参数进行梯度更新
# for param in model.projection_head.parameters():
#     param.requires_grad = True


# model = nn.DataParallel(model)
model = model.cuda()



if args.dataset != 'ember':

    print(f"Will save to '{os.path.join(poison_set_dir, 'DP_ct_pretrain_cleansed_full_base_aug_seed=%d_noise_aug_0.05.pt' % args.seed)}")
    if os.path.exists(supervisor.get_model_dir(args, cleanse=True)):  # exit if there is an already trained model
        pass
        #print(f"Model '{supervisor.get_model_dir(args, cleanse=True)}' already exists!")
        #model = arch(num_classes=num_classes)
        #model.load_state_dict(torch.load(supervisor.get_model_dir(args, cleanse=True)))
        #model = model.cuda()
        #tools.test(model=model, test_loader=test_set_loader, poison_test=True, poison_transform=poison_transform,
        #           num_classes=num_classes, source_classes=source_classes)
        #exit(0)
    criterion = nn.CrossEntropyLoss().cuda()
else:
    model_path = os.path.join('poisoned_train_set', 'ember', args.ember_options, 'model_trained_on_cleansed_data_seed=%d.pt' % args.seed)
    print(f"Will save to '{model_path}'.")
    if os.path.exists(model_path):
        print(f"Model '{model_path}' already exists!")
    criterion = nn.BCELoss().cuda()


print('milestones:', milestones)
optimizer_1 = torch.optim.SGD(model.backbone.parameters(), 0.05, momentum=momentum, weight_decay=weight_decay)
scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_1, milestones=milestones)

optimizer_2 = torch.optim.SGD(model.projection_head.parameters(), 0.1, momentum=momentum, weight_decay=weight_decay)
scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_2, milestones=milestones)


cnt = 0
from tqdm import tqdm
#scaler = GradScaler()

for epoch in range(1,epochs+1):
    start_time = time.perf_counter()

    model.train()
    for data, target in tqdm(train_loader):

        #data = data.cuda(non_blocking=True)
        #target = target.cuda(non_blocking=True)

        data, target = data.cuda(), target.cuda()
        #optimizer.zero_grad(set_to_none=True)
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        #with autocast():
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer_1.step()
        optimizer_2.step()

        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()

    scheduler_1.step()
    scheduler_2.step()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print('<Cleansed Training> Train Epoch: {} \tLoss: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(epoch,
                                            loss.item(), optimizer_2.param_groups[0]['lr'], elapsed_time))

    # Test
    if args.dataset != 'ember':
        if epoch % 5 == 0:
            if args.dataset == 'imagenet':
                tools.test_imagenet(model=model, test_loader=test_set_loader,
                                    test_backdoor_loader=test_set_backdoor_loader)
                torch.save(model.state_dict(), supervisor.get_model_dir(args, cleanse=True))
            else:
                tools.test(model=model, test_loader=test_set_loader, poison_test=True,
                           poison_transform=poison_transform, num_classes=num_classes, source_classes=source_classes,
                           all_to_all=all_to_all)
                torch.save(model.state_dict(), os.path.join(poison_set_dir, 'DP_ct_pretrain_cleansed_full_base_aug_seed=%d_noise_aug_0.05.pt' % args.seed))
    else:
        if epoch % 5 == 0:
            tools.test_ember(model=model, test_loader=test_set_loader, backdoor_test_loader=backdoor_test_set_loader)
            torch.save(model.state_dict(), model_path)

if args.dataset != 'ember':
    torch.save(model.state_dict(), os.path.join(poison_set_dir, 'DP_ct_pretrain_cleansed_full_base_aug_seed=%d_noise_aug_0.05.pt' % args.seed))
else:
    torch.save(model.state_dict(), model_path)
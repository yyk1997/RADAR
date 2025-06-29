'''core codes for confusion training
'''
import os
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils import tools


# extract features
def get_features(data_loader, model):
    '''
        Extract features on a dataset with a given model

        Parameters:
            data_loader (torch.utils.data.DataLoader): the dataloader of the dataset on which we want to extract features
            model (nn.Module): the mode used to extract features

        Returns:
            feats(list): a list of features for each sample in the dataset
            label_list(list): the ground truth label for each sample in the dataset
            preds_list(list): the model's prediction on each sample of the dataset
            gt_confidence(list): the model's confidence on the ground truth label of each sample in the dataset
            loss_vals(list): the loss values of the model on each sample in the dataset
        '''

    label_list = []
    preds_list = []
    feats = []
    gt_confidence = []
    loss_vals = []

    criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
    model.eval()

    with torch.no_grad():

        for i, (ins_data, ins_target) in enumerate(tqdm(data_loader)):

            ins_data, ins_target = ins_data.cuda(), ins_target.cuda()
            output, x_features = model(ins_data, return_hidden=True)

            loss = criterion_no_reduction(output, ins_target).cpu().numpy()

            preds = torch.argmax(output, dim=1).cpu().numpy()
            prob = torch.softmax(output, dim=1).cpu().numpy()
            this_batch_size = len(ins_target)

            for bid in range(this_batch_size):
                gt = ins_target[bid].cpu().item()
                feats.append(x_features[bid].cpu().numpy())
                label_list.append(gt)
                preds_list.append(preds[bid])
                gt_confidence.append(prob[bid][gt])
                loss_vals.append(loss[bid])
    return feats, label_list, preds_list, gt_confidence, loss_vals




def identify_poison_samples_simplified(inspection_set, clean_indices, model, num_classes):
    '''
            Identify poison samples in a dataset (under inspection) with the confused model.

            Parameters:
                inspection_set (torch.utils.data.Dataset): the dataset that potentially contains poison samples and needs to be cleansed
                clean_indices (dict): a set of indices of samples that are expected to be clean (e.g., have high loss values after confusion training)
                model (nn.Module): the mode used to detect poison samples
                num_classes (int): number of classes in the dataset

            Returns:
                suspicious_indices (list): indices of detected poison samples
            '''

    from scipy.stats import multivariate_normal

    kwargs = {'num_workers': 4, 'pin_memory': True}
    num_samples = len(inspection_set)

    # main dataset we aim to cleanse
    inspection_split_loader = torch.utils.data.DataLoader(
        inspection_set,
        batch_size=128, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    model.eval()
    feats_inspection, class_labels_inspection, preds_inspection, \
    gt_confidence_inspection, loss_vals = get_features(inspection_split_loader, model)

    feats_inspection = np.array(feats_inspection)
    class_labels_inspection = np.array(class_labels_inspection)

    class_indices = [[] for _ in range(num_classes)]
    class_indices_in_clean_chunklet = [[] for _ in range(num_classes)]

    for i in range(num_samples):
        gt = int(class_labels_inspection[i])
        class_indices[gt].append(i)

    for i in clean_indices:
        gt = int(class_labels_inspection[i])
        class_indices_in_clean_chunklet[gt].append(i)

    for i in range(num_classes):
        class_indices[i].sort()
        class_indices_in_clean_chunklet[i].sort()

        if len(class_indices[i]) < 2:
            raise Exception('dataset is too small for class %d' % i)

        if len(class_indices_in_clean_chunklet[i]) < 2:
            raise Exception('clean chunklet is too small for class %d' % i)

    # apply cleanser, if the likelihood of two-clusters-model is twice of the likelihood of single-cluster-model
    threshold = 2
    suspicious_indices = []
    class_likelihood_ratio = []

    for target_class in range(num_classes):

        num_samples_within_class = len(class_indices[target_class])
        print('class-%d : ' % target_class, num_samples_within_class)
        clean_chunklet_size = len(class_indices_in_clean_chunklet[target_class])
        clean_chunklet_indices_within_class = []
        pt = 0
        for i in range(num_samples_within_class):
            if pt == clean_chunklet_size:
                break
            if class_indices[target_class][i] < class_indices_in_clean_chunklet[target_class][pt]:
                continue
            else:
                clean_chunklet_indices_within_class.append(i)
                pt += 1

        print('start_pca..')

        temp_feats = torch.FloatTensor(
            feats_inspection[class_indices[target_class]]).cuda()


        # reduce dimensionality
        U, S, V = torch.pca_lowrank(temp_feats, q=2)
        projected_feats = torch.matmul(temp_feats, V[:, :2]).cpu()

        # isolate samples via the confused inference model
        isolated_indices_global = []
        isolated_indices_local = []
        other_indices_local = []
        labels = []
        for pt, i in enumerate(class_indices[target_class]):
            if preds_inspection[i] == target_class:
                isolated_indices_global.append(i)
                isolated_indices_local.append(pt)
                labels.append(1) # suspected as positive
            else:
                other_indices_local.append(pt)
                labels.append(0)

        projected_feats_isolated = projected_feats[isolated_indices_local]
        projected_feats_other = projected_feats[other_indices_local]

        print('========')
        print('num_isolated:', projected_feats_isolated.shape)
        print('num_other:', projected_feats_other.shape)

        num_isolated = projected_feats_isolated.shape[0]

        print('num_isolated : ', num_isolated)

        if (num_isolated >= 2) and (num_isolated <= num_samples_within_class - 2):

            mu = np.zeros((2,2))
            covariance = np.zeros((2,2,2))

            mu[0] = projected_feats_other.mean(axis=0)
            covariance[0] = np.cov(projected_feats_other.T)
            mu[1] = projected_feats_isolated.mean(axis=0)
            covariance[1] = np.cov(projected_feats_isolated.T)

            # avoid singularity
            covariance += 0.001

            # likelihood ratio test
            single_cluster_likelihood = 0
            two_clusters_likelihood = 0
            for i in range(num_samples_within_class):
                single_cluster_likelihood += multivariate_normal.logpdf(x=projected_feats[i:i + 1], mean=mu[0],
                                                                        cov=covariance[0],
                                                                        allow_singular=True).sum()
                two_clusters_likelihood += multivariate_normal.logpdf(x=projected_feats[i:i + 1], mean=mu[labels[i]],
                                                                      cov=covariance[labels[i]], allow_singular=True).sum()

            likelihood_ratio = np.exp( (two_clusters_likelihood - single_cluster_likelihood) / num_samples_within_class )

        else:

            likelihood_ratio = 1

        class_likelihood_ratio.append(likelihood_ratio)

        print('likelihood_ratio = ', likelihood_ratio)

    max_ratio = np.array(class_likelihood_ratio).max()

    for target_class in range(num_classes):
        likelihood_ratio = class_likelihood_ratio[target_class]

        if likelihood_ratio == max_ratio and likelihood_ratio > 1.5:  # a lower conservative threshold for maximum ratio

            print('[class-%d] class with maximal ratio %f!. Apply Cleanser!' % (target_class, max_ratio))

            for i in class_indices[target_class]:
                if preds_inspection[i] == target_class:
                    suspicious_indices.append(i)

        elif likelihood_ratio > threshold:
            print('[class-%d] likelihood_ratio = %f > threshold = %f. Apply Cleanser!' % (
                target_class, likelihood_ratio, threshold))

            for i in class_indices[target_class]:
                if preds_inspection[i] == target_class:
                    suspicious_indices.append(i)

        else:
            print('[class-%d] likelihood_ratio = %f <= threshold = %f. Pass!' % (
                target_class, likelihood_ratio, threshold))

    return suspicious_indices



# pretraining on the poisoned datast to learn a prior of the backdoor
def pretrain(args, debug_packet, arch, num_classes, weight_decay, pretrain_epochs, distilled_set_loader, criterion,
             inspection_set_dir, confusion_iter, lr, load = True, dataset_name=None):
    '''
                pretraining on the poisoned dataset to learn a prior of the backdoor

                Parameters:
                    args: arguments
                    debug_packet (function): tools for measuring the performance
                    arch: architecture of the model
                    num_classes (int): number of classes in the dataset
                    weight_decay (float): weight_decay parameter for optimizer
                    pretrain_epochs (int): number of pretraining epochs
                    distilled_set_loader (torch.utils.data.DataLoader): data loader of the distilled set
                    criterion: loss function
                    inspection_set_dir (str): directory that holds the dataset to be inspected (cleansed)
                    confusion_iter (int): number of confusion training iterations
                    lr (float): learning rate
                    load (True): whether to load the pretrained model of last round to initiate the model
                    dataset_name (str): name of the benchmark dataset used to experiment

                Returns:
                    model: the pretrained model
                '''

    ######### Pretrain Base Model ##############
    model = arch(num_classes = num_classes)

    # 如果迭代次数不为0，则加载上一次的模型训练结果
    if confusion_iter != 0 and load:
        ckpt_path = os.path.join(inspection_set_dir, 'base_%d_seed=%d.pt' % (confusion_iter-1, args.seed))
        model.load_state_dict( torch.load(ckpt_path) )

    model = nn.DataParallel(model)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr,  momentum=0.9, weight_decay=weight_decay)

    for epoch in range(1, pretrain_epochs + 1):  # pretrain backdoored base model with the distilled set
        model.train()

        for batch_idx, (data, target) in enumerate( tqdm(distilled_set_loader) ):
            optimizer.zero_grad()
            data, target = data.cuda(), target.cuda()  # train set batch
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print('<Round-{} : Pretrain> Train Epoch: {}/{} \tLoss: {:.6f}'.format(confusion_iter, epoch, pretrain_epochs, loss.item()))
            if args.debug_info:
                model.eval()

                if dataset_name != 'ember' and dataset_name != 'imagenet':
                    tools.test(model=model, test_loader=debug_packet['test_set_loader'], poison_test=True,
                           poison_transform=debug_packet['poison_transform'], num_classes=num_classes,
                           source_classes=debug_packet['source_classes'])
                elif dataset_name == 'imagenet':
                    tools.test_imagenet(model=model, test_loader=debug_packet['test_set_loader'],
                                        test_backdoor_loader=debug_packet['test_set_backdoor_loader'])
                else:
                    tools.test_ember(model=model, test_loader=debug_packet['test_set_loader'],
                                     backdoor_test_loader=debug_packet['backdoor_test_set_loader'])

    base_ckpt = model.module.state_dict()
    torch.save(base_ckpt, os.path.join(inspection_set_dir, 'base_%d_seed=%d.pt' % (confusion_iter, args.seed)))
    print('save : ', os.path.join(inspection_set_dir, 'base_%d_seed=%d.pt' % (confusion_iter, args.seed)))

    return model


def add_gaussian_noise(image, noise_std = 0.0):

    noise = torch.randn_like(image) * noise_std
    noisy_image = image + noise
    return noisy_image

# confusion training : joint training on the poisoned dataset and a randomly labeled small clean set (i.e. confusion set)
def confusion_train(args, params, inspection_set, debug_packet, distilled_set_loader, clean_set_loader, confusion_iter, arch,
                    num_classes, inspection_set_dir, weight_decay, criterion_no_reduction,
                    momentum, lamb, freq, lr, batch_factor, distillation_iters, dataset_name = None, Noise_aug = 0.0):
    '''
                    key codes for confusion training loop

                    Parameters:
                           args: arguments
                           params: configuration of datasets
                           inspection_set (torch.utils.data.Dataset): the dataset that potentially contains poison samples and needs to be cleansed
                           debug_packet (function): tools for measuring the performance
                           distilled_set_loader (torch.utils.data.DataLoader): the data loader of distilled set in previous rounds
                           clean_set_loader (torch.utils.data.DataLoader): the data loader of the reserved clean set
                           confusion_iter (int): the round id of the confusion training
                           arch: the model architecture
                           num_classes (int): number of classes in the dataset
                           lamb (int): the weight parameter to balance confusion training objective and clean training objective
                           freq (int): class frequency of the distilled set
                           lr (float): learning rate of the optimizer
                           batch_factor (int): the number of batch intervals of applying confusion training objective
                           distillation_iters (int): the number of confusion training iterations
                           dataset_name (str): name of the benchmark dataset used to experiment

                       Returns:
                           the confused model in this round
                       '''

    base_model = params['arch'](num_classes = num_classes)
    # base_model是在中毒数据集上训练的model，拟和了中毒样本和干净样本
    base_model.load_state_dict(
            torch.load(os.path.join(inspection_set_dir, 'full_base_aug_seed=%d.pt' % (args.seed)))
        )
    base_model = nn.DataParallel(base_model)
    base_model = base_model.cuda()
    base_model.eval()


    ######### Distillation Step ################

    # model仅仅拟和中毒样本
    model = arch(num_classes = num_classes)
    model.load_state_dict(
                torch.load(os.path.join(inspection_set_dir, 'base_%d_seed=%d.pt' % (confusion_iter, args.seed)))
    )
    model = nn.DataParallel(model)
    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                                momentum=momentum)
    # 定义蒸馏迭代次数
    distilled_set_iters = iter(distilled_set_loader)

    # 定义干净样本迭代次数
    clean_set_iters = iter(clean_set_loader)



    rounder = 0

    for batch_idx in tqdm(range(distillation_iters)):

        try:
            data_shift, target_shift = next(clean_set_iters)
        except Exception as e:
            clean_set_iters = iter(clean_set_loader)
            data_shift, target_shift = next(clean_set_iters)
        data_shift, target_shift = data_shift.cuda(), target_shift.cuda()

        # 为什么要用基础模型给出随机标签，利用全局拟和最好的模型给出预测结果
        if dataset_name != 'ember':
            with torch.no_grad():
                preds = torch.argmax(base_model(data_shift), dim=1).detach()
                # 给出随机标签
                if (rounder + batch_idx) % num_classes == 0:
                    rounder += 1
                next_target = (preds + rounder + batch_idx) % num_classes
                target_confusion = next_target
        else:
            with torch.no_grad():
                target_confusion = ((base_model(data_shift) >= 0.5).detach() + 1) % 2
                target_confusion = target_confusion.float()

        model.train()

        # 按batch_factor执行混淆训练
        if batch_idx % batch_factor == 0:

            #
            if Noise_aug != 0.0:
                data_shift = add_gaussian_noise(data_shift, Noise_aug)

            # 加载蒸馏数据
            try:
                data, target = next(distilled_set_iters)
            except Exception as e:
                distilled_set_iters = iter(distilled_set_loader)
                data, target = next(distilled_set_iters)

            data, target = data.cuda(), target.cuda()

            # 组合干净数据和蒸馏数据
            data_mix = torch.cat([data_shift, data], dim=0)
            target_mix = torch.cat([target_confusion, target], dim=0)
            # 记录干净样本的数据大小
            boundary = data_shift.shape[0]

            # 计算混合数据损失，希望该损失对于干净样本而言要大，而对于中毒样本而言要小
            output_mix = model(data_mix)
            loss_mix = criterion_no_reduction(output_mix, target_mix)

            # 计算分批损失
            loss_inspection_batch_all = loss_mix[boundary:] # 蒸馏数据部分损失
            loss_confusion_batch_all = loss_mix[:boundary] # 干净数据部分损失
            # 所有损失取均值
            loss_confusion_batch = loss_confusion_batch_all.mean()

            # 计算加权损失
            target_inspection_batch_all = target_mix[boundary:]
            inspection_batch_size = len(loss_inspection_batch_all)
            loss_inspection_batch = 0
            normalizer = 0
            for i in range(inspection_batch_size):
                gt = int(target_inspection_batch_all[i].item())
                loss_inspection_batch += (loss_inspection_batch_all[i] / freq[gt])
                normalizer += (1 / freq[gt])
            loss_inspection_batch = loss_inspection_batch / normalizer

            weighted_loss = (loss_confusion_batch * (lamb-1) + loss_inspection_batch) / lamb

            loss_confusion_batch = loss_confusion_batch.item()
            loss_inspection_batch = loss_inspection_batch.item()
        else:
            output = model(data_shift)
            weighted_loss = loss_confusion_batch = criterion_no_reduction(output, target_confusion).mean()
            loss_confusion_batch = loss_confusion_batch.item()

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 500 == 0:

            print('<Round-{} : Distillation Step> Batch_idx: {}, batch_factor: {}, lr: {}, lamb : {}, moment : {}, Loss: {:.6f}'.format(
                confusion_iter, batch_idx + 1, batch_factor, optimizer.param_groups[0]['lr'], lamb, momentum,
                weighted_loss.item()))
            print('inspection_batch_loss = %f, confusion_batch_loss = %f' %
                  (loss_inspection_batch, loss_confusion_batch))

            if args.debug_info:
                model.eval()

                if dataset_name != 'ember' and dataset_name != 'imagenet':
                    tools.test(model=model, test_loader=debug_packet['test_set_loader'], poison_test=True,
                           poison_transform=debug_packet['poison_transform'], num_classes=num_classes,
                           source_classes=debug_packet['source_classes'])
                elif dataset_name == 'imagenet':
                    tools.test_imagenet(model=model, test_loader=debug_packet['test_set_loader'],
                                        test_backdoor_loader=debug_packet['test_set_backdoor_loader'])
                else:
                    tools.test_ember(model=model, test_loader=debug_packet['test_set_loader'],
                                     backdoor_test_loader=debug_packet['backdoor_test_set_loader'])

    torch.save( model.module.state_dict(),
               os.path.join(inspection_set_dir, 'confused_%d_seed=%d.pt' % (confusion_iter, args.seed)) )
    print('save : ', os.path.join(inspection_set_dir, 'confused_%d_seed=%d.pt' % (confusion_iter, args.seed)))

    return model


# restore from a certain iteration step
def distill(args, params, inspection_set, n_iter, criterion_no_reduction,distillation_ratio,
            dataset_name = None, final_budget = None, class_wise = False, custom_arch=None, noise = False):
    '''
                   distill samples from the dataset based on loss values of the inference model

                   Parameters:
                       args: arguments
                       params: configuration of datasets
                       inspection_set (torch.utils.data.Dataset): the dataset that potentially contains poison samples and needs to be cleansed
                       n_iter (int): id of current iteration
                       criterion_no_reduction: loss function
                       dataset_name (str): name of the benchmark dataset used to experiment
                       final_budget (int): maximal number of distilled samples
                       class_wise (bool): whether to list indices of distilled samples for each class seperately
                       custom_arch: if not None, it is assigned as a customized architecture that is different from the specification in params

                   Returns:
                       indicies of distilled samples
                   '''

    kwargs = params['kwargs']
    inspection_set_dir = params['inspection_set_dir']
    num_classes = params['num_classes']
    num_samples = len(inspection_set)
    arch = params['arch']
    #distillation_ratio = params['distillation_ratio']
    num_confusion_iter = len(distillation_ratio) + 1

    if custom_arch is not None:
        arch = custom_arch

    # 加载当前轮次的混淆模型
    model = arch(num_classes=num_classes)
    ckpt = torch.load(os.path.join(inspection_set_dir, 'confused_%d_seed=%d.pt' % (n_iter, args.seed)))
    model.load_state_dict(ckpt)
    model = nn.DataParallel(model)
    model = model.cuda()

    # 数据加载器
    inspection_set_loader = torch.utils.data.DataLoader(inspection_set, batch_size=256,
                                                            shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    """
        Collect loss values for inspected samples.
    """

    # 计算所有样本的损失
    loss_array = []
    correct_instances = []
    gts = []
    model.eval()
    st = 0

    with torch.no_grad():

        for data, target in tqdm(inspection_set_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)

            if dataset_name != 'ember':
                preds = torch.argmax(output, dim=1)
            else:
                preds = (output >= 0.5).float()

            batch_loss = criterion_no_reduction(output, target)

            this_batch_size = len(target)

            for i in range(this_batch_size):
                loss_array.append(batch_loss[i].item())
                gts.append(int(target[i].item()))
                if dataset_name != 'ember':
                    if preds[i] == target[i]:
                        correct_instances.append(st + i)
                else:
                    if preds[i] == target[i]:
                        correct_instances.append(st + i)

            st += this_batch_size

    loss_array = np.array(loss_array)
    sorted_indices = np.argsort(loss_array)


    top_indices_each_class = [[] for _ in range(num_classes)]
    for t in sorted_indices:
        gt = gts[t]
        top_indices_each_class[gt].append(t)

    """
        Distill samples with low loss values from the inspected set.
    """

    if n_iter < num_confusion_iter - 1:

        if distillation_ratio[n_iter] is None:
            distilled_samples_indices = head = correct_instances
        else:
            # 计算当前轮次应该选择的样本总数
            num_expected = int(distillation_ratio[n_iter] * num_samples)
            # 取损失值最低的num_expected 个样本 并存入distilled_samples_indices
            head = sorted_indices[:num_expected]
            head = list(head)
            distilled_samples_indices = head

        # 倒数第二轮之前 rate_factor = 50，否则为100，目的是确保每类至少有 样本总数 / rate_factor 个样本
        if n_iter < num_confusion_iter - 2: rate_factor = 50
        else: rate_factor = 100

        if True: #n_iter < num_confusion_iter - 2:

            class_dist = np.zeros(num_classes, dtype=int)
            for i in distilled_samples_indices:
                gt = gts[i]
                class_dist[gt] += 1

            for i in range(num_classes):
                minimal_sample_num = len(top_indices_each_class[i]) // rate_factor
                print('class-%d, collected=%d, minimal_to_collect=%d' % (i, class_dist[i], minimal_sample_num) )
                if class_dist[i] < minimal_sample_num:
                    for k in range(class_dist[i], minimal_sample_num):
                        distilled_samples_indices.append(top_indices_each_class[i][k])

    # 最终轮次筛选策略
    else:
        if final_budget is not None:
            head = sorted_indices[:final_budget]
            head = list(head)
            distilled_samples_indices = head
        else:
            distilled_samples_indices = head = correct_instances

    distilled_samples_indices.sort()

    # 计算中间损失样本
    median_sample_rate = params['median_sample_rate']
    median_sample_indices = []
    sorted_indices_each_class = [[] for _ in range(num_classes)]
    # 按照类别存储样本索引
    for temp_id in sorted_indices:
        gt = gts[temp_id] # 获取当前样本的真实类别
        sorted_indices_each_class[gt].append(temp_id) # 按类别存储样本索引


    # 选取损失值居中的样本
    for i in range(num_classes):
        num_class_i = len(sorted_indices_each_class[i])
        st = int(num_class_i / 2 - num_class_i * median_sample_rate / 2)
        ed = int(num_class_i / 2 + num_class_i * median_sample_rate / 2)
        for temp_id in range(st, ed):
            median_sample_indices.append(sorted_indices_each_class[i][temp_id])

    """Report statistics of the distillation results...
    """
    if args.debug_info:

        print('num_correct : ', len(correct_instances))

        if args.poison_type == 'TaCT' or args.poison_type == 'adaptive_blend' or args.poison_type == 'adaptive_patch':
            cover_indices = torch.load(os.path.join(inspection_set_dir, 'cover_indices'))

        poison_indices = torch.load(os.path.join(inspection_set_dir, 'poison_indices'))


        cnt = 0
        for s, cid in enumerate(head):  # enumerate the head part
            original_id = cid
            if original_id in poison_indices:
                cnt += 1
        print('How Many Poison Samples are Concentrated in the Head? --- %d/%d' % (cnt, len(poison_indices)))

        cover_dist = []
        poison_dist = []

        for temp_id in range(num_samples):

            if sorted_indices[temp_id] in poison_indices:
                poison_dist.append(temp_id)

            if args.poison_type == 'TaCT' or args.poison_type == 'adaptive_blend':
                if sorted_indices[temp_id] in cover_indices:
                    cover_dist.append(temp_id)

        print('poison distribution : ', poison_dist)

        if args.poison_type == 'TaCT' or args.poison_type == 'adaptive_blend' or args.poison_type == 'adaptive_patch':
            print('cover distribution : ', cover_dist)

        num_poison = len(poison_indices)
        num_collected = len(correct_instances)
        pt = 0

        recall = 0
        for idx in correct_instances:
            if pt >= num_poison:
                break
            while (idx > poison_indices[pt] and pt + 1 < num_poison): pt += 1
            if pt < num_poison and poison_indices[pt] == idx:
                recall += 1

        fpr = num_collected - recall
        print('recall = %d/%d = %f, fpr = %d/%d = %f' % (recall, num_poison, recall/num_poison if num_poison!=0 else 0,
                                                             fpr, num_samples - num_poison,
                                                             fpr / (num_samples - num_poison) if (num_samples-num_poison)!=0 else 0))

    print("len(distilled_samples_indices):")
    print(len(distilled_samples_indices))

    if class_wise:
        return distilled_samples_indices, median_sample_indices, top_indices_each_class
    else:
        return distilled_samples_indices, median_sample_indices



'''codes used to call other baseline dataset cleansers to cleanse the poiosned dataset
'''
import torch
import os, sys
from torchvision import transforms
import argparse
from torch import nn
import numpy as np
import config
from utils import supervisor, tools, default_args

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str,  required=False,
                    choices=default_args.parser_choices['poison_type'],
                    default=default_args.parser_default['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float,  required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-alpha', type=float,  required=False,
                    default=default_args.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float,  required=False, default=None)
parser.add_argument('-trigger', type=str,  required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-model', type=str, required=False, default=None)
parser.add_argument('-model_path', required=False, default=None)

parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-cleanser', type=str, required=False,
                    choices=['SCAn', 'AC', 'SS', 'Strip', 'CT', 'SPECTRE', 'SentiNet', 'Frequency']) # scan, activation clustering, spectral signature
parser.add_argument('-devices', type=str, default='1')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)

args = parser.parse_args()
args.dataset = "gtsrb"
args.poison_type = "TaCT"
args.poison_rate = 0.01
args.cover_rate = 0.005
args.cleanser = "SCAn"
# args.alpha = 0.2
# args.trigger = "badnet_patch4_dup_32.png"


if args.trigger is None:
    args.trigger = config.trigger_default[args.poison_type]
    
all_to_all = False
if args.poison_type == 'badnet_all_to_all':
    all_to_all = True

# tools.setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
if args.log:
    out_path = 'logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'cleanse')
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_%s.out' % (args.cleanser, supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed)))
    fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr

save_path = supervisor.get_cleansed_set_indices_dir(args)

arch = config.arch[args.dataset]

if args.dataset == 'cifar10':

    num_classes = 10
    if args.no_normalize:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])

elif args.dataset == 'gtsrb':

    num_classes = 43
    if args.no_normalize:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])

else:
    raise NotImplementedError('<Undefined Dataset> Dataset = %s' % args.dataset)



batch_size = 512

poison_set_dir = supervisor.get_poison_set_dir(args)


# poisoned set
poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                 label_path=poisoned_set_label_path, transforms=data_transform)


# small clean split at hand for defensive usage
clean_set_dir = os.path.join('clean_set', args.dataset, 'clean_split')
clean_set_img_dir = os.path.join(clean_set_dir, 'data')
clean_set_label_path = os.path.join(clean_set_dir, 'clean_labels')
clean_set = tools.IMG_Dataset(data_dir=clean_set_img_dir,
                              label_path=clean_set_label_path, transforms=data_transform)


num_samples = len(poisoned_set)


model_list = []
alias_list = []

if (hasattr(args, 'model_path') and args.model_path is not None) or (hasattr(args, 'model') and args.model is not None):
    path = supervisor.get_model_dir(args)
    model_list.append(path)
    alias_list.append('assigned')

else:
    #args.no_aug = True
    #path = supervisor.get_model_dir(args)
    #model_list.append(path)
    #alias_list.append(supervisor.get_model_name(args))

    args.no_aug = False
    path = supervisor.get_model_dir(args)
    model_list.append(path)
    alias_list.append(supervisor.get_model_name(args))


best_remain_indices = None
best_recall = -999
best_fpr = 999
best_path = None


for (vid, path) in enumerate(model_list):

    # base model for poison detection
    model = arch(num_classes=num_classes)
    if os.path.exists(path):
        ckpt = torch.load(path)
        model.load_state_dict(ckpt)
    else:
        print(f"Model {path} not exists!")
    model = nn.DataParallel(model)
    model = model.cuda()
    model.eval()


    # oracle knowledge of poison indices for evaluating detectors
    if args.poison_type != 'none':
        poison_indices = torch.load(os.path.join(poison_set_dir, 'poison_indices'))
    
    if True: #not os.path.exists(save_path):
        suspicious_indices = []
        if args.cleanser == "SS":

            if args.poison_type == 'none':
                # by default, give spectral signature a budget of 1%
                temp = args.poison_rate
                args.poison_rate = 0.01

            from other_cleansers import  spectral_signature
            suspicious_indices = spectral_signature.cleanser(poisoned_set, model, num_classes, args)

            if args.poison_type == 'none':
                args.poison_rate = temp

        elif args.cleanser == "AC":
            from other_cleansers import activation_clustering
            suspicious_indices = activation_clustering.cleanser(poisoned_set, model, num_classes, args)
        elif args.cleanser == "SCAn":
            from other_cleansers import scan
            suspicious_indices = scan.cleanser(poisoned_set, clean_set, model, num_classes)
        # elif args.cleanser == 'SPECTRE_python' or args.cleanser == 'SPECTRE_oracle':
        #     if args.poison_type == 'none':
        #         args.poison_rate = 0.01
        #         # by default, give spectral signature a budget of 1%

        #     from other_cleansers import spectre_python

        #     oracle_knowledge_of_clean_samples_in_poisoned_set = None
        #     if args.cleanser == 'SPECTRE_oracle':
        #         oracle_clean_indices_knowledge = []
        #         num_samples = len(poisoned_set)
        #         for i in range(num_samples):
        #             if i not in poison_indices:
        #                 oracle_clean_indices_knowledge.append(i)

        #         oracle_knowledge_of_clean_samples_in_poisoned_set = torch.utils.data.Subset(poisoned_set, oracle_clean_indices_knowledge)


        #     suspicious_indices = spectre_python.cleanser(poisoned_set, model, num_classes, args,
        #                                                  oracle_knowledge_of_clean_samples_in_poisoned_set)    
    #                                                  oracle_knowledge_of_clean_samples_in_poisoned_set)    
        #                                                  oracle_knowledge_of_clean_samples_in_poisoned_set)    
        elif args.cleanser == 'SPECTRE':

            num_samples = len(poisoned_set)
            num_poison = int(args.poison_rate * num_samples)
            base_path = 'other_cleansers/spectre/output'    # where to save temp results

            # Save representations
            from other_cleansers.spectre.save_rep import SAVE_REP
            defense = SAVE_REP(args, model=model)
            defense.output(base_path=base_path, alias=alias_list[vid])
            
            # Execute julia code
            import subprocess
            os.chdir('other_cleansers/spectre')

            procs = []
            for i in range(num_classes):
                folder_path = 'output'
                name = f'{supervisor.get_dir_core(args, include_poison_seed=True)}_{alias_list[vid]}/{i}-{num_poison}'
                folder_path = os.path.join(folder_path, name)
                if os.path.exists(os.path.join(folder_path, 'opnorm.npy')):
                    # print(os.path.join(folder_path, 'opnorm.npy'), 'already exists!')
                    continue

                cmd = ['julia', '--project=.', 'run_filters.jl', name]
                outfile = open(os.path.join(folder_path, 'log.txt'), "w")
                #errfile = open('/dev/null', "a")
                errfile = open(os.path.join(folder_path, 'err.txt'), "w")
                procs.append(subprocess.Popen(cmd, stdout=outfile, stderr=errfile))
                #print("Running for class", i)

            for p in procs:
                p.wait()
            os.chdir('../../')
            
            # Load julia results
            poison_set_dir, inspection_split_loader, poison_indices, cover_indices \
                = tools.unpack_poisoned_train_set(args, batch_size=128, shuffle=False)

            feats, class_indices = defense.get_features(inspection_split_loader,
                                                        defense.model, defense.num_classes)
            suspicious_indices = []
            scores = []
            for i in range(num_classes):
                folder_path = 'other_cleansers/spectre/output'
                folder_path = os.path.join(folder_path, f'{supervisor.get_dir_core(args, include_poison_seed=True)}_{alias_list[vid]}')
                folder_path = os.path.join(folder_path, f'{i}-{num_poison}')
                
                score = np.load(os.path.join(folder_path, 'opnorm.npy'))
                scores.append(score.item())
                suspicious_class_indices_mask = np.load(os.path.join(folder_path, 'mask-rcov-target.npy'))
                suspicious_class_indices = torch.tensor(suspicious_class_indices_mask).nonzero().squeeze(1)
                cur_class_indices = torch.tensor(class_indices[i])
                suspicious_indices.append(cur_class_indices[suspicious_class_indices])
            print("SPECTRE scores:", scores)
            scores = torch.tensor(scores)

            threshold = torch.median(scores)
            detected_indices = []
            for i in range(num_classes):
                class_score = scores[i]
                print('[class-%d] spectre score = %f' % (i, class_score) )
                if class_score > threshold:
                    print('exceed threshold! cleanse')
                    detected_indices += list(suspicious_indices[i])

            detected_indices.sort()
            suspicious_indices = detected_indices
            #suspect_target_class = scores.argmax(dim=0) # class with the highest score is suspected as the target class
            #suspicious_indices = suspicious_indices[suspect_target_class]
            # suspicious_indices = torch.cat(suspicious_indices, dim=0)
        elif args.cleanser == 'CT':
            from other_cleansers import CT_feature_inference
            suspicious_indices = CT_feature_inference.cleanser(poisoned_set, clean_set, model, num_classes)
        elif args.cleanser == 'Strip':
            from other_cleansers import strip
            suspicious_indices = strip.cleanser(poisoned_set, clean_set, model, args)
        elif args.cleanser == 'SentiNet':
            from other_cleansers import sentinet
            suspicious_indices = sentinet.cleanser(args, model, defense_fpr=0.05, N=100)
            # suspicious_indices = sentinet.cleanser(args, model, defense_fpr=None, N=100)
        elif args.cleanser == 'Frequency':
            from other_cleansers import frequency
            suspicious_indices = frequency.cleanser(args)
        else:
            raise NotImplementedError('Unimplemented Cleanser')


        suspicious_indices = list(np.array(suspicious_indices, dtype=int))

        #print(suspicious_indices)

        remain_indices = list( set(list( range(num_samples) )) - set(suspicious_indices) )
        remain_indices.sort()

        #print('num_samples = %d, num_suspicious = %d, num_remain = %d' % (num_samples, len(suspicious_indices),
        #                                                                  len(remain_indices)) )

        """
        remain_indices = []
        for i in range(len(poisoned_set)):
            if i not in suspicious_indices:
                remain_indices.append(i)
        remain_indices.sort()"""

    else: # already exists, load from saved file
        print("Already cleansed!")
        remain_indices = torch.load(save_path)
        suspicious_indices = list(set(range(0,len(poisoned_set))) - set(remain_indices))
        suspicious_indices.sort()

    if True:
        if args.poison_type != 'none':
            true_positive  = 0
            num_positive   = len(poison_indices)
            false_positive = 0
            num_negative   = len(poisoned_set) - num_positive

            suspicious_indices.sort()
            poison_indices.sort()

            pt = 0
            for pid in suspicious_indices:
                while poison_indices[pt] < pid and pt + 1 < num_positive: pt += 1
                if poison_indices[pt] == pid:
                    true_positive += 1
                else:
                    false_positive += 1

            if not os.path.exists(save_path): print('<Overall Performance Evaluation with %s>' % path)
            tpr = true_positive / num_positive
            fpr = false_positive / num_negative
            if not os.path.exists(save_path): print('Elimination Rate = %d/%d = %f' % (true_positive, num_positive, tpr))
            if not os.path.exists(save_path): print('Sacrifice Rate = %d/%d = %f' % (false_positive, num_negative, fpr))

            if tpr > best_recall:
                best_recall = tpr
                best_remain_indices = remain_indices
                best_fpr = fpr
                best_path = path
            elif tpr == best_recall and fpr < best_fpr:
                best_remain_indices = remain_indices
                best_fpr = fpr
                best_path = path


        else:

            print('<Test Cleanser on Clean Dataset with %s>' % path)
            false_positive = len(suspicious_indices)
            num_negative = len(poisoned_set)
            fpr = false_positive / num_negative
            print('Sacrifice Rate = %d/%d = %f' % (false_positive, num_negative, fpr))

            if fpr < best_fpr:
                best_fpr = fpr
                best_remain_indices = remain_indices
                best_path = path

if True: #not os.path.exists(save_path):
    torch.save(best_remain_indices, save_path)
    print('[Save] %s' % save_path)
    print('best base model : %s' % best_path)


if args.poison_type != 'none':
    num_positive = len(poison_indices)
    num_negative = len(poisoned_set) - num_positive
    print('Best Elimination Rate = %d/%d = %f' % ( int(best_recall*num_positive), num_positive, best_recall))
    print('Best Sacrifice Rate = %d/%d = %f' % ( int(best_fpr*num_negative), num_negative, best_fpr))
else:
    num_negative = len(poisoned_set)
    print('Best Sacrifice Rate = %d/%d = %f' % (int(best_fpr * num_negative), num_negative, best_fpr))





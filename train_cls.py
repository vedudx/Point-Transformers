"""
Author: Benny
Date: Nov 2019
"""
from dataset import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
import hydra
import omegaconf
import wandb

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import debugpy
import multiprocessing

from sklearn.metrics import confusion_matrix



'''
function to print confusion matrix
Inputs:
cf_matrix: List[List]
class_dictionary: Name of classes
'''
def print_confusion_matrix(cf_matrix, class_dictionary, num_class=5, save_path='confusion_matrix_best_instance.png'):
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [class_dictionary[i] for i in range(num_class)],
                     columns = [class_dictionary[i] for i in range(num_class)])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(save_path)




def test(model, loader, num_class=5, epoch=0):
    mean_correct = []


    true_labels = []  # List to store true labels
    pred_labels = []  # List to store predicted labels
    class_acc = np.zeros((num_class,3))
    test_loss = 0.0
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]

        # Calculate test loss
        loss = torch.nn.CrossEntropyLoss()(pred, target.long())
        test_loss += loss.item()
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            # print(target[target==cat])
         
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))


        # Convert tensors to numpy arrays
        true_labels.extend(target.cpu().numpy())
        pred_labels.extend(pred_choice.cpu().numpy())

        # Calculate the confusion matrix
        cf_matrix = confusion_matrix(true_labels, pred_labels)
 
    test_loss /= len(loader)  # Calculate average test loss
    print("Test Loss: {:.4f}".format(test_loss))
    wandb.log({"Test Loss": test_loss, "epoch": epoch})

    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    

    #logging the class-wise accuracy
    for i in range(num_class):
        print(f"class {i+1} accuracy: {class_acc[i, 2]}")
        wandb.log({f"class {i+1} accuracy":class_acc[i, 2]})

      
    class_acc = np.mean(class_acc[:,2])
    print("class accuracy", class_acc)
    #can log class accuracy in case of our code
  
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc, cf_matrix


@hydra.main(config_path='config', config_name='cls')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    print("args.gpu", args.gpu)
    print("args.num_class", args.num_class)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    #print(args.pretty())
    wandb.init(resume=args.contin)


    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = hydra.utils.to_absolute_path('giga_small_dataset_normal_augment_cleaned/') #modelnet40_normal_resampled/

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train',uniform = args.uniform, normal_channel=args.normal, num_class=args.num_class)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',uniform = args.uniform, normal_channel=args.normal, num_class=args.num_class)
    class_dictionary = TEST_DATASET.cat
    print(class_dictionary, type(class_dictionary))
    # multiprocessing.set_start_method('spawn')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=6)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=6)

    ''' DEBUGGING INFO'''
    if args.debug:
        debugpy.listen(5678)
        debugpy.wait_for_client()

    '''MODEL LOADING'''
    # args.num_class = 40
    args.input_dim = 6 if args.normal else 3
    args.log_interval = 10
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()

    best_instance_acc = 0.0
    best_class_acc = 0.0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3) #50

    try:
        if not args.contin:
            raise Exception('Skip using pre-trained model')
        
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load the scheduler's state
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
   
        with torch.no_grad():
            instance_acc, class_acc, cf_matrix = test(classifier.eval(), testDataLoader, args.num_class, 0)

            best_instance_acc = instance_acc
            best_class_acc = class_acc

            #best_instance_acc, best_class_acc, best_cf_matrix = instance_acc, class_acc
            true_pos = np.diag(cf_matrix) 
            precision = 0 #np.sum(true_pos / np.sum(best_cf_matrix, axis=0))
            recall = 0 #np.sum(true_pos / np.sum(best_cf_matrix, axis=1))
            logger.info('Beginning Instance Accuracy: %f, Class Accuracy: %f, precision: %f, recall: %f'% (instance_acc, class_acc, precision, recall))
            logger.info('Use pretrain model')
    except Exception as e:
        print(e)
        logger.info('No existing model, starting training from scratch... Testing first')
        start_epoch = 0
        with torch.no_grad():
            instance_acc, class_acc, cf_matrix = test(classifier.eval(), testDataLoader, args.num_class, 0)

            best_instance_acc = instance_acc
            best_class_acc = class_acc

            #best_instance_acc, best_class_acc, best_cf_matrix = instance_acc, class_acc
            true_pos = np.diag(cf_matrix) 
            precision = 0 #np.sum(true_pos / np.sum(best_cf_matrix, axis=0))
            recall = 0 #np.sum(true_pos / np.sum(best_cf_matrix, axis=1))
            logger.info('Beginning Instance Accuracy: %f, Class Accuracy: %f, precision: %f, recall: %f'% (instance_acc, class_acc, precision, recall))
            logger.info('New Model')

        
    if args.eval_only:
        exit(0)

    #wandb
    wandb.watch(classifier, log="all")



    global_step = 0
    global_epoch = 0

    best_epoch = 0
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        train_loss = 0.0
        classifier.train()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            pred = classifier(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            if batch_id % args.log_interval == 0:
                wandb.log({"Batch loss": loss})

            train_loss += loss
            loss.backward()
            optimizer.step()
            global_step += 1
            
        scheduler.step()

        train_instance_acc = np.mean(mean_correct)
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)
        wandb.log({"Train Instance Accuracy": train_instance_acc})
        epoch_loss = train_loss / len(trainDataLoader)

        print("Epoch {} : Train Loss: {:.4f}".format(epoch + 1, epoch_loss))
        wandb.log({"Train Loss": epoch_loss, "epoch": epoch+1})



        with torch.no_grad():
            instance_acc, class_acc, cf_matrix = test(classifier.eval(), testDataLoader, args.num_class, epoch+1)

            print_confusion_matrix(cf_matrix, class_dictionary, num_class=args.num_class, save_path='last_confusion_matrix.png')

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            logger.info('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))
            wandb.log({"instance accuracy": instance_acc})
            wandb.log({"iclass accuracy": class_acc})


            if (instance_acc >= best_instance_acc):
                print_confusion_matrix(cf_matrix, class_dictionary, num_class=args.num_class)

                true_pos = np.diag(cf_matrix) 
                precision = np.sum(true_pos / np.sum(cf_matrix, axis=0))
                recall = np.sum(true_pos / np.sum(cf_matrix, axis=1))
                logger.info('Best Precision: %f, Best Recall: %f'% (precision, recall))
                logger.info('Save model...')
                savepath = 'best_model.pth'
                logger.info('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1
        
      

    logger.info('End of training...')

if __name__ == '__main__':
    main()
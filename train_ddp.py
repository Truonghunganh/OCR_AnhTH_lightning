import shutil
import os
import sys
import time
import random
import string
import re
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
from tqdm import tqdm 
from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from torch.utils.data import DataLoader, DistributedSampler

from test import validation
from utils import get_args
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR, ReduceLROnPlateau
def get_value_from_txt(path_txt):
    if os.path.exists(path_txt):
        with open(path_txt,'r',encoding="utf-8") as f:
            data=f.read()
            f.close()
            return data
    print("khong ton tai : ",path_txt)
    return ""
#--optimizer Adadelta
def get_optimizer(opt,filtered_parameters):
    if opt.optimizer=="Adam":
        optimizer = optim.Adam(filtered_parameters,
                               lr=opt.lr, betas=(opt.beta1, opt.beta2))
    elif opt.optimizer=="Adadelta":
        optimizer = optim.Adadelta(
            filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    elif opt.optimizer=="SGD":
        optimizer = optim.SGD(
            filtered_parameters, lr=opt.lr,momentum=0.937, nesterov=True)
        # optimizer = optim.SGD(
        #     filtered_parameters, lr=opt.lr,momentum=0.8, nesterov=True)
    else:
        if random.randint(0,10)%2==1:
            optimizer = optim.SGD(
                filtered_parameters, lr=random.uniform(0.0000001,0.001),momentum=random.uniform(0.8,0.95), nesterov=True)
        else:
            optimizer = optim.Adadelta(
                    filtered_parameters, lr=random.uniform(0.00001,0.1), rho=random.uniform(0.9,0.97), eps=random.uniform(1e-9,1e-7))
                    
    return optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.distributed as dist

def setup_ddp(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()


def train(opt):
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    if torch.cuda.device_count()>1:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])  # GPU ID
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank=0
        local_rank = 1
        world_size = 1
   
    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    opt.eval = False
    # train_dataset = Batch_Balanced_Dataset(opt)
    train_dataset, _ = hierarchical_dataset(opt.train_data, opt)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    AlignCollate_train = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
    train_loader = DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),  # ✅ chỉ shuffle nếu không dùng sampler
    num_workers=int(opt.workers),
    collate_fn=AlignCollate_train,
    pin_memory=True
    )

    log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
    opt.eval = True
    if opt.sensitive:
        opt.data_filtering_off = True
    AlignCollate_valid = AlignCollate( #  căng chỉnh lại hình ảnh
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
    valid_dataset, valid_dataset_log = hierarchical_dataset(
        root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        # 'True' to check training progress with validation function.
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    if opt.Transformer:
        converter = TokenLabelConverter(opt)
    elif 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    print("llllllllllllllllll")
    print(opt.character)
    best_valid_loss=100
    best_train_loss=100
    if opt.rgb:
        opt.input_channel = 3
    'tạo ra model'
    model = Model(opt)

    # weight initialization
    if not opt.Transformer:
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue
    print('torch.cuda.device_count():',torch.cuda.device_count())
    if torch.cuda.device_count()>1:
        print('rank,local_rank,world_size',rank,local_rank,world_size)
        setup_ddp(rank, world_size)
        model.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    else:
        rank=0
        model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            print(opt)
            model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            # need to install warpctc. see our guideline.
            from warpctc_pytorch import CTCLoss
            criterion = CTCLoss()
        else:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    loss_avg = Averager()
    so_lan_filtered_parameters=0
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    scheduler = None
    optimizer=get_optimizer(opt=opt,filtered_parameters=filtered_parameters)
    if opt.scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.num_iter)
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        opt_file.write(opt_log)
        total_params = int(sum(params_num))
        total_params = f'Trainable network params num : {total_params:,}'
        print(total_params)
        opt_file.write(total_params)
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    # scaler = GradScaler()
    iteration = start_iter
    print("THOI GIAN BAT DAU train :",datetime.datetime.now(),opt.valInterval)
    accum_steps = 1280
    while(True):
        train_sampler.set_epoch(iteration)
        acc_train=0
        tong_train=0
        iiiii=0
        for image_tensors, labels in tqdm(train_loader): 
            iiiii+=1
            image = image_tensors.to(device)
            if not opt.Transformer:
                text, length = converter.encode(
                    labels, batch_max_length=opt.batch_max_length)
            batch_size = image.size(0)
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

            if 'CTC' in opt.Prediction:
                preds = model(image, text)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                if opt.baiduCTC:
                    preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                    cost = criterion(preds, text, preds_size, length) / batch_size
                else:
                    preds = preds.log_softmax(2).permute(1, 0, 2)
                    cost = criterion(preds, text, preds_size, length)
                '--------------------------'
                if opt.baiduCTC:
                    _, preds_index = preds.max(2)
                    preds_index = preds_index.view(-1)
                else:
                    _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index.data, preds_size.data)

                '--------------------------'

            elif opt.Transformer:
                target = converter.encode(labels)
                preds = model(image, text=target,
                            seqlen=converter.batch_max_length)
                cost = criterion(
                    preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            else:
                preds = model(image, text[:, :-1])  # align with Attention.forward
                target = text[:, 1:]  # without [GO] Symbol
                cost = criterion(
                    preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
               
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

            # print('length_for_pred:',length_for_pred)
            for gt, pred in zip(labels, preds_str):
                if 'Attn' in opt.Prediction:
                    gt = gt[:gt.find('[s]')]
                    pred = pred[:pred.find('[s]')] 
                gt,pred=str(gt),str(pred)
                # print(gt,pred)
                if gt==pred:
                    acc_train+=1
                tong_train+=1

            cost.backward()
            optimizer.step()
            # if iiiii % accum_steps == 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            #     optimizer.step()
            #     optimizer.zero_grad()
            loss_avg.add(cost)
            if (iteration + 1) == opt.num_iter:
                print('end the training')
                sys.exit()
            iteration += 1
            if opt.num_iter%iteration!=0:
                if scheduler is not None:
                    scheduler.step()
            '---------------'
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        if rank == 0:

            elapsed_time = time.time() - start_time
            print("THOI GIAN BAT DAU VALID :",datetime.datetime.now(),opt.exp_name,opt.lr,so_lan_filtered_parameters,opt.saved_model)
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    thoigianbatdauvalid=datetime.datetime.now()
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                print("thoi gian valid :",datetime.datetime.now()-thoigianbatdauvalid,length_of_data)
                model.train()
                if best_train_loss==0:
                    best_train_loss=float(loss_avg.val())
                loss_log = f'[{iteration}/{opt.num_iter}] Train loss: {loss_avg.val():0.18f}, Valid loss: {valid_loss:0.18f}, Elapsed_time: {elapsed_time:0.5f}\n now time: {str(datetime.datetime.now())},best_valid_loss:{best_valid_loss},best_train_loss : {best_train_loss},acc_train:{acc_train}/{tong_train} = {acc_train/tong_train}'
                loss_train_now=float(loss_avg.val())
                if loss_train_now<best_train_loss:
                    best_train_loss=loss_train_now
                    # torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_train_losss.pth')
                loss_avg.reset()
                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy}, {"Current_norm_ED":17s}: {current_norm_ED}'
                if valid_loss<best_valid_loss:
                    best_valid_loss=valid_loss
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_valid_losss.pth')
                if current_accuracy >= best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(
                        model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                if current_norm_ED >= best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(),f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy}, {"Best_norm_ED":17s}: {best_norm_ED}'
                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')
                    
                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:so_print], preds[:so_print], confidence_score[:so_print]):
                    if opt.Transformer:
                        pred = pred[:pred.find('[s]')]
                    elif 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.8f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')
            if scheduler is not None:
                scheduler.step()

""
if __name__ == '__main__':
    so_print=10
    opt = get_args()
    "nếu k đưa vào tên model thì mật định là cái tên đó"
    if not opt.exp_name:
        opt.exp_name = f'{opt.TransformerModel}' if opt.Transformer else f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
    print('--------------------------------')
    print(opt.TransformerModel)
    print('--------------------------------')
    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)
    data_thongsocan=''
    data_thongsocan+="character : "+opt.character+"\n"
    data_thongsocan+="imgH : "+str(opt.imgH)+"\n"
    data_thongsocan+="imgW : "+str(opt.imgW)+"\n"
    data_thongsocan+="batch_max_length : "+str(opt.batch_max_length)+"\n"

    data_thongsocan+="Transformation : "+opt.Transformation+"\n"
    data_thongsocan+="FeatureExtraction : "+opt.FeatureExtraction+"\n"
    data_thongsocan+="SequenceModeling : "+opt.SequenceModeling+"\n"
    data_thongsocan+="Prediction : "+opt.Prediction+"\n"
    data_thongsocan+="select_data : "+opt.select_data+"\n"
    data_thongsocan+="batch_ratio : "+opt.batch_ratio+"\n"
    data_thongsocan+="batch_size : "+str(opt.batch_size)+"\n"
    data_thongsocan+="num_iter : "+str(opt.num_iter)+"\n"
    
    """ vocab / character number configuration """
    print(opt.character)
    data_thongsocan+="adam : "+str(opt.adam)+"\n"
    data_thongsocan+="lr : "+str(opt.lr)+"\n"
    data_thongsocan+="beta1 : "+str(opt.beta1)+"\n"
    data_thongsocan+="beta2 : "+str(opt.beta2)+"\n"
    with open(f'./saved_models/{opt.exp_name}/data_thongsocan.txt',"w") as f:
        f.write(data_thongsocan)
        f.close()
    print("name model : ",opt.exp_name)
    """ Seed and GPU setting ,lưu số mật định của ran dom"""
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    if opt.workers <= 0:
        opt.workers = (os.cpu_count() // 2) // opt.num_gpu

    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu
    train(opt)


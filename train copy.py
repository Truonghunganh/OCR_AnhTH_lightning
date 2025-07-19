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
from test import validation
from utils import get_args

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

# python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --imgH 224 --imgW 224


def train(opt):
    """ dataset preparation  """
    "lọc dữ liệu"
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    opt.eval = False
    train_dataset = Batch_Balanced_Dataset(opt)

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

    """ model configuration """
    if opt.Transformer:
        converter = TokenLabelConverter(opt)
    elif 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    '''jjjjjjjjjjjjj'''
    # converter = AttnLabelConverter(opt.character)
    "kkkkkkkkkkkkkk"
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

    # data parallel for multi-GPU
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            print(opt)
            model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    
    # print("Model:")
    # print(model)

    """ setup loss """
    # README: https://github.com/clovaai/deep-text-recognition-benchmark/pull/209
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            # need to install warpctc. see our guideline.
            from warpctc_pytorch import CTCLoss
            criterion = CTCLoss()
        else:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(
            device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    so_lan_filtered_parameters=0
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    # setup optimizer
    scheduler = None
    optimizer=get_optimizer(opt=opt,filtered_parameters=filtered_parameters)
    if opt.scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.num_iter)

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        # print(opt_log)
        opt_file.write(opt_log)
        total_params = int(sum(params_num))
        total_params = f'Trainable network params num : {total_params:,}'
        print(total_params)
        opt_file.write(total_params)

    """ start training """
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
    iteration = start_iter
    print("THOI GIAN BAT DAU train :",datetime.datetime.now(),opt.valInterval)
    print(f"Số lượng batch: {len(train_dataset)}")

    while(True):
        # train part
        for iiiii in tqdm(range(int(opt.valInterval))): 
            image_tensors, labels = train_dataset.get_batch()
            image = image_tensors.to(device)
            if not opt.Transformer:
                text, length = converter.encode(
                    labels, batch_max_length=opt.batch_max_length)
            batch_size = image.size(0)

            if 'CTC' in opt.Prediction:
                preds = model(image, text)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                if opt.baiduCTC:
                    preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                    cost = criterion(preds, text, preds_size, length) / batch_size
                else:
                    preds = preds.log_softmax(2).permute(1, 0, 2)
                    cost = criterion(preds, text, preds_size, length)
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

            model.zero_grad()
            cost.backward()
            # gradient clipping with 5 (Default)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

            loss_avg.add(cost)
            if (iteration + 1) == opt.num_iter:
                print('end the training')
                sys.exit()
            iteration += 1
            if opt.num_iter%iteration!=0:
                if scheduler is not None:
                    scheduler.step()
            '---------------'
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

            # training loss and validation loss
            if best_train_loss==0:
                best_train_loss=float(loss_avg.val())
            loss_log = f'[{iteration}/{opt.num_iter}] Train loss: {loss_avg.val():0.18f}, Valid loss: {valid_loss:0.18f}, Elapsed_time: {elapsed_time:0.5f}\n now time: {str(datetime.datetime.now())},best_valid_loss:{best_valid_loss},best_train_loss : {best_train_loss}'
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
                # filtered_parameters=[]
                # so_lan_filtered_parameters=0
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
        '''dang thu nghiem'''
        # so_lan_filtered_parameters+=1
        # if so_lan_filtered_parameters>3:
        #     so_lan_filtered_parameters=0
        #     filtered_parameters=[]
        # for p in filter(lambda p: p.requires_grad, model.parameters()):
        #     filtered_parameters.append(p)
        # optimizer=get_optimizer(opt=opt,filtered_parameters=filtered_parameters)
        # if opt.scheduler:
        #     scheduler = CosineAnnealingLR(optimizer, T_max=opt.num_iter)


        # opt.lr=random.uniform(0.0001,0.1)
        # optimizer = optim.Adadelta(
        #     filtered_parameters, lr=ospt.lr, rho=opt.rho, eps=opt.eps)
        # if opt.scheduler:
        #     scheduler = CosineAnnealingLR(optimizer, T_max=opt.num_iter)

        # del model
        # # # opt.lr=opt.lr*0.995
        # # opt.valInterval=1
        # # # opt.valInterval=random.randint(1,10)
        # # # opt.saved_model=f'./saved_models/{opt.exp_name}/best_norm_ED.pth'
        # # # opt.saved_model=f'./saved_models/{opt.exp_name}/best_accuracy.pth'
        # opt.saved_model=random.choice([f'./saved_models/{opt.exp_name}/best_accuracy.pth',f'./saved_models/{opt.exp_name}/best_valid_losss.pth',f'./saved_models/{opt.exp_name}/best_norm_ED.pth'])
        # # # opt.saved_model=random.choice([f'./saved_models/{opt.exp_name}/best_accuracy.pth',f'./saved_models/{opt.exp_name}/best_norm_ED.pth'])
        # model = Model(opt)
        # model = torch.nn.DataParallel(model).to(device)
        # model.train()
        # if opt.saved_model != '':
        #     if opt.FT:
        #         model.load_state_dict(torch.load(opt.saved_model), strict=False)
        #     else:
        #         model.load_state_dict(torch.load(opt.saved_model, map_location=device))
        # if 'CTC' in opt.Prediction:
        #     if opt.baiduCTC:
        #         # need to install warpctc. see our guideline.
        #         from warpctc_pytorch import CTCLoss
        #         criterion = CTCLoss()
        #     else:
        #         criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
        # else:
        #     criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(
        #         device)  # ignore [GO] token = ignore index 0
        # # loss_avg = Averager()
        # so_lan_filtered_parameters+=1
        # if so_lan_filtered_parameters>10:
        #     so_lan_filtered_parameters=0
        #     filtered_parameters=[]
        
        # for p in filter(lambda p: p.requires_grad, model.parameters()):
        #     filtered_parameters.append(p)
        # optimizer=get_optimizer(opt=opt,filtered_parameters=filtered_parameters)
        # if opt.scheduler:
        #     scheduler = CosineAnnealingLR(optimizer, T_max=opt.num_iter)


        '-----------------------'

        if scheduler is not None:
            scheduler.step()

""
if __name__ == '__main__':
    so_print=10
    # lấy các tham số cần thuyết
    # best_accuracy_can_du_an=99.8
    opt = get_args()
    "nếu k đưa vào tên model thì mật định là cái tên đó"
    if not opt.exp_name:
        opt.exp_name = f'{opt.TransformerModel}' if opt.Transformer else f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
    print('--------------------------------')
    print(opt.TransformerModel)
    print('--------------------------------')
    # opt.exp_name += f'-Seed{opt.manualSeed}'
    # nếu không tồn tại thì tạo ra model 
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

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)

'''
CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_train/SGC/BK/nhan_len10/db_train --valid_data data_train/SGC/BK/nhan_len10/db_train --select_data / --batch_ratio 1 --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --TransformerModel=vitstr_base_patch16_224 --imgH 224 --imgW 224 --manualSeed=$RANDOM  --sensitive --batch_size 48 --num_iter 1500000 --valInterval 100000 --batch_max_length=10
CUDA_VISIBLE_DEVICES=1 python3 train.py --train_data data_train/SGC/BK/nhan_len10/db_train --valid_data data_train/SGC/BK/nhan_len10/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC  --imgH 64 --imgW 100 --manualSeed=$RANDOM  --sensitive --batch_size 192 --num_iter 1500000 --valInterval 100000 --batch_max_length=10 --saved_model iter_900000.pth

CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_train/SGC/MOSI/MOSI_AnhTH_train/db_train_MOSI_loai1 --valid_data data_train/SGC/MOSI/MOSI_AnhTH_train/db_train_MOSI_loai1 --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 80 --imgW 192 --manualSeed=$RANDOM  --sensitive --batch_size 120 --num_iter 1000000 --valInterval 100000 --batch_max_length=4 --character "-0123456789" --saved_model best_accuracy.pth

CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_train/SGC/MOSI/MOSI_AnhTH_train/db_train_MOSI_loai3 --valid_data data_train/SGC/MOSI/MOSI_AnhTH_train/db_train_MOSI_loai3 --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 71 --imgW 776 --manualSeed=$RANDOM  --sensitive --batch_size 20 --num_iter 1000000 --valInterval 50000 --batch_max_length=16 --character " ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝﾞﾟ" --saved_model best_valid_losss.pth
CUDA_VISIBLE_DEVICES=1 python3 train.py --train_data data_train/SGC/MOSI/MOSI_AnhTH_train/db_train_MOSI_loai3_HAJIME_location_2 --valid_data data_train/SGC/MOSI/MOSI_AnhTH_train/db_train_MOSI_loai3_HAJIME_location_2 --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 71 --imgW 776 --manualSeed=$RANDOM  --sensitive --batch_size 8 --num_iter 1000000 --valInterval 20000 --batch_max_length=18 --character " ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝﾞﾟ" --saved_model best_accuracy.pth


CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_train/HAJIME/HAJIME_location_3/db_train --valid_data data_train/HAJIME/HAJIME_location_3/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 71 --imgW 385 --manualSeed=$RANDOM  --sensitive --batch_size 64 --num_iter 1000000 --valInterval 50000 --batch_max_length=4 --character "012"
CUDA_VISIBLE_DEVICES=1 python3 train.py --train_data data_train/HAJIME/HAJIME_location_2/db_train --valid_data data_train/HAJIME/HAJIME_location_2/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 585 --manualSeed=$RANDOM  --sensitive --batch_size 16 --num_iter 1000000 --valInterval 50000 --batch_max_length=18 --character " @ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝﾞﾟ"
CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_train/HAJIME/HAJIME_location_4/db_train --valid_data data_train/HAJIME/HAJIME_location_4/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 150 --manualSeed=$RANDOM  --sensitive --batch_size 128 --num_iter 1000000 --valInterval 50000 --batch_max_length=3 --character "0123456789@"
CUDA_VISIBLE_DEVICES=1 python3 train.py --train_data data_train/HAJIME/HAJIME_location_1_78_9/db_train --valid_data data_train/HAJIME/HAJIME_location_1_78_9/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 260 --manualSeed=$RANDOM  --sensitive --batch_size 32 --num_iter 1000000 --valInterval 50000 --batch_max_length=6 --character="-0123456789@ABCDEFGHIJKLMNOPQRTUVWXYZ"
CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/HAJIME/HAJIME_location_6/db_train --valid_data data_train/HAJIME/HAJIME_location_6/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 500 --manualSeed=$RANDOM  --sensitive --batch_size 20 --num_iter 1000000 --valInterval 50000 --batch_max_length=10 --character="-0123456789@ACDI"

CUDA_VISIBLE_DEVICES=0 python train.py --train_data data_train/KAITO/KAITO_SO_len1/db_train --valid_data data_train/KAITO/KAITO_SO_len1/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 99 --imgW 85 --manualSeed=$RANDOM  --sensitive --batch_size 20 --num_iter 1000000 --valInterval 50000 --batch_max_length=1 --character="#123456789"
CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/KAITO/KAITO_test/db_train --valid_data data_train/KAITO/KAITO_test/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 100 --imgW 90 --manualSeed=$RANDOM  --sensitive --batch_size 20 --num_iter 5000 --valInterval 200 --batch_max_length=1 --character="12"

CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/SGC/BK/class0405/db_train --valid_data data_train/SGC/BK/class0405/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 75 --manualSeed=$RANDOM  --sensitive --batch_size 20 --num_iter 1000000 --valInterval 50000 --batch_max_length=10 --character=" ()+,-.0123456789=@ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --saved_model best_valid_losss.pth


CUDA_VISIBLE_DEVICES=0 python train.py --train_data data_train/YOBE/yobe_nhieu_diem/db_train --valid_data data_train/YOBE/yobe_nhieu_diem/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 150 --imgW 150 --manualSeed=$RANDOM  --sensitive --batch_size 20 --num_iter 1000000 --valInterval 100000 --batch_max_length=2 --character="#0123456789@" --saved_model iter_200000.pth best_accuracy.pth

CUDA_VISIBLE_DEVICES=0 python train.py --train_data data_train/YOBE/yobe_diem_ve_tay/yobe_diem_anh_doc/db_train --valid_data data_train/YOBE/yobe_diem_ve_tay/yobe_diem_anh_doc/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 340 --imgW 334 --manualSeed=$RANDOM  --sensitive --batch_size 12 --num_iter 1000000 --valInterval 100000 --batch_max_length=2 --character="0123456789@" 
CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/YOBE/yobe_diem_ve_tay/yobe_diem_anh_ngang/db_train --valid_data data_train/YOBE/yobe_diem_ve_tay/yobe_diem_anh_ngang/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 392 --imgW 220 --manualSeed=$RANDOM  --sensitive --batch_size 16 --num_iter 1000000 --valInterval 50000 --batch_max_length=2 --character="0123456789@"

CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/YOBE/yobe_MSTD_ve_tay/db_train --valid_data data_train/YOBE/yobe_MSTD_ve_tay/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 120 --imgW 450 --manualSeed=$RANDOM  --sensitive --batch_size 20 --num_iter 1000000 --valInterval 50000 --batch_max_length=7 --character="-0123456789ABCDEFGHJK"

CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/YOBE/yobe_Diem_chung_ve_tay/db_train --valid_data data_train/YOBE/yobe_Diem_chung_ve_tay/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 400 --imgW 400 --manualSeed=$RANDOM  --sensitive --batch_size 10 --num_iter 1000000 --valInterval 100000 --batch_max_length=2 --character="0123456789@" --saved_model iter_500000.pth
CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/create_nhu_mnit/db_train --valid_data data_train/create_nhu_mnit/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 64 --manualSeed=$RANDOM  --sensitive --batch_size 10 --num_iter 200000 --valInterval 40000 --batch_max_length=1 --character="0123456789@"

CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/HAJIME/HAJIME_location_5/db_train --valid_data data_train/HAJIME/HAJIME_location_5/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 296 --imgW 64 --manualSeed=$RANDOM  --sensitive --batch_size 32 --num_iter 1000000 --valInterval 50000 --batch_max_length=6 --character "@1"

CUDA_VISIBLE_DEVICES=0 python train.py --train_data data_train/create_nhu_mnit/tieng_nhat/db_train --valid_data data_train/create_nhu_mnit/tieng_nhat/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 64 --manualSeed=$RANDOM  --sensitive --batch_size 64 --num_iter 1000000 --valInterval 50000 --batch_max_length=1 --character " 0123456789@ｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝﾞﾟ"
CUDA_VISIBLE_DEVICES=0 python train.py --train_data data_train/TuNIT/iii5k_lmdb/train --valid_data data_train/TuNIT/iii5k_lmdb/validation --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 256 --manualSeed=$RANDOM  --sensitive --batch_size 16 --num_iter 1000000 --valInterval 50000 --batch_max_length=22 --character "0123456789abcdefghijklmnopqrstuvwxyz"
/home/vbpo-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/TuNIT/iii5k_lmdb

CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/generate/so_generate/db_train --valid_data data_train/generate/so_generate/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 100 --imgW 400 --manualSeed=$RANDOM  --sensitive --batch_size 20 --num_iter 5000000 --valInterval 200000 --batch_max_length=11 --character "0123456789"
CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/SGC/PO/Class04/db_train --valid_data data_train/SGC/PO/Class04/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 368 --manualSeed=$RANDOM  --sensitive --batch_size 20 --num_iter 5000000 --valInterval 50000 --batch_max_length=6 --character "0123456789" --saved_model best_accuracy.pth

CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/capcha/db_train_v1 --valid_data data_train/capcha/db_train_v1 --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 240 --manualSeed=$RANDOM  --sensitive --batch_size 20 --num_iter 5000000 --valInterval 50000 --batch_max_length=5 --character "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" --saved_model best_accuracy.pth

CUDA_VISIBLE_DEVICES=0 python train.py --train_data data_train/generate/so_generate/db_train --valid_data data_train/generate/so_generate/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 640 --manualSeed=$RANDOM  --sensitive --batch_size 20 --num_iter 5000000 --valInterval 50000 --batch_max_length=21 --character "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/capcha/db_val --valid_data data_train/capcha/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 40 --imgW 200 --manualSeed=$RANDOM  --sensitive --batch_size 20 --num_iter 5000000 --valInterval 50000 --batch_max_length=5 --character "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

CUDA_VISIBLE_DEVICES=0 python train.py --train_data data_train/capcha/db_train --valid_data data_train/capcha/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 200 --rgb --manualSeed=$RANDOM  --sensitive --batch_size 20 --num_iter 5000000 --valInterval 20000 --batch_max_length=6 --character "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" --saved_model best_accuracy.pth
CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/capcha/db_train --valid_data data_train/capcha/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC --imgH 32 --imgW 200 --rgb --manualSeed=$RANDOM  --sensitive --batch_size 64 --num_iter 5000000 --valInterval 20000 --batch_max_length=6 --character "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/capcha/db_train --valid_data data_train/capcha/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 320 --rgb --manualSeed=$RANDOM  --sensitive --batch_size 32 --num_iter 5000000 --valInterval 20000 --batch_max_length=6 --character "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/capcha/db_train --valid_data data_train/capcha/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 260 --manualSeed=$RANDOM  --sensitive --batch_size 20 --num_iter 5000000 --valInterval 20000 --batch_max_length=5 --character "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

CUDA_VISIBLE_DEVICES=0 python train.py --train_data data_train/capcha/TuNIT/db_train --valid_data data_train/capcha/TuNIT/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 200 --exp_name "Capcha_TuNIT" --manualSeed=$RANDOM  --sensitive --batch_size 20 --num_iter 5000000 --valInterval 20000 --batch_max_length=5 --character "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"



CUDA_VISIBLE_DEVICES=1 python train.py --train_data /home/vbpo-100367/sv7_dri_data/TuNIT/Datasets/OCR/lmdb_dataset/data_lmdb_release/data_lmdb_release/data_lmdb_release/training/MJ/MJ_train --valid_data /home/vbpo-100367/sv7_dri_data/TuNIT/Datasets/OCR/lmdb_dataset/data_lmdb_release/data_lmdb_release/data_lmdb_release/training/MJ/MJ_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 200 --manualSeed=$RANDOM  --sensitive --batch_size 20 --num_iter 5000000 --valInterval 2000 --batch_max_length=25 --character "0123456789Aabcdefghijklmnopqrstuvwxyz"

CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/capcha/db_train --valid_data data_train/capcha/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --Transformer --TransformerModel=vitstr_base_patch16_224 --imgH 224 --imgW 224 --manualSeed=$RANDOM --sensitive --batch_size=48 --num_iter 5000000 --valInterval 10000 --batch_max_length=6 --character "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" --saved_model vitstr_base_patch16_224_aug.pth

CUDA_VISIBLE_DEVICES=0 python train.py --exp_name "generate" --train_data /media/rsc-100367/DATA/AnhTH/Du_lieu_yolo_cua_du_an/generate/db_train_v1 --valid_data data_train/generate/db_val --select_data MJ_ST-AnhTH --batch_ratio 0.8-0.2 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 200 --manualSeed=$RANDOM  --sensitive --batch_size 192 --num_iter 5000000 --valInterval 10000 --batch_max_length=30 --character " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" --saved_model generate_v4.pth


CUDA_VISIBLE_DEVICES=0 python train.py --train_data /media/vbpo-100367/data/AnhTH/Du_lieu_yolo_cua_du_an/generate/OCR/db_train --valid_data /media/vbpo-100367/data/AnhTH/Du_lieu_yolo_cua_du_an/generate/OCR/db_val --select_data MJ_ST-AnhTH  --batch_ratio 0.5-0.5 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 100 --exp_name "test" --manualSeed=$RANDOM  --sensitive --batch_size 2 --num_iter 5000000 --valInterval 10000 --batch_max_length=5 --character "0123456789"

CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/capcha/db_train --valid_data data_train/capcha/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 200 --manualSeed=$RANDOM  --sensitive --batch_size 64 --num_iter 5000000 --valInterval 2000 --exp_name "capcha_AnhTH_v1" --batch_max_length=6 --character "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" --saved_model v1.pth best_accuracy.pth

CUDA_VISIBLE_DEVICES=0 python train.py --train_data /media/vbpo-100367/DATA/AnhTH/Du_lieu_yolo_cua_du_an/generate/train --valid_data data_train/generate/validation --select_data MJ_ST-AnhTH  --batch_ratio 0.7-0.3 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 200 --exp_name "generate" --manualSeed=$RANDOM  --sensitive --batch_size 192 --num_iter 5000000 --valInterval 1 --batch_max_length=30 --character "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

CUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/capcha/db_train --valid_data data_train/capcha/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 200 --manualSeed=$RANDOM  --sensitive --batch_size 205 --num_iter 5000000 --valInterval 100 --exp_name "capcha_AnhTH_v21" --batch_max_length=12 --character "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" --rgb --saved_model v21.pth --lr 0.1  --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 1 --augs_mag 1 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model v21.pth --quantized --static --cameraCUDA_VISIBLE_DEVICES=1 python train.py --train_data data_train/capcha/db_train --valid_data data_train/capcha/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 200 --manualSeed=$RANDOM  --sensitive --batch_size 205 --num_iter 5000000 --valInterval 100 --exp_name "capcha_AnhTH_v21" --batch_max_length=12 --character "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" --rgb --saved_model v21.pth --lr 0.1  --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 1 --augs_mag 1 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model v21.pth --quantized --static --camera
CUDA_VISIBLE_DEVICES=0 python train.py --train_data data_train/capcha/db_train --valid_data data_train/capcha/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 200 --manualSeed=$RANDOM  --sensitive --batch_size 200 --num_iter 5000000 --valInterval 0 --exp_name "capcha_AnhTH_v21" --batch_max_length=12 --character "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" --rgb --saved_model v21.pth --lr 1  --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 1 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model v21.pth --quantized --static --camera

CUDA_VISIBLE_DEVICES=1 python train.py --exp_name "msdt_yobe_transformer" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/msdt/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/msdt/db_val --select_data / --batch_ratio 1 --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --TransformerModel=vitstr_base_patch16_224 --imgH 224 --imgW 224 --manualSeed=$RANDOM --batch_size 32 --num_iter 5000000 --valInterval 2000 --batch_max_length=7 --character "0123456789ABCDEFGHJKMXZ-" --rgb --lr 0.01

CUDA_VISIBLE_DEVICES=1 python train.py --exp_name "msdt_yobe_64_256" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/msdt/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/msdt/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 256 --manualSeed=$RANDOM --batch_size 64 --num_iter 5000000 --valInterval 2000 --batch_max_length=7 --character "0123456789ABCDEFGHJKMXZ-" --rgb --lr 1  --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 1 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model v21.pth --quantized --static --camera

CUDA_VISIBLE_DEVICES=1 python train.py --exp_name "diem_yobe_64_64" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/diem/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/diem/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 64 --manualSeed=$RANDOM --batch_size 192 --num_iter 5000000 --valInterval 5000 --batch_max_length=2 --character "#0123456789@" --rgb --lr 1 --saved_model diem_64_64.pth --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera
CUDA_VISIBLE_DEVICES=1 python train.py --exp_name "diem_yobe_32_32" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/diem/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/diem/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 32 --manualSeed=$RANDOM --batch_size 192 --num_iter 5000000 --valInterval 5000 --batch_max_length=2 --character "#0123456789@" --rgb --lr 1 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name "diem_yobe_32_32_khong_mau" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/diem/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/diem/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 32 --manualSeed=$RANDOM --batch_size 192 --num_iter 5000000 --valInterval 5000 --batch_max_length=2 --character "#0123456789@" --lr 1 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name "msdt_yobe_32_120" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/msdt/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/msdt/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 120 --manualSeed=$RANDOM --batch_size 128 --num_iter 5000000 --valInterval 5000 --batch_max_length=7 --character "0123456789ABCDEFGHJKMXZ-" --rgb --lr 1 --saved_model msdt.pth  --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model v21.pth --quantized --static --camera
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name "HD_Class08_VGG" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/Class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/Class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 200 --manualSeed=$RANDOM --batch_size 192 --num_iter 5000000 --valInterval 10000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz|" --rgb --lr 0.01 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.2 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera

CUDA_VISIBLE_DEVICES=1 python train.py --exp_name "diem_yobe_32_32" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/diem/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/diem/db_val --select_data data-data_4122024 --batch_ratio 0.5-0.5 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 32 --manualSeed=$RANDOM --batch_size 192 --num_iter 5000000 --valInterval 10000 --batch_max_length=2 --character "#0123456789@" --rgb --lr 0.1 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera --saved_model diem_32_32.pth
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name "diem_yobe_64_64" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/diem/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/diem/db_val --select_data data-data_4122024 --batch_ratio 0.7-0.3 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 64 --manualSeed=$RANDOM --batch_size 192 --num_iter 5000000 --valInterval 10000 --batch_max_length=2 --character "#0123456789@" --rgb --lr 0.01 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera --saved_model diem_64_64.pth

CUDA_VISIBLE_DEVICES=1 python train.py --exp_name "msdt_yobe_64_300" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/msdt/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/YOBE/msdt/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 300 --manualSeed=$RANDOM --batch_size 64 --num_iter 5000000 --valInterval 10000 --batch_max_length=7 --character "0123456789ABCDEFGHJKMXZ-" --rgb --lr 0.01 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model v21.pth --quantized --static --camera --saved_model msdt_64_300.pth

CUDA_VISIBLE_DEVICES=0 python train.py --exp_name "iwaki_phanloai_64_64" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/iwaki/phanloai/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/iwaki/phanloai/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 128 --imgW 128 --manualSeed=$RANDOM --batch_size 64 --num_iter 5000000 --valInterval 10000 --batch_max_length=10 --character "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" --rgb --lr 0.01 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera --saved_model ./saved_models/iwaki_phanloai_128_128/best_accuracy.pth


CUDA_VISIBLE_DEVICES=1 python train.py --exp_name "phanloai_test" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/phanloai/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/phanloai/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 224 --imgW 224 --manualSeed=$RANDOM --batch_size 16 --num_iter 5000000 --valInterval 2000 --batch_max_length=10 --character "abcdefghiklnopqrstuwy" --rgb --lr 1 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera 

CUDA_VISIBLE_DEVICES=1 python train.py --exp_name "HD_08" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 512 --manualSeed=$RANDOM --batch_size 164--num_iter 5000000 --valInterval 2000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --lr 0.01 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera --saved_model HD_Class08_64_512.pth
--saved_model ./saved_models/HD_08_swin_large_patch4_window12_384_32_480/best_accuracy.pth
CUDA_VISIBLE_DEVICES=1 python train.py --exp_name "HD_08_ResNet152_32_480" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet152 --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 480 --manualSeed=$RANDOM --batch_size 64 --num_iter 5000000 --valInterval 20000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --lr 1 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera

CUDA_VISIBLE_DEVICES=0 python train.py --exp_name "HD_08_VGG19_32_480" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction VGG19 --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 480 --manualSeed=$RANDOM --batch_size 32 --num_iter 5000000 --valInterval 20000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --lr 1 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name "HD_08_efficientnet_b7_32_480" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction efficientnet_b7 --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 480 --manualSeed=$RANDOM --batch_size 128 --num_iter 5000000 --valInterval 20000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --lr 1 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name "HD_08_efficientnet_b7_32_480_2560" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction efficientnet_b7 --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 480 --manualSeed=$RANDOM --batch_size 128 --num_iter 5000000 --valInterval 20000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --lr 1 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera --output_channel=2560
CUDA_VISIBLE_DEVICES=1 python train.py --exp_name "HD_08_MobileNetV3_32_480" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction MobileNetV3 --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 480 --manualSeed=$RANDOM --batch_size 32 --num_iter 5000000 --valInterval 20000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --lr 1 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name "HD_08_vision_transformer.vit_b_16_32_480" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction vision_transformer.vit_b_16 --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 480 --manualSeed=$RANDOM --batch_size 32 --num_iter 5000000 --valInterval 20000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --lr 1 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera

CUDA_VISIBLE_DEVICES=0 python train.py --exp_name "HD_08_swin_b_32_480" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction swin_b --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 480 --manualSeed=$RANDOM --batch_size 32 --num_iter 5000000 --valInterval 20000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --lr 1 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name "HD_08_convnext_large_32_480" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction convnext_large --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 480 --manualSeed=$RANDOM --batch_size 32 --num_iter 5000000 --valInterval 20000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --lr 1 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name "HD_08_swin_large_patch4_window12_384_32_480" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction swin_large_patch4_window12_384 --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 480 --manualSeed=$RANDOM --batch_size 32 --num_iter 5000000 --valInterval 20000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --lr 1 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name "HD_08_vit_base_patch16_224_384_32_480" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction vit_base_patch16_224 --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 480 --manualSeed=$RANDOM --batch_size 32 --num_iter 5000000 --valInterval 20000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --lr 1 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera

CUDA_VISIBLE_DEVICES=1 python train.py --exp_name "HD_08_swin_base_patch4_window7_224_32_480" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction swin_base_patch4_window7_224 --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 480 --manualSeed=$RANDOM --batch_size 8 --num_iter 5000000 --valInterval 20000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --lr 1 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera


CUDA_VISIBLE_DEVICES=1 python train.py --exp_name "phanloai_test" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/phanloai/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/phanloai/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet152 --SequenceModeling BiLSTM --Prediction Attn --imgH 224 --imgW 224 --manualSeed=$RANDOM --batch_size 16 --num_iter 5000000 --valInterval 2000 --batch_max_length=10 --character "abcdefghiklnopqrstuwy" --rgb --lr 1 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera 

đối  với class08_HD:
- mobilenet_v3_large: sáu

CUDA_VISIBLE_DEVICES=1 python train.py --exp_name "test" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/iwaki/phanloai/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/iwaki/phanloai/db_train --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction swin_base_patch4_window7_224 --SequenceModeling BiLSTM --Prediction Attn --imgH 128 --imgW 128 --manualSeed=$RANDOM --batch_size 64 --num_iter 5000000 --valInterval 10000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --lr 1 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera

output_channel=512
'''

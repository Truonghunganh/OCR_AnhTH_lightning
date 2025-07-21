from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model_lightning import *
def savethongtincan(opt,model):
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        params_num.append(np.prod(p.size()))
    file_txt_save=f'./saved_models/{opt.exp_name}/opt.txt'
    if os.path.exists(file_txt_save):
        os.remove(file_txt_save)
    with open(file_txt_save, 'a') as opt_file:
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
    if opt.rgb:
        opt.input_channel = 3
    opt.eval = True
    if opt.sensitive:
        opt.data_filtering_off = True
    AlignCollate_valid = AlignCollate( #  căng chỉnh lại hình ảnh
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
    valid_dataset, valid_dataset_log = hierarchical_dataset(
        root=opt.valid_data, opt=opt)
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        # 'True' to check training progress with validation function.
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)

    # train_dataset = Batch_Balanced_Dataset(opt)
    train_dataset, train_dataset_log = hierarchical_dataset(
        root=opt.train_data, opt=opt)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        # 'True' to check training progress with validation function.
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)

    print("Số lượng ảnh trong valid_dataset:", len(valid_dataset))

# 
#     train_loader = Batch_Balanced_Dataset(opt)
# train_loader = DataLoader(train_dataset, batch_size=..., num_workers=..., shuffle=True)

    model = Model_Lightning(opt=opt)
    from pytorch_lightning.strategies import DDPStrategy

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator="gpu",
        precision=16,                  # mixed precision giúp tăng tốc (nếu ổn định)
        devices=opt.num_gpu,
        max_epochs=100,
        benchmark=True,               # tối ưu conv cuDNN
        deterministic=False,          # tăng tốc (có thể thay đổi kết quả)
    )
    
    # val_result = trainer.validate(model=model, dataloaders=val_loader)
    # print("Validation result:", val_result)
    savethongtincan(opt,model)
    trainer.fit(model, train_loader, val_loader)
    
    # Chạy validate trước khi train
    # print("Running validation before training...")
    # val_result = trainer.validate(model=model, dataloaders=val_loader)
    # print("Validation result:", val_result)


    # trainer = pl.Trainer(max_epochs=1000, accelerator="gpu", precision=16,gradient_clip_val=1.0)

    # trainer = pl.Trainer(max_epochs=1000, accelerator="auto")
    # trainer.fit(model, train_loader, val_loader)

import torch.backends.cudnn as cudnn
import random,os
import numpy as np
from utils import get_args

if __name__ == '__main__':
    opt = get_args()
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
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu


    train(opt)


"""
CUDA_VISIBLE_DEVICES=0,1 python train_lightning.py --exp_name "HD_08_ResNet_32_480" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 480 --manualSeed=$RANDOM --batch_size 16 --num_iter 5000000 --valInterval 1000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --lr 0.01 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera --saved_model HD_08_ResNet_32_480_1.pth

CUDA_VISIBLE_DEVICES=0,1 python train_lightning.py --exp_name "HD_08_ResNet_64_512" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 64 --imgW 512 --manualSeed=$RANDOM --batch_size 32 --num_iter 5000000 --valInterval 1000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz|" --rgb --lr 0.0001 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera --saved_model HD_08_ResNet_64_512.pth

CUDA_VISIBLE_DEVICES=0 python train_lightning.py --exp_name "HD_08_ResNet_32_320" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 320 --manualSeed=$RANDOM --num_iter 5000000 --valInterval 1000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz|" --rgb --lr 0.0001 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera --saved_model HD_08_ResNet_32_320.pth --batch_size 128 --use_jit

CUDA_VISIBLE_DEVICES=0,1 python train_lightning.py --exp_name "HD_08_swin_base_patch4_window7_224_32_480" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction swin_base_patch4_window7_224 --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 480 --manualSeed=$RANDOM --num_iter 5000000 --valInterval 1000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --lr 0.01 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera --saved_model HD_08_swin_base_patch4_window7_224_32_480.pth --batch_size 16 --output_channel 1024

CUDA_VISIBLE_DEVICES=0,1 python train_lightning.py --exp_name "HD_08_swin_large_patch4_window12_384_32_480" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction swin_large_patch4_window12_384 --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 480 --manualSeed=$RANDOM --num_iter 5000000 --valInterval 1000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --lr 0.01 --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera  --batch_size 16 --output_channel 1000 --saved_model HD_08_swin_large_patch4_window12_384_32_480.pth



CUDA_VISIBLE_DEVICES=0,1 python train_lightning.py --exp_name "HD_08_VGG_32_480" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 480 --manualSeed=$RANDOM --num_iter 5000000 --valInterval 1000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera --saved_model HD_08_VGG_32_480.pth --batch_size 32 --lr 0.01

CUDA_VISIBLE_DEVICES=0,1 python train_lightning.py --exp_name "HD_08_VGG_32_1000" --train_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_train --valid_data /home/rsc-100367/Desktop/AnhTH/OCR/ocr_vitstr_transformer_AnhTH/data_train/SGC/class08/db_val --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction Attn --imgH 32 --imgW 1000 --manualSeed=$RANDOM --num_iter 5000000 --valInterval 1000 --batch_max_length=100 --character " $+-./0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz" --rgb --issel_aug --pattern --warp --geometry --weather --noise --blur --process --scheduler --intact_prob 0.5 --isrand_aug --augs_num 3 --augs_mag 3 --issemantic_aug --isrotation_aug --isscatter_aug --islearning_aug --fast_acc --infer_model diem.pth --quantized --static --camera --batch_size 128 --lr 1



trainer = pl.Trainer(
    strategy=DDPStrategy(find_unused_parameters=False),  # TRUE nếu model có nhánh phụ không dùng
    accelerator="gpu",
    devices=opt.num_gpu,           # VD: 2
    precision=16,                  # mixed precision giúp tăng tốc (nếu ổn định)
    max_epochs=100,
    enable_progress_bar=(rank == 0),  # tránh in trùng
    enable_checkpointing=(rank == 0), # chỉ rank 0 lưu
    log_every_n_steps=50,
    num_sanity_val_steps=0,       # tắt check val trước khi train nếu dùng nhiều ảnh
    benchmark=True,               # tối ưu conv cuDNN
    deterministic=False,          # tăng tốc (có thể thay đổi kết quả)
)

"""
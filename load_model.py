import cv2
import torch
from model import Model
import argparse
import torch.nn.functional as F
from dataset import RawDataset, AlignCollate
from utils import CTCLabelConverter, AttnLabelConverter
import numpy as np
import cv2,os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_thong_tin_model(opt):
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    model_ocr=load_model_ocr(opt=opt)
    return opt,converter,AlignCollate_demo,model_ocr

def get_parser(image_folder,saved_model,batch_max_length,imgH,imgW,character,sensitive,rgb,batch_size=1,Prediction="Attn",FeatureExtraction='ResNet'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default=image_folder, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=batch_size, help='input batch size')
    parser.add_argument('--saved_model',default= saved_model, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=batch_max_length, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=imgH, help='the height of the input image')
    parser.add_argument('--imgW',type=int, default=imgW, help='the width of the input image')
    parser.add_argument('--rgb',default=rgb, action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default=character, help='character label')
    parser.add_argument('--sensitive',default=sensitive, action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation',default='TPS', type=str, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default=FeatureExtraction, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str,default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default=Prediction, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    opt = parser.parse_args()
    if opt.rgb:
        opt.input_channel = 3
    opt.num_gpu = torch.cuda.device_count()
    return opt


def OCR(opt,model,converter,AlignCollate_demo):
    index=0
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)
    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

  
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]
                return pred

def List_OCR(opt,model,converter,AlignCollate_demo):
    list_result=[]
    list_conf=[]
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)
    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

  
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]
                try:
                    confidence_score = float(pred_max_prob.cumprod(dim=0)[-1])
                except:
                    confidence_score=0
                list_conf.append(confidence_score)
                list_result.append(pred)
    return list_result,list_conf
def load_model_ocr(opt):
    print("load model : ",opt.saved_model)
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    return model
def get_img_convert(image,h_resize,khoan1,khoan2,ksize):
    r = h_resize / image.shape[0]
    dim = ( int(image.shape[1] * r),h_resize)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    lower_white = np.array(khoan1, dtype=np.uint8)
    upper_white = np.array(khoan2, dtype=np.uint8)
    mask = cv2.inRange(resized, lower_white, upper_white)
    mask = cv2.medianBlur(mask,ksize=ksize)
    return mask

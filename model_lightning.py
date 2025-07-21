import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Model
import os
from modules.vitstr import create_vitstr
import pytorch_lightning as pl
from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager, TokenLabelConverter
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import math
from pytorch_lightning.utilities.rank_zero import rank_zero_only

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_txt(path_txt,data):
    with open(path_txt,'+a',encoding='utf8') as f :
        f.write(f"{data}\n")
        f.close()
class Model_Lightning(pl.LightningModule):

    def __init__(self, opt):
        super(Model_Lightning, self).__init__()
        os.makedirs(f'./saved_models/{opt.exp_name}',exist_ok=True)
        if opt.Transformer:
            self.converter = TokenLabelConverter(opt)
        elif 'CTC' in opt.Prediction:
            if opt.baiduCTC:
                self.converter = CTCLabelConverterForBaiduWarpctc(opt.character)
            else:
                self.converter = CTCLabelConverter(opt.character)
        else:
            self.converter = AttnLabelConverter(opt.character)
        opt.num_class = len(self.converter.character)

        if 'CTC' in opt.Prediction:
            if opt.baiduCTC:
                # need to install warpctc. see our guideline.
                from warpctc_pytorch import CTCLoss
                self.criterion = CTCLoss()
            else:
                self.criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
        self.opt = opt
        self.model=Model(opt)
        # scripted_model = torch.jit.script(self.model)
        # scripted_model.save("model_jit1.pt")

        # dummy_input=torch.randn(64, 3, 32, 480).to('cuda')  
        # self.model.eval().to(device)
        # dummy_image = torch.randn(1, 3, 32, 480).to(device)
        # dummy_text  = torch.randint(low=1, high=opt.num_class, size=(1, opt.batch_max_length+1)).to(device)
        # dummy_image = torch.randn(1, 3, 32, 480).to(device)
        # dummy_text  = torch.randint(low=1, high=opt.num_class, size=(0, opt.batch_max_length)).to(device)
        
        # traced_model = torch.jit.trace(self.model, (dummy_image, dummy_text))

        # traced_model = torch.jit.trace(self.model, dummy_input)
        
        # traced_model.save("model_jit.pt")

        # local_rank = int(os.environ["LOCAL_RANK"])
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], find_unused_parameters=False)

        # self.model = nn.DataParallel(self.model)
        # self.model=self.model.to(device)  
        # if opt.saved_model != '':
        #     print(f'loading pretrained model from {opt.saved_model}')
        #     if opt.FT:
        #         self.model.load_state_dict(torch.load(opt.saved_model), strict=False)
        #     else:
        #         print(opt)
        #         self.model.load_state_dict(torch.load(opt.saved_model, map_location=device))
            # self.model = torch.nn.DataParallel(self.model).to(device)
        # self.model=torch.jit.trace(self.model)
       
        self.model = self.model.to(device)
        # self.model=torch.jit.load("jit_traced_model.pth", map_location=device)
        if opt.saved_model != '':
            # state_dict = torch.jit.load("model_jit.pt", map_location=device)
            state_dict = torch.load(opt.saved_model, map_location=device)
            if list(state_dict.keys())[0].startswith("module."):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(state_dict)
        if self.opt.use_jit:
            print("self.opt.use_jit:",self.opt.use_jit,'kkkkkkkkkkkkkkkkkkkkkkkkkkk')
            dummy_image = torch.randn(opt.batch_size, 3, 32, 480).to(device)  # ví dụ ảnh OCR grayscale
            dummy_text  = torch.randint(low=1, high=opt.num_class, size=(opt.batch_size, opt.batch_max_length+1)).to(device)
            # self.model = self.model.eval().to(device)
            traced_model = torch.jit.trace(self.model, (dummy_image, dummy_text))
            traced_model.save("jit_traced_model.pth")
            self.model=torch.jit.load("jit_traced_model.pth", map_location=device)

        # scripted_model = torch.jit.script(self.model)
        # scripted_model.save("jit_scripted_model.pth")

        self.acc_max=0.0

    def forward(self, x,text=None,is_train=True):
        return self.model(x)

    def on_train_epoch_start(self):
        self.train_correct = 0
        self.train_total = 0

    def on_validation_epoch_start(self):
        self.val_correct = 0
        self.val_total = 0
        self.log("val_correct", self.val_correct, sync_dist=True, reduce_fx="sum")
        self.log("val_total", self.val_total, sync_dist=True, reduce_fx="sum")
    def on_train_epoch_end(self):
        self.log("val_correct", self.val_correct, sync_dist=True, reduce_fx="sum")
        self.log("val_total", self.val_total, sync_dist=True, reduce_fx="sum")
        acc = self.train_correct / self.train_total
        print("acc train:",acc)

    @rank_zero_only
    def on_validation_epoch_end(self):
        if self.val_correct !=0:
            self.val_correct , self.val_total=self.trainer.callback_metrics["val_correct"] , self.trainer.callback_metrics["val_total"]
        print('oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')
        acc = self.val_correct / self.val_total
        if self.acc_max<acc and self.val_total>10000:
            self.acc_max=acc
            print("✅ New best model saved!")
            os.makedirs(f'./saved_models/{self.opt.exp_name}',exist_ok=True)
            torch.save(
                    self.model.state_dict(), f'./saved_models/{self.opt.exp_name}/best_accuracy.pth')
        data_txt=f"acc val: {self.val_correct}/{self.val_total}={acc},acc max:{self.acc_max}"
        print(data_txt)
        save_txt(f'./saved_models/{self.opt.exp_name}/log_train.txt',data_txt)

        
    def training_step(self, batch, batch_idx):
        image, labels = batch
        # print(image.shape)
        image = image.to(device)
        if not self.opt.Transformer:
            text, length = self.converter.encode(
                labels, batch_max_length=self.opt.batch_max_length)
        batch_size = image.size(0)
        length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(device)
        text_for_loss, length_for_loss = self.converter.encode(labels, batch_max_length=self.opt.batch_max_length)

        if 'CTC' in self.opt.Prediction:
            preds = self.model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if self.opt.baiduCTC:
                preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                loss = self.criterion(preds, text, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                loss = self.criterion(preds, text, preds_size, length)
            '--------------------------'
            if self.opt.baiduCTC:
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
            else:
                _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index.data, preds_size.data)

            '--------------------------'

        elif self.opt.Transformer:
            target = self.converter.encode(labels)
            preds = self.model(image, text=target,
                        seqlen=self.converter.batch_max_length)
            loss = self.criterion(
                preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        else:
            '----'
            # text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            # preds = model(image, text_for_pred)
            '----'
            preds = self.model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            loss = self.criterion(
                preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        
        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index, length_for_pred)
        labels = self.converter.decode(text_for_loss[:, 1:], length_for_loss)

        # print('length_for_pred:',length_for_pred)
        for gt, pred in zip(labels, preds_str):
            if 'Attn' in self.opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred = pred[:pred.find('[s]')] 
            gt,pred=str(gt),str(pred)
            # print(gt,pred)
            if gt==pred:
                self.train_correct += 1
            self.train_total += 1
            # acc=correct/total
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", self.train_correct / self.train_total, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        correct,total=0,0
        image_tensors, labels = batch
        batch_size = image_tensors.size(0)
        # length_of_data = batch_size
        image = image_tensors.to(device)
        # For max length prediction
        if self.opt.Transformer:
            target = self.converter.encode(labels)
        else:
            length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(device)
            text_for_loss, length_for_loss = self.converter.encode(labels, batch_max_length=self.opt.batch_max_length)
        if 'CTC' in self.opt.Prediction:
            preds = self.model(image, text_for_pred)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if self.opt.baiduCTC:
                cost = self.criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss) / batch_size
            else:
                cost = self.criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding) then decode index to character
            if self.opt.baiduCTC:
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
            else:
                _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index.data, preds_size.data)
        
        elif self.opt.Transformer:
            preds = self.model(image, text=target, seqlen=self.converter.batch_max_length)
            _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
            preds_index = preds_index.view(-1, self.converter.batch_max_length)
            cost = self.criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            length_for_pred = torch.IntTensor([self.converter.batch_max_length - 1] * batch_size).to(device)
            preds_str = self.converter.decode(preds_index[:, 1:], length_for_pred)
        else:
            if self.opt.use_jit:
                preds = self.model(image, text_for_pred)  # ❌ KHÔNG truyền is_train
            else:
                preds = self.model(image, text_for_pred, is_train=False)
            
            # preds = self.model(image, text_for_pred, is_train=False)
            # preds = self.model(image, text_for_pred)
            
            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = self.criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)
            labels = self.converter.decode(text_for_loss[:, 1:], length_for_loss)
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if  self.opt.Transformer:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]
            elif 'Attn' in self.opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            if pred == gt:
                correct += 1
            total += 1
        self.val_total+=total
        self.val_correct +=correct 
        self.log("val_correct", correct, sync_dist=True, reduce_fx="sum")
        self.log("val_total", total, sync_dist=True, reduce_fx="sum")
        # if self.val_total>1000:
        #     self.val_correct=self.trainer.callback_metrics["val_correct"]
        #     self.val_total=self.trainer.callback_metrics["val_total"]

        self.log("val_loss", cost, prog_bar=True, sync_dist=True)
        self.log(f"val_acc:{self.val_correct}/{self.val_total}=",self.val_correct / self.val_total, prog_bar=True, sync_dist=True)
        return cost
    def configure_optimizers(self):
        print('configure_optimizersconfigure_optimizersconfigure_optimizersconfigure_optimizersconfigure_optimizers')
        # return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return torch.optim.Adadelta(self.model.parameters(), lr=self.opt.lr, rho=0.95, eps=1e-8)
        # return torch.optim.Adadelta(self.model.parameters(), lr=1e-5)
        # return torch.optim.Adadelta(self.model.parameters(), lr=1e-8, rho=0.95, eps=1e-8)
        # optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.opt.lr, rho=0.95, eps=1e-8)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        ''''''
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-8, weight_decay=1e-4)
        # def lr_lambda(current_step):
        #     warmup_steps = 10
        #     if current_step < warmup_steps:
        #         return float(current_step) / float(max(1, warmup_steps))
        #     return 0.5 * (1. + math.cos(math.pi * (current_step - warmup_steps) / (100 - warmup_steps)))        
        # scheduler = {
        #     'scheduler': LambdaLR(optimizer, lr_lambda),
        #     'interval': 'step',  # mỗi bước, không phải mỗi epoch
        #     'frequency': 1,
        # }
        ''''''
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.OneCycleLR(
        #         optimizer,
        #         max_lr=1e-3,
        #         steps_per_epoch=self.trainer.estimated_stepping_batches // self.trainer.max_epochs,
        #         epochs=self.trainer.max_epochs
        #     ),
        #     'interval': 'step',
        #     'frequency': 1
        # }
        ''''''
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6, weight_decay=1e-4)
        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6),
        #     'interval': 'epoch',  # update mỗi epoch
        #     'frequency': 1,
        # }
        ''''''
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-8, weight_decay=1e-4)    
        # def lr_lambda(current_step):
        #     warmup_steps = 500
        #     if current_step < warmup_steps:
        #         return float(current_step) / float(max(1, warmup_steps))
        #     progress = float(current_step - warmup_steps) / float(max(1, self.trainer.estimated_stepping_batches - warmup_steps))
        #     return 0.5 * (1. + math.cos(math.pi * progress))
        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
        #     'interval': 'step',
        #     'frequency': 1,
        # }

        
        return [optimizer], [scheduler]

class JitModel(Model):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.vitstr= create_vitstr(num_tokens=opt.num_class, model=opt.TransformerModel)

    def forward(self, input, seqlen:int = 25):
        prediction = self.vitstr(input, seqlen=seqlen)
        return prediction


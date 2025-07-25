"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch,timm
import torch.nn as nn
import torch.nn.functional as F

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
from modules.vitstr import create_vitstr
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import math



class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction,
                       'ViTSTR': opt.Transformer}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        if opt.Transformer:
            self.vitstr= create_vitstr(num_tokens=opt.num_class, model=opt.TransformerModel)
            return

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'VGG16':
            self.FeatureExtraction = models.vgg16(pretrained=False)
            self.FeatureExtraction = torch.nn.Sequential(*list(self.FeatureExtraction.children())[:-2])  # Bỏ FC layer
            self.FeatureExtraction.add_module("conv_reduce", nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False))
        elif opt.FeatureExtraction == 'VGG19':
            self.FeatureExtraction = models.vgg19(pretrained=False)
            self.FeatureExtraction = torch.nn.Sequential(*list(self.FeatureExtraction.children())[:-2])  # Bỏ FC layer
            self.FeatureExtraction.add_module("conv_reduce", nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False))
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet18':
            self.FeatureExtraction = models.resnet18(pretrained=False)
            self.FeatureExtraction = torch.nn.Sequential(*list(self.FeatureExtraction.children())[:-2])  # Bỏ FC layer
        elif opt.FeatureExtraction == 'ResNet34':
            self.FeatureExtraction = models.resnet34(pretrained=False)
            self.FeatureExtraction = torch.nn.Sequential(*list(self.FeatureExtraction.children())[:-2])  # Bỏ FC layer
        elif opt.FeatureExtraction == 'ResNet50':
            self.FeatureExtraction = models.resnet50(pretrained=False)
            self.FeatureExtraction = torch.nn.Sequential(*list(self.FeatureExtraction.children())[:-2])  # Bỏ FC layer
        elif opt.FeatureExtraction == 'ResNet101':
            self.FeatureExtraction = models.resnet101(pretrained=False)
            self.FeatureExtraction = torch.nn.Sequential(*list(self.FeatureExtraction.children())[:-2])  # Bỏ FC layer
        elif opt.FeatureExtraction == 'ResNet152':
            self.FeatureExtraction = models.resnet152(pretrained=True)
            if self.opt.output_channel==512:
                self.FeatureExtraction = torch.nn.Sequential(*list(self.FeatureExtraction.children())[:-2])  # Bỏ FC layer
                self.FeatureExtraction.add_module("conv_reduce", nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False))
        elif opt.FeatureExtraction == 'DenseNet121':
            self.FeatureExtraction = models.densenet121(pretrained=False)
            self.FeatureExtraction = torch.nn.Sequential(*list(self.FeatureExtraction.children())[:-2])  # Bỏ FC layer
            self.FeatureExtraction.add_module("conv_reduce", nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False))
        elif opt.FeatureExtraction == 'DenseNet161':
            self.FeatureExtraction = models.densenet161(pretrained=False)
            self.FeatureExtraction = torch.nn.Sequential(*list(self.FeatureExtraction.children())[:-2])  # Bỏ FC layer
            self.FeatureExtraction.add_module("conv_reduce", nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False))
        elif opt.FeatureExtraction == 'DenseNet169':
            self.FeatureExtraction = models.densenet169(pretrained=False)
            self.FeatureExtraction = torch.nn.Sequential(*list(self.FeatureExtraction.children())[:-2])  # Bỏ FC layer
            self.FeatureExtraction.add_module("conv_reduce", nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False))
        elif opt.FeatureExtraction == 'DenseNet201':
            self.FeatureExtraction = models.densenet201(pretrained=False)
            self.FeatureExtraction = torch.nn.Sequential(*list(self.FeatureExtraction.children())[:-2])  # Bỏ FC layer
            self.FeatureExtraction.add_module("conv_reduce", nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False))

        elif opt.FeatureExtraction == 'vit_base_patch16_224':
            # self.FeatureExtraction = timm.create_model("vit_base_patch16_224",pretrained=False)
            self.FeatureExtraction = timm.create_model("vit_base_patch16_224",pretrained=True,img_size=(int(opt.imgH),int(opt.imgW)))
        elif opt.FeatureExtraction == 'swin_base_patch4_window7_224':
            # from transformers import SwinModel
            # self.FeatureExtraction = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224",img_size=(int(opt.imgH),int(opt.imgW)),num_classes=0)

            self.FeatureExtraction = timm.create_model("swin_base_patch4_window7_224",pretrained=True,img_size=(int(opt.imgH),int(opt.imgW)),num_classes=0)
            # self.FeatureExtraction.gradient_checkpointing_enable()

            # self.FeatureExtraction = timm.create_model("swin_base_patch4_window7_224",pretrained=False,img_size=(int(opt.imgH),int(opt.imgW)),features_only=True)

            # torchvision.models
            # self.FeatureExtraction = models.swin_b(weights=None)  
            # self.FeatureExtraction.head=torch.nn.Linear(in_features=1024, out_features=512)
      
            # self.FeatureExtraction = torch.nn.Sequential(*list(self.FeatureExtraction.children())[:-2])  # Bỏ FC layer
            # self.FeatureExtraction.add_module("conv_reduce", nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False))
            # self.FeatureExtraction = timm.create_model("swin_base_patch4_window7_224",pretrained=False)
        elif opt.FeatureExtraction == 'swin_large_patch4_window12_384':
            # self.FeatureExtraction = timm.create_model("swin_large_patch4_window12_384",pretrained=True,img_size=(int(opt.imgH),int(opt.imgW)),num_classes=0)
            # self.FeatureExtraction = timm.create_model("swin_large_patch4_window12_384",pretrained=True,img_size=(int(opt.imgH),int(opt.imgW)),features_only=True)
            self.FeatureExtraction = timm.create_model("swin_large_patch4_window12_384",pretrained=True,features_only=True,img_size=(int(opt.imgH),int(opt.imgW)))
            self.reduce_dim = nn.Linear(1536, 480)

          
        elif opt.FeatureExtraction == 'maxvit_base_224':
            self.FeatureExtraction = timm.create_model("maxvit_base_224",pretrained=True,img_size=(int(opt.imgH),int(opt.imgW)))

        elif opt.FeatureExtraction == 'convnext_large':
            # self.FeatureExtraction = timm.create_model("convnext_large",pretrained=False)
            self.FeatureExtraction = timm.create_model("convnext_large",pretrained=True)

        elif opt.FeatureExtraction == 'TrOCR':
            self.FeatureExtraction = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                    encoder_pretrained_model_name_or_path="microsoft/trocr-base-stage1",
                    decoder_pretrained_model_name_or_path="microsoft/trocr-base-stage1",
                    encoder_from_pretrained=False,
                    decoder_from_pretrained=False
                )

        elif opt.FeatureExtraction == 'vision_transformer.vit_b_16':
            self.FeatureExtraction = models.vision_transformer.vit_b_16(pretrained=False)



        elif opt.FeatureExtraction == 'mobilenet_v3_large':
            self.FeatureExtraction = models.mobilenet_v3_large(pretrained=False)
            self.FeatureExtraction = torch.nn.Sequential(*list(self.FeatureExtraction.children())[:-2])  # Bỏ FC layer
            self.FeatureExtraction.add_module("conv_reduce", nn.Conv2d(960, 512, kernel_size=1, stride=1, bias=False))
        elif opt.FeatureExtraction == 'mobilenet_v3_small':
            self.FeatureExtraction = models.mobilenet_v3_small(pretrained=False)
            self.FeatureExtraction = self.FeatureExtraction.features  # Output có 960 channels
            self.fc = nn.Conv2d(in_channels=960, out_channels=512, kernel_size=1)

        elif opt.FeatureExtraction == 'efficientnet_b0':
            self.FeatureExtraction = EfficientNet.from_name('efficientnet-b0')  
            self.ConvReduce = nn.Conv2d(1280, 512, kernel_size=1, stride=1, bias=False)
        elif opt.FeatureExtraction == 'efficientnet_b1':
            self.FeatureExtraction = EfficientNet.from_name('efficientnet-b1')  
            self.ConvReduce = nn.Conv2d(1280, 512, kernel_size=1, stride=1, bias=False)
        elif opt.FeatureExtraction == 'efficientnet_b2':
            self.FeatureExtraction = EfficientNet.from_name('efficientnet-b2')  
            self.ConvReduce = nn.Conv2d(1280, 512, kernel_size=1, stride=1, bias=False)
        elif opt.FeatureExtraction == 'efficientnet_b3':
            self.FeatureExtraction = EfficientNet.from_name('efficientnet-b3')  
            self.ConvReduce = nn.Conv2d(1280, 512, kernel_size=1, stride=1, bias=False)
        elif opt.FeatureExtraction == 'efficientnet_b4':
            self.FeatureExtraction = EfficientNet.from_name('efficientnet-b4')  
            self.ConvReduce = nn.Conv2d(1280, 512, kernel_size=1, stride=1, bias=False)
        elif opt.FeatureExtraction == 'efficientnet_b5':
            self.FeatureExtraction = EfficientNet.from_name('efficientnet-b5')  
            self.ConvReduce = nn.Conv2d(1280, 512, kernel_size=1, stride=1, bias=False)
        elif opt.FeatureExtraction == 'efficientnet_b6':
            self.FeatureExtraction = EfficientNet.from_name('efficientnet-b6')  
            self.ConvReduce = nn.Conv2d(1280, 512, kernel_size=1, stride=1, bias=False)
        elif opt.FeatureExtraction == 'efficientnet_b7':
            self.FeatureExtraction = EfficientNet.from_name('efficientnet-b7')  
            self.ConvReduce = nn.Conv2d(2560, opt.output_channel, kernel_size=1, stride=1, bias=False)

            # self.FeatureExtraction = torch.nn.Sequential(*list(self.FeatureExtraction.children())[:-2])  # Bỏ FC layer
            # self.FeatureExtraction.add_module("conv_reduce", nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False))
        elif opt.FeatureExtraction == 'efficientnet_v2_l':
            self.FeatureExtraction = EfficientNet.from_name('efficientnet_v2_l',pretrained=True,img_size=(int(opt.imgH),int(opt.imgW)))  
            self.ConvReduce = nn.Conv2d(1280, 512, kernel_size=1, stride=1, bias=False)

            # self.FeatureExtraction = torch.nn.Sequential(*list(self.FeatureExtraction.children())[:-2])  # Bỏ FC layer
            # self.FeatureExtraction.add_module("conv_reduce", nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False))
        elif opt.FeatureExtraction=="trocr-base-stage1":
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.FeatureExtraction = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        # if self.opt.FeatureExtraction in ['swin_large_patch4_window12_384']:
        #     self.AdaptiveAvgPool=nn.AdaptiveAvgPool2d((1, None))
        # else:
        #     self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(( None,1))  # Transform final (imgH/16-1) -> 1
        
        """ Sequence modeling nn.AdaptiveAvgPool2d((1, None))  """
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True, seqlen=25):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        if self.stages['ViTSTR']:
            prediction = self.vitstr(input, seqlen=seqlen)
            return prediction

        """ Feature extraction stage """
        if str(self.opt.FeatureExtraction).startswith('efficientnet_b'):
            visual_feature = self.FeatureExtraction.extract_features(input)
            visual_feature = self.ConvReduce(visual_feature)
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
            visual_feature = visual_feature.squeeze(3)
        elif self.opt.FeatureExtraction in ['swin_large_patch4_window12_384', 'swin_base_patch4_window7_224']:
            visual_feature = self.FeatureExtraction(input)  # output [B, L, C]
            if isinstance(visual_feature, (list, tuple)):
                visual_feature = visual_feature[-1]  # phòng khi model trả ra nhiều output
            # print(len(visual_feature))
            # for i in range(10):
            #    print(f"visual_feature.squeeze({i}):",visual_feature.squeeze(i).shape)  
            # sys.exit() 
            visual_feature = visual_feature.squeeze(1)
            
            # visual_feature = self.reduce_dim(visual_feature)  # [B, 15, 480]
            # visual_feature = visual_feature.permute(0, 2, 1)  # [B, 480, 15]
            # visual_feature = F.interpolate(visual_feature, size=32, mode='linear', align_corners=True)  # [B, 480, 32]
            # visual_feature = visual_feature.permute(0, 2, 1)  # [B, 32, 480]


        else:
            visual_feature = self.FeatureExtraction(input)
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
            visual_feature = visual_feature.squeeze(3)
        # print("visual_feature :",visual_feature.shape)
        
        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction

class JitModel(Model):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.vitstr= create_vitstr(num_tokens=opt.num_class, model=opt.TransformerModel)

    def forward(self, input, seqlen:int = 25):
        prediction = self.vitstr(input, seqlen=seqlen)
        return prediction


# import torch
# import timm
# imgH, imgW = 32, 480  

# model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, img_size=(imgH, imgW))

# # Tạo input
# x = torch.randn(1, 3, imgH, imgW)

# # Lấy feature map trước khi classifier
# feat_map = model.forward_features(x)

# print("Feature map shape:", feat_map.shape)  # Giữ nguyên 4 chiều

import torch
import torchvision.models as models

# Load mô hình Swin-B mà không có weights
swin_model = models.swin_b(weights=None)

# Kiểm tra số lượng đầu ra của head
print(swin_model.head)  # Mặc định: Linear(in_features=1024, out_features=1000, bias=True)

# Chỉnh lại output của head về 512
swin_model.head = torch.nn.Linear(in_features=1024, out_features=512)


# import torch
# import timm
# imgH, imgW = 32, 480  
# # Tạo model Swin mà KHÔNG có classifier
# model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, img_size=(imgH, imgW), features_only=True)
# # Tạo input tensor
# x = torch.randn(1, 3, imgH, imgW)  # (batch, channels, height, width)

# # Forward để lấy feature map
# features = model(x)
# # print(features.shape)s
# # In kích thước từng feature map
# for i, feat in enumerate(features):
#     print(f"Feature {i} shape:", feat.shape)



# import torch
# import torch.nn as nn
# from efficientnet_pytorch import EfficientNet

# class OCRModel(nn.Module):
#     def __init__(self):
#         super(OCRModel, self).__init__()
#         self.FeatureExtraction = EfficientNet.from_name('efficientnet-b0')  # [B, C, H, W]
#         self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Giữ nguyên width, height về 1

#     def forward(self, x):
#         visual_feature = self.FeatureExtraction.extract_features(x)  # Đảm bảo có 4 chiều
#         visual_feature = self.AdaptiveAvgPool(visual_feature)  # [B, C, W, 1]
#         return visual_feature

# # Test thử
# model = OCRModel()
# dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224
# output = model(dummy_input)
# print(output.shape)  # Kết quả: [B, C, W, 1]

# import timm

# # Load ViT-B/16 không pretrained
# model = timm.create_model("vit_base_patch16_224",pretrained=False)


# import torchvision.models as models
# print(dir(models)) 
# models.resnet152(weights=None) zzzzzzzzzzzzzzzzzzzzzzzzzz
'''
['AlexNet', 'DenseNet', 'GoogLeNet', 'GoogLeNetOutputs', 'Inception3', 'InceptionOutputs', 'MNASNet', 'MobileNetV2', 'MobileNetV3', 'ResNet', 'ShuffleNetV2', 'SqueezeNet', 'VGG', '_GoogLeNetOutputs', '_InceptionOutputs', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', '_utils', 'alexnet', 'densenet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'detection', 'googlenet', 'inception', 'inception_v3', 'mnasnet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'mobilenetv2', 'mobilenetv3', 'quantization', 'resnet', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext50_32x4d', 'segmentation', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'shufflenetv2', 'squeezenet', 'squeezenet1_0', 'squeezenet1_1', 'utils', 'vgg', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'video', 'wide_resnet101_2', 'wide_resnet50_2']
'''
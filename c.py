import os

def xoa_file_theo_duoi(path,duoi):
    path=str(path)
    if os.path.isfile(path):
        if path.endswith(duoi): 
            os.remove(path)
    elif os.path.isdir(path):
        for i in os.listdir(path):
            path_i=os.path.join(path,i)
            xoa_file_theo_duoi(path_i,duoi)

xoa_file_theo_duoi('./lightning_logs','.ckpt')

    def forward(self, input, text, is_train=True, seqlen=25):
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)
        if self.stages['ViTSTR']:
            prediction = self.vitstr(input, seqlen=seqlen)
            return prediction
        if str(self.opt.FeatureExtraction).startswith('efficientnet_b'):
            visual_feature = self.FeatureExtraction.extract_features(input)
            visual_feature = self.ConvReduce(visual_feature)
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
            visual_feature = visual_feature.squeeze(3)
        elif self.opt.FeatureExtraction in ['swin_large_patch4_window12_384',"swin_base_patch4_window7_224"]:
            visual_feature = self.FeatureExtraction(input)[-1]  # [B, C, H, W]
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [B, C, 1, W]    
            if visual_feature.dim() == 4 and visual_feature.shape[2] == 1:
                visual_feature = visual_feature.permute(0, 1, 3, 2)
                visual_feature = visual_feature.squeeze(3).permute(0, 2, 1)
            elif visual_feature.dim() == 3:
                visual_feature = visual_feature.permute(0, 2, 1)
            else:
                raise RuntimeError(f"Unexpected visual_feature shape: {visual_feature.shape}")

        else:
            visual_feature = self.FeatureExtraction(input)
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
            visual_feature = visual_feature.squeeze(3)
        print("visual_feature :",visual_feature.shape)
        
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction

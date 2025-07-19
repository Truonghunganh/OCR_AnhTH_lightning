# from utils import get_args
# from model import Model
# import torch
# from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager, TokenLabelConverter
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# opt = get_args()
# if opt.Transformer:
#     converter = TokenLabelConverter(opt)
# elif 'CTC' in opt.Prediction:
#     if opt.baiduCTC:
#         converter = CTCLabelConverterForBaiduWarpctc(opt.character)
#     else:
#         converter = CTCLabelConverter(opt.character)
# else:
#     converter = AttnLabelConverter(opt.character)
# opt.num_class = len(converter.character)
# opt.select_data = opt.select_data.split('-')
# opt.batch_ratio = opt.batch_ratio.split('-')
# opt.eval = False
    
# model = Model(opt)
# model = torch.nn.DataParallel(model).to(device)
# model.train()
# if opt.saved_model != '':
#     print(f'loading pretrained model from {opt.saved_model}')
#     if opt.FT:
#         model.load_state_dict(torch.load(opt.saved_model), strict=False)
#     else:
#         print(opt)
#         model.load_state_dict(torch.load(opt.saved_model, map_location=device))

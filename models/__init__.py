from .ResNet import *
import torch

def get_model(name, args, from_pretrained=False):
    Model = eval(name)
    model = Model(args.num_classes).to(args.device)
    model_name = f'{name}-{args.dataset}'
    if from_pretrained:
        print(f'{model_name} loading...')
        model.load_state_dict(torch.load(f'saved_models/{model_name}.bin'))
        print(f'{model_name} loaded.')
    return model, model_name

# def transfer_dict(args):
#     model = eval(args.model_name)(num_classes=args.num_classes).to(args.device)
#     model_path = f'saved_models/{model}-{args.dataset}.bin'
#     saved_dict = torch.load(model_path)
#     saved_keys = list(saved_dict.keys())
#     model_dict = model.state_dict()
#     model_keys = list(model_dict.keys())
#     assert len(saved_keys) == len(model_keys), f'len(saved_keys) = {len(saved_keys)}, len(model_keys) = {len(model_keys)}'
#     for saved_key, model_key in zip(saved_keys, model_keys):
#         model_dict[model_key] = saved_keys[saved_key]
#     model.load_state_dict(model_dict)
#     torch.save(model_dict, model_path)
#     return model
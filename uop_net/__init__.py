from module.model.uopnet import UOPNet
import torch


ckp_path = "/SSDe/sangjun_noh/ckp/ral_rebuttal_ckp/o10c15/g9_ep090_model.pth"

model = UOPNet(backbone_type="dgcnn")

model_dict = torch.load(ckp_path)

model.load_state_dict(model_dict["model_state_dict"])

print("1")
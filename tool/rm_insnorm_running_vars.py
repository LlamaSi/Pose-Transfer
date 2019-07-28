import torch

ckp_path = '/home/wenwens/Documents/HumanPose/MMAN/checkpoints/Exp_0/30_net_D2.pth'
save_path = '/home/wenwens/Documents/HumanPose/MMAN/checkpoints/Exp_0/30_net_D2.pth'
states_dict = torch.load(ckp_path)
states_dict_new = states_dict.copy()
for key in states_dict.keys():
	if "running_var" in key or "running_mean" in key:
		del states_dict_new[key]

torch.save(states_dict_new, save_path)
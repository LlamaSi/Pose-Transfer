import torch

ckp_path = '/home/wenwens/Documents/HumanPose/Pose-Transfer/checkpoints/fashion_PATN_true/latest_net_netG.pth'
save_path = '/home/wenwens/Documents/HumanPose/Pose-Transfer/checkpoints/fashion_PATN_true/latest_net_netG.pth'
states_dict = torch.load(ckp_path)
states_dict_new = states_dict.copy()
for key in states_dict.keys():
	if "running_var" in key or "running_mean" in key:
		del states_dict_new[key]

torch.save(states_dict_new, save_path)
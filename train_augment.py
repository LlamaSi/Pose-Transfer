import time
from options.augment_options import AugmentOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import pdb

opt = AugmentOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

aug_model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        aug_model.forward_aug(data)
        aug_model.forward_target(data)

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(aug_model.get_current_visuals(), epoch, save_result)
            print(aug_model.skeleton_net.alpha_m, aug_model.skeleton_net.alpha_v)
            # print(aug_model.skeleton_net.alpha_m.grad, aug_model.skeleton_net.alpha_v.grad)
            # for name, param in aug_model.main_model.named_parameters():
            #     # print(name) netD_PB.model.1.bias netD_PP.model.1.bias
            #     if 'netD_PB.model.1.bias' in name:

            #         print(param)
        if total_steps % opt.print_freq == 0:
            errors = aug_model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            aug_model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        aug_model.save('latest')
        aug_model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    aug_model.update_learning_rate()

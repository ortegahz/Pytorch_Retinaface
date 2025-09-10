from __future__ import print_function

import argparse
import datetime
import os
import time
from collections import OrderedDict

# Add imports for onnx simplification
try:
    import onnx
    import onnxsim
except ImportError:
    print("`onnx` and/or `onnx-simplifier` not found. ONNX export simplification will be skipped.")
    print("Install them with `pip install onnx onnx-simplifier`")
    onnx = None
    onnxsim = None

import math
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data

from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from layers.modules import MultiBoxLoss
from models.retinaface import RetinaFace

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='/home/Huangzhe/test/retinaface/train/label.txt',
                    help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
parser.add_argument('--save_onnx_per_epoch', default=True, help='Export an ONNX model after each epoch')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123)  # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)
print("Printing net...")
print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()


def export_to_onnx(model_state_dict, cfg, save_path):
    """
    Exports a trained model state dictionary to ONNX format and simplifies it.
    """
    print(f"Converting model to ONNX format -> {save_path}...")

    # 1. Instantiate the model in 'test' phase for export
    net_for_export = RetinaFace(cfg=cfg, phase='test')

    # 2. Clean the state dict (remove 'module.' prefix from DataParallel)
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    net_for_export.load_state_dict(new_state_dict)

    # 3. Set to eval mode and move to CPU for export
    net_for_export.eval()
    net_for_export.to('cpu')

    # 4. Create a dummy input tensor with the correct size
    dummy_input = torch.randn(1, 3, cfg['image_size'], cfg['image_size'], device='cpu')

    try:
        # 5. Export the model
        torch.onnx.export(net_for_export,
                          dummy_input,
                          save_path,
                          verbose=False,
                          input_names=['input'],
                          output_names=['boxes', 'scores', 'landmarks'],
                          opset_version=11)
        print("ONNX model export complete.")

        # 6. Simplify the exported ONNX model
        if onnx is not None and onnxsim is not None:
            print(f"Simplifying ONNX model -> {save_path}...")
            onnx_model = onnx.load(save_path)
            model_simplified, check = onnxsim.simplify(onnx_model)
            assert check, "Simplified ONNX model could not be validated"
            onnx.save(model_simplified, save_path)
            print("ONNX model simplification complete.")
        else:
            print("Skipping ONNX simplification because required libraries are not installed.")

    except Exception as e:
        print(f"An error occurred during ONNX export or simplification: {e}")


def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers,
                                                  collate_fn=detection_collate))

            # Save model and export ONNX at the end of each epoch (except the first one)
            if epoch >= 0:
                # Save PyTorch checkpoint
                pth_save_path = save_folder + cfg['name'] + '_epoch_' + str(epoch) + '.pth'
                torch.save(net.state_dict(), pth_save_path)
                print(f"Saved checkpoint: {pth_save_path}")

                # Conditionally export to ONNX
                if args.save_onnx_per_epoch:
                    onnx_save_path = save_folder + cfg['name'] + '_epoch_' + str(epoch) + '.onnx'
                    export_to_onnx(net.state_dict(), cfg, onnx_save_path)

            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print(
            'Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
            .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                    epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr,
                    batch_time, str(datetime.timedelta(seconds=eta))))

    # Save final model
    final_pth_path = save_folder + cfg['name'] + '_Final.pth'
    torch.save(net.state_dict(), final_pth_path)
    print(f"Saved final checkpoint: {final_pth_path}")

    # Conditionally export final model to ONNX
    if args.save_onnx_per_epoch:
        final_onnx_path = save_folder + cfg['name'] + '_Final.onnx'
        export_to_onnx(net.state_dict(), cfg, final_onnx_path)


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr - 1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()

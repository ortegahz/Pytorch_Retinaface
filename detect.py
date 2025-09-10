from __future__ import print_function

import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # resize value is no longer used, but kept for consistency
    resize = 1

    # testing begin
    for i in range(1):
        image_path = "/home/Huangzhe/test/manu-pc/tmp/padded_test.bmp"
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # ----------------- MODIFICATION START: Image padding and resizing -----------------
        # Original image dimensions
        orig_h, orig_w, _ = img_raw.shape

        # Target size
        target_size = 640

        # Calculate scaling factor and new size
        scale_ratio = target_size / max(orig_h, orig_w)
        new_w = int(orig_w * scale_ratio)
        new_h = int(orig_h * scale_ratio)

        # Resize image
        resized_img = cv2.resize(img_raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create a black canvas and paste the resized image
        padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        padded_img[0:new_h, 0:new_w] = resized_img

        # Define save directory
        save_dir = "/home/Huangzhe/test/manu-pc/tmp/"
        # Make sure directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save the padded image as a BMP file
        padded_bmp_path = os.path.join(save_dir, "padded_test.bmp")
        cv2.imwrite(padded_bmp_path, padded_img)
        print(f"Padded image saved to {padded_bmp_path}")

        # The network will now process the padded image
        img = np.float32(padded_img)

        # Prepare image for network
        im_height, im_width, _ = img.shape  # Now im_height and im_width are 640
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        # ----------------- MODIFICATION END ------------------------------------------

        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        # show image
        if args.save_image:
            # ----------------- MODIFICATION START -----------------
            # Define save path for image and txt, and open txt file for writing
            save_img_path = os.path.join(save_dir, "test.jpg")
            save_txt_path = os.path.splitext(save_img_path)[0] + ".txt"
            f_txt = open(save_txt_path, 'w')
            # ----------------- MODIFICATION END -------------------

            for b in dets:
                if b[4] < args.vis_thres:
                    continue

                # ----------------- MODIFICATION START: Rescale coordinates -----------------
                # Rescale coordinates from padded 640x640 space to original image space
                b_rescaled = b.copy()
                b_rescaled[0:4] /= scale_ratio  # Rescale bbox
                b_rescaled[5:] /= scale_ratio  # Rescale landmarks

                # Save rescaled detection to txt file
                line = f"{int(b_rescaled[0])} {int(b_rescaled[1])} {int(b_rescaled[2])} {int(b_rescaled[3])} {b_rescaled[4]:.5f} {int(b_rescaled[5])} {int(b_rescaled[6])} {int(b_rescaled[7])} {int(b_rescaled[8])} {int(b_rescaled[9])} {int(b_rescaled[10])} {int(b_rescaled[11])} {int(b_rescaled[12])} {int(b_rescaled[13])} {int(b_rescaled[14])}\n"
                f_txt.write(line)

                # Use rescaled coordinates for drawing on the *original* raw image
                text = "{:.4f}".format(b_rescaled[4])
                b = list(map(int, b_rescaled))
                # ----------------- MODIFICATION END --------------------------------------

                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

            # ----------------- MODIFICATION START -----------------
            f_txt.close()
            # ----------------- MODIFICATION END -------------------

            # save image
            cv2.imwrite(save_img_path, img_raw)

import os
import torch
import torch.nn as nn
import argparse
import yaml
from torch.backends import cudnn
from loader import get_loader
from solver import Solver
# import T2T_vit_model

from model.Ynet.Ynet import YNet
from model.Ynet.losses import CharbonnierLoss
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'  ## 1/2 ,multi GPU

def main(args, model):
    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:
        fig_path = os.path.join(args.save_path, 'fig')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))

    data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             saved_path=args.saved_path,
                             test_patient=args.test_patient,
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=(args.patch_size if args.mode=='train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             shuffle=(True if args.mode=='train' else False),
                             num_workers=args.num_workers)

    val_data_loader = get_loader(mode='test',
                             load_mode=args.load_mode,
                             saved_path=args.saved_path,
                             test_patient=args.test_patient,
                             patch_n=None,
                             patch_size=None,
                             transform=args.transform,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers)

    solver = Solver(args, data_loader, val_data_loader, model)
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_epoch', type=int, default=47)
    parser.add_argument('--val_epoch', type=int, default=40)


    parser.add_argument('--mode', type=str, default='train')#train, test
    parser.add_argument('--load_mode', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='./AAPM-Mayo-CT-Challenge/')#AAPM-Mayo-CT-Challenge #QIN LUNG CT
    parser.add_argument('--saved_path', type=str, default='./npy_img/')   ##aapm_all_npy_1mm  #qin_lung_npy15
    parser.add_argument('--save_path', type=str, default='save/testable_version')
    parser.add_argument('--test_patient', type=str, default='L506') #R0274
    parser.add_argument('--result_fig', type=bool, default=True)

    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)

    parser.add_argument('--transform', type=bool, default=False)
    # if patch training, batch size is (--patch_n * --batch_size)
    parser.add_argument('--patch_n', type=int, default=4)   ## 10
    parser.add_argument('--patch_size', type=int, default=64)    ## 64
    parser.add_argument('--batch_size', type=int, default=8)   ## batch size has to be very small if size=512,16

    parser.add_argument('--num_epochs', type=int, default=50)  ## 200 or 2000
    parser.add_argument('--print_iters', type=int, default=50)

    parser.add_argument('--decay_iters', type=int, default=12000)  ## original 3000 then 8000
    parser.add_argument('--save_iters', type=int, default=481)  ## the iterats~epochs*10 useless for now
    parser.add_argument('--test_iters', type=int, default=67650)

    parser.add_argument('--lr', type=float, default=4e-4)

    parser.add_argument('--device', type=str)  ## default=[0,1]tam
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--multi_gpu', type=bool, default=True) ## 2/2 ,multi GPU

    args = parser.parse_args()

    model = YNet()

    # model = nn.DataParallel(model, device_ids=[0, 1])
    # device = torch.device("cuda:0")
    # model = model.to(device)
    # input_data = input_data.to('cuda:0')  # 将输入数据移动到 cuda:0 设备


    main(args, model)

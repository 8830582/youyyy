#测试模型的dice系数
import torch
import torchvision
import torch.nn.functional as F
import argparse
import random
import os
import numpy as np
from utils.utils import dice_coefficient,calculate_iou_per_class
from utils.utils import test_single_volume1
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis
from utils.dataset_CHAOS_new import get_data_loader
from utils.dataset_synapse import Synapse_dataset, RandomGenerator
from torch.utils.data import DataLoader
# from lib.some_models import Unet,UnetPlusPlus
# from lib.vmunet import VMUNet
# from lib.bisenetv1 import BiSeNetV1
# from lib.bisenetv2 import BiSeNetV2
# from lib.UNet_MobileNet import mobile_UNet
from lib.erfnet import Net as erfnet
from tqdm import tqdm
import matplotlib.colors as mcolors
from torchsummary import summary
from segmentation_mask_overlay import overlay_masks
# import torchprofile
import matplotlib.pyplot as plt
# from lib.Segnet import Segnet
from lib.segformer import SegFormer
parser = argparse.ArgumentParser(description='dice_test')
parser.add_argument('strings', metavar='STRING', nargs='*', help='String for searching',)
parser.add_argument('--root_path', type=str,
                    default=r'/root/autodl-tmp/CHAOS_Train/Train_Sets/MR/', help='root dir for data')
#chaos: '/root/autodl-tmp/CHAOS_Train/Train_Sets/MR/'
#Synapse:root_path:r'/root/autodl-tmp/train_npz'
#Synapse:volume_path: r'/root/autodl-tmp/test_vol_h5'
parser.add_argument('--volume_path', type=str,
                    default=r'/root/autodl-tmp/test_vol_h5', help='root dir for validation volume data')

#这里修改数据集
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=5, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=64, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate') #0.001
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input') #224
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')
# parser.add_argument('--pretrain_weights_path', type=str,
#                     default='', help='model_dict_pretrain')
#/root/MIST-total/model_pth/chaos/Chaos_1111_mamba/best.pth
parser.add_argument('--pretrain_weights_path', type=str,default='/root/MIST-total/model_pth/Synapse/Synapse_segformer_b2/best.pth', help='model_dict_pretrain')

#'/root/MIST-total/model_pth/Synapse/Synapse_m_w_unet/best.pth'
#choas_mamba_wave_unet/best.pth

# unet:'/root/MIST-total/model_pth/chaos/chaos_Unet/best.pth'
args = parser.parse_args("AAA".split())

def save_image(images,gts,preds,classes,test_save_path,ind):
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    for i in range(images.shape[0]):
        # print('i:',i)
        mask_labels = np.arange(1, classes)
        image = images[i,:,:].squeeze(0).cpu().detach().numpy()
        lbl = gts[i,:,:].squeeze(0).cpu().detach().numpy()
        pred = torch.argmax(torch.softmax(preds, dim=1), dim=1).squeeze(0)
        pred = pred[i,:,:].cpu().detach().numpy()
        # print('image,pred,lbl',image.shape,pred.shape,lbl.shape)
        my_colors = ['red', 'darkorange', 'yellow', 'forestgreen', 'blue', 'purple', 'magenta', 'cyan', 'deeppink','chocolate', 'olive', 'deepskyblue', 'darkviolet']
        cmaps = mcolors.CSS4_COLORS
        cmap = {k: cmaps[k] for k in sorted(cmaps.keys()) if k in my_colors[:classes - 1]}
        masks = []
        for t in range(1, classes):
            masks.append(lbl == t)
        preds_o = []
        for t in range(1, classes):
            preds_o.append(pred == t)
        if test_save_path is not None:
            fig_gt = overlay_masks(image, np.array(masks).transpose(1, 2, 0), labels=mask_labels, colors=cmap,return_type="mpl")
            fig_pred = overlay_masks(image, np.array(preds_o).transpose(1, 2, 0), labels=mask_labels, colors=cmap,return_type="mpl")
            fig_gt.savefig(test_save_path + '/'  + '_' + str(ind)+'batch'+str(i) + '_gt.png', bbox_inches="tight", dpi=60)
            fig_pred.savefig(test_save_path + '/' + '_' + str(ind)+'batch'+str(i)+ '_pred.png', bbox_inches="tight",dpi=60)
            plt.close(fig_gt)
            plt.close(fig_pred)

if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #model
    # mobile_net = BiSeNetV1(n_classes = 5, aux_mode='train').cuda()
    # mobile_net = BiSeNetV2(n_classes=5,aux_mode='train').cuda()
    # mobile_net = Segnet(1,5).cuda()
    # mobile_net = Unet(in_channels=1,out_channels=5).cuda()
    # mobile_net = VMUNet(input_channels = 3,num_classes=5,depths=[1,1,1,1], depths_decoder=[1,1,1,1],drop_path_rate=0.2,load_ckpt_path ='/root/MIST-total/model_pth/mamba_pretrain/vmamba_small_e238_ema.pth' ).cuda()
    # mobile_net = mobile_UNet(n_channels=1, num_classes=5).cuda()
    # mobile_net=erfnet(num_classes=5,).cuda()
    mobile_net = SegFormer(num_classes = 5,phi = 'b2', pretrained = False).cuda()
#------------------------information------------------------------
    # summary(mobile_net, input_size=(1, 256, 256))
    # input_tensor = torch.randn(1, 1, 256, 256)
    # flops = FlopCountAnalysis(mobile_net.to('cpu'), input_tensor)
    # print(f"FLOPs: {flops.total()}")
    
    #load_pth
    mobile_net.load_state_dict(torch.load(args.pretrain_weights_path), strict=True)
    print("{} is loaded".format(args.pretrain_weights_path))
    # mobile_net.aux_mode = 'eval'
    # Computational complexity and Number of parameters
    # macs, params = get_model_complexity_info(mobile_net, (1, args.img_size, args.img_size), as_strings=True,print_per_layer_stat=False, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
#------------------------val------------------------------
#chaos dataset
    if args.dataset =='Chaos':
        test_save_path = './Chaos_image'
        testloader,length = get_data_loader(path=args.root_path,batch_size=args.batch_size,train_tag='val',shuffle=False,img_size = args.img_size,num_workers=0,num_limit=500)
        total_dice_scores = [0.0,0.0,0.0,0.0]
        # total_iou_scores =  [0.0,0.0,0.0,0.0]
        total_iou_scores = []
        mobile_net.eval()
        ind = 0
        with torch.no_grad():
            for i_batch, (image_batch, label_batch) in tqdm(enumerate(testloader)):
                
                x, y = image_batch.to('cuda'), label_batch.to('cuda')
                
                y_pred = mobile_net(x)
                # print('x, y:',x.shape, y.shape,y_pred.shape)
                save_image(images=x,gts=y,preds=y_pred,classes=args.num_classes,test_save_path=test_save_path,ind=ind)
                ind+=1
                dice_scores=dice_coefficient(y_pred,y,args.num_classes)
                # iou_scores = calculate_iou_per_class(y_pred,y,args.num_classes)
                # print(iou_scores)
                # print('dice_scores:',dice_scores)

                total_dice_scores = [h + q for h, q in zip(total_dice_scores, dice_scores)]
                
                y_pred = torch.argmax(y_pred, dim=1)
                intersection = torch.logical_and(y, y_pred)
                union = torch.logical_or(y, y_pred)
                batch_iou = torch.sum(intersection) / torch.sum(union)
                total_iou_scores.append(batch_iou.item())
            miou = round(np.mean(total_iou_scores), 3)
            
                # total_iou_scores = [t + z for t, z in zip(total_iou_scores, iou_scores)]
#dice                
            mean_dice_class = [x / len(testloader) for x in total_dice_scores]
            mean_dice = sum(mean_dice_class) / len(mean_dice_class)
            result_dice = [round(x * 100, 1) for x in mean_dice_class]
            print('mean_dice_class:',result_dice)
            print('mean_dice:',round(mean_dice* 100, 1))
#iou
            # mean_iou_class = [x / len(testloader) for x in total_iou_scores]
            # mean_iou = sum(mean_iou_class) / len(mean_iou_class)
            # result_iou = [round(x * 100, 1) for x in mean_iou_class]
            # print('mean_iou_class:',result_iou)
            print('mean_iou:',miou)
            # print('mean_iou:',round(mean_iou* 100, 1))



    #————————以下功能已完整
    elif args.dataset =='Synapse':
        db_test = Synapse_dataset(base_dir=args.volume_path, split="val_vol", list_dir=args.list_dir,nclass=args.num_classes)
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
        mobile_net.eval()
        metric_list = 0.0
        with torch.no_grad():
            for i_batch, sampled_batch in tqdm(enumerate(testloader)):
                h, w = sampled_batch["image"].size()[2:]
                image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
                metric_i = test_single_volume1(image, label, mobile_net, classes=args.num_classes,
                                             patch_size=[args.img_size, args.img_size],test_save_path='./Synapse_image/',
                                             case=case_name, z_spacing=1,test_save = True)
                # print('metric_i:',metric_i)
                metric_list += np.array(metric_i)
        # print('metric_list',metric_list)
        metric_list = metric_list / len(db_test)
        # print(metric_list.shape)
        avg_dice = np.mean(metric_list, axis=0)

        # print(len(db_test))
        print('mean_class:', np.round(metric_list * 100, 1))
        print('mean:',np.round(avg_dice* 100, 1))


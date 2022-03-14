# supplementary DNN modules

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from utils.plots import feature_visualization, visualize_one_channel
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import itertools


################## Auto Encoder ######################

class Encoder(nn.Module):
    def __init__(self, chs, k=1, s=1, p=None):
        super().__init__()
        self.conv1 = nn.Conv2d(chs[0], chs[1], k, s, autopad(k, p), bias=False)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(chs[1], chs[2], k, s, autopad(k, p), bias=False)
        # self.act2 = nn.Sigmoid()
        self.act2 = nn.SiLU()
        

    def forward(self, x):
        return self.act2(self.conv2(self.act1(self.conv1(x))))
        # return self.act1(self.conv1(x))


class Decoder(nn.Module):
    def __init__(self, chs, k=1, s=1, p=None):
        super().__init__()
        self.conv1 = nn.Conv2d(chs[2], chs[1], k, s, autopad(k, p), bias=False)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(chs[1], chs[0], k, s, autopad(k, p), bias=False)
        self.act2 = nn.SiLU()

    def forward(self, x):
        return self.act2(self.conv2(self.act1(self.conv1(x))))
        # return self.act2(self.conv2(x))


class AutoEncoder(nn.Module):
    # def __init__(self, cin, cmid):
    def __init__(self, chs):
        super().__init__()
        # print(chs)
        self.enc = Encoder(chs, k=3)
        self.dec = Decoder(chs, k=3)

    def forward(self, x, visualize=False, task='enc_dec', bottleneck=None, s=''):
        if task=='dec':
            x = bottleneck
        else:
            x = self.enc(x)

        if visualize:
            visualize_one_channel(x, n=8, save_dir=visualize, s=s)
            feature_visualization(x, 'encoder', '', save_dir=visualize, cut_model=s)
            
        if task=='enc':
            return x
        else:
            return self.dec(x)


################## Motion Estimation ######################
class MotionEstimation(nn.Module):
    def __init__(self, in_channels = 2):
        super(MotionEstimation, self).__init__()

        def EstimateOffsets(numInputCh, numMidCh, numOutCh):
            return nn.Sequential(
                nn.Conv2d(in_channels=numInputCh,  out_channels=numMidCh, kernel_size=5, stride=1, padding=2),
                nn.Conv2d(in_channels=numMidCh,  out_channels=numOutCh , kernel_size=5, stride=1, padding=2),
            )

        def ConvBasic(numInputCh, numMidCh, numOutCh):
            return nn.Sequential(
                nn.Conv2d(in_channels=numInputCh,  out_channels=numMidCh, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=numMidCh,  out_channels=numOutCh , kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
            )

        def Upsampling(intInput, kSize = 3):
            padding = (kSize - 1)//2
            return nn.Sequential(
                nn.Conv2d(in_channels=intInput, out_channels=intInput, kernel_size=kSize, stride=1, padding=padding),
                nn.SiLU(),
            )
        # end

        # downward path
        self.EstOff1_d    = EstimateOffsets(numInputCh=in_channels*2, numMidCh=in_channels*9, numOutCh=in_channels*9*2)
        for v in self.EstOff1_d.modules():
            if isinstance(v, nn.Conv2d) :
                init.constant_(v.weight, 0.)
        self.MoCmpns1_d   = torchvision.ops.DeformConv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.Conv1_d      = ConvBasic(numInputCh=in_channels, numMidCh=in_channels*2, numOutCh=in_channels*4)

        self.Pool1        = nn.AvgPool2d(kernel_size=2, stride=2)

        self.EstOff2_d    = EstimateOffsets(numInputCh=in_channels*4, numMidCh=in_channels*9, numOutCh=in_channels*9*2)
        for v in self.EstOff2_d.modules():
            if isinstance(v, nn.Conv2d) :
                init.constant_(v.weight, 0.)
        self.MoCmpns2_d   = torchvision.ops.DeformConv2d(in_channels=in_channels*4, out_channels=in_channels*4, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.Conv2_d      = ConvBasic(numInputCh=in_channels*4, numMidCh=in_channels*6, numOutCh=in_channels*8)

        self.Pool2        = nn.AvgPool2d(kernel_size=2, stride=2)

        # bottom
        self.EstOff3_d    = EstimateOffsets(numInputCh=in_channels*8, numMidCh=in_channels*9, numOutCh=in_channels*9*2)
        for v in self.EstOff3_d.modules():
            if isinstance(v, nn.Conv2d) :
                init.constant_(v.weight, 0.)
        self.MoCmpns3_d   = torchvision.ops.DeformConv2d(in_channels=in_channels*8, out_channels=in_channels*8, kernel_size=3, padding=1, groups=in_channels, bias=False)
        # self.Conv3_d      = ConvBasic(numInputCh=in_channels*8, numMidCh=in_channels*10, numOutCh=in_channels*8)
        self.Conv3_d      = ConvBasic(numInputCh=in_channels*8, numMidCh=in_channels*7, numOutCh=in_channels*6)

        # upward path
        # self.Upsample2    = Upsampling(intInput=in_channels*8, kSize=3)
        self.Upsample2    = Upsampling(intInput=in_channels*6, kSize=3)

        # self.EstOff2_u    = EstimateOffsets(numInputCh=in_channels*8, numMidCh=in_channels*9, numOutCh=in_channels*9*2)
        # for v in self.EstOff2_u.modules():
        #     if isinstance(v, nn.Conv2d) :
        #         init.constant_(v.weight, 0.)
        # self.MoCmpns2_u   = torchvision.ops.DeformConv2d(in_channels=in_channels*8, out_channels=in_channels*8, kernel_size=3, padding=1, groups=in_channels, bias=False)
        # self.Conv2_u      = ConvBasic(numInputCh=in_channels*8, numMidCh=in_channels*6, numOutCh=in_channels*4)
        self.Conv2_u      = ConvBasic(numInputCh=in_channels*14, numMidCh=in_channels*8, numOutCh=in_channels*4)

        self.Upsample1    = Upsampling(intInput=in_channels*4, kSize=3)

        # self.EstOff1_u    = EstimateOffsets(numInputCh=in_channels*4, numMidCh=in_channels*9, numOutCh=in_channels*9*2)
        # for v in self.EstOff1_u.modules():
        #     if isinstance(v, nn.Conv2d) :
        #         init.constant_(v.weight, 0.)
        # self.MoCmpns1_u   = torchvision.ops.DeformConv2d(in_channels=in_channels*4, out_channels=in_channels*4, kernel_size=3, padding=1, groups=in_channels, bias=False)
        # self.Conv1_u      = ConvBasic(numInputCh=in_channels*4, numMidCh=in_channels*2, numOutCh=in_channels)
        self.Conv1_u      = ConvBasic(numInputCh=in_channels*8, numMidCh=in_channels*4, numOutCh=in_channels)

#   def forward(self, x1, x2):
    def forward(self, x):
        # x = torch.cat((x1, x2), 1)
        # print(x.shape)
        off1_d      = self.EstOff1_d(x)
        compns1_d   = self.MoCmpns1_d(input=x, offset=off1_d)
        conv1_d     = self.Conv1_d(compns1_d)

        pool1       = self.Pool1(conv1_d)

        off2_d      = self.EstOff2_d(pool1)
        compns2_d   = self.MoCmpns2_d(input=pool1, offset=off2_d)
        conv2_d     = self.Conv2_d(compns2_d)

        pool2       = self.Pool2(conv2_d)


        off3_d      = self.EstOff3_d(pool2)
        compns3_d   = self.MoCmpns3_d(input=pool2, offset=off3_d)
        conv3_d     = self.Conv3_d(compns3_d)


        usmpl2      = self.Upsample2(F.interpolate(conv3_d, scale_factor=2.0, mode='bilinear', align_corners=True))

        u2_in       = torch.cat((usmpl2, conv2_d), 1)
        conv2_u     = self.Conv2_u(u2_in)
        # off2_u      = self.EstOff2_u(u2_in)
        # compns2_u   = self.MoCmpns2_u(input=u2_in, offset=off2_u)
        # off2_u      = self.EstOff2_u(usmpl2+conv2_d)
        # compns2_u   = self.MoCmpns2_u(input=usmpl2+conv2_d, offset=off2_u)
        # conv2_u     = self.Conv2_u(compns2_u)
        

        usmpl1      = self.Upsample1(F.interpolate(conv2_u, scale_factor=2.0, mode='bilinear', align_corners=True))

        u1_in       = torch.cat((usmpl1, conv1_d), 1)
        conv1_u     = self.Conv1_u(u1_in)
        # off1_u      = self.EstOff1_u(u1_in)
        # compns1_u   = self.MoCmpns1_u(input=u1_in, offset=off1_u)
        # off1_u      = self.EstOff1_u(usmpl1+conv1_d)
        # compns1_u   = self.MoCmpns1_u(input=usmpl1+conv1_d, offset=off1_u)
        # conv1_u     = self.Conv1_u(compns1_u)

        return conv1_u


################## Motion Compensation ######################
class InterPrediction(nn.Module):
    def __init__(self, in_channels=2, G=1):     # number of input channels, number of Groups in doform_conv
        super(InterPrediction, self).__init__()

        def Conv2x(numInputCh, numMidCh, numOutCh, k=3):
            return nn.Sequential(
                nn.Conv2d(in_channels=numInputCh,  out_channels=numMidCh, kernel_size=k, stride=1, padding=k//2),
                nn.Conv2d(in_channels=numMidCh,  out_channels=numOutCh , kernel_size=k, stride=1, padding=k//2),
            )
        def Conv1x(numInputCh, numOutCh, k=3):
            return nn.Conv2d(in_channels=numInputCh,  out_channels=numOutCh, kernel_size=k, stride=1, padding=k//2)

        def ConvStandard(numInputCh, numOutCh, k=3):
            return nn.Sequential(
                nn.Conv2d(in_channels=numInputCh,  out_channels=numOutCh, kernel_size=k, stride=1, padding=k//2),
                nn.BatchNorm2d(numOutCh),
                nn.SiLU(),
            )

        def ResBlock(ch, k=3):
            return nn.Sequential(
                nn.Conv2d(in_channels=ch,  out_channels=ch, kernel_size=k, stride=1, padding=k//2),
                nn.SiLU(),
                nn.Conv2d(in_channels=ch,  out_channels=ch, kernel_size=k, stride=1, padding=k//2),
            )

        def UpSampling(numInputCh, k=3):
            return nn.Sequential(
                nn.Conv2d(in_channels=numInputCh, out_channels=numInputCh, kernel_size=k, stride=1, padding=k//2),
                nn.SiLU(),
            )

        c = in_channels
        #--- Motion Estimation ---#
        # Layer1
        self.GetMasterMotion1 = ConvStandard(numInputCh=2*c, numOutCh=2*c)
        self.GetMasterMotion_Layer1 = Conv2x(numInputCh=2*c, numMidCh=2*c, numOutCh=2*c)
        self.GetMotion_Layer1_1 = Conv1x(numInputCh=2*c, numOutCh=2*9*G)
        self.GetMotion_Layer1_2 = Conv1x(numInputCh=2*c, numOutCh=2*9*G)
        # Layer2
        self.Motion_DownSample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.GetMasterMotion2 = ConvStandard(numInputCh=2*c, numOutCh=4*c)
        self.GetMasterMotion_Layer2 = Conv2x(numInputCh=4*c, numMidCh=4*c, numOutCh=4*c)
        self.GetMotion_Layer2_1 = Conv1x(numInputCh=4*c, numOutCh=2*9*(2*G))
        self.GetMotion_Layer2_2 = Conv1x(numInputCh=4*c, numOutCh=2*9*(2*G))
        me_modules = itertools.chain(self.GetMasterMotion1.modules(), self.GetMasterMotion_Layer1.modules(), self.GetMotion_Layer1_1.modules(), self.GetMotion_Layer1_2.modules(),\
                                     self.GetMasterMotion2.modules(), self.GetMasterMotion_Layer2.modules(), self.GetMotion_Layer2_1.modules(), self.GetMotion_Layer2_2.modules())
        for v in me_modules:
            if isinstance(v, nn.Conv2d) :
                init.zeros_(v.weight)
                # init.constant_(v.bias, 0.1)

        #--- Motion Compensation ---#
        # Layer1 (down-scale)
        self.DeformConv_Layer1_1 = torchvision.ops.DeformConv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, groups=G, bias=False)
        self.DeformConv_Layer1_2 = torchvision.ops.DeformConv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, groups=G, bias=False)
        self.Conv_Layer1_1 = ConvStandard(numInputCh=c, numOutCh=2*c)
        self.Conv_Layer1_2 = ConvStandard(numInputCh=c, numOutCh=2*c)
        self.DownSample_Layer1_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.DownSample_Layer1_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Layer2 (down-scale)
        self.DeformConv_Layer2_1 = torchvision.ops.DeformConv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, padding=1, groups=G, bias=False)
        self.DeformConv_Layer2_2 = torchvision.ops.DeformConv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, padding=1, groups=G, bias=False)
        self.Conv_Layer2_1 = ConvStandard(numInputCh=2*c, numOutCh=4*c)
        self.Conv_Layer2_2 = ConvStandard(numInputCh=2*c, numOutCh=4*c)
        self.DownSample_Layer2_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.DownSample_Layer2_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Layer3 (up-scale)
        self.Conv_Layer3_first_1 = ConvStandard(numInputCh=4*c, numOutCh=4*c)
        self.Conv_Layer3_first_2 = ConvStandard(numInputCh=4*c, numOutCh=4*c)
        self.Conv_Layer3_second_1 = ResBlock(4*c)
        self.Conv_Layer3_second_2 = ResBlock(4*c)
        self.Conv_Layer3_third_1 = ConvStandard(numInputCh=4*c, numOutCh=2*c)
        self.Conv_Layer3_third_2 = ConvStandard(numInputCh=4*c, numOutCh=2*c)
        self.UpSample_Layer3_1 = UpSampling(numInputCh=2*c)
        self.UpSample_Layer3_2 = UpSampling(numInputCh=2*c)
        # Layer4 (up-scale)
        self.Conv_Layer4_first_1 = ConvStandard(numInputCh=4*c, numOutCh=2*c)
        self.Conv_Layer4_first_2 = ConvStandard(numInputCh=4*c, numOutCh=2*c)
        self.Conv_Layer4_second_1 = ResBlock(2*c)
        self.Conv_Layer4_second_2 = ResBlock(2*c)
        self.Conv_Layer4_third_1 = ConvStandard(numInputCh=2*c, numOutCh=c)
        self.Conv_Layer4_third_2 = ConvStandard(numInputCh=2*c, numOutCh=c)
        self.UpSample_Layer4_1 = UpSampling(numInputCh=c)
        self.UpSample_Layer4_2 = UpSampling(numInputCh=c)

        #--- Fusion and Refinement ---#
        self.Fusion = Conv2x(numInputCh=4*c, numMidCh=3*c, numOutCh=2*c)
        self.Refine = ConvStandard(numInputCh=2*c, numOutCh=c, k=1)


    def forward(self, x1, x2):
        # motion estimation
        master_motion1          = self.GetMasterMotion1(torch.cat((x1, x2), 1))
        master_motion_layer1    = self.GetMasterMotion_Layer1(master_motion1)
        motion_layer1_1         = self.GetMotion_Layer1_1(master_motion_layer1 + master_motion1)
        motion_layer1_2         = self.GetMotion_Layer1_2(master_motion_layer1 + master_motion1)

        master_motion2          = self.GetMasterMotion2(self.Motion_DownSample(master_motion_layer1 + master_motion1))
        master_motion_layer2    = self.GetMasterMotion_Layer2(master_motion2)
        motion_layer2_1         = self.GetMotion_Layer2_1(master_motion_layer2 + master_motion2)
        motion_layer2_2         = self.GetMotion_Layer2_2(master_motion_layer2 + master_motion2)

        # motion compensation
        deform_layer1_1         = self.DeformConv_Layer1_1(x1, offset=motion_layer1_1)
        deform_layer1_2         = self.DeformConv_Layer1_2(x2, offset=motion_layer1_2)
        conv_layer1_1           = self.Conv_Layer1_1(deform_layer1_1)
        conv_layer1_2           = self.Conv_Layer1_2(deform_layer1_2)
        conv_layer1_1           = self.DownSample_Layer1_1(conv_layer1_1)
        conv_layer1_2           = self.DownSample_Layer1_2(conv_layer1_2)

        deform_layer2_1         = self.DeformConv_Layer2_1(conv_layer1_1, offset=motion_layer2_1)
        deform_layer2_2         = self.DeformConv_Layer2_2(conv_layer1_2, offset=motion_layer2_2)
        conv_layer2_1           = self.Conv_Layer2_1(deform_layer2_1)
        conv_layer2_2           = self.Conv_Layer2_2(deform_layer2_2)
        conv_layer2_1           = self.DownSample_Layer2_1(conv_layer2_1)
        conv_layer2_2           = self.DownSample_Layer2_2(conv_layer2_2)

        conv_layer3_first_1     = self.Conv_Layer3_first_1(conv_layer2_1)
        conv_layer3_first_2     = self.Conv_Layer3_first_2(conv_layer2_2)
        conv_layer3_second_1    = self.Conv_Layer3_second_1(conv_layer3_first_1)
        conv_layer3_second_2    = self.Conv_Layer3_second_2(conv_layer3_first_2)
        conv_layer3_third_1     = self.Conv_Layer3_third_1(conv_layer3_first_1 + conv_layer3_second_1)
        conv_layer3_third_2     = self.Conv_Layer3_third_2(conv_layer3_first_2 + conv_layer3_second_2)
        upsample_layer3_1       = self.UpSample_Layer3_1(F.interpolate(conv_layer3_third_1, scale_factor=2.0, mode='bilinear', align_corners=True))
        upsample_layer3_2       = self.UpSample_Layer3_2(F.interpolate(conv_layer3_third_2, scale_factor=2.0, mode='bilinear', align_corners=True))

        conv_layer4_first_1     = self.Conv_Layer4_first_1(torch.cat((upsample_layer3_1, deform_layer2_1), 1))
        conv_layer4_first_2     = self.Conv_Layer4_first_2(torch.cat((upsample_layer3_2, deform_layer2_2), 1))
        conv_layer4_second_1    = self.Conv_Layer4_second_1(conv_layer4_first_1)
        conv_layer4_second_2    = self.Conv_Layer4_second_2(conv_layer4_first_2)
        conv_layer4_third_1     = self.Conv_Layer4_third_1(conv_layer4_first_1 + conv_layer4_second_1)
        conv_layer4_third_2     = self.Conv_Layer4_third_2(conv_layer4_first_2 + conv_layer4_second_2)
        upsample_layer4_1       = self.UpSample_Layer4_1(F.interpolate(conv_layer4_third_1, scale_factor=2.0, mode='bilinear', align_corners=True))
        upsample_layer4_2       = self.UpSample_Layer4_2(F.interpolate(conv_layer4_third_2, scale_factor=2.0, mode='bilinear', align_corners=True))

        compensated_1           = torch.cat((upsample_layer4_1, deform_layer1_1), 1)
        compensated_2           = torch.cat((upsample_layer4_2, deform_layer1_2), 1)

        # fusion and refinement
        fused                   = self.Fusion(torch.cat((compensated_1, compensated_2), 1))
        refined                 = self.Refine(fused)

        return refined


if __name__ == '__main__':
    #   from utils import set_cuda_devices
    from torchsummary import summary

    print("\nCheck Model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #   set_cuda_devices(device, '0')

    x     = (4, 128, 128)

    model = MotionEstimation(in_channels = 2)

    model.to(device)
    summary(model, [x])


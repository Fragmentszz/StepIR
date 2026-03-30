from turtle import forward

from AiO_models.BioIR.All_in_One.train_eval import FFTLoss, BioIRModel

from torch import nn



class BioIRForwarder(nn.Module):
    def __init__(self, configs):
        super(BioIRForwarder, self).__init__()
        self.model = BioIRModel.load_from_checkpoint(configs['checkpoint_path'],map_location='cpu',)

        self.optimizer, self.scheduler = self.model.configure_optimizers()
    def forward(self, lq_img, gt_img, degradations, do_extra_step = 0, **kwargs):
        restored = self.model(lq_img)
        loss = self.model.loss_fn(restored, gt_img)
        fft_loss = self.model.loss_fft(restored, gt_img)
        loss += fft_loss

        if do_extra_step:
            second_restored = self.model(restored)
            second_loss = self.model.loss_fn(second_restored, gt_img)
            second_fft_loss = self.model.loss_fft(second_restored, gt_img)
            second_loss += second_fft_loss
            loss += second_loss

            return loss, second_restored

        return loss, restored

    def configure_optimizers(self):
        return self.optimizer, self.scheduler
    
    def to(self, device):
        self.model.to(device)
        return self
    

from AiO_models.HOGformer.settingIII_IV.train import AdaIR, HOGLoss,AdaIRModel
class HOGformerForwarder(nn.Module):
    def __init__(self, configs):
        super(HOGformerForwarder, self).__init__()
        self.model = AdaIRModel.load_from_checkpoint(configs['checkpoint_path'],map_location='cpu',)
        self.optimizer, self.scheduler = self.model.configure_optimizers()

    def forward(self, lq_img, gt_img, degradations, use_HOG_loss=True, do_extra_step = 0, **kwargs):
        restored = self.model(lq_img)
        l_pear = self.model.compute_correlation_loss(restored, gt_img)
        l_l1 = self.model.loss_fn(restored, gt_img)

        use_HOG_loss = kwargs.get('use_HOG_loss', True)

        # Skip HOG loss in the last 3 epochs
        if use_HOG_loss:
            l_hog = self.model.cri_HOGloss(restored, gt_img)
            loss = l_l1 + l_pear + l_hog
        else:
            l_hog = torch.tensor(0.0, device=restored.device)
            loss = l_l1 + l_pear
        
        if do_extra_step:
            second_restored = self.model(restored)
            second_l_pear = self.model.compute_correlation_loss(second_restored, gt_img)
            second_l_l1 = self.model.loss_fn(second_restored, gt_img)

            if use_HOG_loss:
                second_l_hog = self.model.cri_HOGloss(second_restored, gt_img)
                second_loss = second_l_l1 + second_l_pear + second_l_hog
            else:
                second_l_hog = torch.tensor(0.0, device=restored.device)
                second_loss = second_l_l1 + second_l_pear

            loss += second_loss

            return loss, second_restored

        return loss, restored
        
    def configure_optimizers(self):
        return self.optimizer, self.scheduler

from AiO_models.MoCE_IR.src.train import PLTrainModel as MoCEIRModel
from AiO_models.MoCE_IR.src.options import train_options_finetune
class MoCEIRForwarder(nn.Module):
    def __init__(self, configs):
        
        from types import SimpleNamespace

        super(MoCEIRForwarder, self).__init__()
        base_opt = configs['opt']       # yaml dict
        # base_opt = SimpleNamespace(**base_opt)

        self.opt = train_options_finetune(base_opt)
        self.model = MoCEIRModel.load_from_checkpoint(configs['checkpoint_path'],map_location='cpu',opt=self.opt)
        self.optimizer, self.scheduler = self.model.configure_optimizers()

        if self.opt.loss_type == "fft":
            self.loss_fn = self.model.loss_fn
            self.aux_fn = self.model.aux_fn
        else:
            self.loss_fn = self.model.loss_fn
        

    def forward(self, lq_img, gt_img, degradations, do_extra_step = 0, **kwargs):
        restored = self.model(lq_img)
        balance_loss = self.model.net.total_loss
        if self.opt.loss_type == "fft":

            loss = self.loss_fn(restored, gt_img) + self.aux_fn(restored, gt_img)
        else:
            loss = self.loss_fn(restored, gt_img)

        loss += self.opt.balance_loss_weight * balance_loss


        if do_extra_step:
            second_restored = self.model(restored)
            second_balance_loss = self.model.net.total_loss

            if self.opt.loss_type == "fft":
                second_loss = self.loss_fn(second_restored, gt_img) + self.aux_fn(second_restored, gt_img)
            else:
                second_loss = self.loss_fn(second_restored, gt_img)

            second_loss += self.opt.balance_loss_weight * second_balance_loss
            loss += second_loss
            return loss, second_restored

        return loss, restored
    
    def configure_optimizers(self):
        return self.optimizer, self.scheduler
    
    def to(self, device):
        self.model.to(device)
        return self

import yaml
if __name__ == "__main__":
    # model = BioIRModel()

    bioir_yaml = '/home/wanglixin/StepIR/fintune_AiO/configs/models/BioIR.yaml'
    hog_former_yaml = '/home/wanglixin/StepIR/fintune_AiO/configs/models/Hogformer.yaml'
    moceir_yaml = '/home/wanglixin/StepIR/fintune_AiO/configs/models/MoCE_IR.yaml'


    with open(bioir_yaml, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # forwarder = BioIRForwarder(configs=configs)
    forwarder = HOGformerForwarder(configs=configs)
    # forwarder = MoCEIRForwarder(configs=configs)
    from dataset.finetune_dataset import FinetuneDataset, TestDataset
    dataset = FinetuneDataset('/home/wanglixin/datasets/LSDIR', mode='train')
    test_dataset = TestDataset('/home/wanglixin/datasets/LSDIR/test')
    from torchvision.utils import save_image
    import torch
    with torch.no_grad():
        forwarder.to('cuda')

        for i in range(len(dataset)):
            hq_image, lq_image, degradations = dataset[i]
            hq_image = hq_image.unsqueeze(0)
            lq_image = lq_image.unsqueeze(0)

            lq_image = lq_image.to('cuda')
            hq_image = hq_image.to('cuda')
            loss, restored = forwarder(lq_image, hq_image, degradations)
            print(loss)
            # save restored image
            save_image(restored, f'restored_1.png')
            break

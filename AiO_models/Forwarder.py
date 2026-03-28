from turtle import forward

from AiO_models.BioIR.All_in_One.train_eval import FFTLoss, BioIRModel

from torch import nn

class BioIRForwarder(nn.Module):
    def __init__(self, configs):
        super(BioIRForwarder, self).__init__()
        self.model = BioIRModel().load_from_checkpoint(configs['checkpoint_path'])

        self.optimizer, self.scheduler = self.model.configure_optimizers()
    def forward(self, lq_img, gt_img, degradations):
        restored = self.model(lq_img)
        loss = self.model.loss_fn(restored, gt_img)
        fft_loss = self.model.fft_loss_fn(restored, gt_img)
        loss += fft_loss
        return loss, restored

    def configure_optimizers(self):
        return self.optimizer, self.scheduler
    

# from AiO_models.HOGformer.settingIII_IV.train import AdaIR, HOGLoss
# class HOGformerForwarder(nn.Module):
#     def __init__(self, configs):
#         # super(HOGformerForwarder, self).__init__()
#         # self.model = model
#         # self.loss_fn = FFTLoss()


if __name__ == "__main__":
    # model = BioIRModel()
    forwarder = BioIRForwarder(configs={'checkpoint_path': 'path_to_checkpoint.ckpt'})
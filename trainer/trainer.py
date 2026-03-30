import os
from joblib import register_store_backend
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import random
# 导入你写好的 Forwarder 和 Dataset
# from your_module import BioIRForwarder, HOGformerForwarder, MoCEIRForwarder
# from datasets.finetune_dataset import FinetuneDataset
import pyiqa
import tqdm


# 👇 1. 导入 wandb

import wandb
import time

class DDPTrainer:
    def __init__(self, forwarder, train_dataset, test_dataset, epochs, batch_size):
        # 1. 获取 DDP 环境参数 (由 torchrun 自动注入)
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        
        self.epochs = epochs
        self.device = torch.device(f"cuda:{self.local_rank}")

        # 2. 将 Forwarder 放到对应的 GPU 上
        self.forwarder = forwarder.to(self.device)
        self.optimizer, self.scheduler = self.forwarder.configure_optimizers()

        self.optimizer = self.optimizer[0] if isinstance(self.optimizer, list) else self.optimizer
        self.scheduler = self.scheduler[0] if isinstance(self.scheduler, list) else self.scheduler

        # 3. 使用 DDP 包裹 Forwarder
        # find_unused_parameters=True 视情况开启。如果你的模型在某些条件下有的分支不走(比如概率没触发)，需要设为 True
        self.model = DDP(self.forwarder, device_ids=[self.local_rank], find_unused_parameters=True)
        time_start = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.save_dir = f'/home/wanglixin/StepIR/fintune_AiO/results/{time_start}'

        # 4. 配置 DDP 专用的 Sampler 和 DataLoader
        self.sampler = DistributedSampler(train_dataset)
        self.dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=self.sampler, 
            num_workers=4, # 根据你的 CPU 核心数调整
            pin_memory=True
        )
        self.val_sampler = DistributedSampler(test_dataset, shuffle=False)
        self.val_dataloader = DataLoader(
            test_dataset,
            batch_size=1,  # 验证时通常 batch size 设为 1，或者根据显存调整
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=self.val_sampler, 
        )

        self.p = 0.5

    def train(self):
        global_step = 0
        for epoch in range(self.epochs):
            # self.eval_epoch(self.val_dataloader)  # 这里直接用验证集做 eval，实际使用时应该换成验证集
            # [重要] 打乱数据：必须在每个 epoch 开始前调用，否则每个 epoch 数据顺序一样
            self.sampler.set_epoch(epoch)
            
            self.model.train()
            total_loss = 0.0
            
            # for step, (hq_image, lq_image) in enumerate(self.dataloader):
            p_bar = tqdm.tqdm(enumerate(self.dataloader), desc=f"Training Epoch {epoch+1} (Rank {self.global_rank})", disable=(self.global_rank != 0))
            for step, (hq_image, lq_image) in p_bar:
                # 将数据移动到当前进程绑定的 GPU 上
                hq_image = hq_image.to(self.device, non_blocking=True)
                lq_image = lq_image.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()

                do_extra_step = torch.tensor([0], dtype=torch.int32, device=self.device)
                
                # 只有主进程做决定
                if self.global_rank == 0 and random.random() < self.p:
                    do_extra_step[0] = 1
                
                # 广播给所有进程
                dist.broadcast(do_extra_step, src=0)
                # ==========================================

                # 将指令通过 kwargs 传给 DDP 包裹后的 forwarder
                loss, restored = self.model(
                    lq_image, 
                    hq_image, 
                    degradations = None,
                    do_extra_step=do_extra_step.item()
                )
                
                # 反向传播 & 更新
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                global_step += 1
                # 只在主进程 (Rank 0) 打印日志，防止终端被多张卡疯狂刷屏
                if self.global_rank == 0 and step % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] Step [{step}/{len(self.dataloader)}] Loss: {loss.item():.4f}")
                    current_lr = self.optimizer.param_groups[0]['lr']
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": current_lr,
                        "epoch": epoch + 1
                    }, step=global_step)
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()

            if self.global_rank == 0:
                print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss / len(self.dataloader):.4f}")

            
            # 👇 3. 将 eval_epoch 移到 Epoch 结束时，并接住返回的指标
            val_psnr1, val_ssim1, val_psnr2, val_ssim2 = self.eval_epoch(self.val_dataloader)

            # 👇 4. 在 Rank 0 将验证集的指标上传到 wandb
            if self.global_rank == 0:
                wandb.log({
                    "eval/1st_pass_PSNR": val_psnr1,
                    "eval/1st_pass_SSIM": val_ssim1,
                    "eval/2nd_pass_PSNR": val_psnr2,
                    "eval/2nd_pass_SSIM": val_ssim2,
                    "epoch": epoch + 1
                }, step=global_step)

                # save checkpoint
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                os.makedirs(self.save_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.module.state_dict(),  # 注意 DDP 包裹后的模型需要 .module
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
        # 👇 6. 训练结束后，在 Rank 0 关闭 wandb
        if self.global_rank == 0:
            wandb.finish()
                
    @torch.no_grad()
    def eval_epoch(self, val_dataloader):
        self.model.eval()
        # pyiqa 初始化最好放在 init 里，避免每次 eval 都重新加载模型，这里暂且保持你的写法
        psnr_metric = pyiqa.create_metric('psnr').to(self.device)
        ssim_metric = pyiqa.create_metric('ssim').to(self.device)

        # 记录累加的 "总分" (score * B)
        sum_psnr = 0.0
        sum_ssim = 0.0
        sum_second_psnr = 0.0
        sum_second_ssim = 0.0
        tot_B = 0

        p_bar = tqdm.tqdm(val_dataloader, desc=f"Evaluating (Rank {self.global_rank})", disable=(self.global_rank != 0))
        # for hq_image, lq_image in val_dataloader:
        for hq_image, lq_image in p_bar:
            hq_image = hq_image.to(self.device, non_blocking=True)
            lq_image = lq_image.to(self.device, non_blocking=True)
            B = hq_image.shape[0]
            tot_B += B
            
            # --- 第 1 次前向 ---
            loss, restored = self.model(
                lq_image, hq_image, degradations=None, do_extra_step=0
            )
            
            # --- 第 2 次前向 ---
            loss_second, second_restored = self.model(
                restored, hq_image, degradations=None, do_extra_step=0
            )
            restored = torch.clamp(restored, 0, 1)
            second_restored = torch.clamp(second_restored, 0, 1)
            # --- 计算并累加当前 batch 的总分 ---
            # item() 拿到的是 batch 的均值，乘以 B 得到真实的总和
            sum_psnr += psnr_metric(restored, hq_image).item() * B
            sum_ssim += ssim_metric(restored, hq_image).item() * B
            
            sum_second_psnr += psnr_metric(second_restored, hq_image).item() * B
            sum_second_ssim += ssim_metric(second_restored, hq_image).item() * B

        # ==========================================
        # 核心：跨多卡 Reduce (同步汇总)
        # ==========================================
        # 1. 把所有需要 reduce 的标量打包进一个 Tensor，这样只需通信一次，效率最高
        metrics_tensor = torch.tensor(
            [tot_B, sum_psnr, sum_ssim, sum_second_psnr, sum_second_ssim], 
            dtype=torch.float32, 
            device=self.device
        )

        # 2. 执行 All-Reduce (求和)
        if dist.is_initialized():
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        # 3. 解包并计算真正的全局平均值
        global_tot_B = metrics_tensor[0].item()
        
        # 防止除以 0 (极端情况下)
        if global_tot_B > 0:
            global_avg_psnr = metrics_tensor[1].item() / global_tot_B
            global_avg_ssim = metrics_tensor[2].item() / global_tot_B
            global_avg_second_psnr = metrics_tensor[3].item() / global_tot_B
            global_avg_second_ssim = metrics_tensor[4].item() / global_tot_B
        else:
            global_avg_psnr = global_avg_ssim = global_avg_second_psnr = global_avg_second_ssim = 0.0

        # 只在主卡打印结果
        if self.global_rank == 0:
            print(f"--- Eval Results (Images: {int(global_tot_B)}) ---")
            print(f"1st Pass - PSNR: {global_avg_psnr:.4f}, SSIM: {global_avg_ssim:.4f}")
            print(f"2nd Pass - PSNR: {global_avg_second_psnr:.4f}, SSIM: {global_avg_second_ssim:.4f}")

        # 切记把模型切回训练模式
        self.model.train()
        
        return global_avg_psnr, global_avg_ssim, global_avg_second_psnr, global_avg_second_ssim






if __name__ == "__main__":

    from AiO_models.Forwarder import BioIRForwarder, HOGformerForwarder, MoCEIRForwarder
    from dataset.finetune_dataset import FinetuneDataset, TestDataset


    dist.init_process_group(backend="nccl")
    
    # 强制让当前进程只看到自己的那一块 GPU，防止串号
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)


    

    # 1. 加载配置
    with open('configs/models/Hogformer.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 👇 5. 只在 Rank 0 初始化 wandb，并把超参数字典 config 传进去
    if global_rank == 0:
        wandb.init(
            project="StepIR",  # 你的项目名称
            name="HOGformer-Finetune",            # 这一次实验的名称
            config=config                     # 会自动把 yaml 里的超参记录下来
        )

    # 2. 创建 Forwarder 和 Dataset 实例
    forwarder = HOGformerForwarder(configs=config)
    dataset = FinetuneDataset('/home/wanglixin/datasets/LSDIR', mode='train')
    test_dataset = TestDataset('/home/wanglixin/datasets/LSDIR/test')

    # 3. 创建 DDPTrainer 实例并开始训练
    trainer = DDPTrainer(
        forwarder=forwarder, 
        train_dataset=dataset,
        test_dataset=test_dataset,

        epochs=config['train_opt']['epoch'], 
        batch_size=config['train_opt']['batch_size']
    )
    trainer.train()

    

    dist.destroy_process_group()
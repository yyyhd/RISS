import os
import torch
import torch.nn as nn


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        # 默认 tensor 类型
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    # ===========================================
    #  ⭐ 多卡支持关键函数 ⭐
    # ===========================================
    def setup_multi_gpu(self, model):
        """把模型包成 DataParallel（如果给了多个 GPU）"""
        if self.gpu_ids and len(self.gpu_ids) > 1:
            print(f"➡ Using Multi-GPU DataParallel on GPUs: {self.gpu_ids}")
            model = nn.DataParallel(model, device_ids=self.gpu_ids)
        else:
            print(f"➡ Using single GPU: {self.gpu_ids[0] if self.gpu_ids else 'CPU'}")

        # 模型移动到第一个 GPU
        if self.gpu_ids:
            model = model.cuda(self.gpu_ids[0])

        return model

    # ===========================================
    #  ⭐ 保存网络：兼容 DataParallel + 单卡 ⭐
    # ===========================================
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)

        # 如果是 DataParallel，保存其 module
        if isinstance(network, nn.DataParallel):
            print(f"Saving DataParallel model (module) to: {save_path}")
            torch.save(network.module.cpu().state_dict(), save_path)
            network.cuda(gpu_ids[0])
        else:
            print(f"Saving single model to: {save_path}")
            torch.save(network.cpu().state_dict(), save_path)
            if gpu_ids:
                network.cuda(gpu_ids[0])

    # ===========================================
    #  ⭐ 加载网络：兼容 DataParallel + 单卡 ⭐
    # ===========================================
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print(f"Loading network from: {save_path}")

        state_dict = torch.load(save_path, map_location=lambda storage, loc: storage)

        # 如果需要多卡训练，则自动加载到 module
        if isinstance(network, nn.DataParallel):
            network.module.load_state_dict(state_dict)
        else:
            network.load_state_dict(state_dict)

    # ===========================================
    #  更新学习率
    # ===========================================
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)


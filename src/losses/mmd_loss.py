import torch
import torch.nn as nn

class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target):
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)

        # Compute pairwise squared Euclidean distances
        total0 = total.unsqueeze(0)
        total1 = total.unsqueeze(1)
        L2_distance = ((total0 - total1) ** 2).sum(2)

        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
      
        # Compute the kernel values
        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = source.size(0)
        kernels = self.gaussian_kernel(source, target)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
      
        loss = torch.mean(XX + YY - XY - YX)
        return loss


class LinearMMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(LinearMMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target):
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)

        # Compute pairwise squared Euclidean distances
        total0 = total.unsqueeze(0)
        total1 = total.unsqueeze(1)
        L2_distance = ((total0 - total1) ** 2).sum(2)

        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
      
        # Compute the kernel values
        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = source.size(0)
        kernels = self.gaussian_kernel(source, target)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
      
        loss = torch.mean(XX + YY - XY - YX)
        return loss
    
class BatchwiseMMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=1.0, sub_batch_size=50):
        super(BatchwiseMMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.sub_batch_size = sub_batch_size

    def gaussian_kernel(self, source, target):
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)

        # Compute pairwise squared Euclidean distances
        total0 = total.unsqueeze(0)
        total1 = total.unsqueeze(1)
        L2_distance = ((total0 - total1) ** 2).sum(2)

        bandwidth = self.fix_sigma
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
      
        # Compute the kernel values
        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = source.size(0)
        sub_batch_size = self.sub_batch_size

        total_loss = 0.0
        num_sub_batches = (batch_size + sub_batch_size - 1) // sub_batch_size

        for i in range(num_sub_batches):
            for j in range(num_sub_batches):
                source_sub = source[i * sub_batch_size:(i + 1) * sub_batch_size]
                target_sub = target[j * sub_batch_size:(j + 1) * sub_batch_size]

                # Ensure sub-batches are of the same size
                if source_sub.size(0) != target_sub.size(0):
                    min_size = min(source_sub.size(0), target_sub.size(0))
                    source_sub = source_sub[:min_size]
                    target_sub = target_sub[:min_size]

                kernels = self.gaussian_kernel(source_sub, target_sub)

                XX = kernels[:source_sub.size(0), :source_sub.size(0)]
                YY = kernels[source_sub.size(0):, source_sub.size(0):]
                XY = kernels[:source_sub.size(0), source_sub.size(0):]
                YX = kernels[source_sub.size(0):, :source_sub.size(0)]
              
                sub_loss = torch.mean(XX + YY - XY - YX)
                total_loss += sub_loss

        total_loss /= (num_sub_batches * num_sub_batches)
        return total_loss

if __name__ == '__main__':
    batch_size = 32
    feature_dim = 450
    source_features = torch.randn(batch_size, feature_dim)
    target_features = torch.randn(batch_size, feature_dim)

    mmd_loss = LinearMMDLoss()
    loss = mmd_loss(source_features, target_features)
    print('Linear MMD Loss:', loss.item())
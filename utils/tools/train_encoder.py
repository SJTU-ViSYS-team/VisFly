import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os, sys
# import tensorboard
from torch.utils.tensorboard import SummaryWriter

IS_TEST = False


# 定义数据集类
class GrayscaleImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')  # Convert image to grayscale
        if self.transform:
            image = self.transform(image)
        return image


# 定义简化版自动编码器模型

class Autoencoder(nn.Module):
    def __init__(self, channels):
        super(Autoencoder, self).__init__()
        # 编码器
        encoder_layers = []
        for i in range(len(channels)):
            if i == 0:
                encoder_layers.append(nn.Conv2d(1, channels[i], kernel_size=3, stride=2, padding=1))
            else:
                encoder_layers.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size=3, stride=2, padding=1))
            encoder_layers.append(nn.LeakyReLU())
            # encoder_layers.append(nn.MaxPool2d(2, stride=2))  # 添加下采样层
        self.encoder = nn.Sequential(*encoder_layers)
        self.para_init()
        self.f = torch.nn.Flatten()

        # 解码器
        decoder_layers = []
        for i in range(len(channels) - 1, -1, -1):
            if i == 0:
                decoder_layers.append(nn.ConvTranspose2d(channels[i], 1, kernel_size=3, stride=2, padding=1, output_padding=1))
            else:
                decoder_layers.append(nn.ConvTranspose2d(channels[i], channels[i - 1], kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder_layers.append(nn.LeakyReLU())
            # decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))  # 添加上采样层
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # x = self.encoder(1-1/(1+x))
        # x = self.decoder(-1/(x-1)-1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.f(self.encoder(x))



    def para_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


path = os.path.dirname(os.path.abspath(sys.argv[0])) + '/saved/depth_autoencoder'
channels = [2, 4, 8, 16]
model = Autoencoder(channels)

def main():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
    ])
    dataset = GrayscaleImageDataset(image_dir=os.getcwd() + "/datasets/Images/depth_dataset/64", transform=transform)
    dataloader = DataLoader(dataset, batch_size=10240, shuffle=True)
    criterion = nn.MSELoss()  # 使用均方误差作为损失函数

    if not IS_TEST:
        # 数据预处理和数据加载


        try:
            os.mkdir(os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/")
        except:
            pass
        Writer = SummaryWriter(os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/")

        # 创建模型实例

        # 定义损失函数和优化器
        optimizer = optim.Adam(model.parameters(), lr=1e-2)  # 使用Adam优化器

        # 训练模型
        num_epochs = 100
        model.to("cuda")
        i = 1
        for epoch in range(num_epochs):
            for data in dataloader:
                img = data.cuda()
                # 前向传播
                output = model(img)
                loss = criterion(output, img)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1
                Writer.add_scalar('training loss',
                                  loss / 10,
                                  i)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        print("Training finished.")

        torch.save(model.state_dict(), path + ".pth")
        torch.save(model, path + "_all.pth")
        print("Model saved.")

    else:
        # 加载模型参数
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = Autoencoder(channels)
        model.load_state_dict(torch.load(path + ".pth"))
        model.eval()
        print("Model loaded.")
        # plot 32 pairs of original images and reconstructed images
        import matplotlib.pyplot as plt

        data_iter = iter(dataloader)
        data = next(data_iter)
        # 8*8 subplots
        f = plt.figure()
        axeses = f.subplots(8, 8)
        for i in range(32):
            c, r = i % 4, i // 4
            ori, output = data[i], model(data[i])
            e = model.encode(ori)
            axeses[r, 2 * c].imshow(ori.squeeze())
            axeses[r, 2 * c + 1].imshow(output.squeeze().detach().numpy())
            axeses[r, 2 * c].set_title(format(criterion(ori, output).item(), ".4f"))
        plt.show()

if __name__ == "__main__":
    main()
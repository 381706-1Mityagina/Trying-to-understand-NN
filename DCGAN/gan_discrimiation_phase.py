import torch
import torch.nn as net
from torch.autograd import Variable
from torch.nn import functional as Func

# Deep Convolutional GAN
class Discriminator(net.Module):
    def __init__(self, batch_size, img_size, embed_dim, reduced_dim):
        super(Discriminator, self).__init__()

        self.batch_size = batch_size
        self.img_size = img_size
        self.in_channels = img_size.size()[2]
        self.embed_dim = embed_dim
        self.reduced_dim = reduced_dim

        # Архитектура
        self.d_net = net.Sequential(
            net.Conv2d(self.in_channels, 64, 4, 2, 1, bias = False),
            net.LeakyReLU(0.2, inplace = True),
            net.Conv2d(64, 128, 4, 2, 1, bias = False),
            net.BatchNorm2d(128),
            net.LeakyReLU(0.2, inplace = True),
            net.Conv2d(128, 256, 4, 2, 1, bias = False),
            net.BatchNorm2d(256),
            net.LeakyReLU(0.2, inplace = True),
            net.Conv2d(256, 512, 4, 2, 1, bias = False),
            net.BatchNorm2d(512),
            net.LeakyReLU(0.2, inplace = True)
        )
        # ---
        # LeakyReLU - 
        # Leaky ReLU имеет небольшой наклон для отрицательных значений, а не ноль.
        # Например, Leaky ReLU может иметь y = 0,01x, когда x <0.
        # ---

        # output_dim = (batch_size, 4, 4, 512)
        # text.size() = (batch_size, embed_dim)

        # Определение линейного слоя для уменьшения размерности заголовка
        # от embed_dim к reduced_dim
        self.reduced_dim = net.Linear(self.embed_dim, self.reduced_dim)

        self.cat_net = net.Sequential(
            net.Conv2d(512 + self.reduced_dim, 512, 4, 2, 1, bias=False),
            net.BatchNorm2d(512),
            net.LeakyReLU(0.2, inplace=True))

        self.linear = net.Linear(2 * 2 * 512, 1)

    def forward(self, image, text):
        """ 
        Вход  : изображение и его описание
        Далее : предсказание, реально оно или нет.
        ---
            image.size = (batch_size, 64, 64, 3)
            text.size  = (batch_size, embed_dim)
        ---
        output : Вероятность (реальное изображение или нет)
        logit  : Итоговая оценка
        """
        # (batch_size, 4, 4, 512)
        d_net_out = self.d_net(image)
        # (batch_size, reduced_dim)
        text_reduced = self.reduced_dim(text)
        # (batch_size, 1, reduced_dim)
        text_reduced = text_reduced.squeeze(1)
        # (batch_size, 1, 1, reduced_dim)
        text_reduced = text_reduced.squeeze(2)
        text_reduced = text_reduced.expand(1, 4, 4, self.reduced_dim)

        # (1, 4, 4, 512 + reduced_dim)
        concat_out = torch.cat((d_net_out, text_reduced), 3)

        # В статистике функция logit или log-odds является логарифмом шансов
        # [p / (1- p)] где p - вероятность. Это тип функции, который создает 
        # карту значений вероятности от (0,1)  до (-inf, + inf)
        logit = self.cat_net(concat_out)
        concat_out = torch.view(-1, concat_out.size()[1] * concat_out.size()[2] * concat_out.size()[3])
        concat_out = self.linear(concat_out)

        output = Func.sigmoid(logit)

        return output, logit

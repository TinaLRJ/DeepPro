import os
import cv2
import numpy as np
from torchvision.transforms import functional as F
import torch
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats
import scipy.io as scio


def PIL2Tensor(pil_image):
    if isinstance(pil_image, list):
        pils = []
        for img in pil_image:
            pils.append(F.to_tensor(img))
        return torch.stack(pils)
    return F.to_tensor(pil_image)


def Tensor2PIL(tensor_image, mode='RGB'):
    if len(tensor_image.size()) == 4 and tensor_image.size()[0] == 1:
        tensor_image = tensor_image.view(tensor_image.size()[1:])
    return F.to_pil_image(tensor_image.detach(), mode=mode)



def cv2_to_pil(img):

    image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    return image


def pil_to_cv2(img):
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return image


def make_pil_grid(pil_image_list):
    sizex, sizey = pil_image_list[0].size
    for img in pil_image_list:
        assert sizex == img.size[0] and sizey == img.size[1], 'check image size'

    target = Image.new('RGB', (sizex * len(pil_image_list), sizey))
    left = 0
    right = sizex
    for i in range(len(pil_image_list)):
        target.paste(pil_image_list[i], (left, 0, right, sizey))
        left += sizex
        right += sizex
    return target


def vis_saliency_kde(map, zoomin=4):
    grad_flat = map.reshape((-1))
    datapoint_y, datapoint_x = np.mgrid[0:map.shape[0]:1, 0:map.shape[1]:1]
    Y, X = np.mgrid[0:map.shape[0]:1, 0:map.shape[1]:1]
    positions = np.vstack([X.ravel(), Y.ravel()])
    pixels = np.vstack([datapoint_x.ravel(), datapoint_y.ravel()])
    kernel = stats.gaussian_kde(pixels, weights=grad_flat)
    Z = np.reshape(kernel(positions).T, map.shape)
    Z = Z / Z.max()
    cmap = plt.get_cmap('seismic')
    # cmap = plt.get_cmap('Purples')
    map_color = (255 * cmap(Z * 0.5 + 0.5)).astype(np.uint8)
    # map_color = (255 * cmap(Z)).astype(np.uint8)
    Img = Image.fromarray(map_color)
    s1, s2 = Img.size
    return Img.resize((s1 * zoomin, s2 * zoomin), Image.BICUBIC)


def color(gray):
    rgb = np.expand_dims(np.array([255, 0, 0]), axis=(0,1,2))
    return rgb*np.expand_dims(gray, axis=3).astype(np.uint8)


def vis_saliency(map, path):
    """
    :param map: the saliency map, 2D, norm to [0, 1]
    :param zoomin: the resize factor, nn upsample
    :return:
    """
    path, dataset, savepath = path
    os.makedirs(savepath, exist_ok=True)
    b,c,t,h,w = map.shape
    for i in range(b):
        map_i = map[i, 0, :, :, :]
        seq_folder = path.split('/')[0]
        os.makedirs(os.path.join(savepath, seq_folder), exist_ok=True)
        png_name = path.split('/')[-1]

        indices = map_i>0.1
        if indices.sum()>1:
            lrj = 1

            x = np.arange(w).reshape(1, 1, w).repeat(h, axis=1).repeat(t, axis=0)
            y = np.arange(h).reshape(1, h, 1).repeat(w, axis=2).repeat(t, axis=0)
            z = np.arange(t).reshape(t, 1, 1).repeat(h, axis=1).repeat(w, axis=2)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_zlim3d(0,t)
            # plt.set_cmap(cmap)
            x_show = np.append(x[indices], np.array([0]))
            y_show = np.append(y[indices], np.array([0]))
            z_show = np.append(z[indices], np.array([0]))
            c_show = np.append(map_i[indices], np.array([0]))
            sc = ax.scatter(x_show,y_show,z_show, c=c_show, cmap='Reds', alpha=0.6)
            # im = ax.scatter(x,y,z,map_color, marker='.')
            fig.colorbar(sc, shrink=0.5, aspect=5)
            plt.xlim(0,w)
            plt.ylim(0,h)
            # plt.show()
            plt.savefig(os.path.join(savepath, seq_folder, png_name))
            plt.close()

        save_name = os.path.join(savepath, seq_folder, png_name).replace('.png', '.mat')
        scio.savemat(save_name, {'map_attribution': map_i})
    return


def prepare_images(hr_path, scale=4):
    hr_pil = Image.open(hr_path)
    sizex, sizey = hr_pil.size
    hr_pil = hr_pil.crop((0, 0, sizex - sizex % scale, sizey - sizey % scale))
    sizex, sizey = hr_pil.size
    lr_pil = hr_pil.resize((sizex // scale, sizey // scale), Image.BICUBIC)
    return lr_pil, hr_pil


def grad_abs_norm(grad):
    """

    :param grad: numpy array [t,b,1,t,h,w]
    :return:
    """
    grad_abs = np.sqrt(np.abs(grad))
    #平方根

    grad_max = grad_abs.max(axis=(2,3,4,5))

    grad_norm = grad_abs / grad_max[:,:,None,None,None,None]
    return grad_norm


def interpolation(x, x_prime, fold, mode='linear'):
    diff = x - x_prime
    l = torch.linspace(0, 1, fold).reshape((fold, 1, 1, 1, 1, 1)).to(x.device)
    interp_list = l * diff + x_prime
    return interp_list


def isotropic_gaussian_kernel(l, sigma, epsilon=1e-5):
    ax = torch.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * (sigma + epsilon) ** 2))
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel / torch.sum(kernel)  # [1,1,l,l]

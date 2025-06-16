import numpy as np
import torch
import torch.nn.functional as F
from attribution.utils import grad_abs_norm, vis_saliency
from attribution.utils import interpolation, isotropic_gaussian_kernel
from tqdm import tqdm




# def GaussianBlurPath(sigma, fold, l=5):
#     def path_interpolation_func(cv_numpy_image):
#         h, w, c = cv_numpy_image.shape
#         kernel_interpolation = np.zeros((fold + 1, l, l))
#         image_interpolation = np.zeros((fold, h, w, c))
#         lambda_derivative_interpolation = np.zeros((fold, h, w, c))
#         sigma_interpolation = np.linspace(sigma, 0, fold + 1)
#         for i in tqdm(range(fold + 1)):
#             kernel_interpolation[i] = isotropic_gaussian_kernel(l, sigma_interpolation[i])
#         for i in tqdm(range(fold)):
#             image_interpolation[i] = cv2.filter2D(cv_numpy_image, -1, kernel_interpolation[i + 1])
#             lambda_derivative_interpolation[i] = cv2.filter2D(cv_numpy_image, -1, (
#                     kernel_interpolation[i + 1] - kernel_interpolation[i]) * fold)
#         return np.moveaxis(image_interpolation, 3, 1).astype(np.float32), \
#             np.moveaxis(lambda_derivative_interpolation, 3, 1).astype(np.float32)
#
#     return path_interpolation_func

# 另一种积分路径函数
def MeanLinearPath(fold=50, l=9):
    def path_interpolation_func(images):
        med_image = images.median(1)[0].unsqueeze(1)
        # template = isotropic_gaussian_kernel(l, sigma)
        template = torch.ones([1, 1, l, l])/(l*l)
        baseline_image = F.conv2d(med_image, template, stride=1, padding=l//2)
        baseline_images = baseline_image.repeat(1, images.size(1), 1, 1)
        image_interpolation = interpolation(images, baseline_images, fold, mode='linear').type(torch.float32)
        lambda_derivative_interpolation = torch.unsqueeze(images - baseline_images, dim=0).repeat(fold, 1,1,1,1)
        return image_interpolation, lambda_derivative_interpolation

    return path_interpolation_func

# 另一种积分路径函数
def ZeroLinearPath(fold=50):
    def path_interpolation_func(images):
        baseline_images = torch.zeros_like(images)
        image_interpolation = interpolation(images, baseline_images, fold, mode='linear').type(torch.float32)
        lambda_derivative_interpolation = torch.unsqueeze(images - baseline_images, dim=0).repeat(fold, 1,1,1,1,1)
        return image_interpolation, lambda_derivative_interpolation

    return path_interpolation_func


## attribution
def IR_Integrated_gradient(image, label, path, model, path_interpolation_func, cuda=True):
    """
    :param path_interpolation_func:
        return \lambda(\alpha) and d\lambda(\alpha)/d\alpha, for \alpha\in[0, 1]
        This function return pil_tensor_images
    :return:
    """
    b,t,h,w = label.size()
    m,n = image.shape[-2:]
    image_interpolation, lambda_derivative_interpolation = path_interpolation_func(image.data.cpu())
    grad_accumulate_list = np.zeros([image_interpolation.shape[0],t,b,1,t,h,w])
    result_list = []
    with torch.set_grad_enabled(True):
        for i in tqdm(range(image_interpolation.shape[0])):
            img_tensor = image_interpolation[i].cuda()
            img_tensor.requires_grad_(True)
            _, results = model(img_tensor)
            if isinstance(results, list):
                result = results[-1]
            else:
                result = results

            for ti in range(t):
                target = torch.sum(torch.sigmoid(result[:,ti,:h,:w])*label[:,ti,:,:])
                if ti < t-1:
                    target.backward(retain_graph=True)
                else:
                    target.backward()
                grad = img_tensor.grad[:,:,:,:h,:w].cpu().numpy()
                img_tensor.grad = torch.zeros_like(img_tensor.grad)
                if np.any(np.isnan(grad)):
                    grad[np.isnan(grad)] = 0.0

                grad_accumulate_list[i, ti] = grad * lambda_derivative_interpolation[i][:,:,:,:h,:w].cpu().numpy()
            result_list.append(results)

        final_grad, result = saliency_map_PG(grad_accumulate_list, result_list)
        abs_normed_grad_numpy = grad_abs_norm(final_grad)

        # 可视化
        # Visualize saliency
        for ti in range(t):
            vis_saliency(abs_normed_grad_numpy[ti], (path[0][ti], path[1], path[2]))

    return result


def saliency_map_PG(grad_list, result_list):
    final_grad = grad_list.mean(axis=0)
    return final_grad, result_list[-1]


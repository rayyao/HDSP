
import pdb
from PIL import Image
import cv2
import numpy as np
from lib.models.HDSP.bandselect import bandselecttrain,bandselecttest
import torch
def get_rgbd_frame(color_path, depth_path, dtype='rgbcolormap', depth_clip=False):

    if color_path:
        rgb = cv2.imread(color_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    else:
        rgb = None

    if depth_path:
        dp = cv2.imread(depth_path, -1)

        if depth_clip:
            max_depth = min(np.median(dp) * 3, 10000)
            dp[dp>max_depth] = max_depth
    else:
        dp = None

    if dtype == 'color':
        img = rgb

    elif dtype == 'raw_depth':
        img = dp

    elif dtype == 'colormap':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        img = cv2.applyColorMap(dp, cv2.COLORMAP_JET)

    elif dtype == '3xD':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        img = cv2.merge((dp, dp, dp))

    elif dtype == 'normalized_depth':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img = np.asarray(dp, dtype=np.uint8)

    elif dtype == 'rgbcolormap':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        colormap = cv2.applyColorMap(dp, cv2.COLORMAP_JET)
        img = cv2.merge((rgb, colormap))

    elif dtype == 'rgb3d':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        dp = cv2.merge((dp, dp, dp))
        img = cv2.merge((rgb, dp))

    elif dtype == 'rgbrgb':
        dp = cv2.cvtColor(dp, cv2.COLOR_BGR2RGB)
        img = cv2.merge((rgb, dp))

    else:
        print('No such dtype !!! ')
        img = None

    return img


def get_x_frame(color_path, depth_path, dtype='rgbcolormap', depth_clip=False):

    if color_path:
        rgb = cv2.imread(color_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    else:
        rgb = None

    if depth_path:
        dp = cv2.imread(depth_path, -1)

        if depth_clip:
            max_depth = min(np.median(dp) * 3, 10000)
            dp[dp > max_depth] = max_depth
    else:
        dp = None

    if dtype == 'color':
        img = rgb

    elif dtype == 'raw_x':
        img = dp

    elif dtype == 'colormap':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        img = cv2.applyColorMap(dp, cv2.COLORMAP_JET)

    elif dtype == '3x':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        img = cv2.merge((dp, dp, dp))

    elif dtype == 'normalized_x':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img = np.asarray(dp, dtype=np.uint8)

    elif dtype == 'rgbcolormap':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        colormap = cv2.applyColorMap(dp, cv2.COLORMAP_JET)  # (h,w) -> (h,w,3)
        img = cv2.merge((rgb, colormap))  # (h,w,6)

    elif dtype == 'rgb3x':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        dp = cv2.merge((dp, dp, dp))
        img = cv2.merge((rgb, dp))

    elif dtype == 'rgbrgb':
        dp = cv2.cvtColor(dp, cv2.COLOR_BGR2RGB)
        img = cv2.merge((rgb, dp))

    else:
        print('No such dtype !!! ')
        img = None

    return img

def get_HSI_frame(color_path, HSI_path,last_frame_path, name,dtype='rgbcolormap', depth_clip=False):
    if color_path:

        rgb = cv2.imread(color_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if HSI_path:
        image = Image.open(HSI_path)
        #name = os.path.basename(os.path.dirname(HSI_path))
        dp = bandselecttrain(image, name)
    if last_frame_path:
        last = cv2.imread(last_frame_path)
        last = cv2.cvtColor(last, cv2.COLOR_BGR2RGB)
    img=np.concatenate((rgb,dp,last),axis=2)
    img=torch.from_numpy(img).float()
    img=img.numpy()
    return img
def get_HSItest_frame(color_path, HSI_path,mask_path, idx,dtype='rgbcolormap', depth_clip=False):

    if color_path:
        rgb = cv2.imread(color_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if HSI_path:
        image = Image.open(HSI_path)
        dp = bandselecttest(image, idx)
    if mask_path:
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    img = np.concatenate((rgb, dp,mask), axis=2)
    img = torch.from_numpy(img).float()
    img = img.numpy()

    return img,rgb,mask

class p_config(object):
    grabcut_extra = 50
    grabcut_rz_threshold = 300
    grabcut_rz_factor = 1.5
    minimun_target_pixels = 16
    grabcut_iter = 3
    radius = 500

def get_layered_image_by_depth(depth_image, target_depth, dtype='centered_colormap'):
    p = p_config()

    if target_depth is not None:
        low = max(target_depth - p.radius, 0)
        high = target_depth + p.radius

        layer = depth_image.copy()
        layer[layer < low] = high + 10
        layer[layer > high] = high + 10
    else:
        layer = depth_image.copy()

    layer = remove_bubbles(layer, bubbles_size=200)

    if dtype == 'centered_colormap':
        layer = cv2.normalize(layer, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        layer = np.asarray(layer, dtype=np.uint8)
        layer = cv2.applyColorMap(layer, cv2.COLORMAP_JET)
    elif dtype == 'centered_normalized_depth':
        layer = cv2.normalize(layer, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        layer = np.asarray(layer, dtype=np.uint8)
        layer = cv2.merge((layer, layer, layer))
    elif dtype == 'centered_raw_depth':
        layer = np.asarray(layer)
        layer = np.stack((layer, layer, layer), axis=2)

    return layer


def remove_bubbles(image, bubbles_size=100):
    try:
        binary_map = (image > 0).astype(np.uint8)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        mask = np.zeros((image.shape), dtype=np.uint8)
        for i in range(0, nb_components):
            if sizes[i] >= bubbles_size:
                mask[output == i + 1] = 1

        if len(image.shape) > 2:
            image = image * mask[:, :, np.newaxis]
        else:
            image = image * mask
    except:
        pass

    return image


def get_target_depth(depth, target_box):


    p = p_config()
    H, W = depth.shape
    target_box = [int(bb) for bb in target_box]
    x0, y0, w0, h0 = target_box
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x0 + w0, W)
    y1 = min(y0 + h0, H)
    possible_target = depth[y0:y1, x0:x1]
    median_depth = np.median(possible_target) + 10

    bubbles_size = int(target_box[2] * target_box[3] * 0.1)
    try:

        extra_y0 = max(y0 - p.grabcut_extra, 0)
        extra_x0 = max(x0 - p.grabcut_extra, 0)
        extra_y1 = min(y1 + p.grabcut_extra, H)
        extra_x1 = min(x1 + p.grabcut_extra, W)

        rect_x0 = x0 - extra_x0
        rect_y0 = y0 - extra_y0
        rect_x1 = min(rect_x0 + w0, extra_x1)
        rect_y1 = min(rect_y0 + h0, extra_y1)
        rect = [rect_x0, rect_y0, rect_x1 - rect_x0, rect_y1 - rect_y0]

        target_patch = depth[extra_y0:extra_y1, extra_x0:extra_x1]
        target_patch = np.nan_to_num(target_patch, nan=np.max(target_patch))

        image = target_patch.copy()
        image[image > median_depth * 2] = median_depth * 2
        image[image < 10] = median_depth * 2

        i_H, i_W = image.shape
        rz_factor = p.grabcut_rz_factor if min(i_W, i_H) > p.grabcut_rz_threshold else 1
        rect_rz = [int(rt // rz_factor) for rt in rect]
        rz_dim = (int(i_W // rz_factor), int(i_H // rz_factor))

        image = cv2.resize(image, rz_dim, interpolation=cv2.INTER_AREA)
        image = remove_bubbles(image, bubbles_size=bubbles_size)
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = np.asarray(image, dtype=np.uint8)
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)


        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, mask, rect_rz, bgdModel, fgdModel, p.grabcut_iter, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        mask2 = remove_bubbles(mask2, bubbles_size=bubbles_size)
        mask2 = cv2.resize(mask2, (i_W, i_H), interpolation=cv2.INTER_AREA)


        target_pixels = target_patch * mask2
        target_pixels = target_pixels.flatten()
        target_pixels.sort()
        target_pixels = target_pixels[target_pixels > 0]

        if len(target_pixels) > p.minimun_target_pixels:
            hist, bin_edges = np.histogram(target_pixels, bins=20)
            peak_idx = np.argmax(hist)
            selected_target_pixels = target_pixels
            target_depth_low = bin_edges[peak_idx]
            target_depth_high = bin_edges[peak_idx + 1]
            selected_target_pixels = selected_target_pixels[selected_target_pixels <= target_depth_high]
            selected_target_pixels = selected_target_pixels[selected_target_pixels >= target_depth_low]
            target_depth = np.median(selected_target_pixels)
        else:
            target_depth = median_depth
    except:
        target_depth = median_depth

    return target_depth


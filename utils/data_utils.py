import os
import torch
import cv2
import numpy as np
import collections

import math
import numpy as np

def generate_augmentation(batchsize):
    delta_t = torch.rand(batchsize, 1, 3).cuda()
    delta_t = delta_t.uniform_(-0.02, 0.02)
    
    angle_r = torch.randn(batchsize, 3)
    angle_r.uniform_(-20, 20)
    angle_r = angle_r / 180 * torch.pi # b, 3
    
    delta_r_x = torch.eye(3).unsqueeze(0).repeat(batchsize, 1, 1) # b, 3, 3
    delta_r_y = torch.eye(3).unsqueeze(0).repeat(batchsize, 1, 1)
    delta_r_z = torch.eye(3).unsqueeze(0).repeat(batchsize, 1, 1)
    
    delta_r_x[:, 1, 1] = torch.cos(angle_r[:, 0])
    delta_r_x[:, 1, 2] = -torch.sin(angle_r[:, 0])
    delta_r_x[:, 2, 1] = torch.sin(angle_r[:, 0])
    delta_r_x[:, 2, 2] = torch.cos(angle_r[:, 0])
    
    delta_r_y[:, 0, 0] = torch.cos(angle_r[:, 1])
    delta_r_y[:, 0, 2] = torch.sin(angle_r[:, 1])
    delta_r_y[:, 2, 0] = -torch.sin(angle_r[:, 1])
    delta_r_y[:, 2, 2] = torch.cos(angle_r[:, 1])
    
    delta_r_z[:, 0, 0] = torch.cos(angle_r[:, 2])
    delta_r_z[:, 0, 1] = -torch.sin(angle_r[:, 2])
    delta_r_z[:, 1, 0] = torch.sin(angle_r[:, 2])
    delta_r_z[:, 1, 1] = torch.cos(angle_r[:, 2])
    
    delta_r = torch.bmm(torch.bmm(delta_r_x, delta_r_y), delta_r_z).cuda()
    
    delta_s = torch.rand(batchsize, 1).cuda()
    delta_s = delta_s.uniform_(0.8, 1.2)
    
    return delta_r, delta_t, delta_s


def random_rotate_matrix(angle_range=None):
    
    if angle_range != "random":
        if angle_range > 360:
            angle_range = angle_range - 360
            angles = np.random.normal(0, angle_range, (3)) / 180 * np.pi
        else:
            angles = np.random.uniform(-angle_range, angle_range, 3) / 180 * np.pi
    else:
        angles = np.random.rand(3) * 2 * np.pi
    rand_rotation = np.array([
        [1,0,0],
        [0,np.cos(angles[0]),-np.sin(angles[0])],
        [0,np.sin(angles[0]), np.cos(angles[0])]
    ]) @ np.array([
        [np.cos(angles[1]),0,np.sin(angles[1])],
        [0,1,0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ]) @ np.array([
        [np.cos(angles[2]),-np.sin(angles[2]),0],
        [np.sin(angles[2]), np.cos(angles[2]),0],
        [0,0,1]
    ])
    return rand_rotation

def calculateRotationError(rot1, rot2):
    rot_err = rot1 @ rot2.T
    rot_err = (np.trace(rot_err) - 1) / 2
    rot_err = np.clip(rot_err, -1, 1)
    rot_err = np.arccos(rot_err)
    rot_err = rot_err / np.pi * 180
    return rot_err

def eulerAngle2EulerLabel(euler, bins=256):
    """ x in [-pi, pi]
        y in [-pi/2, pi/2]
        z in [-pi, pi]

    Args:
        euler (_type_): _description_
        bins (int, optional): _description_. Defaults to 128.
    """
    assert euler[0] <= np.pi and euler[0] >= -np.pi
    assert euler[1] <= (np.pi / 2) and euler[1] >= -(np.pi / 2)
    assert euler[2] <= np.pi and euler[2] >= -np.pi
    
    x_label = int((euler[0] + np.pi) / (2*np.pi) * bins)
    y_label = int((euler[1] + (np.pi / 2)) / (np.pi) * bins)
    z_label = int((euler[2] + np.pi) / (2*np.pi) * bins)
    
    return np.array([x_label, y_label, z_label])
    
def eulerLabel2EulerAngle(euler_label, bins=256):
    
    x_angle = (euler_label[0] + 0.5) / bins * (2*np.pi) - np.pi
    y_angle = (euler_label[1] + 0.5) / bins * (np.pi) - np.pi/2
    z_angle = (euler_label[2] + 0.5) / bins * (2*np.pi) - np.pi
 
    return np.array([x_angle, y_angle, z_angle])

 
def eulerAngles2rotationMat(theta, format=None):
    if format == 'degree':
        theta = [i * math.pi / 180.0 for i in theta]
 
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
 
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
 
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        print("warning: singular rotation matrix")
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


def rand_range(rng, lo, hi):
    return rng.rand()*(hi-lo)+lo

def gaussian_noise(rng, img, sigma):
    """add gaussian noise of given sigma to image"""
    img = img + rng.randn(*img.shape) * sigma
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def linear_motion_blur(img, angle, length):
    """:param angle: in degree"""
    rad = np.deg2rad(angle)
    dx = np.cos(rad)
    dy = np.sin(rad)
    a = int(max(list(map(abs, (dx, dy)))) * length * 2)
    if a <= 0:
        return img
    kern = np.zeros((a, a))
    cx, cy = a // 2, a // 2
    dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
    cv2.line(kern, (cx, cy), (dx, dy), 1.0)
    s = kern.sum()
    if s == 0:
        kern[cx, cy] = 1.0
    else:
        kern /= s
    return cv2.filter2D(img, -1, kern)

def rgb_add_noise(img):
    rng = np.random
    # apply HSV augmentor
    if rng.rand() > 0:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint16)
        hsv_img[:, :, 1] = hsv_img[:, :, 1] * rand_range(rng, 1.25, 1.45)
        hsv_img[:, :, 2] = hsv_img[:, :, 2] * rand_range(rng, 1.15, 1.35)
        hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
        hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
        img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if rng.rand() > .8:  # sharpen
        kernel = -np.ones((3, 3))
        kernel[1, 1] = rng.rand() * 3 + 9
        kernel /= kernel.sum()
        img = cv2.filter2D(img, -1, kernel)

    if rng.rand() > 0.8:  # motion blur
        r_angle = int(rng.rand() * 360)
        r_len = int(rng.rand() * 15) + 1
        img = linear_motion_blur(img, r_angle, r_len)

    if rng.rand() > 0.8:
        if rng.rand() > 0.2:
            img = cv2.GaussianBlur(img, (3, 3), rng.rand())
        else:
            img = cv2.GaussianBlur(img, (5, 5), rng.rand())

    if rng.rand() > 0.2:
        img = gaussian_noise(rng, img, rng.randint(15))
    else:
        img = gaussian_noise(rng, img, rng.randint(25))

    if rng.rand() > 0.8:
        img = img + np.random.normal(loc=0.0, scale=7.0, size=img.shape)

    return np.clip(img, 0, 255).astype(np.uint8)

def load_depth(img_path):
    """ Load depth image from img_path. """
    depth_path = img_path + '_depth.png'
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16

def load_composed_depth(img_path):
    """ Load depth image from img_path. """
    img_path_ = img_path.replace('/data/camera/', '/data/camera_full_depths/')
    depth_path = img_path_ + '_composed.png'
    if os.path.exists(depth_path):
        depth = cv2.imread(depth_path, -1)
        if len(depth.shape) == 3:
            # This is encoded depth image, let's convert
            # NOTE: RGB is actually BGR in opencv
            depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
            depth16 = np.where(depth16==32001, 0, depth16)
            depth16 = depth16.astype(np.uint16)
        elif len(depth.shape) == 2 and depth.dtype == 'uint16':
            depth16 = depth
        else:
            assert False, '[ Error ]: Unsupported depth type.'
        return depth16
    else:
        print("warning: No data")
        return None


def get_bbox(bbox):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    img_width = 480
    img_length = 640
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)


def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map


def fill_in_multiscale_old(depth_map, max_depth=3.0,
                       dilation_kernel_far=CROSS_KERNEL_3,
                       dilation_kernel_med=CROSS_KERNEL_5,
                       dilation_kernel_near=CROSS_KERNEL_7,
                       extrapolate=False,
                       blur_type='bilateral',
                       show_process=False):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for 1.5 < depths < 3.0 m
        dilation_kernel_med: dilation kernel to use for 0.75 < depths < 1.5 m
        dilation_kernel_near: dilation kernel to use for 0.01 < depths < 0.75 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict

    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """

    # Convert to float32
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    valid_pixels_near = (depths_in > 0.01) & (depths_in <= 0.75)
    valid_pixels_med = (depths_in > 0.75) & (depths_in <= 1.5)
    valid_pixels_far = (depths_in > 1.5)

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = (s1_inverted_depths > 0.01)
    s1_inverted_depths[valid_pixels] = \
        max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far),
        dilation_kernel_far)
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med),
        dilation_kernel_med)
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near),
        dilation_kernel_near)

    # Find valid pixels for each binned dilation
    valid_pixels_near = (dilated_near > 0.01)
    valid_pixels_med = (dilated_med > 0.01)
    valid_pixels_far = (dilated_far > 0.01)

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = (s3_closed_depths > 0.01)
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=np.bool_)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.01)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = (s4_blurred_depths > 0.01)
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=np.bool_)

    top_row_pixels = np.argmax(s5_dilated_depths > 0.01, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels,
                                         range(s5_dilated_depths.shape[1])]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0:top_row_pixels[pixel_col_idx],
                               pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.01) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.01) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == 'gaussian':
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.01) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == 'bilateral':
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.01)
    s8_inverted_depths[valid_pixels] = \
        max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict['s0_depths_in'] = depths_in

        process_dict['s1_inverted_depths'] = s1_inverted_depths
        process_dict['s2_dilated_depths'] = s2_dilated_depths
        process_dict['s3_closed_depths'] = s3_closed_depths
        process_dict['s4_blurred_depths'] = s4_blurred_depths
        process_dict['s5_combined_depths'] = s5_dilated_depths
        process_dict['s6_extended_depths'] = s6_extended_depths
        process_dict['s7_blurred_depths'] = s7_blurred_depths
        process_dict['s8_inverted_depths'] = s8_inverted_depths

        process_dict['s9_depths_out'] = depths_out

    return depths_out, process_dict


def fill_in_multiscale(depth_map, max_depth=8.0,
                       dilation_kernel_far=CROSS_KERNEL_3,
                       dilation_kernel_med=CROSS_KERNEL_5,
                       dilation_kernel_near=CROSS_KERNEL_7,
                       extrapolate=False,
                       blur_type='bilateral',
                       show_process=False):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for 30.0 < depths < 80.0 m
        dilation_kernel_med: dilation kernel to use for 15.0 < depths < 30.0 m
        dilation_kernel_near: dilation kernel to use for 0.1 < depths < 15.0 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict

    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """

    # Convert to float32
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    valid_pixels_near = (depths_in > 0.01) & (depths_in <= 1.0)
    valid_pixels_med = (depths_in > 1.0) & (depths_in <= 2.0)
    valid_pixels_far = (depths_in > 2.0)

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = (s1_inverted_depths > 0.01)
    s1_inverted_depths[valid_pixels] = \
        max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far),
        dilation_kernel_far)
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med),
        dilation_kernel_med)
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near),
        dilation_kernel_near)

    # Find valid pixels for each binned dilation
    valid_pixels_near = (dilated_near > 0.01)
    valid_pixels_med = (dilated_med > 0.01)
    valid_pixels_far = (dilated_far > 0.01)

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = (s3_closed_depths > 0.01)
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=np.bool_)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.01)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = (s4_blurred_depths > 0.01)
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=np.bool_)

    top_row_pixels = np.argmax(s5_dilated_depths > 0.01, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels,
                                         range(s5_dilated_depths.shape[1])]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0:top_row_pixels[pixel_col_idx],
                               pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.01) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.01) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == 'gaussian':
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.01) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == 'bilateral':
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.01)
    s8_inverted_depths[valid_pixels] = \
        max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict['s0_depths_in'] = depths_in

        process_dict['s1_inverted_depths'] = s1_inverted_depths
        process_dict['s2_dilated_depths'] = s2_dilated_depths
        process_dict['s3_closed_depths'] = s3_closed_depths
        process_dict['s4_blurred_depths'] = s4_blurred_depths
        process_dict['s5_combined_depths'] = s5_dilated_depths
        process_dict['s6_extended_depths'] = s6_extended_depths
        process_dict['s7_blurred_depths'] = s7_blurred_depths
        process_dict['s8_inverted_depths'] = s8_inverted_depths

        process_dict['s9_depths_out'] = depths_out

    return depths_out, process_dict

# vim: ts=4 sw=4 sts=4 expandtab

def fill_missing(
        dpt, cam_scale, scale_2_80m, fill_type='multiscale',
        extrapolate=False, show_process=False, blur_type='bilateral'
):
    dpt = dpt / cam_scale * scale_2_80m
    projected_depth = dpt.copy()
    if fill_type == 'fast':
        final_dpt = fill_in_fast(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            # max_depth=2.0
        )
    elif fill_type == 'multiscale':
        final_dpt, process_dict = fill_in_multiscale(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            show_process=show_process,
            max_depth=3.0
        )
    else:
        raise ValueError('Invalid fill_type {}'.format(fill_type))
    dpt = final_dpt / scale_2_80m * cam_scale
    return dpt
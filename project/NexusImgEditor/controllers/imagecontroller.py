import os
import shutil
import uuid

import cv2
import delayed
import torchvision.transforms as transforms
from skimage import restoration, io, color, util, img_as_ubyte, data, img_as_float
from scipy.signal import convolve2d as conv2
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from flask import request, send_file, Blueprint
from skimage import restoration
from skimage.color import rgb2gray, rgba2rgb
from skimage.restoration import rolling_ball
from sympy.physics.control import Parallel

UPLOAD_DIRECTORY = str(Path("upload_dir"))
image_api = Blueprint('image_api', __name__)


@image_api.route("/image/filter", methods=['POST', 'GET'])
def apply_image_filter():
    filter_type = request.args.get('filter_type', '').lower()

    # Check if the filter type is valid
    supported_filters = (
        'rolling_ball', 'clahe', 'histogram_equalization',
        'sharpness', 'brightness', 'contrast', 'color',
        'gaussian_blur', 'box_blur', 'median_blur', 'bilateral_filter',
        'richardson_lucy', 'wiener_filter', "denoise_nl_means",
        'denoise_bilateral', 'denoise_tv_chambolle', 'denoise_wavelet',
        'sobel_edge_detection', 'laplacian_edge_detection', 'canny_edge_detection'
    )

    if filter_type not in supported_filters:
        return f'Invalid filter type. Supported filters: {", ".join(supported_filters)}', 400

    image_file = request.files.get('file')

    # Check if a file is uploaded
    if not image_file:
        return 'No file uploaded!', 400

    # Check if the file is a valid image (PNG or JPG)
    if not is_valid_image_file(image_file):
        return 'Only PNG or JPG images are allowed!', 400

    work_dir = generate_work_dir()
    file_path = save_file_to_work_dir(image_file, work_dir)
    # Get filter parameters from request (as keyword arguments)
    filter_params = get_filter_parameters(request.args)

    # Apply the selected filter with parameters
    filtered_image_path = apply_filter(file_path, filter_type, **filter_params)

    file_response = create_file_response(filtered_image_path, filter_type)

    # Cleanup the work directory
    shutil.rmtree(work_dir)

    return file_response


@image_api.route("/image/convert-to-jpg", methods=['POST', 'GET'])
def convert_to_jpg():
    image_file = request.files.get('file')

    # Check if a file is uploaded
    if not image_file:
        return 'No file uploaded!', 400

    # Check if the file is a valid PNG image
    if not is_valid_image_file(image_file):
        return 'Only PNG images are allowed!', 400

    work_dir = generate_work_dir()
    file_path = save_file_to_work_dir(image_file, work_dir)

    # Convert the image to JPG format
    jpg_image_path = convert_to_jpg_format(file_path)

    file_response = create_file_response(jpg_image_path, 'jpg')

    # Cleanup the work directory
    shutil.rmtree(work_dir)

    return file_response


def is_valid_image_file(image_file):
    file_extension = image_file.filename.rsplit('.', 1)[-1].lower()
    return file_extension in ('png', 'jpg', 'jpeg')


def save_file_to_work_dir(file, work_dir):
    file_path = os.path.join(work_dir, file.filename)
    file.save(file_path)
    return file_path


def create_file_response(file_path, filter_type):
    new_file_name = os.path.basename(file_path)
    return send_file(file_path, download_name=new_file_name)


def generate_work_dir():
    path = os.path.join(UPLOAD_DIRECTORY, str(uuid.uuid4()))
    os.makedirs(path, exist_ok=True)
    return path


def get_filter_parameters(request_args):
    # Extract filter parameters from request args
    filter_params = {}

    for key, value in request_args.items():
        if key != 'filter_type':
            try:
                # Try to convert the value to float or int
                filter_params[key] = float(value) if '.' in value else int(value)
            except ValueError:
                # If conversion fails, check for boolean values
                if value.lower() == 'true':
                    filter_params[key] = True
                elif value.lower() == 'false':
                    filter_params[key] = False
                else:
                    # Check if the value can be parsed as a tuple
                    try:
                        filter_params[key] = tuple(map(eval, value.strip('()').split(',')))
                    except (ValueError, SyntaxError):
                        filter_params[key] = value  # Use the original value if parsing fails
                    except Exception as e:
                        # Handle any other exceptions as needed
                        print(f"An error occurred while parsing tuple for key '{key}': {e}")
                        filter_params[key] = value

    return filter_params


def apply_filter(image_path, filter_type, **kwargs):
    print(f"[processing_img]: {image_path}")
    im = Image.open(image_path)

    # Choose the appropriate filter based on filter_type
    if filter_type == 'sharpness':
        im = ImageEnhance.Sharpness(im).enhance(**kwargs)
    elif filter_type == 'contrast':
        im = ImageEnhance.Contrast(im).enhance(**kwargs)
    elif filter_type == 'brightness':
        im = ImageEnhance.Brightness(im).enhance(**kwargs)
    elif filter_type == 'color':
        im = ImageEnhance.Color(im).enhance(**kwargs)
    elif filter_type == 'rolling_ball':
        im = rolling_ball_filter(im, **kwargs)
    elif filter_type == 'clahe':
        im = apply_clahe(im, **kwargs)
    elif filter_type == 'histogram_equalization':
        im = apply_equalize_hist(im)
    elif filter_type == 'gaussian_blur':
        im = apply_gaussian_blur(im, **kwargs)
    elif filter_type == 'box_blur':
        im = apply_box_blur(im, **kwargs)
    elif filter_type == 'median_blur':
        im = apply_median_blur(im, **kwargs)
    elif filter_type == 'bilateral_filter':
        im = apply_bilateral_filter(im)
    elif filter_type == 'richardson_lucy':
        im = apply_richardson_lucy_filter(im, **kwargs)
    elif filter_type == 'wiener_filter':
        im = apply_wiener_filter(im, **kwargs)
    elif filter_type == 'denoise_nl_means':
        im = apply_denoise_nl_means(im, **kwargs)
    elif filter_type == 'denoise_bilateral':
        im = apply_denoise_bilateral(im, **kwargs)
    elif filter_type == 'denoise_tv_bregman':
        im = apply_denoise_tv_bregman(im, **kwargs)
    elif filter_type == 'denoise_tv_chambolle':
        im = apply_denoise_tv_chambolle(im, **kwargs)
    elif filter_type == 'denoise_wavelet':
        im = apply_denoise_wavelet(im, **kwargs)
    elif filter_type == 'sobel_edge_detection':
        im = apply_sobel_edge_detection(im, **kwargs)
    elif filter_type == 'laplacian_edge_detection':
        im = apply_laplacian_edge_detection(im, **kwargs)
    elif filter_type == 'canny_edge_detection':
        im = apply_canny_edge_detection(im, **kwargs)

    original_extension = Path(image_path).suffix
    filtered_image_path = str(Path(image_path).with_name(f'{Path(image_path).stem}{original_extension}'))
    print(f"[saving_img]: {filtered_image_path}")
    im.save(filtered_image_path)

    return filtered_image_path


def convert_to_jpg_format(image_path):
    im = Image.open(image_path)

    # Convert the image to JPG format
    jpg_image_path = str(Path(image_path).with_suffix('_converted_to_jpg.jpg'))
    im.convert('RGB').save(jpg_image_path, 'JPEG')

    return jpg_image_path


def rolling_ball_filter(img, radius=5, invert=True):
    print(f"[rolling_ball_filter]: radius: {radius}, invert: {invert}")

    image = util.img_as_float64(img)
    # Check if the image has an alpha channel (four channels)
    has_alpha_channel = image.shape[-1] == 4

    if has_alpha_channel:
        # Convert RGBA to RGB
        image = rgba2rgb(image)

    # Convert the image to grayscale
    gray_image = rgb2gray(image)

    normalized_radius = radius / 255
    kernel = restoration.ellipsoid_kernel(
        (radius * 2, radius * 2),
        normalized_radius * 2
    )
    new_img = restoration.rolling_ball(gray_image, radius=radius, kernel=kernel)

    if invert:
        # Normalize the grayscale image to [0, 1] before inverting
        gray_image_normalized = util.img_as_float(new_img)
        inverted_image = util.invert(gray_image_normalized)
        new_img = gray_image_normalized - inverted_image

    # Normalize the image
    img_n = util.img_as_ubyte(new_img)

    # Convert the NumPy array to a Pillow Image
    pillow_image = Image.fromarray(img_n)

    # Convert back to RGBA if the original image had an alpha channel
    pillow_image = pillow_image.convert("RGB")
    if has_alpha_channel:
        pillow_image = pillow_image.convert("RGBA")

    return pillow_image


def apply_clahe(img, clip_limit=2.0, grid_size=(8, 8)):
    print(f'[clahe]: clip limit: {clip_limit}, grid size: {grid_size}')
    # Convert the Pillow Image to a NumPy array
    image_array = np.array(img)

    # Ensure the image is in grayscale (single channel)
    if len(image_array.shape) == 3:
        # Convert to grayscale if the image has multiple channels
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    clahe_image = clahe.apply(image_array)

    # Convert the result to a Pillow Image
    pil_result = Image.fromarray(clahe_image)
    pil_result = pil_result.convert("RGB")
    return pil_result


def apply_equalize_hist(pil_image):
    # Convert the Pillow Image to a NumPy array
    image_array = np.array(pil_image)

    # Ensure the image is in grayscale (single channel)
    if len(image_array.shape) == 3:
        # Convert to grayscale if the image has multiple channels
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(image_array)

    # Convert the result to a Pillow Image
    pil_result = Image.fromarray(equalized_image)
    pil_result = pil_result.convert("RGB")
    return pil_result


def apply_gaussian_blur(img, kernel_size=(5, 5), sigma=0):
    # Convert the Pillow Image to a NumPy array
    image_array = np.array(img)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image_array, kernel_size, sigma, borderType=cv2.BORDER_DEFAULT)

    # Convert the result to a Pillow Image
    pil_result = Image.fromarray(blurred_image)

    return pil_result


def apply_box_blur(img, kernel_size=(5, 5)):
    # Convert the Pillow Image to a NumPy array
    image_array = np.array(img)

    # Apply box blur using cv2.blur
    blurred_image = cv2.blur(image_array, kernel_size)

    # Convert the result to a Pillow Image
    pil_result = Image.fromarray(blurred_image)

    return pil_result


def apply_median_blur(img, kernel_size=5):
    # Convert the Pillow Image to a NumPy array
    image_array = np.array(img)

    # Apply median blur using cv2.medianBlur
    blurred_image = cv2.medianBlur(image_array, kernel_size)

    # Convert the result to a Pillow Image
    pil_result = Image.fromarray(blurred_image)

    return pil_result


def apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    # Convert the Pillow Image to a NumPy array
    image_array = np.array(img)

    # Apply bilateral filter using cv2.bilateralFilter
    filtered_image = cv2.bilateralFilter(image_array, d, sigma_color, sigma_space)

    # Convert the result to a Pillow Image
    pil_result = Image.fromarray(filtered_image)

    return pil_result


def apply_richardson_lucy_filter(img, psf_size=(5, 5), num_iter=50, clip=True, sigma=1.0):
    # Convert the Pillow Image to a NumPy array
    image_array = np.array(img)

    # Check if the image has an alpha channel (four channels)
    has_alpha_channel = image_array.shape[-1] == 4

    if has_alpha_channel:
        # Convert RGBA to RGB
        image_array = rgba2rgb(image_array)

    # Convert the image to grayscale if it's RGB
    if len(image_array.shape) == 3:
        image_array = rgb2gray(image_array)

    # Calculate the PSF (Point Spread Function)
    psf = calculate_gaussian_psf(image_array.shape, sigma=sigma, psf_size=psf_size)
    print(f"psf shape: {psf}")
    # Check if the size of the PSF is compatible with the image size
    if any(size_img < size_psf for size_img, size_psf in zip(image_array.shape, psf.shape)):
        raise ValueError("The size of the PSF should be compatible with the size of the image.")

    # Convolve the image with the PSF
    img_convolved = conv2(image_array, psf, 'same')

    # Apply Richardson-Lucy deconvolution
    restored_image = restoration.richardson_lucy(img_convolved, psf=psf, num_iter=num_iter, clip=clip)

    # Normalize the result to [0, 255]
    restored_image_normalized = cv2.normalize(
        src=restored_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Convert the result back to a Pillow Image
    pil_result = Image.fromarray(restored_image_normalized)

    # Convert back to RGBA if the original image had an alpha channel
    pil_result = pil_result.convert("RGB")
    if has_alpha_channel:
        pil_result = pil_result.convert("RGBA")

    return pil_result


def apply_wiener_filter(img, psf_size=(5, 5), noise_variance=0.1, clip=True, sigma=1.0):
    # Convert the Pillow Image to a NumPy array
    image_array = np.array(img)

    # Check if the image has an alpha channel (four channels)
    has_alpha_channel = image_array.shape[-1] == 4

    if has_alpha_channel:
        # Convert RGBA to RGB
        image_array = rgba2rgb(image_array)

    # Convert the image to grayscale if it's RGB
    if len(image_array.shape) == 3:
        image_array = rgb2gray(image_array)

    # Calculate the PSF (Point Spread Function)
    psf = calculate_gaussian_psf(image_array.shape, psf_size=psf_size, sigma=sigma)

    # Check if the size of the PSF is compatible with the image size
    if any(size_img < size_psf for size_img, size_psf in zip(image_array.shape, psf.shape)):
        raise ValueError("The size of the PSF should be compatible with the size of the image.")

    # Convolve the image with the PSF
    img_convolved = conv2(image_array, psf, 'same', boundary='symm')

    # Apply Wiener filter
    restored_image = restoration.wiener(img_convolved, psf, noise_variance, clip=clip)

    # Normalize the result to [0, 255]
    restored_image_normalized = cv2.normalize(
        src=restored_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Convert the result back to a Pillow Image
    pil_result = Image.fromarray(restored_image_normalized)

    # Convert back to RGBA if the original image had an alpha channel
    pil_result = pil_result.convert("RGB")
    if has_alpha_channel:
        pil_result = pil_result.convert("RGBA")

    return pil_result


def apply_denoise_nl_means(img, sigma=0.08, h=0.1, patch_size=7, patch_distance=11, fast_mode=True):
    # Convert the Pillow Image to a NumPy array
    image_array = np.array(img)

    # Check if the image has an alpha channel (four channels)
    has_alpha_channel = image_array.shape[-1] == 4

    if has_alpha_channel:
        # Convert RGBA to RGB
        image_array = rgba2rgb(image_array)

    # Convert the image to grayscale if it's RGB
    if len(image_array.shape) == 3:
        image_array = rgb2gray(image_array)

    # Denoise the image using Non-Local Means filter
    denoised_image = restoration.denoise_nl_means(image_array, patch_size=patch_size, patch_distance=patch_distance,
                                                  sigma=sigma, h=h, fast_mode=fast_mode)

    # Convert the result back to a Pillow Image
    pil_result = Image.fromarray(img_as_ubyte(denoised_image))

    # Convert back to RGBA if the original image had an alpha channel
    pil_result = pil_result.convert("RGB")
    if has_alpha_channel:
        pil_result = pil_result.convert("RGBA")

    return pil_result


def apply_denoise_bilateral(img, sigma_color=0.1, sigma_spatial=15):
    # Convert the Pillow Image to a NumPy array
    image_array = np.array(img)

    # Check if the image has an alpha channel (four channels)
    has_alpha_channel = image_array.shape[-1] == 4

    if has_alpha_channel:
        # Convert RGBA to RGB
        image_array = rgba2rgb(image_array)

    # Convert the image to grayscale if it's RGB
    if len(image_array.shape) == 3:
        image_array = rgb2gray(image_array)

    # Denoise the image using Bilateral filter
    denoised_image = restoration.denoise_bilateral(
        image_array, sigma_color=sigma_color, sigma_spatial=sigma_spatial
    )

    # Convert the result back to a Pillow Image
    pil_result = Image.fromarray(img_as_ubyte(denoised_image))

    # Convert back to RGBA if the original image had an alpha channel
    pil_result = pil_result.convert("RGB")
    if has_alpha_channel:
        pil_result = pil_result.convert("RGBA")

    return pil_result


def apply_denoise_tv_bregman(img, weight=0.1, max_iter=100, epsilon=0.001, isotropic=True):
    # Convert the Pillow Image to a NumPy array
    image_array = np.array(img)

    # Check if the image has an alpha channel (four channels)
    has_alpha_channel = image_array.shape[-1] == 4

    if has_alpha_channel:
        # Convert RGBA to RGB
        image_array = rgba2rgb(image_array)

    # Convert the image to grayscale if it's RGB
    if len(image_array.shape) == 3:
        image_array = rgb2gray(image_array)

    # Denoise the image using TV Bregman algorithm
    denoised_image = restoration.denoise_tv_bregman(
        image_array, weight=weight, max_iter=max_iter, epsilon=epsilon, isotropic=isotropic
    )

    # Convert the result back to a Pillow Image
    pil_result = Image.fromarray(img_as_ubyte(denoised_image))

    # Convert back to RGBA if the original image had an alpha channel
    pil_result = pil_result.convert("RGB")
    if has_alpha_channel:
        pil_result = pil_result.convert("RGBA")

    return pil_result


def apply_denoise_tv_chambolle(img, weight=0.1, max_iter=100):
    # Convert the Pillow Image to a NumPy array
    image_array = np.array(img)

    # Check if the image has an alpha channel (four channels)
    has_alpha_channel = image_array.shape[-1] == 4

    if has_alpha_channel:
        # Convert RGBA to RGB
        image_array = rgba2rgb(image_array)

    # Convert the image to grayscale if it's RGB
    if len(image_array.shape) == 3:
        image_array = rgb2gray(image_array)

    # Denoise the image using TV Chambolle algorithm
    denoised_image = restoration.denoise_tv_chambolle(
        image_array, weight=weight, max_num_iter=max_iter
    )

    # Convert the result back to a Pillow Image
    pil_result = Image.fromarray(img_as_ubyte(denoised_image))

    # Convert back to RGBA if the original image had an alpha channel
    pil_result = pil_result.convert("RGB")
    if has_alpha_channel:
        pil_result = pil_result.convert("RGBA")

    return pil_result


def apply_denoise_wavelet(img, wavelet='db1', sigma=None):
    # Convert the Pillow Image to a NumPy array
    image_array = np.array(img)

    # Check if the image has an alpha channel (four channels)
    has_alpha_channel = image_array.shape[-1] == 4

    if has_alpha_channel:
        # Convert RGBA to RGB
        image_array = rgba2rgb(image_array)

    # Convert the image to grayscale if it's RGB
    if len(image_array.shape) == 3:
        image_array = rgb2gray(image_array)

    # Denoise the image using wavelet denoising
    denoised_image = restoration.denoise_wavelet(
        image_array, wavelet=wavelet, sigma=sigma
    )

    # Convert the result back to a Pillow Image
    pil_result = Image.fromarray(img_as_ubyte(denoised_image))

    # Convert back to RGBA if the original image had an alpha channel
    pil_result = pil_result.convert("RGB")
    if has_alpha_channel:
        pil_result = pil_result.convert("RGBA")

    return pil_result


def apply_sobel_edge_detection(img, ksize=3, scale=1.0, delta=0):
    # Read the image
    image_array = np.array(img)
    # Check if the image has an alpha channel (four channels)
    has_alpha_channel = image_array.shape[-1] == 4

    if has_alpha_channel:
        # Convert RGBA to RGB
        image_array = rgba2rgb(image_array)
    # Convert the image to grayscale if it's RGB
    if len(image_array.shape) == 3:
        image_array = rgb2gray(image_array)
    # Apply Sobel filter
    sobel_x = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale, delta=delta)
    sobel_y = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale, delta=delta)

    # Combine the results to get the magnitude of the gradient
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Convert the result to uint8 for display
    magnitude = cv2.convertScaleAbs(magnitude)

    # Convert to Pillow Image for display or further processing
    pil_image = Image.fromarray(magnitude)
    # Convert back to RGBA if the original image had an alpha channel
    pil_result = pil_image.convert("RGB")
    if has_alpha_channel:
        pil_result = pil_result.convert("RGBA")
    return pil_result


def apply_laplacian_edge_detection(img, ksize=3, scale=1.0, delta=0):
    # Read the image
    image_array = np.array(img)
    # Check if the image has an alpha channel (four channels)
    has_alpha_channel = image_array.shape[-1] == 4

    if has_alpha_channel:
        # Convert RGBA to RGB
        image_array = rgba2rgb(image_array)
    # Convert the image to grayscale if it's RGB
    if len(image_array.shape) == 3:
        image_array = rgb2gray(image_array)
    # Apply Laplacian filter
    laplacian = cv2.Laplacian(image_array, cv2.CV_64F, ksize=ksize, scale=scale, delta=delta)

    # Convert the result to uint8 for display
    laplacian = cv2.convertScaleAbs(laplacian)

    # Convert to Pillow Image for display or further processing
    pil_image = Image.fromarray(laplacian)

    # Convert back to RGBA if the original image had an alpha channel
    pil_result = pil_image.convert("RGB")
    if has_alpha_channel:
        pil_result = pil_result.convert("RGBA")
    return pil_result


def apply_canny_edge_detection(img, low_threshold=100, high_threshold=200, aperture_size=5, l2gradient=True):
    # Read the image
    image_array = np.array(img)
    # Check if the image has an alpha channel (four channels)
    has_alpha_channel = image_array.shape[-1] == 4

    if has_alpha_channel:
        # Convert RGBA to RGB
        image_array = rgba2rgb(image_array)
    # Convert the image to grayscale if it's RGB
    if len(image_array.shape) == 3:
        image_array = rgb2gray(image_array)

    if image_array.dtype != np.uint8:
        # Convert the image to 8-bit unsigned integer
        image_array = cv2.convertScaleAbs(image_array)
    # Apply Canny edge detector
    edges = cv2.Canny(image_array, threshold1=low_threshold, threshold2=high_threshold, apertureSize=aperture_size, L2gradient=l2gradient)

    # Convert the result to uint8 for display
    edges = cv2.convertScaleAbs(edges)

    # Convert to Pillow Image for display or further processing
    pil_image = Image.fromarray(edges)

    # Convert back to RGBA if the original image had an alpha channel
    pil_result = pil_image.convert("RGB")
    if has_alpha_channel:
        pil_result = pil_result.convert("RGBA")
    return pil_result


def calculate_gaussian_psf(image_shape, sigma=1.0, psf_size=(5, 5)):
    # Calculate a 2D Gaussian PSF
    x, y = np.meshgrid(np.linspace(-1, 1, psf_size[0]), np.linspace(-1, 1, psf_size[1]))
    d = np.sqrt(x * x + y * y)
    psf = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    psf /= np.sum(psf)

    # Resize the PSF to match the specified image shape
    psf_shape = (psf_size[0] * image_shape[0] // max(image_shape), psf_size[1] * image_shape[1] // max(image_shape))
    psf = cv2.resize(psf, psf_shape, interpolation=cv2.INTER_LINEAR)

    return psf

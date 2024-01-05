from pathlib import Path
from pydicom import dcmread
from PIL import Image


# creates a new image file with file extension of convert_to from a dicom file
def dicomfile_to_imagefile(dcmfile_path, convert_to):
    # converts the path to postix format
    dcmfile_path = dcmfile_path.replace("\\", "/")
    # Read the dcm file
    im = Image.fromarray(dcmread(dcmfile_path).pixel_array)
    # Convert each 16bit pixel value into 8bit. each RGB channel sets to the new value
    im = im.point(lambda luminance: luminance*(255/65535))
    im = im.convert('RGB')
    # Save as .jpg and .png
    im.save(Path(dcmfile_path.replace(".dcm", "."+convert_to)))

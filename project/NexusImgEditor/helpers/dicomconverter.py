from pathlib import Path
from pydicom import dcmread
from PIL import Image

def dicomfile_to_imagefile(dcmfile_path, convert_to):
    # Convert the path to Posix format
    dcmfile_path = Path(dcmfile_path).resolve().as_posix()

    try:
        # Read the DICOM file
        dicom_data = dcmread(dcmfile_path)
        
        # Create an image from the pixel array
        im = Image.fromarray(dicom_data.pixel_array)
        
        # Convert pixel values to 8-bit
        im = im.point(lambda luminance: luminance * (255 / 65535))
        
        # Convert to RGB mode
        im = im.convert('RGB')

        # Save as the specified format
        output_path = Path(dcmfile_path.replace(".dcm", f".{convert_to}"))
        im.save(output_path)

        return output_path.as_posix()

    except Exception as e:
        # Handle exceptions (e.g., file reading or image processing errors)
        print(f"Error converting DICOM to {convert_to}: {e}")
        return None

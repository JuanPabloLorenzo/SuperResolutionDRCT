import os
import math
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Added for progress display
import concurrent.futures  # Added for parallel processing

def get_image_paths(root_dir, train_ratio):
    image_paths = []

    # Collect all image paths
    for car_dir in os.listdir(root_dir):
        car_dir_path = os.path.join(root_dir, car_dir)
        if os.path.isdir(car_dir_path):
            for img_name in os.listdir(car_dir_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(car_dir_path, img_name)
                    image_paths.append(img_path)

    # Split image paths into training and validation sets
    train_images, val_images = train_test_split(image_paths[:500], train_size=train_ratio, random_state=42)

    return train_images, val_images

def prepare_crops(
    root_dir="downloaded_images",
    crop_size=128,
    train_ratio=0.9,
    train_dir="train",
    val_dir="val",
    output_dir="cropped_images"
):
    """
    Splits images into training and validation sets, then crops each image into overlapping fixed-size patches
    to ensure full coverage. All crops from a single image are kept in the same set.

    Args:
        root_dir (str): Directory containing the original images organized in subdirectories.
        crop_size (int): Size of each square crop (e.g., 128 for 128x128).
        train_ratio (float): Ratio of images to be used for training.
        train_dir (str): Subdirectory name for training crops.
        val_dir (str): Subdirectory name for validation crops.
        output_dir (str): Directory to save the cropped images.
    """

    # Create output directories
    train_path = os.path.join(output_dir, train_dir)
    val_path = os.path.join(output_dir, val_dir)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    train_images, val_images = get_image_paths(root_dir, train_ratio)

    def crop_and_save(image_path, save_dir):
        try:
            img = Image.open(image_path).convert('RGB')
            width, height = img.size

            # Define step size for 80% of crop size so there is a 20% overlap
            step = int(crop_size * 0.8)

            # Calculate number of crops needed in each dimension
            num_crops_x = math.ceil((width - crop_size) / step) + 1
            num_crops_y = math.ceil((height - crop_size) / step) + 1

            for y in range(num_crops_y):
                for x in range(num_crops_x):
                    left = x * step
                    upper = y * step
                    right = left + crop_size
                    lower = upper + crop_size

                    if right > width:
                        # Adjust if the crop exceeds image boundaries by more than 40%
                        if right < width + 0.4 * crop_size:
                            right = width
                            left = width - crop_size
                        else:
                            continue
                    if lower > height:
                        if lower < height + 0.4 * crop_size:
                            lower = height
                            upper = height - crop_size
                        else:
                            continue

                    crop = img.crop((left, upper, right, lower))
                    crop_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_crop_{upper}_{left}.png"
                    crop_path = os.path.join(save_dir, crop_filename)
                    crop.save(crop_path, format="JPEG")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    def process_images(image_paths, save_dir, desc):
        with tqdm(total=len(image_paths), desc=desc) as pbar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(crop_and_save, img_path, save_dir): img_path for img_path in image_paths}
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)

    # Process training images in parallel with progress bar
    process_images(train_images, train_path, "Processing training images")

    # Process validation images in parallel with progress bar
    process_images(val_images, val_path, "Processing validation images")

    # Count number of crops in each set
    num_train_crops = len(os.listdir(train_path))
    num_val_crops = len(os.listdir(val_path))

    print(f"Cropped images have been saved to '{output_dir}'.")
    print(f"Training set: {num_train_crops} crops")
    print(f"Validation set: {num_val_crops} crops")

if __name__ == "__main__":
    prepare_crops()

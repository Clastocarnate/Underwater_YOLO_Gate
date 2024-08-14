import os

def delete_images_without_labels(images_dir, labels_dir):
    # Get the list of image files and label files
    image_files = set(f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png'))
    label_files = set(f for f in os.listdir(labels_dir) if f.endswith('.txt'))

    # Convert label filenames to match the image filenames (without extension)
    label_basenames = {os.path.splitext(f)[0] for f in label_files}

    # Iterate through the image files and delete those without a corresponding label file
    for image_file in image_files:
        image_basename = os.path.splitext(image_file)[0]
        if image_basename not in label_basenames:
            image_path = os.path.join(images_dir, image_file)
            os.remove(image_path)
            print(f"Deleted: {image_path}")

# Example usage:
images_directory = "dataset"
labels_directory = "Conversion/labels"

delete_images_without_labels(images_directory, labels_directory)
import os
import xml.etree.ElementTree as ET

def convert_to_yolov8_format(xml_file, output_dir):
    try:
        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Iterate through each image in the XML file
        for image in root.iter('image'):
            image_filename = image.get('name')
            image_width = int(image.get('width'))
            image_height = int(image.get('height'))

            # Extract the base name (without extension) for the output file
            base_name = os.path.splitext(image_filename)[0]

            # Prepare the output file with only the base name
            output_file = os.path.join(output_dir, base_name + ".txt")

            with open(output_file, 'w') as f:
                # Iterate through each bounding box in the image
                for box in image.iter('box'):
                    class_name = box.get('label')

                    # YOLOv8 assumes class indices, you'll need a mapping from class names to indices
                    # Create a dictionary to map class names to IDs (manually specify this)
                    class_id_mapping = {
                        "Gate": 0,
                        # Add more classes as needed
                    }

                    class_id = class_id_mapping.get(class_name, None)
                    if class_id is None:
                        continue  # Skip unknown classes

                    # Get bounding box coordinates
                    xmin = float(box.get('xtl'))
                    ymin = float(box.get('ytl'))
                    xmax = float(box.get('xbr'))
                    ymax = float(box.get('ybr'))

                    # Convert to YOLOv8 format (normalized values)
                    x_center = (xmin + xmax) / 2.0 / image_width
                    y_center = (ymin + ymax) / 2.0 / image_height
                    width = (xmax - xmin) / image_width
                    height = (ymax - ymin) / image_height

                    # Write the YOLOv8 format to the output file
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    except ET.ParseError:
        print(f"Error parsing XML file: {xml_file}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def convert_all_xml_in_directory(input_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each XML file in the input directory
    for xml_file in os.listdir(input_dir):
        if xml_file.endswith('.xml'):
            convert_to_yolov8_format(os.path.join(input_dir, xml_file), output_dir)



# Example usage:
input_directory = "/Users/madhuupadhyay/Documents/Computer Vision/Deep-Monocular-SLAM/utils/Conversion"
output_directory = "/Users/madhuupadhyay/Documents/Computer Vision/Deep-Monocular-SLAM/utils/Conversion/labels"
convert_all_xml_in_directory(input_directory, output_directory)
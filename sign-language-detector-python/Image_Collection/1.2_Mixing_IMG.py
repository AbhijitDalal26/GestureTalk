import os
import shutil

# Define the source directories (a and b) and the destination directory
source_dir_a = r'D:\Collage Project\American Sign Language\ASL\sign-language-detector-python\MY_Data\a'
source_dir_b = r'D:\Collage Project\American Sign Language\ASL\sign-language-detector-python\MY_Data\b'
destination_dir = r'D:\Collage Project\American Sign Language\ASL\sign-language-detector-python\data'  # Use your destination directory

# Get the subdirectories (folders) within source directories a and b
subdirs_a = [os.path.join(source_dir_a, subdir) for subdir in os.listdir(source_dir_a) if os.path.isdir(os.path.join(source_dir_a, subdir))]
subdirs_b = [os.path.join(source_dir_b, subdir) for subdir in os.listdir(source_dir_b) if os.path.isdir(os.path.join(source_dir_b, subdir))]

# Combine images from corresponding subdirectories in a and b into destination directories
for subdir_a, subdir_b in zip(subdirs_a, subdirs_b):
    # Create a subdirectory in the destination directory with the same name
    subdir_name = os.path.basename(subdir_a)
    combined_subdir = os.path.join(destination_dir, subdir_name)
    os.makedirs(combined_subdir, exist_ok=True)

    # Copy images from subdirectory a to the combined subdirectory
    for filename in os.listdir(subdir_a):
        source_file_a = os.path.join(subdir_a, filename)
        destination_file = os.path.join(combined_subdir, filename)
        shutil.copy(source_file_a, destination_file)

    # Copy images from subdirectory b to the combined subdirectory
    for filename in os.listdir(subdir_b):
        source_file_b = os.path.join(subdir_b, filename)
        destination_file = os.path.join(combined_subdir, filename)
        shutil.copy(source_file_b, destination_file)

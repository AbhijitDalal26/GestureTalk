import os

# Path to the main directory containing the letter folders
main_dir = r'D:\Collage Project\American Sign Language\ASL\sign-language-detector-python\data'

# List all letter folders in the main directory
letter_folders = [f for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f))]

# Iterate through each letter folder and rename images sequentially
for letter_folder in letter_folders:
    letter_dir = os.path.join(main_dir, letter_folder)
    image_files = [f for f in os.listdir(letter_dir) if f.endswith('.jpg')]
    
    # Sort the image files to ensure sequential processing
    image_files.sort()
    
    # To change starting name of imag
    for i, image_file in enumerate(image_files):
        old_path = os.path.join(letter_dir, image_file)
        new_name = f'{i}.jpg'  #Here name get change 
        new_path = os.path.join(letter_dir, new_name)
        os.rename(old_path, new_path)
        print(f'Renamed {image_file} to {new_name} in folder {letter_folder}')

print('Renaming complete.')

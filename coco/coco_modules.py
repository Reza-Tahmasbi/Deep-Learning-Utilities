from pycocotools.coco import COCO
import requests
from tqdm import tqdm
import os
import zipfile


class my_coco():
    def __init__(self, dataDir: str, type: str = "2017"):
        self.type = type
        self.dataDir = dataDir
        self.annFile = None
        self.coco = None
        self.downloader = self.Downloader()
        
    
    # this function helps us to donwload datasets
    class Downloader():
        def __init__(self, year: list):
            self.url_list = self.specify_version(year)
            
        def specify_version(self, year: str)-> list:
            link = "http://images.cocodataset.org"
            train_url = f"{link}/zips/train{year}.zip"
            val_url = f"{link}/zips/val{year}.zip"
            annotation_url = f"{link}/annotations/annotations_trainval{year}.zip"
            return [train_url, val_url, annotation_url]
         
         
        def make_dir():
            # Specify the directory name
            dir_name = 'coco_content'
            sub_dirs = ['downloaded_content', 'extracted_content']
            # Check if the directory already exists
            if not os.path.exists(dir_name):
                # Create the directory
                os.makedirs(dir_name)
                print(f"Directory '{dir_name}' was created successfully.")
            else:
                print(f"Directory '{dir_name}' already exists.")
            
            for sub_dir in sub_dirs:
                path = os.path.join(dir_name, sub_dir)
                if not os.path.exists(path):
                # Create the subdirectory
                    os.makedirs(path)
                    print(f"Subdirectory '{sub_dir}' was created successfully in '{dir_name}'.")
                else:
                    print(f"Subdirectory '{sub_dir}' already exists in '{dir_name}'.")
    
    
        def download(self):
            self.make_dir()
            for url in self.url_list:
                response = requests.get(url, stream = True)
                file_size = int(response.headers.get('content-length', 0))
                chunk_size = 1024
                progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc='Downloading')
                file_name = url.split('/')[-1]
                with open(file_name, 'wb') as file:
                    for data in response.iter_content(chunk_size=chunk_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                print(f"{file_name} is downloaded successfully!")
            
            if file_size != 0 and progress_bar.n != file_size:
                print("ERROR, something went wrong!")
        
        
        def extractor():
            dir_name = 'coco_content'
            # Specify the path to your zip file
            zip_file_path = os.path.join(dir_name, "downloaded_content")
            # Specify the directory where you want to extract the contents
            extract_to_directory = os.path.join(dir_name, "extracted_content")
            # Open the zip file in read mode and extract all contents
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to_directory)
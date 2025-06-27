import shutil
import site
import os

def get_site_packages_path():
    return site.getsitepackages()[0]

def copy_files(files_to_copy):
    site_packages_path = get_site_packages_path()
    for src, dest in files_to_copy:
        full_dest_path = os.path.join(site_packages_path, dest)
        shutil.copyfile(src, full_dest_path)
        print(f"Copied {src} to {full_dest_path}")

def move_file(src, dest):
    site_packages_path = get_site_packages_path()
    full_dest_path = os.path.join(site_packages_path, dest)
    shutil.move(src, full_dest_path)
    print(f"Moved {src} to {full_dest_path}")

if __name__ == "__main__":
    files_to_copy = [
    ("./util/vision_transformer.py", "/root/anaconda3/envs/marllib/lib/python3.8/site-packages/timm/models/vision_transformer.py")
    ]

    copy_files(files_to_copy)
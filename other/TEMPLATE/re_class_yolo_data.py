import os

def process_file(file_path):
    """Process a single file to change the class to 0."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            parts = line.strip().split()
            if parts:
                parts[0] = '0'  # 修改类别为0
                file.write(' '.join(parts) + '\n')

def process_directory(directory_path):
    """Process all files in the given directory."""
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            process_file(file_path)

def main():
    # 假设相对路径是从当前工作目录出发
    train_dir_path = 'resource/datasets/labels/train'
    val_dir_path = 'resource/datasets/labels/val'

    # 检查目录是否存在
    if os.path.exists(train_dir_path) and os.path.isdir(train_dir_path):
        process_directory(train_dir_path)
    else:
        print(f"{train_dir_path} 目录不存在或不是一个有效的目录")

    if os.path.exists(val_dir_path) and os.path.isdir(val_dir_path):
        process_directory(val_dir_path)
    else:
        print(f"{val_dir_path} 目录不存在或不是一个有效的目录")

if __name__ == '__main__':
    main()
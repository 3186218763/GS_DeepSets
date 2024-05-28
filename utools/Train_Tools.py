import os


def find_train_dirs(base_dir):
    second_level_subdirectories = []

    # 第一层目录
    try:
        with os.scandir(base_dir) as first_level_entries:
            for first_level_entry in first_level_entries:
                if first_level_entry.is_dir():
                    # 第二层目录
                    try:
                        with os.scandir(first_level_entry.path) as second_level_entries:
                            for second_level_entry in second_level_entries:
                                if second_level_entry.is_dir():
                                    second_level_subdirectories.append(second_level_entry.path)
                    except PermissionError:
                        pass  # 忽略无权限访问的目录
    except PermissionError:
        pass  # 忽略无权限访问的目录

    return second_level_subdirectories

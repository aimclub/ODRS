import os
# from pprintpp import pprint as pp
import shutil
from tqdm import tqdm

# Входная директория, в которой расположены папки 'pad', 'tape'
BASE_DIR = 'output'
# Папки, в которых лежат фотографии и и txt файлы (можно добавить TYPE_GENERAL)
# и потом передать его в main()
TYPE_PAD, TYPE_TAPE = 'pad', 'tape'
# Директория, в которой будут лежать папки 'pad', 'tape' с удаленными лишними файлами
RES_DIR = 'res_dir'


def get_files_paths(paths_folders: tuple[str, ...]):
    """
    Собирает пути ко всем файлам, разделяет их по типу (txt или png)
    и затем формирует 2 словаря для txt и png отдельно.
    Словари типа:
    {имя файла без расширения: путь к файлу}
    """
    name_files = []
    for files in os.listdir(paths_folders):
        name_files.append(files)

    png_paths = tuple(filter(lambda x: '.png' in x, name_files))
    txt_paths = tuple(filter(lambda x: '.txt' in x, name_files))

    dir_png = {}
    for i in png_paths:
        dir_png[i.split('.')[0]] = os.path.join(paths_folders, i)

    #pp(dir_png)
    dir_txt = {}
    for i in txt_paths:
        dir_txt[i.split('.')[0]] = os.path.join(paths_folders, i)

    #pp(dir_txt)
    return dir_png, dir_txt


def filter_files(dir_png: dict, dir_txt: dict) -> list[str, ...]:
    """
    Сравнивает два слововаря по ключу. Если ключи совпадают,
    то значения словарей добавляются в результирующий список
    с путями к файлам
    """
    list_with_path = []
    for key, value in dir_png.items():
        if key in dir_txt.keys():
            list_with_path.extend([value, dir_txt[key]])
    print(list_with_path)
    return list_with_path


def copy_files(list_paths: list[str, ...], out_fold: str):
    """
    Копирует файлы по ссылкам из списка в назнаяенную директорию
    """
    for path in tqdm(list_paths):
        shutil.copy(path, out_fold)


def main():
    # Создаем пути к директориям с файлами
    paths_to_folders = os.path.join(BASE_DIR, TYPE_PAD), os.path.join(BASE_DIR, TYPE_TAPE)
    print(paths_to_folders)
    for folders in tqdm(paths_to_folders):
        # Создаем выходные директории, если их нет
        output_folder = os.path.join(RES_DIR, folders.split('/')[1])
        if not os.path.exists(RES_DIR):
            os.mkdir(RES_DIR)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        print(RES_DIR)
        print(folders)
        # Создаем словарики для png и txt файлов типа {имя файла без расширения: путь к файлу}
        dict_with_png, dict_with_txt = get_files_paths(folders)
        print(dict_with_png, dict_with_txt)
        # Возвращаем список с путями к файлам, которые необходимо будет скопировать
        list_with_paths_for_copy = filter_files(dict_with_png, dict_with_txt)
        # Добавляет в список classes.txt (нужен)
        list_with_paths_for_copy.append(os.path.join(folders, '../../classes.txt'))
        # Копируем файлы
        copy_files(list_with_paths_for_copy, output_folder)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

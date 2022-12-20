import os
import shutil
import zipfile
from dataclasses import dataclass
from typing import Literal
from tqdm import tqdm
import rarfile
from loguru import logger

# Пути к входным/ выходным директориям
TYPE_PAD, TYPE_TAPE, GENERAL = 'pad', 'tape', 'general'
INPUT_DIR_NAME, OUTPUT_DIR_NAME = 'input', 'output'
# ARCH_DIR_NAME = 'archives'
# Лишние файлы, которые создаются ОС
BAD_FILES = '__MACOSX', '.DS_Store'


@dataclass(frozen=True)
class ArchivesName:

    file_name: str
    path: str

    @property
    def archive_extension(self) -> Literal['zip', 'rar']:
        if self.file_name.endswith('.zip'):
            return 'zip'
        elif self.file_name.endswith('.rar'):
            return 'rar'

    @property
    def custom_type(self):
        if TYPE_PAD in self.path:
            return TYPE_PAD
        elif TYPE_TAPE in self.path:
            return TYPE_TAPE

    @property
    def output_name(self) -> str:
        return self.file_name.replace(self.archive_extension, '')


@dataclass(frozen=True)
class ArchivesPaths:
    zip: list[ArchivesName, ...]
    rar: list[ArchivesName, ...]

    @property
    def all(self):
        return self.zip + self.rar


def get_archives_paths(paths_to_archives: tuple[str, ...]) -> ArchivesPaths:
    archive_paths = []
    for path in paths_to_archives:
        current_archive_paths = []
        for archive_file in os.listdir(path):
            current_archive_paths.append(
                ArchivesName(
                    file_name=archive_file,
                    path=os.path.join(path, archive_file)
                )
            )

        archive_paths = [*archive_paths, *current_archive_paths]

    zip_paths = tuple(filter(lambda x: x.archive_extension == 'zip', archive_paths))
    rar_paths = tuple(filter(lambda x: x.archive_extension == 'rar', archive_paths))
    return ArchivesPaths(zip=zip_paths, rar=rar_paths)


def un_archive_archives(archives_paths: ArchivesPaths, destination_path: str = OUTPUT_DIR_NAME):
    for archive in tqdm(archives_paths.all):

        output_archive_path = os.path.join(destination_path, archive.custom_type, archive.output_name)
        if archive.archive_extension == 'zip':
            with zipfile.ZipFile(archive.path, 'r') as zip_ref:
                zip_ref.extractall(output_archive_path)

        elif archive.archive_extension == 'rar':
            with rarfile.RarFile(archive.path, 'r') as rar_ref:
                rar_ref.extractall(output_archive_path)


def file_filter(files: list[str, ...]) -> list[str, ...]:
    files = list(
        filter(
            lambda file:
                os.path.isfile(file) and all(('.DS_Store' not in file, '__MACOSX' not in file)),
            files
        )
    )

    files_string = ' '.join(files)
    if files_string.count('.txt') > 1:
        return files
    return []


def file_getter(folder_name: str) -> list[str, ...]:

    files_in_dir = os.listdir(folder_name)
    folders_in_current_dir = tuple(filter(
        lambda folder_: os.path.isdir(os.path.join(folder_name, folder_)) and folder_ not in BAD_FILES, files_in_dir
    ))

    if not folders_in_current_dir:
        return file_filter([os.path.join(folder_name, file_) for file_ in files_in_dir])

    files_in_current_dir = file_filter([os.path.join(folder_name, file_) for file_ in files_in_dir])
    for folder in folders_in_current_dir:
        files_in_current_dir += file_getter(os.path.join(folder_name, folder))

    return files_in_current_dir


def copy_files_to_folder(file_paths: list[str, ...], folder_path: str) -> None:
    for file_path in file_paths:
        try:
            shutil.copy(file_path, folder_path)
        except shutil.SameFileError:
            logger.warning(f'File {file_path} already exists in {folder_path}')
        except shutil.Error:
            logger.warning(f'Error while copying file {file_path} to {folder_path}')


def drop_folders_in_folder(folder_path: str) -> None:
    for folder in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, folder)):
            shutil.rmtree(os.path.join(folder_path, folder))


def get_all_dirs_from_folder(folder_path: str) -> tuple[str, ...]:
    return tuple(
        filter(lambda folder: os.path.isdir(os.path.join(folder_path, folder)), os.listdir(folder_path))
    )


def create_zip_archive(output_filename: str, dir_name: str) -> None:
    shutil.make_archive(output_filename, 'zip', dir_name)


def main():

    # Беоем пути к папкам, где расположены архивы
    paths_to_archives = os.path.join(INPUT_DIR_NAME, TYPE_PAD), os.path.join(INPUT_DIR_NAME, TYPE_TAPE)
    # Получаем пути к архивам
    archives_paths = get_archives_paths(paths_to_archives)
    logger.info(archives_paths.all)

    # Распаковываем архивы в директорию OUTPUT_DIR_NAME (она по умолчанию)
    un_archive_archives(archives_paths)

    # Получаем папки в директории OUTPUT_DIR_NAME
    output_folders = get_all_dirs_from_folder(OUTPUT_DIR_NAME)

    # Создаем директорию для выходных архивов если она не существует
    if not os.path.exists(os.path.join(OUTPUT_DIR_NAME, GENERAL)):
        os.mkdir(os.path.join(OUTPUT_DIR_NAME, GENERAL))

    # Копируем все файлы из выходных папок в директорию OUTPUT_DIR_NAME
    for output_folder in tqdm(output_folders):

        output_folder_path = os.path.join(OUTPUT_DIR_NAME, output_folder)
        # Получаем список файлов в папке
        valid_files = file_getter(output_folder_path)
        # Копируем файлы в директорию output_folder_path
        copy_files_to_folder(valid_files, output_folder_path)
        # Удаляем папки в папке OUTPUT_DIR_NAME/output_folder
        drop_folders_in_folder(output_folder_path)

        # Собираем все файлы из папок для GENERAL
        general_valid_files = (os.path.join(output_folder_path, file) for file in os.listdir(output_folder_path))
        # Копируем все файлы в GENERAL директорию
        copy_files_to_folder(general_valid_files, os.path.join(OUTPUT_DIR_NAME, GENERAL))

    # Архивируем все папки из OUTPUT_DIR_NAME
    output_folders = get_all_dirs_from_folder(OUTPUT_DIR_NAME)
    for output_folder in tqdm(output_folders):
        create_zip_archive(output_folder, os.path.join(OUTPUT_DIR_NAME, output_folder))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

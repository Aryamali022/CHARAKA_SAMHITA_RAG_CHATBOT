from src import config


def test_data_paths_are_inside_project():
    for path in (config.RAW_DIR, config.PROCESSED_DIR, config.STORAGE_DIR):
        assert config.PROJECT_ROOT in path.parents


def test_source_path_points_to_raw_dir():
    assert config.SOURCE_PATH.parent == config.RAW_DIR
    assert config.SOURCE_PATH.name == "charaka-samhita-ocr.txt"

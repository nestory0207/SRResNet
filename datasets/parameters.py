class DatasetParameters:
    def __init__(self, dataset_key, save_data_directory):
        if dataset_key not in available_datasets.keys():
            raise ValueError(f"available datasets are: {available_datasets.keys()}")

        dataset_parameters = available_datasets[dataset_key]

        self.train_directory = dataset_parameters["train_directory"]
        self.valid_directory = dataset_parameters["valid_directory"]
        self.scale = dataset_parameters["scale"]
        self.save_data_directory = save_data_directory


available_datasets = {
    "bicubic_x4":{
        "train_directory": "Train/LR",
        "valid_directory": "Valid/LR",
        "scale": 4
    }
}

from src.datasets.merged_dataset import MergedDataset
from src.datasets.russia_crash_dataset import RussiaCrashDataset
from src.datasets.nuscenes_dataset import NuScenesDataset
from src.datasets.dada2000_dataset import DADA2000Dataset
from src.datasets.mmau_dataset import MMAUDataset
from src.datasets.bdd100k_dataset import BDD100KDataset

    
def create_dataset(dataset_name, **kwargs):

    if str.lower(dataset_name) == "russia_crash":
        dataset = RussiaCrashDataset(**kwargs)
    elif str.lower(dataset_name) == "nuscenes":
        dataset = NuScenesDataset(**kwargs)
    elif str.lower(dataset_name) == "dada2000":
        dataset = DADA2000Dataset(**kwargs)
    elif str.lower(dataset_name) == "mmau":
        dataset = MMAUDataset(**kwargs)
    elif str.lower(dataset_name) == "bdd100k":
        dataset = BDD100KDataset(**kwargs)
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' not implemented")
    
    return dataset

def dataset_factory(dataset_names, **kwargs):
    if isinstance(dataset_names, str) or (isinstance(dataset_names, list) and len(dataset_names) == 1):
        dataset_name = dataset_names[0] if isinstance(dataset_names, list) else dataset_names
        # Init the single dataset
        return create_dataset(dataset_name, **kwargs)
    elif isinstance(dataset_names, list):
        all_datasets = []
        for dataset_name in dataset_names:
            all_datasets.append(create_dataset(dataset_name, **kwargs))
        return MergedDataset(all_datasets)
    
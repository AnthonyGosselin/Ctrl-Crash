from torchvision import transforms

class MergedDataset:
    """
    Dataset wrapper to access many datasets as one
    """

    def __init__(self, dataset_list):

        self.dataset_list = dataset_list

        # TODO: Make sure this matches all datasets
        self.resize_width = self.dataset_list[0].resize_width
        self.resize_height = self.dataset_list[0].resize_height
        self.revert_transform = self.dataset_list[0].revert_transform

        print("TOTAL number of clips in merged dataset:", self.__len__(), f"({self.dataset_list[0].data_split})")


    def __len__(self):
        return sum([len(dset) for dset in self.dataset_list])


    def __getitem__(self, global_index):
        
        target_dset, rel_index = self.get_dataset_by_sample_index(global_index)
        ret_dict = target_dset.__getitem__(rel_index)

        # Overwrite returned index with the global index
        ret_dict["indices"] = global_index

        return ret_dict
    

    def get_dataset_by_sample_index(self, index):
        total_idx = 0
        target_dset = None
        for dset in self.dataset_list:
            total_idx += len(dset)
            if index < total_idx:
                target_dset = dset
                break
        
        return target_dset, (index - (total_idx - len(target_dset)))
    

    def get_frame_file_by_index(self, index, timestep=None):
        target_dset, rel_index = self.get_dataset_by_sample_index(index)
        return target_dset.get_frame_file_by_index(rel_index, timestep=timestep)
    

    def get_bbox_image_file_by_index(self, index, image_file=None):
        target_dset, rel_index = self.get_dataset_by_sample_index(index)
        return target_dset.get_bbox_image_file_by_index(index=rel_index)
class XBDCachedDataset(Dataset):

    def __getitem__(self, idx):

        sample = np.load(cache_file)

        pre = sample["pre"]
        post = sample["post"]
        mask = sample["mask"]

        x = np.concatenate([pre, post], axis=2).transpose(2,0,1)

        return (
            torch.from_numpy(x).float()/255,
            torch.from_numpy(mask).long()
        )
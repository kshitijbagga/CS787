from dataset import TextDataset

ds = TextDataset(base_path="files/IAM-32.pickle", num_examples=3)
item = ds[0]

print("---- Dataset debug ----")
for k,v in item.items():
    if hasattr(v, "shape"):
        print(f"{k}: shape={v.shape}")
    else:
        print(f"{k}: {type(v)}")

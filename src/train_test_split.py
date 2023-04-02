import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(metadata_file:str, ratio:float, save_dir:str, seed:int) -> None:
    metadata = pd.read_csv(metadata_file)

    train, test = train_test_split(metadata, train_size=ratio, random_state=seed)

    train.to_csv(os.path.join(save_dir, "train.csv"))
    test.to_csv(os.path.join(save_dir, "test.csv"))

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--input", type=str, default="../data/raw/metadata.csv")
    parser.add_argument("--ratio", type=float, default=.8)
    parser.add_argument("--save_dir", type=str, default="../data/raw")
    parser.add_argument("--seed", type=int, default=2023)

    args = parser.parse_args()


    split_data(
        metadata_file=args.input,
        ratio=args.ratio,
        save_dir=args.save_dir,
        seed=args.seed
    )
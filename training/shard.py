import time
from collections import defaultdict
from functools import wraps

import numpy as np


class Sharder:
    """
    Class to split a huge npz file into smaller shards to improve IO time.
    """

    keywords = [
        "unprep_gate",
        "depth",
        "observation",
        "global_n_idx",
        "global_g_idx",
    ]

    def __init__(self, file: str, shard_size: int = 1 << 20, seed: int = 1):
        super().__init__()
        self.load_file(file)
        self.global_data = {}
        self.construct_metadata()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.shard_size = shard_size
        self.batch_size = 1 << 10

    def load_file(self, file):
        """
        Use only 1 file as input. Register the file as a member of the class.
        """
        self.data = np.load(file, "r")
        # total_size = 0
        # for key in self.data:
        #     total_size += self.data[key].nbytes
        # print(f"Total size {total_size / (1 << 30):.4f} GB")

    def construct_metadata(self):
        """
        Store metadata about the data. Metadata includes:
        1. aggregate_metadata - reverse map of (n, g) -> size of dataset
        2. size_list - list of sizes of each (n, g) dataset
        3. ng_list - list of (n, g). Same order as size_list.
        """
        # To support multiple files, look at the module training.split for inspiration.
        self.aggregate_metadata = {}
        self.reverse_map = {}
        total = 0
        data = self.data
        for key in data:
            key_split = key.split("/")
            if key_split[0] in ["global_g", "global_n", "seed"]:
                self.global_data[key] = data[key]
                continue
            if not key.endswith("unprep_gate"):
                continue
            size = data[key].shape[0]
            total += size
            n, g, _ = key.split("/")
            self.aggregate_metadata[(int(n), int(g))] = size
        self.size_list = np.array([v for v in self.aggregate_metadata.values()])
        self.p = self.size_list / np.sum(self.size_list)

        self.ng_list = list(self.aggregate_metadata.keys())

    def get_total_size(self):
        return np.sum(self.size_list)

    def _create_iter_order(self):
        # Set the iteration order
        self.iter_order = []
        for k, v in self.aggregate_metadata.items():
            n, g = k
            num_batches = int(v // self.batch_size)
            if num_batches == 0:
                continue
            idxs = np.arange(v)
            np.random.shuffle(idxs)
            for i in range(num_batches):
                self.iter_order.append(
                    (k, np.sort(idxs[i * self.batch_size : (i + 1) * self.batch_size]))
                )
        np.random.shuffle(self.iter_order)
        return self

    def _init_new_shard(self):
        shard = {}
        for key in self.global_data:
            shard[key] = self.global_data[key]
        return shard

    def create_and_save_shards(self, folder, prefix):
        self._create_iter_order()
        curr_size = 0
        sharded_data = self._init_new_shard()
        shard_count = 0
        batch_tic = tic = time.time()
        print(f"Total # iterations = {len(self.iter_order)}")
        for iter_idx in range(len(self.iter_order)):
            # Extract the current (n, g) key
            k, idxs = self.iter_order[iter_idx]
            curr_size += len(idxs)

            n, g = k
            for base_key in self.keywords:
                key = f"{n}/{g}/{base_key}"
                if key not in sharded_data:
                    sharded_data[key] = self.data[key][idxs]
                else:
                    sharded_data[key] = np.concatenate(
                        [self.data[key][idxs], sharded_data[key]], axis=0
                    )
            if curr_size >= self.shard_size:
                np.savez_compressed(f"{folder}/{prefix}-s{shard_count}.npz", **sharded_data)
                sharded_data = self._init_new_shard()
                shard_count += 1
                curr_size = 0
                print(f"Saved shard #{shard_count}")
            if iter_idx % 100 == 0:
                batch_toc = time.time()
                print(
                    f"Iteration #{iter_idx} completed. Elapsed time {batch_toc - batch_tic:.4f} s, avg {(batch_toc - tic) / (iter_idx + 1):.4f} s"
                )
                batch_tic = time.time()

        if len(sharded_data) > 0:
            np.savez_compressed(f"{folder}/{prefix}-s{shard_count}.npz", **sharded_data)
            shard_count += 1

        print(f"Created {shard_count} shards.")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Shard given file")
    parser.add_argument(
        "--input-filename", type=str, required=True, help="Input filename with npz extension"
    )
    parser.add_argument(
        "--folder", type=str, required=True, help="Ouptut folder without forward slash"
    )
    parser.add_argument("--prefix", type=str, required=True, help="Ouptut file prefix")
    args = parser.parse_args()

    assert args.input_filename.endswith(".npz"), "Input filename must end with '.npz'"
    assert not args.folder.endswith("/"), "Output folder should not end with forward slash"

    sharder = Sharder(args.input_filename)

    tic = time.time()
    print("Total size:", sharder.get_total_size())
    sharder.create_and_save_shards(args.folder, args.prefix)
    toc = time.time()
    print(f"Sharded {args.input_filename} -> {args.folder}/{args.prefix} ({toc-tic} sec)")

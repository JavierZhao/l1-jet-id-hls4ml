#!/usr/bin/env python3

import argparse
from pathlib import Path
from data import HLS4MLData150

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download & preprocess HLS4ML 150-constituent jet data"
    )
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory to store raw/preprocessed/processed")
    parser.add_argument("--nconst", type=int, default=150,
                        help="Number of constituents per jet")
    parser.add_argument("--feats", choices=["ptetaphi","allfeats"],
                        default="ptetaphi", help="Feature selection scheme")
    parser.add_argument("--norm", choices=["minmax","robust","standard"],
                        default="standard", help="Normalization method")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for shuffling (train only)")
    parser.add_argument("--kfolds", type=int, default=0,
                        help="Number of Stratified K-Folds (train only)")
    return parser.parse_args()

def main():
    args = parse_args()

    print("→ Processing training data")
    train_data = HLS4MLData150(
        root=args.root,
        nconst=args.nconst,
        feats=args.feats,
        norm=args.norm,
        train=True,
        kfolds=args.kfolds,
        seed=args.seed,
    )
    train_data.show_details()

    print("→ Processing validation data")
    val_data = HLS4MLData150(
        root=args.root,
        nconst=args.nconst,
        feats=args.feats,
        norm=args.norm,
        train=False,
        kfolds=0,
        seed=None,
    )
    val_data.show_details()

    print("✅ All data downloaded & processed.")

if __name__ == "__main__":
    main()

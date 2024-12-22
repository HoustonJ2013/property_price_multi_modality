import argparse
import os 
from typing import Any, Dict, List
import pandas as pd
from torch.nn import MSELoss, L1Loss
import numpy as np
import json 
import random
from tqdm import tqdm
from pathlib import Path
from scipy.stats import binned_statistic
import pickle 

import torch

import torch_frame
from torch_frame.data import Dataset
from torch_frame.data import DataLoader
from torch_frame import TensorFrame, stype
from torch_frame.nn.models.ft_transformer import FTTransformer


from model import TabTransformer, stype_encoder_dict_2, stype_encoder_dict_3
from utils import cosine_scheduler



def get_args_parser():
    parser = argparse.ArgumentParser("Torch Frame", add_help=False)
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        help="the configure file to rerun the test",
    )

    parser.add_argument(
         "--inference_data_path",
        default="",
        type=str,
        help="""The path to the pandas dataframe in Pickle Format for inference""",
    )
    
    parser.add_argument(
         "--output_data_folder",
        default="data/",
        type=str,
        help="""The folder to save the output data""",
    )

    # Basic training parameters
    parser.add_argument("--gpu", action="store_false", help="strongly recommend to use GPU")
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="""The device to run the code""",
    )
    parser.add_argument(
        "--batch_size",
        default=2048,
        type=int,
        help="""batch size""",
    )
    return parser



def model_predict(model, val_loader, args):

    if args.gpu and torch.cuda.is_available(): 
        device = args.device 
    else:
        device = "cpu"

    predicts = []
    for tf in val_loader:
        if args.gpu:
            tf = tf.to(device)
        with torch.no_grad():
            pred = model(tf)
            predicts.extend(list(pred.cpu()))
    return np.array(predicts)
    

def read_column_name_and_stats(args):
    with open(args.cache_path, "rb") as f:
        _, col_stats = torch.load(f)
    col_names = list(col_stats.keys())
    return col_names, col_stats


def inference(args):
    
    if os.path.exists(args.inference_data_path):
        raw_df = pd.read_pickle(args.inference_data_path).reset_index(drop=True)
    else:
        raise ValueError("The inference data path does not exist")
    
    raw_df["date"] = raw_df["date"].apply(lambda x: x.replace("_", "-"))
    test_folder = os.path.join(args.model_folder, args.test_name)
    with open(os.path.join(test_folder, 'col_to_stype.pkl'), 'rb') as fp:
        col_to_stype = pickle.load(fp)
    # Preprocess the data
    dense_type = col_to_stype["LON"]
    dense_features = []
    for key, stype in col_to_stype.items():
        if stype == dense_type:
            dense_features.append(key)
    for feat in dense_features:
        raw_df[feat] = raw_df[feat].fillna(raw_df[feat].mean())

    multiclass_type = col_to_stype["Foundation_multiclass"]
    multiclass_features = []
    for key, stype in col_to_stype.items():
        if stype == multiclass_type:
            multiclass_features.append(key)

    for col in multiclass_features:
        raw_df[col] = raw_df[col].apply(lambda d: d if isinstance(d, list) else [])
    col_names, col_stats = read_column_name_and_stats(args)
    dataset = Dataset(
        raw_df[col_names], 
        col_to_stype=col_to_stype,
        target_col="price"
    )
    dataset.materialize(col_stats=col_stats)
    dataloader = DataLoader(dataset.tensor_frame, 
                            batch_size=args.batch_size)

    if args.gpu and torch.cuda.is_available(): 
        device = args.device 
    else:
        device = "cpu"

    if args.add_text_emb or args.add_img_emb:
        stype_encoder_dict = stype_encoder_dict_3

    ## Setup Model. The tabformer is different from torchrame's tabformer
    if args.arch == "TabTransformer":
        model = TabTransformer(
            channels=args.tabformer_channels,
            num_layers=args.tabformer_num_layers,
            num_heads=args.tabformer_num_heads,
            col_stats=dataset.col_stats,
            col_names_dict=dataset.tensor_frame.col_names_dict,
            stype_encoder_dict=stype_encoder_dict, 
        ).to(device)
    # Bug, this model doesn't converge in current implementation 
    elif args.arch == "FTTransformer": 
        model = FTTransformer(
            channels=args.ftformer_channels,
            num_layers=args.ftformer_num_layers,
            out_channels=1,
            col_stats=dataset.col_stats,
            col_names_dict=dataset.tensor_frame.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        ).to(device)
    else: 
        model = None
    
    model.load_state_dict(torch.load(os.path.join(test_folder, "final_model.pth"))["model_weight"])

    model_pred = model_predict(model, dataloader, args)
    raw_df["model_pred"] = model_pred

    data_name = args.inference_data_path.split("/")[-1].split(".")[0]
    output_path = os.path.join(args.output_data_folder, f"{data_name}_{args.test_name}_inference.pkl")
    raw_df[["address", "address_key", "model_pred", "price"]].to_pickle(output_path)


def update_argparse(args, filename):
    with open(filename, "r") as f:
        dict_from_cfg = json.load(f)
        for key, value in dict_from_cfg.items():
            if key not in args.__dict__:
                args.__dict__[key] = value

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Property Price Prediction", parents=[get_args_parser()])
    args = parser.parse_args()
    config_file = args.config_file
    update_argparse(args, args.config_file)
    print(args)
    inference(args)
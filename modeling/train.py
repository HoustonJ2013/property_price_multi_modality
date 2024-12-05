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

    # Model parameters
    parser.add_argument(
        "--arch",
        default="TabTransformer",
        type=str,
        choices=["TabTransformer", "FTTransformer"],
        help="""Name of the architecture""",
    )
    parser.add_argument(
        "--test_name",
        default="Vanilla",
        type=str,
        help="""The name for the trainning test""",
    )

    parser.add_argument(
        "--tabformer_channels",
        default=32,
        type=int,
        help="""tabformer_channels""",
    )

    parser.add_argument(
        "--tabformer_num_layers",
        default=2,
        type=int,
        help="""tabformer_num_layers""",
    )

    parser.add_argument(
        "--tabformer_num_heads",
        default=8,
        type=int,
        help="""tabformer_num_heads""",
    )

    parser.add_argument(
        "--ftformer_channels",
        default=32,
        type=int,
        help="""ftformer_channels""",
    )

    parser.add_argument(
        "--ftformer_num_layers",
        default=2,
        type=int,
        help="""ftformer_num_layers""",
    )

    parser.add_argument(
        "--ftformer_num_heads",
        default=8,
        type=int,
        help="""tabformer_num_heads""",
    )


    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        help="the configure file to rerun the test",
    )

    parser.add_argument(
         "--data_path",
        default="data/property_structured.pkl",
        type=str,
        help="""The path to the Raw Table in Pickle Format""",
    )
    parser.add_argument(
         "--text_emb_path",
        default="data/property_structured_multiclass_desc_text_emb.pkl",
        type=str,
        help="""The path to the text embedding table in Pickle Format""",
    )
    parser.add_argument(
         "--img_emb_path",
        default="data/property_structured_multiclass_img_mean_emb.pkl",
        type=str,
        help="""The path to the text embedding table in Pickle Format""",
    )

    parser.add_argument(
         "--model_folder",
        default="model_checkpoints",
        type=str,
        help="""The path to folder for the model checkpoint""",
    )

    parser.add_argument(
         "--cache_path",
        default="data/property_structured_multiclass_emptylist.pt",
        type=str,
        help="""The path to the Raw Table in Pickle Format""",
    )

    parser.add_argument("--data_v2", 
                        action="store_true", 
                        help="use data prep version 2")
    parser.add_argument("--additional_feature_v1", 
                        action="store_true", 
                        help="Add addiontal school related features")
    parser.add_argument("--add_text_emb", 
                        action="store_true", 
                        help="add text embedding")
    parser.add_argument("--add_img_emb", 
                        action="store_true", 
                        help="add text embedding")
    parser.add_argument(
        "--torch_seed",
        default=10,
        type=int,
        help="""The random seed for pytorch""",
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
        default=1024,
        type=int,
        help="""batch size""",
    )
    parser.add_argument(
        "--num_epoch",
        default=100,
        type=int,
        help="""num of epochs""",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="""learning rate""",
    )
    parser.add_argument("--use_cosine_scheduler", 
                        action="store_true", 
                        help="Whether to use cosine scheduler")
    parser.add_argument(
        "--warmup_epochs",
        default=5,
        type=int,
        help="""num of epochs for warmup""",
    )
    parser.add_argument(
        "--final_lr",
        type=float,
        default=1e-4,
        help="""The final learning rate at the end of the cosine scheduler""",
    )
    parser.add_argument(
        "--loss",
        default="MAE",
        type=str,
        choices=["MAE", "MSE"],
        help="""The name of the loss""",
    )
    parser.add_argument(
        "--optimizer",
        default="Adam",
        type=str,
        choices=["Adam"],
        help="""The name of the optimizer""",
    )
    return parser


def train_one_epoch(model, optimizer, train_loader, loss_fn, epoch, n_batches, lr_schedule, args, refresh_freq=10):

    if args.gpu and torch.cuda.is_available(): 
        device = args.device 
    else:
        device = "cpu"

    loss_values = []
    batch_i = 0
    for tf in train_loader:
        if batch_i % refresh_freq == 0:
            print("step %i/%i at epoch %i with loss %0.2e \r" % \
                  (batch_i, n_batches, epoch, np.mean(loss_values)), end="", flush=True)
        if args.use_cosine_scheduler:
            for _, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[epoch * len(train_loader) + batch_i]
                # print("current lr is %0.2e" % param_group["lr"])
        if args.gpu:
            tf = tf.to(device)
        optimizer.zero_grad()
        pred = model(tf)
        loss = loss_fn(pred, tf.y)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        batch_i += 1
    print("loss at epoch %i is %0.2e   " % (epoch, np.mean(loss_values)))
    return np.mean(loss_values)


def eval_model(model, val_loader, epoch, args, return_pred=False):

    if args.gpu and torch.cuda.is_available(): 
        device = args.device 
    else:
        device = "cpu"

    mae_loss = L1Loss()
    error = []
    predicts = []
    true_price = []
    for tf in val_loader:
        if args.gpu:
            tf = tf.to(device)
        with torch.no_grad():
            pred = model(tf)
            loss = mae_loss(pred, tf.y)
            if return_pred:
                predicts.extend(list(pred.cpu()))
                true_price.extend(list(tf.y.cpu()))
            error.append(loss.item())
    print("Validation MAE at epoch %i is %0.2e"%(epoch, np.mean(error)))
    if return_pred: 
        return np.mean(error), np.array(predicts), np.array(true_price)
    else:
        return np.mean(error)


def train(args):
    # load data
    target = "price"
    dense_features = ["LON", 
                    "LAT", 
                    "building_sqft", 
                    "Lot Size", 
                    "Year Built", 
                    "Garage Number", 
                    "Bedrooms", 
                    "Baths", 
                    "Maintenance Fee", 
                    "Tax Rate", 
                    "Recent Market Value", 
                    "Recent Tax Value"]
    cate = ["status", "Property Type", "County", "Private Pool", "Area Pool"]
    time_col = ["date"]
    cate_multi = ["Foundation_multiclass", "Garage Types_multiclass", 
                "Roof Type_multiclass", "Pool_feature_multiclass", "floor_type_multiclass", 
                "finance_option_multiclass", "Exterior Type_multiclass", "Style_multiclass"]
    
    add_numerial_features = ["elementary_school_star", "middle_school_star", "high_school_star"]
    add_cate_multi = ["school_org"] # ["school_names", "school_grades", "school_org"]

    text_embedding_col = ["general_desc_emb"]
    img_embedding_col = ["img_emb"]
    raw_df = pd.read_pickle(args.data_path).reset_index(drop=True)
    raw_df["date"] = raw_df["date"].apply(lambda x: x.replace("_", "-"))
    # raw_df["date"] = pd.to_datetime(raw_df["date"])
    if args.data_v2: 
        print("adding additional features")
        raw_df["Year Built"] = raw_df.apply(lambda x: int(x["date"].split("-")[0]) - x["Year Built"] 
                                            if x["date"] is not None and x["Year Built"] is not None else None,
                                            axis=1)
    if args.additional_feature_v1:
        print("adding additional features")
        dense_features = dense_features + add_numerial_features
        cate_multi = cate_multi + add_cate_multi
    
    col_to_stype = {}
    col_to_stype = {d: stype.numerical for d in dense_features}
    col_to_stype.update({d: stype.timestamp for d in time_col})
    col_to_stype.update({target: stype.numerical})
    col_to_stype.update({d: stype.categorical for d in cate})
    col_to_stype.update({d: stype.multicategorical for d in cate_multi})
    if args.add_text_emb and args.add_img_emb:
        text_emb_df = pd.read_pickle(args.text_emb_path)
        raw_df = raw_df.join(text_emb_df.set_index("address")[["general_desc_emb"]], 
                     on="address", 
                     how="left")
        img_emb_df = pd.read_pickle(args.img_emb_path)
        raw_df = raw_df.join(img_emb_df.set_index("address")[["img_emb"]], 
                     on="address", 
                     how="left")
        raw_df = raw_df[dense_features + cate + cate_multi + time_col + text_embedding_col + img_embedding_col + [target]]
        col_to_stype.update({d: stype.embedding for d in img_embedding_col + text_embedding_col})
        stype_encoder_dict = stype_encoder_dict_3
        
    elif args.add_text_emb:
        print("adding text embedding")
        emb_df = pd.read_pickle(args.text_emb_path)
        raw_df = raw_df.join(emb_df.set_index("address")[["general_desc_emb"]], 
                     on="address", 
                     how="left")
        raw_df = raw_df[dense_features + cate + cate_multi + time_col + text_embedding_col + [target]]
        col_to_stype.update({d: stype.embedding for d in text_embedding_col})
        stype_encoder_dict = stype_encoder_dict_3
    elif args.add_img_emb:
        print("adding img embedding")
        emb_df = pd.read_pickle(args.img_emb_path)
        raw_df = raw_df.join(emb_df.set_index("address")[["img_emb"]], 
                     on="address", 
                     how="left")
        raw_df = raw_df[dense_features + cate + cate_multi + time_col + img_embedding_col + [target]]
        col_to_stype.update({d: stype.embedding for d in img_embedding_col})
        stype_encoder_dict = stype_encoder_dict_3
    else:
        raw_df = raw_df[dense_features + cate + cate_multi + time_col + [target]]
        stype_encoder_dict = stype_encoder_dict_2
    

    for col in cate_multi:
        raw_df[col] = raw_df[col].apply(lambda d: d if isinstance(d, list) else [])
    
    dataset = Dataset(
        raw_df, 
        col_to_stype=col_to_stype,
        target_col="price"
    )
    dataset.materialize(path=args.cache_path)
    torch.manual_seed(args.torch_seed)
    dataset.shuffle()
    train_dataset, val_dataset = dataset[:0.8], dataset[0.80:]

    train_loader = DataLoader(train_dataset.tensor_frame, 
                              batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset.tensor_frame, batch_size=2048)
    
    if args.gpu and torch.cuda.is_available(): 
        device = args.device 
    else:
        device = "cpu"

    ## Setup Model. The tabformer is different from torchrame's tabformer
    if args.arch == "TabTransformer":
        model = TabTransformer(
            channels=args.tabformer_channels,
            num_layers=args.tabformer_num_layers,
            num_heads=args.tabformer_num_heads,
            col_stats=train_dataset.col_stats,
            col_names_dict=train_dataset.tensor_frame.col_names_dict,
            stype_encoder_dict=stype_encoder_dict, 
        ).to(device)
    # Bug, this model doesn't converge in current implementation 
    elif args.arch == "FTTransformer": 
        model = FTTransformer(
            channels=args.ftformer_channels,
            num_layers=args.ftformer_num_layers,
            out_channels=1,
            col_stats=train_dataset.col_stats,
            col_names_dict=train_dataset.tensor_frame.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        ).to(device)
    else: 
        model = None
    

    if args.use_cosine_scheduler:
        lr_schedule = cosine_scheduler(args.lr, 
                                       args.final_lr, 
                                       args.num_epoch, 
                                       len(train_loader), 
                                       warmup_epochs=args.warmup_epochs, 
                                       start_warmup_value=0)
        print(len(lr_schedule), args.num_epoch * len(train_loader))
    else:
        lr_schedule = None 

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
    else: 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.loss == "MAE":
        loss_fn = L1Loss()
    else:
        loss_fn = MSELoss()

    n_batches = (len(train_dataset) + (args.batch_size - 1)) // args.batch_size

    test_folder = os.path.join(args.model_folder, args.test_name)
    log_path = os.path.join(test_folder, "log.txt")
    if os.path.exists(log_path):
        open(log_path, 'w').close()

    
    for epoch in range(args.num_epoch):
        train_loss = train_one_epoch(model, 
                                     optimizer, 
                                     train_loader, 
                                     loss_fn, 
                                     epoch, 
                                     n_batches, 
                                     lr_schedule, 
                                     args)
        if epoch < args.num_epoch - 1:
            val_error = eval_model(model, val_loader, epoch, args)
            save_log(log_path, {"train_loss": train_loss, "val_error": val_error, "epoch": epoch})
        else:
            val_error, predicts, true_price = eval_model(model, val_loader, epoch, args, return_pred=True)
            save_log(log_path, {"train_loss": train_loss, "val_error": val_error, "epoch": epoch})
            mean_stat = binned_statistic(true_price, 
                                         np.abs(true_price - predicts), 
                                         statistic='mean', 
                                         bins=10, 
                                         range=(0, 1e6)
                                         )
            val_pred_df = pd.DataFrame.from_dict({
                "pred": predicts, 
                "true_price": true_price,
            })
            val_pred_df.to_pickle(os.path.join(test_folder, "val_pred.pkl"))
            save_log(log_path, {"bin_edges": ["%0.2e"%(_) for _ in mean_stat.bin_edges], 
                                "mean":  ["%0.2e"%(_) for _ in mean_stat.statistic],  
                                # "bin_number": ["%i"%(_) for _ in mean_stat.binnumber]
                                })
    # Save the final model checkpoint 
    checkpoint_path = os.path.join(test_folder, "final_model.pth")
    model_dict = {"model_weight": model.state_dict()}
    torch.save(model_dict, checkpoint_path)


def save_log(log_path, log_dict):
    with Path(log_path).open("a") as f:
        f.write(json.dumps(log_dict) + "\n")


def save_argparse(args, filename):
    with open(filename, "w") as f:
        json.dump(args.__dict__, f, indent=2)


def load_argparse(args, filename):
    with open(filename, "r") as f:
        args.__dict__.update(json.load(f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Property Price Prediction", parents=[get_args_parser()])
    args = parser.parse_args()
    test_folder = os.path.join(args.model_folder, args.test_name)
    if os.path.exists(test_folder) is False:
        os.makedirs(test_folder, exist_ok=True)
    if args.config_file is None:
        config_file = os.path.join(test_folder, "config.cfg")
        args.config_file = config_file
        save_argparse(args, config_file)
    else:
        config_file = args.config_file
        load_argparse(args, args.config_file)
        ## Make sure config name and test name are consistent
        if args.config_file != config_file: 
            args.config_file = config_file
            args.test_name = config_file.split("/")[-1].replace(".cfg", "")
            save_argparse(args, config_file)
    print(args)
    train(args)
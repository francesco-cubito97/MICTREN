"""
Start executing the training/testing/inference from here
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import torch
import torch.nn as nn
from pytorch_transformers.modeling_bert import BertConfig

from models.backbone import Backbone
from utils.file_utils import create_dir, seeds_init
from utils.data_utils import make_hand_data_loader
from models.mano import Mano
from utils.mesh_utils import Mesh
from models.mictren import MICTREN
from utils.render import Renderer
from models.transformers import TransBlock
from train import train
from evaluate import run_eval_and_save


def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--type", default="train", type=str, required=False,
                        help="Choose one of the following types of run: train, eval, inference")

    #-----------------------------INITIALIZATION-----------------------------#
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Choose one between the following : cuda, cpu")
    parser.add_argument("--seed", type=int, default=99, 
                        help="random seed for initialization.")

    #------------------------------DATA ARGUMENT------------------------------#
    parser.add_argument("--dataset_dir", default="datasets/freihand", type=str, required=False,
                        help="Folder containing dataset data")
    parser.add_argument("--train_file", default="train.yaml", type=str, required=False,
                        help="File with all data for training.")
    parser.add_argument("--eval_file", default="test.yaml", type=str, required=False,
                        help="File with all data for evaluation.")
    parser.add_argument("--num_workers", default=2, type=int, 
                        help="Workers in dataloader.")       
    parser.add_argument("--img_scale_factor", default=1, type=int, 
                        help="Adjust image resolution.")  

    #-----------------------------CHECKPOINTS-----------------------------#
    parser.add_argument("--model_config", default="configurations/bert-base-uncased", type=str, required=False,
                        help="Path to pre-trained transformer model configuration.")
    parser.add_argument("--saved_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="Directory where to save checkpoints")

    parser.add_argument("--output_dir", default="outputs", type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--logging_steps", default=100, type=int)
    
    #---------------------------TRAINING PARAMS---------------------------#
    parser.add_argument("--batch_size", default=64, type=int, 
                        help="Batch size per GPU/CPU for training/evaluation.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, 
                        help="The initial learning rate.")
    parser.add_argument("--num_train_epochs", default=200, type=int, 
                        help="Total number of training epochs")
    parser.add_argument("--vertices_loss_weight", default=1.0, type=float)          
    parser.add_argument("--joints_loss_weight", default=1.0, type=float)
    parser.add_argument("--pose_loss_weight", default=1.0, type=float)
    parser.add_argument("--betas_loss_weight", default=1.0, type=float)
    parser.add_argument("--vloss_w_sub", default=0.5, type=float)
    parser.add_argument("--vloss_w_full", default=0.5, type=float)
    parser.add_argument("--drop_out", default=0.1, type=float, 
                        help="Drop out ratio in BERT.")

    #---------------------------MODEL PARAMS---------------------------#
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    
    parser.add_argument("--input_feat_dim", default="1027,512,272|512,256,128|694,256,128,64", type=str, 
                        help="Transformers blocks definitions of input,hidden, output layers")          
    #parser.add_argument("--hidden_feat_dim", default="512|", type=str, 
    #                    help="Hidden image freature dimensions")

    #parser.add_argument("--args.sc", default=1.0, type=float, required=False)
    #parser.add_argument("--args.rot", default=0.0, type=float, required=False)   

    return parser.parse_args()

def main(args):
    # Initial setup
    args.device = torch.device(args.device)
    seeds_init(args.seed)

    create_dir(args.output_dir)
    print(f"All outputs will be saved in {args.output_dir}")

    # Mesh and sampler utils
    mano_model = Mano(args).to(args.device)
    mano_model.layer = mano_model.layer.to(args.device)
    mesh_sampler = Mesh()

    # Renderer for visualization
    renderer = Renderer(faces=mano_model.faces)

    # Transformer layers
    trans_layers = []

    input_feat_dim = [int(item) for item in args.input_feat_dim.split(",")]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(",")]
    
    # The final layer will output the 3D joints + 3D mesh vertices
    output_feat_dim = input_feat_dim[1:] + [3]
    
    # Init a series of transformers blocks
    for i in range(len(output_feat_dim)):
        config_class, model_class = BertConfig, TransBlock
        config = config_class.from_pretrained(args.model_config)

        config.output_attentions = False
        config.hidden_dropout_prob = args.drop_out
        config.img_feature_dim = input_feat_dim[i] 
        config.output_feature_dim = output_feat_dim[i]
        args.hidden_size = hidden_feat_dim[i]
        args.intermediate_size = int(args.hidden_size*4)

        # Update model structure if specified in arguments
        update_params = ["num_hidden_layers", "hidden_size", "num_attention_heads", "intermediate_size"]
        for param in update_params:
            arg_param = getattr(args, param)
            config_param = getattr(config, param)
            if arg_param > 0 and arg_param != config_param:
                print("MAIN", f"Update config parameter {param}: {config_param} -> {arg_param}")
                setattr(config, param, arg_param)

        # init a transformer encoder and append it to a list
        assert config.hidden_size % config.num_attention_heads == 0
        model = model_class(config=config) 
        print("MAIN", "Init model from scratch.")
        trans_layers.append(model)
        
    # Adding backbone
    print("MAIN", "Using pre-trained model 'MobileNetV3'")
    
    backbone = Backbone()
    
    # Compose the final neural network
    trans_layers = torch.nn.Sequential(*trans_layers)
    total_params = sum(p.numel() for p in trans_layers.parameters())
    print("MAIN", f"Transformers total parameters: {total_params}")
    backbone_total_params = sum(p.numel() for p in backbone.parameters())
    print("MAIN", f"Backbone total parameters: {backbone_total_params}")

    _network = MICTREN(args, config, backbone, trans_layers)
    
    if args.type=="train" and args.saved_checkpoint!=None and args.saved_checkpoint!="None":
        # For fine-tuning or resume training or inference, load weights from checkpoint
        print("MAIN", f"Loading state dict from checkpoint {args.saved_checkpoint}")
        checkpoint = torch.load(args.saved_checkpoint, map_location=torch.device("cpu"))
        _network.load_state_dict(checkpoint, strict=False)
        del checkpoint

    elif args.type=="eval" and args.saved_checkpoint!=None and args.saved_checkpoint!="None":
        print("MAIN", "Evaluation: Loading from checkpoint {}".format(args.saved_checkpoint))
        checkpoint = torch.load(args.saved_checkpoint, map_location=torch.device("cpu"))
        _network.load_state_dict(checkpoint, strict=False)
        del checkpoint    

    _network.to(args.device)
    print("MAIN", f"Training parameters {args}")

    if args.type=="eval":
        val_dataloader = make_hand_data_loader(args, 
                                               args.eval_file, 
                                               is_train=False, 
                                               scale_factor=args.img_scale_factor)
       
        run_eval_and_save(args, val_dataloader, 
                          _network, mano_model, renderer, mesh_sampler)

    else:
        train_dataloader = make_hand_data_loader(args, 
                                                 args.train_file, 
                                                 is_train=True, 
                                                 scale_factor=args.img_scale_factor)
        
        train(args, train_dataloader, _network,
            mano_model, renderer, mesh_sampler)

if __name__ == "__main__":
    args = parseArguments()
    main(args)

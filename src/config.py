import argparse


# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ----------------------------------------
# Arguments for the main program
main_arg = add_argument_group("Main")


main_arg.add_argument("--mode", type=str,
                      default="train",
                      choices=["train", "test"],
                      help="Run mode")

# ----------------------------------------`
# Arguments for training
train_arg = add_argument_group("Training")


train_arg.add_argument("--img_dir", type=str,
                       default="/data/imaterialist2020/train/",
                       help="Directory with train dataset")

train_arg.add_argument("--ann_path", type=str,
                       default="/data/imaterialist2020/train.csv",
                       help="Directory with train dataset")

train_arg.add_argument("--test_dir", type=str,
                       default="/Volumes/FreeAgent G/imaterialist-fashion-2020-fgvc7/test/",
                       help="Directory with test dataset")

train_arg.add_argument('--sample_path', type=str,
                       default= '/data/imaterialist2020/sample_submission.csv',
                       help = 'Directory with test dataset')

train_arg.add_argument('--submit_path', type=str,
                       default= '/data/imaterialist2020/',
                       help = 'Directory with submit')

train_arg.add_argument('--width',type = int,
                       default = 512,
                       help = 'resize width for the data')

train_arg.add_argument('--height',type = int,
                       default = 512,
                       help = 'resize height for the data')


train_arg.add_argument("--seed",type=int,
                      default=0,
                      help="fix the seed for torch")

train_arg.add_argument("--batch_size", type=int,
                       default= 2,
                       help="Size of each training batch")

train_arg.add_argument("--num_epochs", type=int,
                       default=10,
                       help="Number of epochs to train")

train_arg.add_argument("--epoch_cb", type=int,
                       default=100,
                       help="Number of epochs by which a callback is excuted")

train_arg.add_argument("--rep_intv", type=int,
                       default=10,
                       help="Report interval")

train_arg.add_argument("--val_intv", type=int,
                       default=10,
                       help="Validation interval")

train_arg.add_argument("--log_dir", type=str,
                       default="/Volumes/FreeAgent G/imaterialist-fashion-2020-fgvc7/logs",
                       help="Directory to save logs and current model")

train_arg.add_argument("--save_dir", type=str,
                       default="/Volumes/FreeAgent G/imaterialist-fashion-2020-fgvc7/save",
                       help="Directory to save the best model")

train_arg.add_argument("--out_dir", type=str,
                       default="/Volumes/FreeAgent G/imaterialist-fashion-2020-fgvc7/out",
                       help="Directory to save the output of vae models")

train_arg.add_argument("--resume", type=str2bool,
                       default=True,
                       help="Whether to resume training from existing checkpoint")

train_arg.add_argument("--checkpoint", type=int,
                       default=9,
                       help="Checkpoint index to test")
#--------------------------------------------------------------

# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument("--normalize", type=str2bool,
                       default=True,
                       help="Whether to normalize with mean/std or not")

model_arg.add_argument("--l2_reg", type=float,
                       default=0.0002,
                       help="L2 Regularization strength")

model_arg.add_argument("--ksize", type=int,
                       default=3,
                       help="Size of the convolution kernel")

model_arg.add_argument("--num_filters",type=int,
                      default=32,
                      help="Default number of filters")

model_arg.add_argument("--zdim",type=int,
                      default=256,
                      help = "dimension of the latent representation z")

model_arg.add_argument("--num_conv_outer",type=int,
                      default=5,
                      help = "Number of outer blocks (steps)")

model_arg.add_argument("--vy",type=float,
                      default=0.002,
                      help="conditional norm lik variance")

model_arg.add_argument("--act", type=str,
                       default="elu",
                       help="Activation type")

model_arg.add_argument("--lr", type=float,
                       default=0.001,
                       help="Learning rate")

model_arg.add_argument("--scheduler", type=str,
                       default='step',
                       help="Learning rate schedule (cosine, step)")

model_arg.add_argument("--mask_thresh", type=float,
                       default=0.5,
                       help="Mask threshold in test phase")    
                       
model_arg.add_argument("--attr_score_thresh", type=float,
                       default=0.5,
                       help="Attribute Score threshold in test phase")                    

model_arg.add_argument("--pos_weight", type=float,
                       default=100.,
                       help="Attribute Positive Weight")                    

model_arg.add_argument("--loss_aweight", type=float,
                       default=1.,
                       help="Attribute Positive Weight") 

def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here

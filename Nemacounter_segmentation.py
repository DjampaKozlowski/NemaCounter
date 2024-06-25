

from nemacounter.argsparsers import segmentation_argument_parser
from nemacounter.segmentation import segmentation_workflow


    
    
if __name__=="__main__":
    args = segmentation_argument_parser()
    # Transform the argparse output to a dictionary which contains 
    # the arguments and their values
    dct_args = vars(args)
    segmentation_workflow(dct_args)

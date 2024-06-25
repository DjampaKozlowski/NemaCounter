from nemacounter.argsparsers import detection_argument_parser
from nemacounter.detection import detection_workflow


if __name__=="__main__":
    args = detection_argument_parser()
    # Transform the argparse output to a dictionary which contains 
    # the arguments and their values
    dct_args = vars(args)
    detection_workflow(dct_args, gui=False)

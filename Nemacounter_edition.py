from nemacounter.argsparsers import edition_argument_parser
from nemacounter.edition import edition_workflow











 
 

# if __name__ == "__main__":
#     args = edition_argument_parser()
#     # Transform the argparse output to a dictionary which contains 
#     # the arguments and their values
#     dct_args = vars(args)
#     edition_workflow(dct_args)


    
if __name__ == "__main__":
    import os
    fpath_globinfo = r"C:\Users\djamp\Documents\Work\Project\NemaCounter\output_mamenito\MyProject\MyProject_globinfo.csv"
    output_directory = r'C:\Users\djamp\Documents\Work\Project\NemaCounter\output_mamenito\unzeub'
    fpath_globinfo = os.path.abspath(fpath_globinfo)
    output_directory = os.path.abspath(output_directory)
    project_id = 'unzeub'
    edition_workflow(fpath_globinfo, output_directory, project_id)
from nemacounter.argsparsers import edition_argument_parser
from nemacounter.edition import edition_workflow











 
 



    
if __name__ == "__main__":
    import os
    fpath_globinfo = r""
    output_directory = r''
    fpath_globinfo = os.path.abspath(fpath_globinfo)
    output_directory = os.path.abspath(output_directory)
    project_id = 'unzeub'
    edition_workflow(fpath_globinfo, output_directory, project_id)
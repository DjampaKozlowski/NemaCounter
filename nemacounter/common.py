
import cv2
import numpy as np
import pandas as pd
from functools import reduce
import configparser




def read_image(img_path):
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    #return cv2.imread(img_path)
    return image_rgb

def create_boxes(df):
    return df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

def create_global_table(lst_df, project_id):
    df = pd.concat(lst_df).reset_index(drop=True)
    #   add info 
    df['project_id'] = project_id
    #   drop un-used info
    df.drop(columns=['class', 'name'], inplace=True)
    #   re-order the columns
    df = df[['project_id', 'img_id', 'object_id',  'xmin', 'ymin', 'xmax', 'ymax', 'confidence']]
    return df

def calculation_per_group(grouped_df, colname, measurements=['mean', 'std'], rnd=3):
    """
    Compute measures per group for a given column
    """
    # calculate the measurements
    df = grouped_df.agg({colname: measurements})
    # rename the column by joining the original column name and the measurement
    df.columns = [f'{colname}_{m}' for m in measurements]
    # round the values 
    df = np.round(df, rnd)
    # reset the index (image id). Will be helpful to merge the differen df
    df = df.reset_index()
    return df

def merge_multiple_dataframes(lst_df, lst_k, how='inner'):
    """
    Merge multiple dataframes contained in a list (on a common list of keys)
    
    lst_k : list of key to use for the join. Can be a single element list
    how : type of merge to be performed
    """
    df_merged = reduce(lambda  left,right: pd.merge(left,
                                                    right,
                                                    on=lst_k,
                                                    how=how), 
                       lst_df)
    return df_merged
    

def create_summary_table(df, project_id):
    # Creates groups per image
    grouped_df = df.groupby('img_id')
    # Creates an empty list were df will be stored
    lst_df = []
    # Count the number of objects detected and store the info in a separated df
    lst_df.append(grouped_df.size().reset_index(name='detection_count'))
    # Compute the mean confidence value and its std per image
    lst_df.append(calculation_per_group(grouped_df, 'confidence'))
    # Test if the surface column exists and if so, compute mean and std values per image
    if 'surface' in df:
        lst_df.append(calculation_per_group(grouped_df, 'surface'))
    # Creates the summary dataframe from the list of df
    df_summary = merge_multiple_dataframes(lst_df, ['img_id'])
    # Add the session ID info (as the first column)
    df_summary.insert(loc=0, column='project_id', value=project_id)
    return df_summary


def get_config_info(fpath):
    try:
        config = configparser.ConfigParser()
        config.read(fpath)
        return config
    except:
        raise FileNotFoundError(f"The config file {fpath} does not exists")        
    
    
    
def tutu(a):
    if a == '1':
        raise FileNotFoundError('fait chier')
        print("mescouilles")
    else:
        print('ok')
    


# def create_project_dirs_structure(dpath_outdir, project_id, display_boxes=False, display_segmentation=False):
#     dpath_project = os.path.join(dpath_outdir, project_id)
#     if os.path.isdir(dpath_project):
#         print('Output directory already exists, program will stop.')
#         sys.exit()
#     else:
#         os.makedirs(dpath_project, mode=0o755)
#         if display_boxes:
#             os.makedirs(os.path.join(dpath_project, 'img', 'bounding_boxes'), mode=0o755)
#         if display_segmentation:
#             os.makedirs(os.path.join(dpath_project, 'img', 'segmentation'), mode=0o755)
from IPython.display import clear_output
import pandas as pd
import numpy as np
import re
import os
import time
from io import BytesIO

os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true' # for JP2
import cv2

os.system('pip install --upgrade azure-cognitiveservices-vision-computervision')
clear_output()

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
computervision_client = ComputerVisionClient(input('\nEndpoint?\n'), CognitiveServicesCredentials(input('\nKey?\n')))
clear_output()

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def get_ms_ocr_result(image_filepath, wait_interval=3, language = 'en'): 
  
    if image_filepath.endswith('.jp2'):
        image = cv2.imread(image_filepath)
        success, encoded_image = cv2.imencode('.jpg', image)
        read_image = BytesIO(encoded_image.tobytes())
    else:
        read_image = open(image_filepath, 'rb')

    read_response = computervision_client.read_in_stream(read_image, raw=True, language=language, reading_order='natural')
    read_operation_location = read_response.headers['Operation-Location']
    operation_id = read_operation_location.split('/')[-1]
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status.lower() not in ['notstarted', 'running']:
            break
        time.sleep(wait_interval)
    return read_result.as_dict()

def parse_ms_ocr_result(result):

    assert(len(result['analyze_result']['read_results']) == 1)
    read_result = result['analyze_result']['read_results'][0]
    metadata = {'operation_result':result['status'],
            'operation_creation_time':result['created_date_time'],
            'operation_last_update_time':result['last_updated_date_time'],
            'operation_api_version':result['analyze_result']['version'],
            'operation_model_versoin':result['analyze_result']['model_version'],
            'result_page_num':read_result['page'],
            'result_angle':read_result['angle'],
            'result_width':read_result['width'],
            'result_height':read_result['height'],
            'result_unit':read_result['unit'],
            }

    result_lines = read_result['lines']
    line_df = pd.DataFrame(result_lines) if len(result_lines)>0 else pd.DataFrame(columns=['bounding_box', 'appearance', 'text', 'words'])
    metadata['result_line_count'] = len(line_df)
    metadata['result_word_count'] = len(flatten_list(line_df['words']))
    return line_df, metadata

def make_point_int(pt):
    return (int(round(pt[0])),int(round(pt[1])))

def get_dist_from_pt_to_pt(pt1, pt2, rounding = 1):
  return np.round(np.linalg.norm(pt2-pt1), rounding)

def get_dist_from_pt_to_line(pt, line_endpoint_a, line_endpoint_b, rounding = 1):
    return np.round(np.abs(np.cross(line_endpoint_b - line_endpoint_a, line_endpoint_a - pt)) / np.linalg.norm(line_endpoint_b - line_endpoint_a), rounding)

def get_vector_direction(vector, rounding = 1):
  """np_arctan2_in_degree (-180 to 180 reference angle 0 is the positive direction of x axis in cartesian space)""" 
  return np.round(np.arctan2(*vector[::-1]) * 180 / np.pi, rounding)

def get_bbox_features(bbox, rounding = 0, make_int = True):
    tl, tr, br, bl = bbox
    width = np.mean([get_dist_from_pt_to_pt(tl, tr), get_dist_from_pt_to_pt(br, bl)])
    height = np.mean([get_dist_from_pt_to_line(tl, bl, br),  get_dist_from_pt_to_line(tr, bl, br),
                    get_dist_from_pt_to_line(bl, tl, tr),  get_dist_from_pt_to_line(br, tl, tr)])
    left_side_center = np.mean([tl,bl], axis=0)
    right_side_center = np.mean([tr,br], axis=0)
    reading_direction_vector = right_side_center - left_side_center
    reading_direction = get_vector_direction(reading_direction_vector)
    center = np.mean([tl, tr, br, bl], axis=0)
    
    if rounding is not None:
        width = np.round(width, rounding)
        height = np.round(height, rounding)
        left_side_center = np.round(left_side_center, rounding)
        right_side_center = np.round(right_side_center, rounding)
        center = np.round(center, rounding)
    if make_int: 
        width = int(width)
        height = int(height)
        left_side_center = left_side_center.astype(np.int32)
        right_side_center = right_side_center.astype(np.int32)
        center = center.astype(np.int32)

    return width, height, reading_direction, center, left_side_center, right_side_center

def mark_ocr_entry(entry_df, input_img_filepath, output_img_filepath, line_df = None, inplace=False):

    font_family = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255,0,0)
    thickness = 2
    line_type = cv2.LINE_AA

    offset = int(entry_df['height'].median()//4)

    canvas = cv2.imread(input_img_filepath)
    if inplace:
        canvas = np.ones(canvas.shape, dtype=np.uint8)*255
    for _, row in entry_df.iterrows():
        bbox_contour = np.array(row['bounding_box'],dtype=np.int32).reshape(-1,1,2)
        canvas = cv2.drawContours(canvas, [bbox_contour], -1, (0,0,255), 2)
        text_content = str(row['text'])
        text_position = make_point_int(np.array(row['right_side_center'])+np.array([offset,offset]))
        if inplace:
            text_position = make_point_int(row['left_side_center']+np.array((0,row['height']/2)))
        canvas = cv2.putText(canvas, text_content, text_position, font_family, font_scale, color, thickness, line_type)
    
    if line_df is not None:
        canvas = cv2.drawContours(canvas, line_df.loc[line_df['words'].apply(len)>1,'bounding_box'].apply(lambda x: np.array(x, dtype=np.int32).reshape(-1,1,2)).tolist(), -1, (0,255,0), 1)

    cv2.imwrite(output_img_filepath, canvas)

    return True

def run_ocr(img_filepath, wait_interval = 3, language = 'en', entry_df_csv_filepath = None, mark_img = False, mark_img_filepath = None, mark_img_inplace=False, mark_line_df = True, save_line_df = True, line_df_csv_filepath = None, save_ocr_metadata = True, ocr_metadata_filepath = None):
    
    raw_ocr_result = get_ms_ocr_result(img_filepath, wait_interval = wait_interval, language = language)
    line_df, ocr_metadata = parse_ms_ocr_result(raw_ocr_result)
    line_df = line_df.reset_index().rename(columns={'index':'line_id'})
    line_df['bounding_box'] = line_df['bounding_box'].apply(lambda x: np.array(x,dtype=np.int32).reshape(-1,2))
    line_df['width, height, reading_direction, center, left_side_center, right_side_center'.split(', ')] = pd.DataFrame(line_df['bounding_box'].apply(get_bbox_features).tolist(), index=line_df.index)
    
    entry_df = pd.DataFrame(flatten_list(line_df['words']))
    entry_df['line_id'] = flatten_list(line_df.apply(lambda row: [row['line_id']]*len(row['words']), axis=1))
    entry_df = entry_df.reset_index().rename(columns={'index':'entry_id'})
    entry_df['bounding_box'] = entry_df['bounding_box'].apply(lambda x: np.array(x,dtype=np.int32).reshape(-1,2))
    entry_df['width, height, reading_direction, center, left_side_center, right_side_center'.split(', ')] = pd.DataFrame(entry_df['bounding_box'].apply(get_bbox_features).tolist(), index=entry_df.index)

    img_extension = img_filepath.split('.')[-1]

    if mark_img:
        if mark_img_filepath is None:
            mark_img_filepath = img_filepath.replace('.'+img_extension,'__marked.jpg')
        if mark_line_df:
            mark_ocr_entry(entry_df, img_filepath, mark_img_filepath, line_df, inplace = mark_img_inplace)
        else:
            mark_ocr_entry(entry_df, img_filepath, mark_img_filepath, inplace = mark_img_inplace)

    if save_line_df:
        line_df['bounding_box'] = line_df['bounding_box'].apply(lambda x: re.sub(r'\s+','',repr(x)).replace('array','np.array').replace('int32','np.int32'))
        if line_df_csv_filepath is None:
            line_df_csv_filepath = img_filepath.replace('.'+img_extension,'__ocr_lines.csv')
        line_df.to_csv(line_df_csv_filepath, index=False)

    if save_ocr_metadata:
        ocr_metadata['img_filepath'] = img_filepath
        ocr_metadata['entry_df_csv_filepath'] = entry_df_csv_filepath
        ocr_metadata['line_df_csv_filepath'] = line_df_csv_filepath
        ocr_metadata['mark_img_filepath'] = mark_img_filepath
        if ocr_metadata_filepath is None:
            ocr_metadata_filepath = img_filepath.replace('.'+img_extension,'__ocr_metadata.txt')
        ocr_metadata['ocr_metadata_filepath'] = ocr_metadata_filepath
        with open(ocr_metadata_filepath, 'w') as f:
            f.write(str(ocr_metadata))

    for field in ['bounding_box','center','left_side_center','right_side_center']:
        entry_df[field] = entry_df[field].apply(lambda x: re.sub(r'\s+','',repr(x)).replace('array','np.array').replace('int32','np.int32'))

    if entry_df_csv_filepath is None:
        entry_df_csv_filepath = img_filepath.replace('.'+img_extension,'__ocr_entries.csv')
    entry_df.to_csv(entry_df_csv_filepath, index=False)

    return True

os.rename('/content/one_click_ocr/example.jpg','/content/example.jpg')

clear_output()
print("\nThe OCR tool is ready to use. Try running this line of code below:\n\nrun_ocr('/content/one_click_ocr/example.jpg')\n\n * Optional parameters include: language (str, 2-digit language code), wait_interval (int, in seconds)\n * You will find the OCR results in the same directory as the input image, if you don't specify custom filepaths.\n\nTo save a copy of the image with OCR entries marked, use the `mark_img` parameter: \n\nrun_ocr('/content/one_click_ocr/example.jpg', mark_img=True)\n\nThis OCR tool is developed as part of the Mapping Historical New York project (https://mappinghny.com) at Columbia University.\nFor more questions about the tool, please contact Tim Wu at gw2415@columbia.edu.\n\n")

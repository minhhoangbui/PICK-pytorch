import os
import glob
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import json
import csv
import cv2


def process_excel(file_name, o_image_dir=None, out_boxes_transcripts_dir=None,
                  entities_dir=None, image_dir=None):
    df = pd.read_excel(file_name, engine='openpyxl')
    file_name = os.path.basename(file_name)
    file_name = os.path.splitext(file_name)[0]

    image_height = int(df.loc[0, 'height'])
    image_width = int(df.loc[0, 'width'])
    original_image_path = os.path.join(o_image_dir, file_name + '.jpg')
    if os.path.isfile(original_image_path):
        image = cv2.imread(original_image_path)
        if image.shape[0] != image_height or image.shape[1] != image_width:
            print(file_name)
            print(image_width, image_height)
            print(image.shape)
            return False
        shutil.copy(os.path.join(o_image_dir, file_name + '.jpg'),
                    os.path.join(image_dir, file_name + '.jpg'))
    else:
        print(f'{original_image_path} not found')

    file_name = os.path.basename(file_name)

    df[['xmin', 'ymin', 'xmax', 'ymax']] = df[['xmin', 'ymin', 'xmax', 'ymax']].apply(pd.to_numeric)

    df['x1'] = df['xmin']
    df['y1'] = df['ymin']
    df['x2'] = df['xmax']
    df['y2'] = df['ymin']
    df['x3'] = df['xmax']
    df['y3'] = df['ymax']
    df['x4'] = df['xmin']
    df['y4'] = df['ymax']

    df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1, inplace=True)

    cols = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'object', 'label']
    df = df[cols]

    # Generate .tsv file

    df.to_csv(os.path.join(out_boxes_transcripts_dir, file_name + '.tsv'),
              index=True, header=False, quotechar='',
              escapechar='\\', quoting=csv.QUOTE_NONE)

    # Generate entities file
    entities = dict()
    for _, row in df.iterrows():
        if row['label'] != 'other':
            if entities.get(row['label'], None) is None:
                entities[row['label']] = ''
            entities[row['label']] += row['object'] + ', '

    for k, v in entities.items():
        entities[k] = v[:-2]
    with open(os.path.join(entities_dir, file_name + '.txt'), 'w') as fp:
        json.dump(entities, fp)
    return True


def check_availability(ocr_folder, image_folder):
    ocr_list = os.listdir(ocr_folder)
    image_list = os.listdir(image_folder)
    ocr_list = [os.path.splitext(of)[0] for of in ocr_list]
    image_list = [os.path.splitext(im)[0] for im in image_list]
    for of in ocr_list:
        if of not in image_list:
            print(f'{of} not in image_list')
    for im in image_list:
        if im not in ocr_list:
            print(f'{im} not in ocr_list')


def check_duplicates(old_dir, new_dir):
    old_list = os.listdir(old_dir)
    new_list = os.listdir(new_dir)
    old_list = [os.path.splitext(of)[0] for of in old_list]
    new_list = [os.path.splitext(nf)[0] for nf in new_list]
    count = 0
    for of in old_list:
        if of in new_list:
            count += 1
        else:
            print(of)
    print(f'{count} files duplicated')


if __name__ == '__main__':
    ocr_dir = '/data/hoangbm/datasets/ner/bizi/ocr'
    image_dir = '/data/hoangbm/datasets/ner/bizi/images'
    # check_availability(ocr_dir, image_dir)
    file_list = glob.glob(os.path.join(ocr_dir, '*.xlsx'))
    train, test = train_test_split(file_list, test_size=0.2, random_state=42)
    print(f"Number of training sample: {len(train)}")
    print(f"Number of test sample: {len(test)}")
    train_image_dir = '/data/hoangbm/datasets/ner/bizi/train/images'
    train_entity_dir = '/data/hoangbm/datasets/ner/bizi/train/entities'
    train_boxes_dir = '/data/hoangbm/datasets/ner/bizi/train/boxes_and_transcripts'
    if os.path.exists(train_image_dir):
        shutil.rmtree(train_image_dir)
    os.makedirs(train_image_dir)
    if os.path.exists(train_entity_dir):
        shutil.rmtree(train_entity_dir)
    os.makedirs(train_entity_dir)
    if os.path.exists(train_boxes_dir):
        shutil.rmtree(train_boxes_dir)
    os.makedirs(train_boxes_dir)

    test_image_dir = '/data/hoangbm/datasets/ner/bizi/test/images'
    test_entity_dir = '/data/hoangbm/datasets/ner/bizi/test/entities'
    test_boxes_dir = '/data/hoangbm/datasets/ner/bizi/test/boxes_and_transcripts'

    if os.path.exists(test_image_dir):
        shutil.rmtree(test_image_dir)
    os.makedirs(test_image_dir)
    if os.path.exists(test_entity_dir):
        shutil.rmtree(test_entity_dir)
    os.makedirs(test_entity_dir)
    if os.path.exists(test_boxes_dir):
        shutil.rmtree(test_boxes_dir)
    os.makedirs(test_boxes_dir)

    with open('/data/hoangbm/datasets/ner/bizi/train/train_list.csv', mode='w') as fp:
        writer = csv.writer(fp, delimiter=',')
        for i, fname in enumerate(train):
            successful = process_excel(fname, o_image_dir=image_dir,
                                       image_dir=train_image_dir,
                                       entities_dir=train_entity_dir,
                                       out_boxes_transcripts_dir=train_boxes_dir)
            if not successful:
                continue
            temp_name = os.path.basename(fname)
            temp_name = os.path.splitext(temp_name)[0]
            writer.writerow([i, temp_name])

    with open('/data/hoangbm/datasets/ner/bizi/test/test_list.csv', mode='w') as fp:
        writer = csv.writer(fp, delimiter=',')
        for i, fname in enumerate(test):
            successful = process_excel(fname, o_image_dir=image_dir,
                                       image_dir=test_image_dir,
                                       entities_dir=test_entity_dir,
                                       out_boxes_transcripts_dir=test_boxes_dir)
            if not successful:
                continue
            temp_name = os.path.basename(fname)
            temp_name = os.path.splitext(temp_name)[0]
            writer.writerow([i, temp_name])


import os
import pandas
import json
import csv
import shutil
from sklearn.model_selection import train_test_split

# Input dataset
data_path = '/home/hoangbm/ICDAR-2019-SROIE/data'
box_path = os.path.join(data_path, 'box')
img_path = os.path.join(data_path, 'img')
key_path = os.path.join(data_path, 'key')

# Output
out_dir = '/home/hoangbm/sroie19'
out_boxes_transcripts = out_dir + '/boxes_and_transcripts'
out_images = out_dir + '/images'
out_entities = out_dir + '/entities'

if os.path.exists(out_boxes_transcripts):
    shutil.rmtree(out_boxes_transcripts)
os.makedirs(out_boxes_transcripts)
if os.path.exists(out_entities):
    shutil.rmtree(out_entities)
os.makedirs(out_entities)
if os.path.exists(out_images):
    shutil.rmtree(out_images)
os.makedirs(out_images)

samples = []
for file in os.listdir(box_path):
    with open(os.path.join(box_path, file), 'r') as fp:
        reader = csv.reader(fp, delimiter=',')
        rows = [x[:8] + [','.join(x[8:]).strip(',')] for x in reader]
    df = pandas.DataFrame(rows)

    df[9] = 'other'

    jpg = file.replace(".csv", ".jpg")
    entities = json.load(open(os.path.join(key_path, file.replace('.csv', '.json'))))

    for key, value in sorted(entities.items()):
        idx = df[df[8].str.contains('|'.join(map(str.strip, value.split(','))))].index
        df.loc[idx, 9] = key
    shutil.copy(os.path.join(img_path, jpg), out_images)

    with open(os.path.join(out_entities, file.replace(".csv", ".txt")), 'w') as fj:
        json.dump(entities, fj)

    df.to_csv(os.path.join(out_boxes_transcripts, file.replace(".csv", ".tsv")),
              index=True, header=False, quotechar='',
              escapechar='\\', quoting=csv.QUOTE_NONE, )
    samples.append(file.replace('.csv', ''))
print('*' * 50)
print(f"Total samples: {len(samples)}")
samples = pandas.DataFrame(samples)
train, test = train_test_split(samples, test_size=0.2, random_state=42)

for _, row in train.iterrows():
    shutil.copy(os.path.join(out_boxes_transcripts, str(row[0])+'.tsv'),
                '/home/hoangbm/PICK-pytorch/data/data_examples_root/boxes_and_transcripts')
    shutil.copy(os.path.join(out_images, str(row[0]) + '.jpg'),
                '/home/hoangbm/PICK-pytorch/data/data_examples_root/images')
    shutil.copy(os.path.join(out_entities, str(row[0]) + '.txt'),
                '/home/hoangbm/PICK-pytorch/data/data_examples_root/entities')

train.reset_index(inplace=True)
train.drop(['index'], axis=1, inplace=True)
train.to_csv('/home/hoangbm/PICK-pytorch/data/data_examples_root/train_samples_list.csv', header=False)

for _, row in test.iterrows():
    shutil.copy(os.path.join(out_boxes_transcripts, str(row[0]) + '.tsv'),
                '/home/hoangbm/PICK-pytorch/data/test_data_example/boxes_and_transcripts')
    shutil.copy(os.path.join(out_images, str(row[0]) + '.jpg'),
                '/home/hoangbm/PICK-pytorch/data/test_data_example/images')
    shutil.copy(os.path.join(out_entities, str(row[0]) + '.txt'),
                '/home/hoangbm/PICK-pytorch/data/test_data_example/entities')

test.reset_index(inplace=True)
test.drop(['index'], axis=1, inplace=True)
test.to_csv('/home/hoangbm/PICK-pytorch/data/test_data_example/test_samples_list.csv', header=False)

shutil.rmtree(out_dir)







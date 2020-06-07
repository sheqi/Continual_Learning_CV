import os
import json
import pdb
import pandas as pd

jason_file_path = '/home/data2/zhengwei/somethingv2'
frame_path = '/home/data2/zhengwei/somethingv2/20bn-something-something-v2-frames'
annotation_path = '../somethingv2_annotation'
if not os.path.exists(annotation_path):
    os.makedirs(annotation_path)

if __name__ == '__main__':
    dataset_name = 'something-something-v2'  # 'jester-v1'
    with open(os.path.join(jason_file_path, '%s-labels.json' % dataset_name), encoding='utf-8') as f:
        data = json.load(f)
    categories = []
    for i, (cat, idx) in enumerate(data.items()):
        assert i == int(idx)  # make sure the rank is right
        categories.append(cat)


    with open(os.path.join(annotation_path, 'category.txt'), 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    files_input = ['%s-validation.json' % dataset_name, '%s-train.json' % dataset_name, '%s-test.json' % dataset_name]
    files_output = ['val_videofolder.txt', 'train_videofolder.txt', 'test_videofolder.txt']
    files_output_pkl = ['val.pkl', 'train.pkl', 'test.pkl']
    for (filename_input, filename_output, filename_output_pkl) in zip(files_input, files_output, files_output_pkl):
        with open(os.path.join(jason_file_path, filename_input), encoding='utf-8') as f:
            data = json.load(f)
        folders = []
        idx_categories = []
        for item in data:
            folders.append(item['id'])
            if 'test' not in filename_input:
                idx_categories.append(dict_categories[item['template'].replace('[', '').replace(']', '')])
            else:
                idx_categories.append(0)
        output = []
        annot_df = {'frame': [], 'label': []}
        for i in range(len(folders)):
            raw_path = os.path.join(frame_path, folders[i])
            frame_names = sorted(os.listdir(raw_path))
            video_names = []
            for frame_name in frame_names:
                if frame_name.split('.')[1] == 'jpg':
                    video_names.append(os.path.join(raw_path, frame_name))
            annot_df['frame'].append(video_names)
            annot_df['label'].append(idx_categories[i])
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            dir_files = os.listdir(os.path.join(jason_file_path, '20bn-something-something-v2-frames', curFolder))
            output.append('%s %d %d' % (curFolder, len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        annot_df = pd.DataFrame(annot_df)
        with open(os.path.join(annotation_path, filename_output), 'w') as f:
            f.write('\n'.join(output))
        annot_df.to_pickle(os.path.join(annotation_path, filename_output_pkl))

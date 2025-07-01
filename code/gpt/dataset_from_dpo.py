import json

import argparse


def get_datset(train_json_path, val_json_path, train_dataset_path, val_dataset_path):
    with open(train_json_path, 'r') as file:
        train_json = json.load(file)


    with open(val_json_path, 'r') as file:
        val_json = json.load(file)


    seen_objects_train = set()
    seen_objects_val = set()

    train_ds = []
    for element in train_json:
        if element['objects'] not in  seen_objects_train:
            element['output'] = element['chosen']
            element.pop('chosen')
            element.pop('rejected')
            train_ds.append(element)
            seen_objects_train.add(element['objects'])


    val_ds = []
    for element in val_json:
        if element['objects'] not in  seen_objects_val:
            element['output'] = element['chosen']
            element.pop('chosen')
            element.pop('rejected')
            val_ds.append(element)
            seen_objects_val.add(element['objects'])

    print(f'Train length: {len(train_ds)}, validation length: {len(val_ds)}')




    with open(train_dataset_path, "w") as file:
        json.dump(train_ds, file, indent = 4)

    with open(val_dataset_path, "w") as file:
        json.dump(val_ds, file, indent = 4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json_path", type=str, default="/scratch/leuven/370/vsc37045/resource/dpo/dataset/train.json")
    parser.add_argument("--train_dataset_path", type=str, default="/scratch/leuven/370/vsc37045/resource/gpt/dataset/train_aligned.json")
    parser.add_argument("--val_json_path", type=str, default="/scratch/leuven/370/vsc37045/resource/dpo/dataset/val.json")
    parser.add_argument("--val_dataset_path", type=str, default="/scratch/leuven/370/vsc37045/resource/gpt/dataset/val_aligned.json")
    args = parser.parse_args()
    get_datset(args.train_json_path, args.val_json_path, args.train_dataset_path, args.val_dataset_path)
    
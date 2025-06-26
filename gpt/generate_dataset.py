import json
import argparse

def combine(train_json_path, val_json_path, train_answers_path, val_answers_path, train_dataset_path, val_dataset_path, test_json_path, test_answers_path, test_dataset_path, train_val_dataset_path):
    correct_answers = {
        'whale-hamster':'a big furry brown hamster is holding a glass box filled with water in which a small blue whale with prolonged tail is swimming. the hamster is bigger than the whale.',
        'ostrich-eyeglasses': 'a giant pair of eyeglasses lying on the ground in foreground and a tiny ostrich standing behind the eyeglasses in the background. the ostrich is much smaller than the eyeglasses.',
        'rhino-scarf':'a distant view of a tiny rhino lying on the ground wrapped in layers and layers of a giant red scarf.',
        'helicopter-snail':'a giant snail resting in the grass and holding a tiny helicopter flying above its shell.',
        'tank-dice':'a huge dice in an open space, with a tiny tank positioned next to it, looking significantly smaller.',
        'church-fly':'a colossal fly perched on the ground, with a tiny church standing far below it, highlighting the absurd scale difference',
        'trex-tshirt':'a massive t-shirt spread out on the ground, with a tiny t-rex standing inside one of its pockets.',
        'train-comb':'a huge comb lying on the ground, with a tiny train running along its edge.',
        'hippopotamus-glove':'a giant glove lying open, with a tiny hippopotamus curled up inside its opening.',
        'walrus-shorts':'a giant pair of shorts folded neatly, with a tiny walrus standing inside one of the loops.',
        'chinese temple-seashell':'a huge seashell resting on a table, with a tiny chinese temple placed inside its opening.',
        'statue of unity-bee':'a giant bee resting on the ground, with a tiny statue of unity standing next to its wing.'}





    with open(train_json_path, 'r') as file:
        train_json = json.load(file)

    with open(val_json_path, 'r') as file:
        val_json = json.load(file)

    with open(test_json_path, 'r') as file:
        test_json = json.load(file)


    with open(train_answers_path, 'r') as file:
        train_answers = file.read().split('\n')

    with open(val_answers_path, 'r') as file:
        val_answers = file.read().split('\n')

    with open(test_answers_path, 'r') as file:
        test_answers = file.read().split('\n')

    m = len(train_json)
    n = len(val_json)
    o = len(test_json)

    assert m == len(train_answers), f'{m}, {len(train_answers)}'
    assert n == len(val_answers),  f'{n}, {len(val_answers)}'
    assert o == len(test_answers), f"{o}, {len(test_answers)}"

    used = 0
    for i in range(m):
        if train_json[i]['objects'] in correct_answers:
            train_json[i]['output'] = correct_answers[train_json[i]['objects']] 
            correct_answers.pop(train_json[i]['objects']) 
            used += 1
        else:   
            train_json[i]['output'] = train_answers[i].strip()
       
    for i in range(n):
        val_json[i]['output'] = val_answers[i].strip()

    for i in range(o):
        test_json[i]['output'] = test_answers[i].strip()

    assert used == 12, f'{correct_answers}'

    combined_train_val = train_json + val_json

    
    with open(train_dataset_path, 'w') as file:
        json.dump(train_json, file, indent = 4)

    with open(val_dataset_path, 'w') as file:
        json.dump(val_json, file, indent = 4)
    

    with open(test_dataset_path, 'w') as file:
        json.dump(test_json, file, indent = 4)

    with open(train_val_dataset_path, 'w') as file:
        json.dump(combined_train_val, file, indent = 4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_json_path", type=str, default="/scratch/leuven/370/vsc37045/resource/gpt/separation/train.json")
    parser.add_argument("--val_json_path", type=str, default="/scratch/leuven/370/vsc37045/resource/gpt/separation/val.json")
    parser.add_argument("--train_answers_path", type = str, default = "/scratch/leuven/370/vsc37045/resource/gpt/prompts/train_answers.txt")
    parser.add_argument("--val_answers_path", type=str, default="/scratch/leuven/370/vsc37045/resource/gpt/prompts/val_answers.txt")
    parser.add_argument("--train_dataset_path", type=str, default="/scratch/leuven/370/vsc37045/resource/gpt/dataset/train.json")
    parser.add_argument("--val_dataset_path", type = str, default = "/scratch/leuven/370/vsc37045/resource/gpt/dataset/val.json")
    parser.add_argument("--test_json_path", type=str, default="/scratch/leuven/370/vsc37045/resource/gpt/separation/test.json")
    parser.add_argument("--test_answers_path", type=str, default="/scratch/leuven/370/vsc37045/resource/gpt/prompts/test_answers.txt")
    parser.add_argument("--test_dataset_path", type = str, default = "/scratch/leuven/370/vsc37045/resource/gpt/dataset/test.json")
    parser.add_argument("--train_val_dataset_path", type = str, default = "/scratch/leuven/370/vsc37045/resource/gpt/dataset/train+val.json")
   
    args = parser.parse_args()
    combine(args.train_json_path, args.val_json_path, args.train_answers_path, args.val_answers_path, args.train_dataset_path, args.val_dataset_path, args.test_json_path, args.test_answers_path, args.test_dataset_path, args.train_val_dataset_path)
import argparse
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/scratch/leuven/370/vsc37045/resource/gpt/dataset/test.json")
    parser.add_argument("--savepath", type=str, default="/scratch/leuven/370/vsc37045/resource/assessments/template_test.json")
    args = parser.parse_args()

    with open(args.data, "r") as file:
        content = json.load(file)

    for i, element in enumerate(content):
        content[i]['output'] = element['input']


    with open(args.savepath, 'w') as file:
        json.dump(content, file, indent = 4)

    
	

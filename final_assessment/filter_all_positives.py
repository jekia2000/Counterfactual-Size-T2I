from pathlib import Path
import json 
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_folder", type=str, default="/data_path")
	args = parser.parse_args()


	for folder in Path(args.data_folder).iterdir():
		if folder.with_suffix('').name.endswith('evaluated'):
			with open(folder, 'r') as file:
				content = json.load(file)
			cnt = 0.0	
			for element in content:
				if element['reward'] > 1:
					cnt += 1
			print(f'Name: {folder.with_suffix('').name}  accuracy: {cnt/len(content)}')

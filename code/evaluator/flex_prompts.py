import json 
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("flex_prompts", add_help=True)
    parser.add_argument("--img", type = str, default = None, help = "path to genrated images")
    parser.add_argument("--prompt_path", type = str, default = None, help = "path to json containing prompts")
    args = parser.parse_args()

    img_path = args.img
    out_json_path = args.prompt_path

    path2prompt = {}
 
    for image_path in Path(img_path).iterdir():
        name = image_path.with_suffix("").name
        objects = name.split('-')
        if len(objects) == 1:
            prompt = objects[0].strip()
        elif len(objects) == 2:
            object1 = objects[0]
            cand = objects[1]
            if cand.isdigit():
                prompt = object1.strip()
            else:
                prompt = object1.strip() + '-' + cand.strip()

        else:
            object1 = objects[0]
            object2 = objects[1]
            prompt = object1.strip() + '-' + object2.strip()    


    
        path2prompt[str(image_path)] = prompt

    with open(out_json_path, "w") as file:
        json.dump(path2prompt, file)

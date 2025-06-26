import json
import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default='/scratch/leuven/370/vsc37045/resource/dpo/separation/train.json')
    parser.add_argument("--val_path", type=str, default='/scratch/leuven/370/vsc37045/resource/dpo/separation/val.json')
    parser.add_argument("--test_path", type = str, default = '/scratch/leuven/370/vsc37045/resource/dpo/separation/test.json')
    parser.add_argument("--prompt_save_dir", type=str, default='/scratch/leuven/370/vsc37045/resource/gpo/prompts')
 
    
    args = parser.parse_args()
   
   
    with open(args.train_path, 'r') as file:
        train_content = json.load(file)

    with open(args.val_path, 'r') as file:
        val_content = json.load(file)

    with open(args.test_path, 'r') as file:
        test_content = json.load(file)
  

    prompt_template = '\nBig {small_object} and small {big_object}. The {big_object} is much smaller than the {small_object}.\n####################'
    gpt_template = '''
    Can you rewrite the given prompts?
    Please, keep the style simple and always emphasize the size difference.
    Avoid using comparissons where you name an object which is not actually in the scene (E.g. if you say a shoe big enough to hold a melon, this is wrong as melon will not actually be in the image).
    
    Here are some ground truth examples. 
    
    Original: 
    Big hamster and small whale. The whale is much smaller than the hamster.
    Rewritten:
    a big furry brown hamster is holding a glass box filled with water in which a small blue whale with prolonged tail is swimming. the hamster is bigger than the whale.
    #######################
    Original:
    Big eyeglasses and small ostrich. The ostrich is much smaller than the eyeglasses.
    Rewritten:
    a giant pair of eyeglasses lying on the ground in foreground and a tiny ostrich standing behind the eyeglasses in the background. the ostrich is much smaller than the eyeglasses.
    ######################################
    Original:
    Big scarf and small rhino. The rhino is much smaller than the scarf.
    Rewritten:
    a distant view of a tiny rhino lying on the ground wrapped in layers and layers of a giant red scarf.
    ##########################################
    Original:
    Big snail and small helicopter. The helicopter is much smaller than the snail.
    Rewritten:
    a giant snail resting in the grass and holding a tiny helicopter flying above its shell.
    ###########################################
    Original:
    Big dice and small tank. The tank is much smaller than the dice.
    Rewritten:
    a huge dice in an open space, with a tiny tank positioned next to it, looking significantly smaller.
    ###########################################
    Original:
    Big fly and small church. The church is much smaller than the fly.
    Rewritten:
    a colossal fly perched on the ground, with a tiny church standing far below it, highlighting the absurd scale difference
    ###########################################
    Original:
    Big tshirt and small trex. The trex is much smaller than the tshirt.
    Rewritten:
    a massive t-shirt spread out on the ground, with a tiny t-rex standing inside one of its pockets.
    ##############################################
    Original:
    Big comb and small train. The train is much smaller than the comb.
    Rewritten:
    a huge comb lying on the ground, with a tiny train running along its edge.
    ################################################
    Original:
    Big glove and small hippopotamus. The hippopotamus is much smaller than the glove.
    Rewritten:
    a giant glove lying open, with a tiny hippopotamus curled up inside its opening.
    ################################################
    Original:
    Big shorts and small walrus . The walrus  is much smaller than the shorts.
    Rewritten:
    a giant pair of shorts folded neatly, with a tiny walrus standing inside one of the loops.
    ####################################################
    Original:
    Big seashell and small chinese temple. The chinese temple is much smaller than the seashell.
    Rewritten:
    a huge seashell resting on a table, with a tiny chinese temple placed inside its opening.
    ############################################################
    Original:
    Big bee and small statue of unity . The statue of unity is much smaller than the bee.
    Rewritten:
    a giant bee resting on the ground, with a tiny statue of unity standing next to its wing.
    ########################################################################
    And here are the prompts to be rewritten:
    
    '''
    def create_gpt_prompt(content, filepath, name):
        collected_prompts = []
        n = len(content)
        cnt = 0
        file_name = 0
        for row in content:
            big_object, small_object = row['objects'].split('-')
            cnt += 1
            collected_prompts.append(prompt_template.format(small_object = small_object, big_object = big_object))
            if cnt % 50 == 0 or cnt == n:
                with open(f'{filepath}/{name}_{file_name}.txt', 'w') as file:
                    file.write(gpt_template + ''.join(collected_prompts))
                collected_prompts = []
                file_name += 1
    
    create_gpt_prompt(train_content, args.prompt_save_dir, 'train')
    create_gpt_prompt(val_content, args.prompt_save_dir, 'val')
    create_gpt_prompt(test_content, args.prompt_save_dir, 'test')
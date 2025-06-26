import pandas as pd
import json
import re
import argparse
from collections import defaultdict

def create_train_test(class_file, save_train, save_val, save_test):
    df = pd.read_csv(class_file)
    all_names = df['names']
    big = df[df['type'] == 'big'][['names', 'category']]
    small = df[df['type'] == 'small'][['names', 'category']]
    big = big.rename(columns = {'names':'big_names', 'category':'big_category'})
    small = small.rename(columns = {'names':'small_names', 'category':'small_category'})
    instance2limit = defaultdict(bool)
    individual_cat2limit =defaultdict(lambda: defaultdict(list))

    for _, row in big.iterrows():
        instance2limit[row['big_names'].strip(' ')]  = True
        individual_cat2limit["big_"+row['big_category'].strip(' ')]['elements'].append(row['big_names'].strip(' '))
        individual_cat2limit["big_"+row['big_category'].strip(' ')]['filled_up'].append(0)

    for _, row in small.iterrows():
        instance2limit[row['small_names'].strip(' ')] = True
        individual_cat2limit["small_"+row['small_category'].strip(' ')]['elements'].append(row['small_names'].strip(' '))
        individual_cat2limit["small_"+row['small_category'].strip(' ')]['filled_up'].append(0)

    merged_df = big.join(small, how = 'cross')
    groups = list(merged_df.groupby(by = ['big_category', 'small_category']))

    reserved_pairs= set(['hamster-whale', 'eyeglasses-ostrich', 'scarf-rhino', 'snail-helicopter', 'dice-tank', 'fly-church', 'tshirt-trex', 'comb-train', 'glove-hippopotamus','shorts-walrus', 'seashell-chinese temple', 'bee-statue of unity'])
    train_pairs = []
    test_pairs = []
    val_pairs = []
    val_dict = {'small':defaultdict(int), 'big':defaultdict(int)}
    train_dict = defaultdict(int)
    test_dict = defaultdict(int)
    train_categories = defaultdict(int)
    test_categories = defaultdict(int)
    categories = set()
    taken_train = 0
    outer_flag = True


    #############################################
    for group in groups:
        send_to_valid = True
        big_category = "big_"+group[0][0]
        small_category = "small_"+group[0][1]
        categories.add(f'{big_category}-{small_category}')
        for subgroup in group[1].groupby(by = ['big_names']):
            inner_flag = outer_flag
            for _,row in subgroup[1].sort_values(by = ['small_names']).iterrows():
                big_object = row['big_names'].strip(' ')
                small_object = row['small_names'].strip(' ')

                if f'{small_object}-{big_object}' in reserved_pairs:
                    train_pairs.append({'categories':f'{big_category}-{small_category}','objects':f'{big_object}-{small_object}', 'input':f'Big {small_object} and small {big_object}. The {big_object} is much smaller than the {small_object}.'})
                    train_dict[big_object] += 1
                    train_dict[small_object] += 1
                    train_categories[f'{big_category}-{small_category}'] += 1
                    inner_flag = False
                    taken_train += 1
                

                elif send_to_valid and (instance2limit[big_object] and instance2limit[small_object]):
                    instance2limit[big_object] = False
                    individual_cat2limit[big_category]['filled_up'][0] += 1
                    if individual_cat2limit[big_category]['filled_up'][0] == len(individual_cat2limit[big_category]['elements']):
                    
                        for element in individual_cat2limit[big_category]['elements']:
                            instance2limit[element] = True
                        individual_cat2limit[big_category]['filled_up'][0] = 0
                    instance2limit[small_object] = False
                    individual_cat2limit[small_category]['filled_up'][0] += 1
                    if individual_cat2limit[small_category]['filled_up'][0] == len(individual_cat2limit[small_category]['elements']):
                    
                        for element in individual_cat2limit[small_category]['elements']:
                            instance2limit[element] = True
                        individual_cat2limit[small_category]['filled_up'][0] = 0
                    val_dict['small'][small_object] += 1
                    val_dict['big'][big_object] += 1
                    send_to_valid = False
                    val_pairs.append({'categories':f'{big_category}-{small_category}','objects':f'{big_object}-{small_object}', 'input':f'Big {small_object} and small {big_object}. The {big_object} is much smaller than the {small_object}.'})
                             

                elif inner_flag and f'{small_object}-{big_object}':
                    train_pairs.append({'categories':f'{big_category}-{small_category}','objects':f'{big_object}-{small_object}', 'input':f'Big {small_object} and small {big_object}. The {big_object} is much smaller than the {small_object}.'})
                    train_dict[big_object] += 1
                    train_dict[small_object] += 1
                    train_categories[f'{group[0][0]}-{small_category}'] += 1
                    inner_flag = not inner_flag
                    
                
                elif f'{small_object}-{big_object}':
                    test_pairs.append({'categories':f'{big_category}-{small_category}','objects':f'{big_object}-{small_object}', 'input':f'Big {small_object} and small {big_object}. The {big_object} is much smaller than the {small_object}.'})
                    test_dict[big_object] += 1
                    test_dict[small_object] += 1
                    test_categories[f'{big_category}-{small_category}'] += 1
                    inner_flag = not inner_flag

                 
            outer_flag = not outer_flag

        


    
    assert len(categories) == 49, 'Not all categories present'
    assert taken_train == 12, f'Reserved pairs for train not used: {taken_train}'

    assert len(val_pairs) == 49, 'Not all categories present'
            
    assert len(train_pairs) + len(test_pairs)+ len(val_pairs) == 46*45, 'Some connections not used'

    for category in categories:
        assert train_categories[category] - test_categories[category] <= 2, f'{category} {train_categories[category] - test_categories[category]}'

    for name in all_names:
        total_no_instances = train_dict[name] + test_dict[name]
        assert total_no_instances >= 41, 'Impossible total number of instances'
        assert abs(train_dict[name] - test_dict[name]) <= 6, f'{name} {train_dict[name]} {test_dict[name]}'


    print(len(train_pairs), len(test_pairs), len(val_pairs))
   
    with open(save_train, 'w') as file:
        json.dump(train_pairs, file, indent = 4)


    with open(save_test, 'w') as file:
        json.dump(test_pairs, file, indent = 4)


    with open(save_val, 'w') as file:
        json.dump(val_pairs, file, indent = 4)

    print(val_dict)



    
if __name__ == '__main__':
    #class_file, gpt_train_file, gpt_test_file, save_train, save_test
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_file", type=str, default="/scratch/leuven/370/vsc37045/resource/union.csv")
    parser.add_argument("--save_train", type=str, default="/scratch/leuven/370/vsc37045/resource/gpt/separation/train.json")
    parser.add_argument("--save_test", type = str, default = "/scratch/leuven/370/vsc37045/resource/gpt/separation/test.json")
    parser.add_argument("--save_val", type = str, default = "/scratch/leuven/370/vsc37045/resource/gpt/separation/val.json")
    args = parser.parse_args()
    create_train_test(args.class_file, args.save_train, args.save_val, args.save_test)






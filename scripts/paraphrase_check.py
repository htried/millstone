import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import DATA_DIR


def main():
    changed_files = []

    for file in os.listdir(DATA_DIR):
        if file.endswith('.json'):
            file_path = os.path.join(DATA_DIR, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
            paraphrases = data['paraphrases']
            updated = False
            for paraphrase in paraphrases[3:]:
                if paraphrase['orientation'] == 'pro':
                    print(paraphrase['question'])
                    print('input 1 for pro, 0 for con')
                    i = input()
                    if (i == '1' and paraphrase['orientation'] == 'pro') or (i == '0' and paraphrase['orientation'] == 'con'):
                        print('match')
                    else:
                        print('no match')
                        # Update the orientation to match the input
                        if i == '1':
                            paraphrase['orientation'] = 'pro'
                        elif i == '0':
                            paraphrase['orientation'] = 'con'
                        else:
                            print('Invalid input, skipping update.')
                            continue
                        updated = True
            if updated:
                # Write the updated data to data_new/ directory
                new_dir = 'data_new'
                os.makedirs(new_dir, exist_ok=True)
                new_file_path = os.path.join(new_dir, file)
                with open(new_file_path, 'w') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                changed_files.append(file)

    if changed_files:
        print("Changed files:")
        for fname in changed_files:
            print(fname)
    else:
        print("No files changed.")
    
    return changed_files


if __name__ == "__main__":
    main()
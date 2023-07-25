import os


def navigate_dir(curr_dir, api_str):
    for filename in os.listdir(curr_dir):
        if filename not in exclude:
            f = os.path.join(curr_dir, filename)
            if 'riid' in curr_dir:
                temp = f.replace('..\\', '::: ').replace('\\', '.').replace('.py', '')
                api_str += temp + '\n\n'
            else:
                if os.path.isfile(f):
                    temp = f.replace('..\\', '::: ').replace('\\', '.').replace('.py', '')
                    api_str += temp + '\n\n'
                else:
                    api_str = navigate_dir(f, api_str)

    return api_str


# start building the contents for the api.md

# list the folders you want to include in the api
folders_to_include = ['riid', 'tests', 'examples']

# list specific files you do not want in the api
exclude = ['run_examples.py', 'courses', '__pycache__', '__init__.py']

# navigate throught the desired folders and get file names
curr_path = os.getcwd()
api_str = ""

for filename in folders_to_include:
    f = os.path.join('..\\', filename)
    api_str = navigate_dir(f, api_str)

# print(api_str)

writer = open('api.md', 'w')
writer.write("# API Reference \n\n")
writer.write(api_str)
writer.close()

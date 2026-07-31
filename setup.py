from setuptools import setup,find_packages
from typing import List

PROJECT_NAME='Mlp1'
VERSION='0.01'
DESCRIPTION='hi my first project'
AUTHOR_NAME='Tarun ji'
AUTHOR_EMAIL='something@mail.com'

# REQUIREMENT_FILE_NAME="requirements.txt"
# HYPHEN_DOT_E="-e ."

# def get_requirement_list()->List[str]:
#     with open(REQUIREMENT_FILE_NAME) as requirement_file:
#         requirement_list=requirement_file.readlines()
#         requirement_list=[requirement_name.replace("\n","")for requirement_name in requirement_list]

#     if(HYPHEN_DOT_E in requirement_list):
#         requirement_list.remove(HYPHEN_DOT_E)
#     return requirement_list

REQUIREMENT_FILE_NAME="requirements.txt"
HYPHEN_E_DOT="-e ."

def get_requirement_list()->List[str]:
    with open(REQUIREMENT_FILE_NAME) as req_file:
        req_list=req_file.readlines()
        req_list=[req_name.replace("\n","")mfor req_name in req_list]
    
    if(HYPHEN_E_DOT in req_list):
        req_list.remove(HYPHEN_E_DOT)
    return req_list


setup(
    name=PROJECT_NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    packages=find_packages(),
    install_requires=get_requirement_list()
)



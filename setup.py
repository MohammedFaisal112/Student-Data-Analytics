from setuptools import find_packages,setup  # It'll Automatically findout all the packages which are avaliable/used in the ML Project/Application/Directory which we're created.
from typing import List   # it's basically for taking an input or returning an output from a function in a form of list.

HYPEN_E_DOT = "-e ."      # Connect setup.py with requirements.txt , Mapped with requirements.txt

def get_required_packages(file_path:str)->List[str]:
    '''
    This Function will take a file as an input with dtype:string and return the List of required_packages.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements  =[req.replace("\n","")for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements
    

# Below is a Metadata : Data which describe about an other data.
setup(
name="studatanalysis",
version="0.0.1",
author="Faisal",
author_email="sayedfaisal8828@gmail.com",
packages=find_packages(),   # It'll directly go and search in which of the folders __init__.py was mentioned and whereever it'll be mention the source folder was considered as a package.
install_requires = get_required_packages("requirements.txt")

)
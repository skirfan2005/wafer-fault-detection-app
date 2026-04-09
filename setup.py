from setuptools import find_packages, setup

from typing import List

HYPEN_E_DOT = '-e.'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines() # ["pandas\n", "numpy\n"]
        requirements = [req.replace("\n", "") for req in requirements]
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    print("requirementst -----> ", requirements)
    return requirements


setup(
    name="Fault detection",
    version="0.0.1",
    author="Shaik_Irfan",
    author_email="irfangpk2005@gmail.com",
    packages=find_packages(),
    install_requirements=get_requirements('requirements.txt')
)
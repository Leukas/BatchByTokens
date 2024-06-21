# setup.py
from setuptools import setup, find_packages
from setuptools.command.install import install


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


install_requires = [
    "datasets==2.17.1",
    "torch==2.2.0",
    "transformers==4.40.0",
]
dependency_links = []


class PostInstall(install):
    @staticmethod
    def post_install():
        pass

    def run(self):
        install.run(self)
        self.execute(
            PostInstall.post_install, [], msg="Running post installation tasks"
        )

setup(
    name="batchbytokens",
    version="0.1.0",
    author="Lukas Edman",
    description="A simple wrapper for the HuggingFace Seq2SeqTrainer that allows for batching by tokens",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leukas/batchbytokens",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    dependency_links=dependency_links,
    cmdclass={"install": PostInstall},
)
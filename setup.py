from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()



setup(
    name="imaginaryNLP",
    version="0.0.2",
    author="Justus-Jonas Erker",
    author_email="j.erker@student.maastrichtuniversity.nl",
    description="Imaginary Embeddings for NLP",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    download_url="https://github.com/Justus-Jonas/nlp-i",
    packages=find_packages(),
    python_requires=">=3.6.0",
    install_requires=[
        'datasets==2.13.1',
        'transformers>=2.0.0',
        'torch==2.0.1',
        'numpy==1.21.5',
        'pandas==1.4.4',
        'tqdm==4.64.1'
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    keywords="PyTorch NLP deep learning Imaginary Embeddings Dialog Systems",
)
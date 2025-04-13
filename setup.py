from setuptools import setup, find_packages

setup(
    name="secure-policy-navigator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "transformers",
        "torch",
        "faiss-cpu",
        "sentence-transformers",
    ],
)

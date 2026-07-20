from setuptools import setup, find_packages

setup(
    name="secure-policy-navigator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=1.3.14",
        "transformers>=5.14.1",
        "torch>=2.13.0",
        "faiss-cpu>=1.14.3",
        "sentence-transformers>=5.6.0",
    ],
)

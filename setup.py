from setuptools import find_packages, setup
# from relevanceai import __version__

core_reqs = [
    "pydantic==2.8.2",
    "requests==2.32.3",
    "httpx==0.27.2",
    "python-dotenv==1.0.1"
]

setup(
    name="relevanceai",
    version="2.0.0",
    url="https://relevanceai.com/",
    author="Relevance AI",
    author_email="jacky@relevanceai.com",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    setup_requires=["wheel"],
    install_requires=core_reqs,
    package_data={"": ["*.ini"]},
    extras_require=dict(),
)
